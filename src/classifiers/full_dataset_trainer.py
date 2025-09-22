"""
Full Dataset Trainer for Wave Height Bias Correction Research

This trainer loads all data into memory and trains models on the complete dataset,
providing better convergence and more robust results for research purposes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import xgboost as xgb

from src.commons.memory_monitor import MemoryMonitor
from src.commons.aws.s3_results_saver import S3ResultsSaver
from src.data_engineering.data_loader import DataLoader
from src.data_engineering.feature_engineer import FeatureEngineer
from src.data_engineering.sampling_manager import SamplingManager
from src.evaluation.metrics_calculator import MetricsCalculator
from src.data_engineering.data_splitter import DataSplitter
from src.commons.preprocessing_manager import PreprocessingManager
from src.classifiers.eqm_corrector import EQMCorrector
from src.classifiers.delta_corrector import DeltaCorrector
from src.commons.region_mapping import RegionMapper

# Set up logger
logger = logging.getLogger(__name__)

from src.evaluation.metrics import evaluate_model
from src.evaluation.diagnostic_plotter import DiagnosticPlotter
from src.evaluation.experiment_logger import ExperimentLogger


def _load_single_file_worker(args):
    """
    Worker function for parallel file loading.
    This function needs to be at module level for multiprocessing.
    
    Args:
        args: Tuple of (file_path, feature_config)
        
    Returns:
        Tuple of (file_path, df, success_flag)
    """
    file_path, feature_config = args
    
    try:
        # Import everything at the top to avoid import issues
        import sys
        import os
        import logging
        
        # Add the project root to the path to ensure imports work
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Now import our modules
        from src.features.helpers import extract_features_from_parquet
        from src.data_engineering.split import stratified_sample_by_location, temporal_sample_within_file
        
        # Load the file
        df = extract_features_from_parquet(file_path, use_dask=False)
        
        # Create lag features BEFORE sampling to preserve temporal sequences
        use_lag_features = feature_config.get("lag_features", {}).get("enabled", False)
        if use_lag_features:
            logger.debug(f"Creating lag features for {file_path} before sampling...")
            lag_config = feature_config.get("lag_features", {}).get("lags", {})
            
            # Check if we have temporal data (time column)
            if "time" in df.columns or "timestamp" in df.columns:
                time_col = "time" if "time" in df.columns else "timestamp"
                
                # Sort by time to ensure proper lag calculation
                df = df.sort([time_col, "lat", "lon"])
                
                # Create lag features for each variable
                for variable, lags in lag_config.items():
                    if variable in df.columns:
                        for lag in lags:
                            lag_col_name = f"{variable}_lag_{lag}h"
                            # Create lag feature by shifting values within each location
                            df = df.with_columns([
                                pl.col(variable).shift(lag).over(["lat", "lon"]).alias(lag_col_name)
                            ])
                    else:
                        logger.warning(f"Variable {variable} not found in data for lag features")
            else:
                logger.warning("Lag features enabled but no time column found. Skipping lag features.")
        
        # Apply sampling AFTER lag features are created
        max_samples = feature_config.get("max_samples_per_file", None)
        sampling_strategy = feature_config.get("sampling_strategy", "none")
        
        if max_samples is not None and sampling_strategy != "none":
            original_size = len(df)
            if original_size <= max_samples:
                pass  # No sampling needed
            else:
                sampling_seed = feature_config.get("sampling_seed", 42)
                
                if sampling_strategy == "random":
                    df = df.sample(n=max_samples, seed=sampling_seed)
                elif sampling_strategy == "per_location":
                    samples_per_location = feature_config.get("samples_per_location", 20)
                    # Use correct column names based on data format
                    location_cols = ["lat", "lon"] if "lat" in df.columns else ["latitude", "longitude"]
                    df = stratified_sample_by_location(df, max_samples, samples_per_location, location_cols, sampling_seed)
                elif sampling_strategy == "temporal":
                    samples_per_hour = feature_config.get("samples_per_hour", 100)
                    df = temporal_sample_within_file(df, samples_per_hour, sampling_seed)
                else:
                    # Default to random sampling
                    df = df.sample(n=max_samples, seed=sampling_seed)
        
        return (file_path, df, True)
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error loading file {file_path}: {e}")
        import traceback
        logging.getLogger(__name__).warning(f"Traceback: {traceback.format_exc()}")
        return (file_path, None, False)


class FullDatasetTrainer:
    """
    Flexible full dataset trainer for wave height bias correction research.
    
    Features:
    - Loads all data into memory for better convergence
    - Configurable train/validation/evaluation splits
    - Multiple model algorithms (XGBoost, Random Forest, Linear models)
    - Early stopping with validation monitoring
    - Comprehensive evaluation and diagnostics
    - Stratified sampling options
    - Feature selection and preprocessing
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the full dataset trainer.
        
        Args:
            config: Configuration dictionary containing all training parameters
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        self.feature_config = config.get("feature_block", {})
        self.evaluation_config = config.get("evaluation", {})
        self.diagnostics_config = config.get("training_diagnostics", {})
        
        # Initialize components
        self.model = None
        self.scaler = None
        self.regional_scaler = None
        self.feature_selector = None
        self.dimension_reducer = None
        self.feature_names = None
        self.selected_features = None
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.regions_train = None
        self.regions_val = None
        self.regions_test = None
        self.coords_train = None
        self.coords_val = None
        self.coords_test = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Store current metrics as class attributes
        self.current_train_metrics = None
        self.current_val_metrics = None
        self.current_test_metrics = None
        
        # Initialize modular components
        self.sampling_manager = SamplingManager(config)
        self.data_loader = DataLoader(config, self.sampling_manager)
        self.feature_engineer = FeatureEngineer(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.data_splitter = DataSplitter(config)
        self.preprocessing_manager = PreprocessingManager(config)
        self.diagnostic_plotter = DiagnosticPlotter(self.config)
        self.experiment_logger = ExperimentLogger(self.config)
        self.memory_monitor = MemoryMonitor(self.config)
        self.s3_results_saver = S3ResultsSaver(self.config)
        
        # Initialize model
        self._initialize_model()
        
        # Preprocessing is now handled by PreprocessingManager
        
        logger.info(f"FullDatasetTrainer initialized with model: {self.model_config.get('type', 'xgb')}")
        
        # Set up memory monitoring with experiment
        if hasattr(self.experiment_logger, 'experiment'):
            self.memory_monitor.set_experiment(self.experiment_logger.experiment)
        
        # Log initial memory info
        self.memory_monitor.log_comprehensive_memory("initialization")
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage for monitoring."""
        self.memory_monitor.log_memory_usage(stage)
    
    def _log_system_memory_info(self):
        """Log system-wide memory information."""
        self.memory_monitor.log_system_memory()
    
    @property
    def train_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return self.current_train_metrics or {}
    
    @property
    def val_metrics(self) -> Dict[str, float]:
        """Get current validation metrics."""
        return self.current_val_metrics or {}
    
    @property
    def test_metrics(self) -> Dict[str, float]:
        """Get current test metrics."""
        return self.current_test_metrics or {}
    
    @property
    def all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all current metrics."""
        return {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'test': self.test_metrics
        }
    
    def _initialize_model(self):
        """Initialize the model based on configuration."""
        model_type = self.model_config.get("type", "xgb")
        
        if model_type == "xgb":
            self.model = xgb.XGBRegressor(
                n_estimators=self.model_config.get("n_estimators", 1000),
                max_depth=self.model_config.get("max_depth", 6),
                learning_rate=self.model_config.get("learning_rate", 0.1),
                subsample=self.model_config.get("subsample", 0.8),
                colsample_bytree=self.model_config.get("colsample_bytree", 0.8),
                reg_alpha=self.model_config.get("reg_alpha", 0.01),
                reg_lambda=self.model_config.get("reg_lambda", 1.0),
                min_child_weight=self.model_config.get("min_child_weight", 1),
                random_state=self.model_config.get("random_state", 42),
                n_jobs=self.model_config.get("n_jobs", -1),
                early_stopping_rounds=self.model_config.get("early_stopping_rounds", 50),
                eval_metric=self.model_config.get("eval_metric", ["rmse", "mae"]),
                objective=self.model_config.get("objective", "reg:squarederror"),
                tree_method=self.model_config.get("tree_method", "hist"),
                max_bin=self.model_config.get("max_bin", 256),
                gamma=self.model_config.get("gamma", 0.5)
            )
        elif model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=self.model_config.get("n_estimators", 100),
                max_depth=self.model_config.get("max_depth", None),
                min_samples_split=self.model_config.get("min_samples_split", 2),
                min_samples_leaf=self.model_config.get("min_samples_leaf", 1),
                random_state=self.model_config.get("random_state", 42),
                n_jobs=self.model_config.get("n_jobs", -1)
            )
        elif model_type == "elasticnet":
            self.model = ElasticNet(
                alpha=self.model_config.get("alpha", 1.0),
                l1_ratio=self.model_config.get("l1_ratio", 0.5),
                max_iter=self.model_config.get("max_iter", 1000),
                random_state=self.model_config.get("random_state", 42)
            )
        elif model_type == "lasso":
            self.model = Lasso(
                alpha=self.model_config.get("alpha", 1.0),
                max_iter=self.model_config.get("max_iter", 1000),
                random_state=self.model_config.get("random_state", 42)
            )
        elif model_type == "ridge":
            self.model = Ridge(
                alpha=self.model_config.get("alpha", 1.0),
                max_iter=self.model_config.get("max_iter", 1000),
                random_state=self.model_config.get("random_state", 42)
            )
        elif model_type == "eqm":
            # EQM is not a traditional ML model, so we'll handle it differently
            eqm_config = self.model_config.get("eqm", {})
            self.model = EQMCorrector(
                quantile_resolution=eqm_config.get("quantile_resolution", 1000),
                extrapolation_method=eqm_config.get("extrapolation_method", "constant"),
                kde_bandwidth=eqm_config.get("kde_bandwidth", None)
            )
            self.eqm_variables = eqm_config.get("variables", ["VHM0"])
            logger.info(f"Initialized EQM corrector for variables: {self.eqm_variables}")
        elif model_type == "delta":
            # Delta corrector is a simple bias correction method
            delta_config = self.model_config.get("delta", {})
            delta_method = delta_config.get("method", "mean_bias")
            self.model = DeltaCorrector(method=delta_method)
            self.delta_variables = delta_config.get("variables", ["VHM0"])
            self.delta_method = delta_method
            logger.info(f"Initialized Delta corrector for variables: {self.delta_variables} (method: {self.delta_method})")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _validate_config(self):
        """Validate configuration for logical consistency."""
        scaler_type = self.feature_config.get("scaler", "standard")
        use_regional_scaling = self.feature_config.get("regional_scaling", {}).get("enabled", False)
        
        # Check for conflicting configurations
        if (scaler_type is None or scaler_type == "null") and use_regional_scaling:
            logger.warning("Configuration conflict: scaler is 'null' but regional_scaling is enabled.")
            logger.warning("Regional scaling requires a base scaler. Consider:")
            logger.warning("  - Set scaler to 'standard', 'robust', or 'minmax' for regional scaling")
            logger.warning("  - Set regional_scaling.enabled to false for no scaling")
    
    # def _initialize_preprocessing(self):
    #     """Initialize preprocessing components."""
    #     # Validate configuration first
    #     self._validate_config()
        
    #     # Scaler
    #     scaler_type = self.feature_config.get("scaler", "standard")
    #     use_regional_scaling = self.feature_config.get("regional_scaling", {}).get("enabled", False)
        
    #     # Check if scaler is null first
    #     if scaler_type is None or scaler_type == "null":
    #         if use_regional_scaling:
    #             logger.warning("Cannot use regional scaling with null scaler. Disabling regional scaling.")
    #             use_regional_scaling = False
            
    #         logger.info("No scaling applied - using raw features")
    #         self.scaler = None
    #         self.regional_scaler = None
    #     elif use_regional_scaling:
    #         logger.info("Using regional scaling for geographic regions")
    #         self.regional_scaler = RegionalScaler(
    #             base_scaler=scaler_type,
    #             region_column="atlantic_region"
    #         )
    #         # No need for regular scaler when using regional scaling
    #         self.scaler = None
    #     else:
    #         logger.info(f"Using {scaler_type} scaling")
    #         if scaler_type == "standard":
    #             self.scaler = StandardScaler()
    #         elif scaler_type == "robust":
    #             self.scaler = RobustScaler()
    #         elif scaler_type == "minmax":
    #             self.scaler = MinMaxScaler()
    #         else:
    #             logger.warning(f"Unknown scaler type '{scaler_type}', using StandardScaler")
    #             self.scaler = StandardScaler()
    #         # No regional scaler when using standard scaling
    #         self.regional_scaler = None
        
    #     # Feature selection
    #     if self.feature_config.get("feature_selection", {}).get("enabled", False):
    #         selection_type = self.feature_config["feature_selection"].get("type", "kbest")
    #         k = self.feature_config["feature_selection"].get("k", 100)
            
    #         if selection_type == "kbest":
    #             self.feature_selector = SelectKBest(score_func=f_regression, k=k)
    #         elif selection_type == "rfe":
    #             # RFE will be initialized after model is available
    #             pass
        
    #     # Dimension reduction
    #     if self.feature_config.get("dimension_reduction", {}).get("enabled", False):
    #         reduction_type = self.feature_config["dimension_reduction"].get("type", "pca")
    #         n_components = self.feature_config["dimension_reduction"].get("n_components", 50)
            
    #         if reduction_type == "pca":
    #             self.dimension_reducer = PCA(n_components=n_components, random_state=42)
    #         elif reduction_type == "svd":
    #             self.dimension_reducer = TruncatedSVD(n_components=n_components, random_state=42)
    
    def load_data(self, data_paths: Union[str, List[str]], target_column: str = "vhm0_y") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load data using the DataLoader."""
        self._log_memory_usage("before loading data")
        
        # Use DataLoader to load and combine data (with per-file sampling)
        combined_df, successful_files = self.data_loader.load_data(data_paths, target_column)
        
        # Prepare features using FeatureEngineer
        X, y, regions, coords = self.feature_engineer.prepare_features(combined_df, target_column)
        
        # Get feature names from FeatureEngineer
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Log dataset info to Comet
        dataset_info = self._prepare_dataset_info(X, y, successful_files)
        self.experiment_logger.log_dataset_info(dataset_info)
        
        self._log_memory_usage("after loading data")
        return X, y, regions, coords, successful_files
    
    def _prepare_dataset_info(self, X: np.ndarray, y: np.ndarray, successful_files: List[str]) -> Dict[str, Any]:
        """Prepare dataset information for logging."""
        dataset_info = {
            "total_samples": len(X),
            "total_features": X.shape[1],
            "files_processed": len(successful_files),
            "target_stats": {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y))
            }
        }
        
        # Add sampling info using SamplingManager
        sampling_reduction = self.sampling_manager.estimate_sampling_reduction(len(successful_files), len(X))
        if sampling_reduction["reduction_percent"] > 0:
            dataset_info["sampling_info"] = sampling_reduction
        
        return dataset_info
    
    def _apply_regional_weights(self, y: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """
        Apply regional weights to training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            regions: Region information for each sample
            
        Returns:
            Sample weights array
        """
        if not self.feature_config.get("regional_weighting", {}).get("enabled", False):
            # Return uniform weights if regional weighting is disabled
            sample_weights = np.ones(len(y))
            return sample_weights
        
        weights_config = self.feature_config.get("regional_weighting", {}).get("weights", {})
        
        # Create weight array
        sample_weights = np.ones(len(y))
        
        # Apply weights based on region
        for region, weight in weights_config.items():
            region_mask = regions == region
            region_count = np.sum(region_mask)
            if region_count > 0:
                sample_weights[region_mask] = weight
                logger.info(f"Applied weight {weight} to {region_count:,} {region} samples")
        
        # Log weight statistics
        total_weighted_samples = np.sum(sample_weights)
        logger.info(f"Regional weighting applied - Total weighted samples: {total_weighted_samples:,.0f}")
        
        # Log effective sample counts per region
        for region_id in weights_config.keys():
            region_mask = regions == region_id
            if np.sum(region_mask) > 0:
                region_count = np.sum(region_mask)
                effective_count = np.sum(sample_weights[region_mask])
                region_name = RegionMapper.get_display_name(region_id)
                logger.info(f"  {region_name}: {region_count:,} samples × {weights_config[region_id]} = {effective_count:,.0f} effective samples")
        
        return sample_weights
    
    def _calculate_regional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, regions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per region using MetricsCalculator."""
        return self.metrics_calculator.calculate_regional_metrics(y_true, y_pred, regions)

    def _calculate_sea_bin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for different sea state bins using MetricsCalculator."""
        return self.metrics_calculator.calculate_sea_bin_metrics(y_true, y_pred, enable_logging=True)

    def split_data(self, X: np.ndarray, y: np.ndarray, regions: np.ndarray = None, coords: np.ndarray = None, file_paths: List[str] = None) -> None:
        """
        Split data into train/validation/test sets using DataSplitter.
        
        Args:
            X: Feature matrix
            y: Target vector
            regions: Regional classification array (optional)
            coords: Coordinate array (lat, lon) (optional)
            file_paths: List of file paths (required for year-based splitting)
        """
        self._log_memory_usage("before splitting data")
        
        # Use DataSplitter to perform the split
        split_data = self.data_splitter.split_data(X, y, regions, coords, file_paths)
        
        # Store splits as instance variables
        self.X_train = split_data['X_train']
        self.X_val = split_data['X_val']
        self.X_test = split_data['X_test']
        self.y_train = split_data['y_train']
        self.y_val = split_data['y_val']
        self.y_test = split_data['y_test']
        self.regions_train = split_data['regions_train']
        self.regions_val = split_data['regions_val']
        self.regions_test = split_data['regions_test']
        self.coords_train = split_data['coords_train']
        self.coords_val = split_data['coords_val']
        self.coords_test = split_data['coords_test']
        
        # Log split information
        self.data_splitter.log_split_info(split_data)
        
        # Apply regional weighting to training data
        self.sample_weights = self._apply_regional_weights(
            self.y_train, self.regions_train
        )
        
        self._log_memory_usage("after splitting data")
    
    # def _split_by_years(self, X: np.ndarray, y: np.ndarray, regions: np.ndarray, coords: np.ndarray, file_paths: List[str], split_config: Dict[str, Any]) -> None:
    #     """
    #     Split data by years: 2017-2022 for train/val, 2023 for test.
        
    #     Args:
    #         X: Feature matrix
    #         y: Target vector
    #         file_paths: List of file paths
    #         split_config: Split configuration
    #     """
    #     train_end_year = split_config.get("train_end_year", 2022)
    #     test_start_year = split_config.get("test_start_year", 2023)
    #     val_months = split_config.get("val_months", [10, 11, 12])  # Default: Oct-Dec
    #     eval_months = split_config.get("eval_months", list(range(1, 13)))  # Default: All months
        
    #     logger.info(f"Year-based split: Train up to {train_end_year}, Val months {val_months} of {train_end_year}, Test months {eval_months} from {test_start_year}")
        
    #     # Create file-to-indices mapping
    #     file_indices = {}
    #     current_idx = 0
        
    #     for file_path in file_paths:
    #         try:
    #             # Extract date from filename
    #             date = extract_date_from_filename(file_path)
    #             year = date.year
    #             month = date.month
                
    #             # Count samples in this file (we need to estimate this)
    #             # For now, we'll assume equal distribution across files
    #             samples_per_file = len(X) // len(file_paths)
    #             file_indices[file_path] = (current_idx, current_idx + samples_per_file)
    #             current_idx += samples_per_file
                
    #             # Debug: log first few files
    #             if len(file_indices) <= 5:
    #                 logger.info(f"File: {file_path} -> Year: {year}, Month: {month}")
                
    #         except Exception as e:
    #             logger.warning(f"Could not extract date from {file_path}: {e}")
    #             continue
        
    #     # Log total files and their distribution
    #     logger.info(f"Total files processed: {len(file_indices)}")
    #     logger.info(f"Total samples: {len(X)}")
    #     logger.info(f"Estimated samples per file: {samples_per_file}")
        
    #     # Create test mask based on eval_months
    #     test_mask = np.zeros(len(X), dtype=bool)
        
    #     for file_path, (start_idx, end_idx) in file_indices.items():
    #         try:
    #             date = extract_date_from_filename(file_path)
    #             year = date.year
    #             month = date.month
                
    #             if year >= test_start_year and month in eval_months:
    #                 test_mask[start_idx:end_idx] = True
                    
    #         except Exception as e:
    #             logger.warning(f"Error processing {file_path}: {e}")
    #             continue
        
    #     # Split train/val data by years and months
    #     train_files = []
    #     val_files = []
        
    #     for file_path, (start_idx, end_idx) in file_indices.items():
    #         try:
    #             date = extract_date_from_filename(file_path)
    #             year = date.year
    #             month = date.month
                
    #             if year <= train_end_year:
    #                 # Determine if this file should be train or validation
    #                 if year < train_end_year:
    #                     # All years before the last year go to training
    #                     train_files.append((file_path, start_idx, end_idx))
    #                     logger.debug(f"Train file: {file_path} -> {year}-{month:02d}")
    #                 elif year == train_end_year:
    #                     # Last year: check if month is in validation months
    #                     if month in val_months:
    #                         val_files.append((file_path, start_idx, end_idx))
    #                         logger.info(f"Validation file: {file_path} -> {year}-{month:02d}")
    #                     else:
    #                         train_files.append((file_path, start_idx, end_idx))
    #                         logger.debug(f"Train file: {file_path} -> {year}-{month:02d}")
                        
    #         except Exception as e:
    #             logger.warning(f"Error processing {file_path}: {e}")
    #             continue
        
    #     # Create masks for train and validation
    #     train_mask = np.zeros(len(X), dtype=bool)
    #     val_mask = np.zeros(len(X), dtype=bool)
        
    #     for _, start_idx, end_idx in train_files:
    #         train_mask[start_idx:end_idx] = True
            
    #     for _, start_idx, end_idx in val_files:
    #         val_mask[start_idx:end_idx] = True
        
    #     # Store splits
    #     self.X_train = X[train_mask]
    #     self.X_val = X[val_mask]
    #     self.X_test = X[test_mask]
        
    #     self.y_train = y[train_mask]
    #     self.y_val = y[val_mask]
    #     self.y_test = y[test_mask]
        
    #     # Store region information (always available now)
    #     self.regions_train = regions[train_mask]
    #     self.regions_val = regions[val_mask]
    #     self.regions_test = regions[test_mask]
    #     # Log regional distribution with readable names
    #     train_regions, train_counts = np.unique(self.regions_train, return_counts=True)
    #     val_regions, val_counts = np.unique(self.regions_val, return_counts=True)
    #     test_regions, test_counts = np.unique(self.regions_test, return_counts=True)
        
    #     train_names = [RegionMapper.get_display_name(rid) for rid in train_regions]
    #     val_names = [RegionMapper.get_display_name(rid) for rid in val_regions]
    #     test_names = [RegionMapper.get_display_name(rid) for rid in test_regions]
        
    #     logger.info(f"Regional distribution - Train: {dict(zip(train_names, train_counts))}")
    #     logger.info(f"Regional distribution - Val: {dict(zip(val_names, val_counts))}")
    #     logger.info(f"Regional distribution - Test: {dict(zip(test_names, test_counts))}")
        
    #     # Store coordinate information if available
    #     if coords is not None:
    #         self.coords_train = coords[train_mask]
    #         self.coords_val = coords[val_mask]
    #         self.coords_test = coords[test_mask]
        
    #     # Apply regional weighting to training data
    #     self.sample_weights = self._apply_regional_weights(
    #         self.X_train, self.y_train, self.regions_train
    #     )
        
    #     # Log split statistics BEFORE deleting file_indices
    #     train_years_months = []
    #     val_years_months = []
    #     test_years_months = []
        
    #     for file_path, (start_idx, end_idx) in file_indices.items():
    #         try:
    #             date = extract_date_from_filename(file_path)
    #             year = date.year
    #             month = date.month
                
    #             if year < train_end_year:  # Training years
    #                 train_years_months.append((year, month))
    #             elif year == train_end_year and month in val_months:  # Validation months
    #                 val_years_months.append((year, month))
    #             elif year >= test_start_year and month in eval_months:  # Test months
    #                 test_years_months.append((year, month))
    #         except:
    #             continue
        
    #     # Group by year for cleaner logging
    #     train_years = sorted(set(year for year, month in train_years_months))
    #     val_years = sorted(set(year for year, month in val_years_months))
    #     test_years = sorted(set(year for year, month in test_years_months))
        
    #     logger.info(f"Year-based split completed:")
    #     logger.info(f"  Train: {len(self.X_train)} samples, years: {train_years}")
    #     logger.info(f"  Val: {len(self.X_val)} samples, months {val_months} of {val_years}")
    #     logger.info(f"  Test: {len(self.X_test)} samples, months {eval_months} of {test_years}")
        
    #     # Check for empty splits
    #     if len(self.X_val) == 0:
    #         logger.warning("Validation set is empty! Check your val_months configuration.")
    #     if len(self.X_test) == 0:
    #         logger.warning("Test set is empty! Check your eval_months and test_start_year configuration.")
        
    #     # Log split information to Comet
    #     split_info = {
    #         "train_samples": len(self.X_train),
    #         "val_samples": len(self.X_val),
    #         "test_samples": len(self.X_test),
    #         "train_ratio": len(self.X_train) / (len(self.X_train) + len(self.X_val) + len(self.X_test)) if (len(self.X_train) + len(self.X_val) + len(self.X_test)) > 0 else 0,
    #         "val_ratio": len(self.X_val) / (len(self.X_train) + len(self.X_val) + len(self.X_test)) if (len(self.X_train) + len(self.X_val) + len(self.X_test)) > 0 else 0,
    #         "test_ratio": len(self.X_test) / (len(self.X_train) + len(self.X_val) + len(self.X_test)) if (len(self.X_train) + len(self.X_val) + len(self.X_test)) > 0 else 0
    #     }
    #     self.experiment_logger.log_split_info(split_info)
        
    #     # 🚀 MEMORY OPTIMIZATION: Delete masks and file indices AFTER logging
    #     del train_mask, val_mask, test_mask, file_indices
    #     import gc; gc.collect()
    
    def preprocess_data(self) -> None:
        """Apply preprocessing to the data splits using PreprocessingManager."""
        self._log_memory_usage("before preprocessing")
        
        # Use PreprocessingManager to preprocess data splits
        self.X_train, self.X_val, self.X_test = self.preprocessing_manager.preprocess_splits_memory_efficient(
            self.X_train, self.X_val, self.X_test,
            self.regions_train, self.regions_val, self.regions_test
        )
        
        # Log preprocessing info
        self.preprocessing_manager.log_preprocessing_info()
        
        self._log_memory_usage("after preprocessing")
    
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model with early stopping and validation monitoring.
        
        Returns:
            Dictionary containing training results and metrics
        """
        self._log_memory_usage("before training")
        logger.info("Starting model training...")
        
        # Handle EQM training differently
        if isinstance(self.model, EQMCorrector):
            logger.info("Training EQM corrector...")
            # For EQM, we need to create a DataFrame with model predictions and observed values
            # We'll use the training data to fit the EQM corrector
            
            # Create training DataFrame for EQM
            train_df = pl.DataFrame({
                'VHM0': self.y_train,  # Model predictions (target variable)
                'corrected_VHM0': self.y_train  # For EQM, we need observed values
            })
            
            # Note: In a real scenario, you'd have both model predictions and observed values
            # For now, we'll use the same values as a placeholder
            # TODO: This needs to be updated when you have actual model vs observed data
            
            self.model.fit(train_df, self.eqm_variables, corrected_suffix="corrected_")
            logger.info("EQM corrector fitted successfully")
            
            # For EQM, we don't have traditional training metrics
            train_metrics = {'rmse': 0.0, 'mae': 0.0, 'pearson': 1.0, 'bias': 0.0}
            val_metrics = {'rmse': 0.0, 'mae': 0.0, 'pearson': 1.0, 'bias': 0.0}
            
            # Log EQM statistics
            eqm_stats = self.model.get_correction_stats()
            for var, stats in eqm_stats.items():
                logger.info(f"EQM {var}: bias={stats['bias']:.4f}, relative_bias={stats['bias_relative']:.1f}%")
            
            return {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'training_history': self.training_history,
                'eqm_stats': eqm_stats
            }
        
        # Handle Delta corrector training
        elif isinstance(self.model, DeltaCorrector):
            logger.info("Training Delta corrector...")
            # For Delta, we need to create a DataFrame with model predictions and observed values
            
            # Create training DataFrame for Delta
            train_df = pl.DataFrame({
                'VHM0': self.y_train,  # Model predictions (target variable)
                'corrected_VHM0': self.y_train  # For Delta, we need observed values
            })
            
            # Note: In a real scenario, you'd have both model predictions and observed values
            # For now, we'll use the same values as a placeholder
            # TODO: This needs to be updated when you have actual model vs observed data
            
            self.model.fit(train_df, self.delta_variables, corrected_suffix="corrected_")
            logger.info("Delta corrector fitted successfully")
            
            # For Delta, we don't have traditional training metrics
            train_metrics = {'rmse': 0.0, 'mae': 0.0, 'pearson': 1.0, 'bias': 0.0}
            val_metrics = {'rmse': 0.0, 'mae': 0.0, 'pearson': 1.0, 'bias': 0.0}
            
            # Log Delta statistics
            delta_stats = self.model.get_correction_stats()
            for var, stats in delta_stats.items():
                logger.info(f"Delta {var}: bias={stats['bias']:.4f} (method: {stats['method']})")
            
            return {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'training_history': self.training_history,
                'delta_stats': delta_stats
            }
        
        # Train traditional ML models
        if hasattr(self.model, 'early_stopping_rounds'):
            # XGBoost with early stopping
            if len(self.X_val) > 0:
                self.model.fit(
                    self.X_train, self.y_train,
                    sample_weight=self.sample_weights,  # Add sample weights
                    eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                    verbose=self.model_config.get("verbose", False)
                )
                
                # Extract training history
                self.training_history['train_loss'] = self.model.evals_result_['validation_0']['rmse']
                self.training_history['val_loss'] = self.model.evals_result_['validation_1']['rmse']
            else:
                # No validation set, train without early stopping
                logger.warning("No validation set available, training without early stopping")
                self.model.fit(
                    self.X_train, self.y_train,
                    sample_weight=self.sample_weights,  # Add sample weights
                    eval_set=[(self.X_train, self.y_train)],
                    verbose=self.model_config.get("verbose", False)
                )
                
                # Extract training history
                self.training_history['train_loss'] = self.model.evals_result_['validation_0']['rmse']
            
        else:
            # Other models (check if they support sample_weight)
            try:
                self.model.fit(self.X_train, self.y_train, sample_weight=self.sample_weights)
            except TypeError:
                # Model doesn't support sample_weight, train without it
                logger.warning("Model doesn't support sample_weight, training without regional weighting")
                self.model.fit(self.X_train, self.y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(self.X_train)
        train_metrics = evaluate_model(self.y_train, train_pred)
        
        # Calculate regional training metrics (always available now)
        logger.info("Calculating regional training metrics...")
        regional_train_metrics = self._calculate_regional_metrics(
            self.y_train, train_pred, self.regions_train
        )
        
        # Calculate sea-bin training metrics
        sea_bin_train_metrics = self._calculate_sea_bin_metrics(self.y_train, train_pred)
        
        # 🚀 MEMORY OPTIMIZATION: Delete train predictions immediately
        del train_pred
        import gc; gc.collect()
        
        # Calculate validation metrics if validation set exists
        if len(self.X_val) > 0:
            val_pred = self.model.predict(self.X_val)
            val_metrics = evaluate_model(self.y_val, val_pred)
            
            # Calculate regional validation metrics (always available now)
            logger.info("Calculating regional validation metrics...")
            regional_val_metrics = self._calculate_regional_metrics(
                self.y_val, val_pred, self.regions_val
            )
            
            # Calculate sea-bin validation metrics
            sea_bin_val_metrics = self._calculate_sea_bin_metrics(self.y_val, val_pred)
            
            # 🚀 MEMORY OPTIMIZATION: Delete val predictions immediately
            del val_pred
            gc.collect()
        else:
            val_metrics = {'rmse': 0.0, 'mae': 0.0, 'bias': 0.0, 'pearson': 0.0, 'snr': 0.0, 'snr_db': 0.0}
            regional_val_metrics = {}
        
        # Store current metrics as class attributes
        self.current_train_metrics = train_metrics
        self.current_val_metrics = val_metrics
        
        self.training_history['train_metrics'].append(train_metrics)
        self.training_history['val_metrics'].append(val_metrics)
        
        logger.info(f"Training completed - Train RMSE: {self.train_metrics.get('rmse', 0):.4f}, Val RMSE: {self.val_metrics.get('rmse', 0):.4f}")
        logger.info(f"Training SNR - Train: {self.train_metrics.get('snr', 0):.1f} ({self.train_metrics.get('snr_db', 0):.1f} dB), Val: {self.val_metrics.get('snr', 0):.1f} ({self.val_metrics.get('snr_db', 0):.1f} dB)")
        
        # Log training results to Comet
        self.experiment_logger.log_training_results(train_metrics, val_metrics)
        
        self._log_memory_usage("after training")
        return {
            'train_metrics': train_metrics,
            'regional_train_metrics': regional_train_metrics,
            'sea_bin_train_metrics': sea_bin_train_metrics,
            'val_metrics': val_metrics,
            'regional_val_metrics': regional_val_metrics,
            'sea_bin_val_metrics': sea_bin_val_metrics if len(self.X_val) > 0 else {},
            'training_history': self.training_history
        }

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on test set.
        
        Returns:
            Dictionary containing evaluation results
        """
        self._log_memory_usage("before evaluation")
        logger.info("Evaluating model on test set...")
        
        # Check if test set exists
        if len(self.X_test) == 0:
            logger.warning("Test set is empty! Cannot evaluate model.")
            return {
                'test_metrics': {'rmse': 0.0, 'mae': 0.0, 'bias': 0.0, 'pearson': 0.0},
                'diagnostic_plots': None
            }
        
        # Make predictions
        if isinstance(self.model, EQMCorrector):
            # For EQM, we need to create a DataFrame with the test data
            test_df = pl.DataFrame({
                'VHM0': self.y_test  # Model predictions to be corrected
            })
            
            # Apply EQM correction
            corrected_df = self.model.predict(test_df, self.eqm_variables)
            test_pred = corrected_df['eqm_corrected_VHM0'].to_numpy()
            
            logger.info(f"Applied EQM correction to {len(test_pred)} test samples")
        elif isinstance(self.model, DeltaCorrector):
            # For Delta, we need to create a DataFrame with the test data
            test_df = pl.DataFrame({
                'VHM0': self.y_test  # Model predictions to be corrected
            })
            
            # Apply Delta correction
            corrected_df = self.model.predict(test_df, self.delta_variables)
            test_pred = corrected_df['delta_corrected_VHM0'].to_numpy()
            
            logger.info(f"Applied Delta correction to {len(test_pred)} test samples")
        else:
            # Traditional ML model prediction
            test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        test_metrics = evaluate_model(self.y_test, test_pred)
        
        # Calculate regional test metrics (always available now)
        logger.info("Calculating regional test metrics...")
        regional_test_metrics = self._calculate_regional_metrics(
            self.y_test, test_pred, self.regions_test
        )
        
        # Calculate sea-bin test metrics
        sea_bin_test_metrics = self._calculate_sea_bin_metrics(self.y_test, test_pred)
        
        # Store current test metrics as class attribute
        self.current_test_metrics = test_metrics
        
        # Store regional and sea-bin metrics as attributes for DiagnosticPlotter
        self.regional_test_metrics = regional_test_metrics
        self.sea_bin_test_metrics = sea_bin_test_metrics
        
        # Create diagnostic plots
        if self.diagnostics_config.get("enabled", False):
            self.diagnostic_plotter.create_diagnostic_plots(self, test_pred)
            # Log diagnostic plots to Comet
            plots_dir = Path(self.diagnostics_config.get("plots_save_path", "diagnostic_plots"))
            self.experiment_logger.log_diagnostic_plots(plots_dir)
        
        # 🚀 MEMORY OPTIMIZATION: Delete test predictions after plotting
        del test_pred
        import gc; gc.collect()
        
        logger.info(f"Test evaluation - RMSE: {self.test_metrics.get('rmse', 0):.4f}, MAE: {self.test_metrics.get('mae', 0):.4f}, Pearson: {self.test_metrics.get('pearson', 0):.4f}")
        logger.info(f"Test SNR: {self.test_metrics.get('snr', 0):.1f} ({self.test_metrics.get('snr_db', 0):.1f} dB)")
        
        # Log evaluation results to Comet
        self.experiment_logger.log_evaluation_results(test_metrics)
        
        # 🚀 MEMORY OPTIMIZATION: Return only metrics, not data
        self._log_memory_usage("after evaluation")
        return {
            'test_metrics': test_metrics,
            'regional_test_metrics': regional_test_metrics,
            'sea_bin_test_metrics': sea_bin_test_metrics
            # Removed 'predictions' and 'actual' to save memory
        } 
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model and preprocessing components."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if isinstance(self.model, EQMCorrector):
            # EQM has its own save method
            self.model.save_model(str(save_path / "eqm_model.pkl"))
            # Also save as regular pickle for compatibility
            joblib.dump(self.model, save_path / "model.pkl")
        elif isinstance(self.model, DeltaCorrector):
            # Delta has its own save method
            self.model.save_model(str(save_path / "delta_model.pkl"))
            # Also save as regular pickle for compatibility
            joblib.dump(self.model, save_path / "model.pkl")
        else:
            joblib.dump(self.model, save_path / "model.pkl")
        
        # Save preprocessing components
        if self.scaler is not None:
            joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        if self.regional_scaler is not None:
            joblib.dump(self.regional_scaler, save_path / "regional_scaler.pkl")
        
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, save_path / "feature_selector.pkl")
        
        if self.dimension_reducer is not None:
            joblib.dump(self.dimension_reducer, save_path / "dimension_reducer.pkl")
        
        # Save feature names and selected features
        joblib.dump(self.feature_names, save_path / "feature_names.pkl")
        joblib.dump(self.selected_features, save_path / "selected_features.pkl")
        
        # Save training history
        joblib.dump(self.training_history, save_path / "training_history.pkl")
        
        logger.info(f"Model and components saved to {save_path}")
        
        # Log model artifacts to Comet
        self.experiment_logger.log_model_artifacts(save_path)
    
    def load_model(self, load_path: str) -> None:
        """Load a trained model and preprocessing components."""
        load_path = Path(load_path)
        
        # Load model
        self.model = joblib.load(load_path / "model.pkl")
        
        # Load preprocessing components
        if (load_path / "scaler.pkl").exists():
            self.scaler = joblib.load(load_path / "scaler.pkl")
        else:
            self.scaler = None
        
        if (load_path / "regional_scaler.pkl").exists():
            self.regional_scaler = joblib.load(load_path / "regional_scaler.pkl")
        else:
            self.regional_scaler = None
        
        if (load_path / "feature_selector.pkl").exists():
            self.feature_selector = joblib.load(load_path / "feature_selector.pkl")
        
        if (load_path / "dimension_reducer.pkl").exists():
            self.dimension_reducer = joblib.load(load_path / "dimension_reducer.pkl")
        
        # Load feature names and selected features
        self.feature_names = joblib.load(load_path / "feature_names.pkl")
        self.selected_features = joblib.load(load_path / "selected_features.pkl")
        
        # Load training history
        if (load_path / "training_history.pkl").exists():
            self.training_history = joblib.load(load_path / "training_history.pkl")
        
        logger.info(f"Model and components loaded from {load_path}")
    
    def save_results_to_s3(self, results: Dict[str, Any], model_path: str = None, plots_dir: str = None) -> Dict[str, bool]:
        """
        Save training results, model, and plots to S3.
        
        Args:
            results: Results dictionary to save
            model_path: Path to saved model directory
            plots_dir: Path to diagnostic plots directory
            
        Returns:
            Dictionary with upload status for each component
        """
        if not self.s3_results_saver.enabled:
            logger.warning("S3 saving is disabled. Enable it in config to save results to S3.")
            return {}
        
        upload_results = {}
        
        logger.info(f"Saving results to S3: {self.s3_results_saver.prefix}")
        
        # Save results JSON
        logger.info("Uploading results JSON to S3...")
        upload_results['results_json'] = self.s3_results_saver.save_results_json(results)
        
        # Save model artifacts if model path provided
        if model_path:
            logger.info("Uploading model artifacts to S3...")
            model_uploads = self.s3_results_saver.save_model_artifacts(model_path)
            upload_results['model_artifacts'] = model_uploads
        
        # Save diagnostic plots if plots directory provided
        if plots_dir:
            logger.info("Uploading diagnostic plots to S3...")
            plots_uploads = self.s3_results_saver.save_diagnostic_plots(plots_dir)
            upload_results['diagnostic_plots'] = plots_uploads
        
        # Log summary
        total_uploads = sum(1 for status in upload_results.values() if status)
        total_attempts = len(upload_results)
        
        if total_uploads == total_attempts:
            logger.info(f"✅ All results successfully uploaded to S3")
        else:
            logger.warning(f"⚠️  {total_uploads}/{total_attempts} uploads successful")
        
        return upload_results
    
    def end_experiment(self):
        """End the Comet ML experiment."""
        # Log final memory usage
        self.memory_monitor.log_comprehensive_memory("experiment_end")
        
        self.experiment_logger.end_experiment()
