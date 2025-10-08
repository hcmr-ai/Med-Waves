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
from src.classifiers.sample_weighting import SampleWeighting

# Set up logger
logger = logging.getLogger(__name__)

from src.evaluation.metrics import evaluate_model
from src.evaluation.diagnostic_plotter import DiagnosticPlotter
from src.evaluation.experiment_logger import ExperimentLogger
from src.classifiers.helpers import reconstruct_vhm0_values

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
        self.diagnostics_config = config.get("diagnostics", {})
        
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
        
        # Raw input data for baseline metrics
        self.vhm0_x_train = None
        self.vhm0_x_val = None
        self.vhm0_x_test = None
        
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
        self.sample_weighting = SampleWeighting(self.feature_config)
        self.diagnostic_plotter = DiagnosticPlotter(self.config)
        self.experiment_logger = ExperimentLogger(self.config)
        self.memory_monitor = MemoryMonitor(self.config)
        self.s3_results_saver = S3ResultsSaver(self.config)

        self._validate_config()
        
        # Initialize model
        self._initialize_model()
        
        # Preprocessing is now handled by PreprocessingManager
        
        logger.info(f"FullDatasetTrainer initialized with model: {self.model_config.get('type', 'xgb')}")
        
        # Log weighting configuration
        weighting_info = self.sample_weighting.get_weight_function_info()
        if weighting_info:
            logger.info("Sample weighting configuration:")
            for weight_type, description in weighting_info.items():
                logger.info(f"  {weight_type}: {description}")
        else:
            logger.info("No sample weighting enabled")
        
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
    
    @property
    def predict_bias(self) -> bool:
        """Check if we're predicting bias instead of vhm0_y directly."""
        return self.feature_config.get("predict_bias", False)
    
    @property
    def predict_bias_log_space(self) -> bool:
        """Check if we're using log-space bias prediction."""
        return self.feature_config.get("predict_bias_log_space", False)
    
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
        scaler_type = self.feature_config.get("scaler", None)
        use_regional_scaling = self.feature_config.get("regional_scaling", {}).get("enabled", False)
        
        # Check for conflicting configurations
        if (scaler_type is None or scaler_type == "null") and use_regional_scaling:
            logger.warning("Configuration conflict: scaler is 'null' but regional_scaling is enabled.")
            logger.warning("Regional scaling requires a base scaler. Consider:")
            logger.warning("  - Set scaler to 'standard', 'robust', or 'minmax' for regional scaling")
            logger.warning("  - Set regional_scaling.enabled to false for no scaling")
    
    def load_data(self, data_paths: Union[str, List[str]], target_column: str = "vhm0_y") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Load data using the DataLoader."""
        self._log_memory_usage("before loading data")
        
        # Use DataLoader to load and combine data (with per-file sampling)
        combined_df, successful_files = self.data_loader.load_data(data_paths)
        
        # Prepare features using FeatureEngineer
        X, y, regions, coords = self.feature_engineer.prepare_features(combined_df, target_column)
        
        # Extract raw input data for baseline metrics
        if 'vhm0_x' in combined_df.columns:
            self.vhm0_x_raw = combined_df['vhm0_x'].to_numpy()
        else:
            logger.warning("vhm0_x column not found in data. Baseline metrics will be empty.")
            self.vhm0_x_raw = None
        
        # Extract actual wave heights for proper binning (needed when predict_bias=True)
        if target_column in combined_df.columns:
            actual_wave_heights = combined_df[target_column].to_numpy()
        else:
            logger.warning(f"{target_column} column not found in data. Wave height binning may be incorrect.")
            actual_wave_heights = None
        
        # Get feature names from FeatureEngineer
        self.feature_names = self.feature_engineer.get_feature_names()
        
        # Log dataset info to Comet
        dataset_info = self._prepare_dataset_info(X, y, successful_files)
        self.experiment_logger.log_dataset_info(dataset_info)
        
        self._log_memory_usage("after loading data")
        return X, y, regions, coords, successful_files, actual_wave_heights
    
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
    
    def _calculate_regional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, regions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per region using MetricsCalculator."""
        return self.metrics_calculator.calculate_regional_metrics(y_true, y_pred, regions)

    def _calculate_sea_bin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for different sea state bins using MetricsCalculator."""
        return self.metrics_calculator.calculate_sea_bin_metrics(y_true, y_pred, enable_logging=True)

    def split_data(self, X: np.ndarray, y: np.ndarray, regions: np.ndarray = None, coords: np.ndarray = None, file_paths: List[str] = None, vhm0_x: np.ndarray = None, actual_wave_heights: np.ndarray = None) -> None:
        """
        Split data into train/validation/test sets using DataSplitter.
        
        Args:
            X: Feature matrix
            y: Target vector (may be bias values if predict_bias=true)
            regions: Regional classification array (optional)
            coords: Coordinate array (lat, lon) (optional)
            file_paths: List of file paths (required for year-based splitting)
            vhm0_x: Raw input wave heights for baseline metrics (optional)
            actual_wave_heights: Actual wave heights for proper binning (optional)
        """
        self._log_memory_usage("before splitting data")
        
        # Use DataSplitter to perform the split
        split_data = self.data_splitter.split_data(X, y, regions, coords, file_paths, actual_wave_heights)
        
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
        
        # Split raw input data for baseline metrics
        if vhm0_x is not None:
            # Use the same indices as the main split
            train_indices = split_data.get('train_indices', None)
            val_indices = split_data.get('val_indices', None)
            test_indices = split_data.get('test_indices', None)
            
            if train_indices is not None:
                self.vhm0_x_train = vhm0_x[train_indices]
                self.vhm0_x_val = vhm0_x[val_indices] if val_indices is not None else np.array([])
                self.vhm0_x_test = vhm0_x[test_indices] if test_indices is not None else np.array([])
            else:
                # Fallback: assume equal distribution
                logger.warning("DataSplitter didn't return indices. Using fallback splitting for vhm0_x.")
                n_samples = len(vhm0_x)
                n_train = len(self.X_train)
                n_val = len(self.X_val)
                
                self.vhm0_x_train = vhm0_x[:n_train]
                self.vhm0_x_val = vhm0_x[n_train:n_train+n_val] if n_val > 0 else np.array([])
                self.vhm0_x_test = vhm0_x[n_train+n_val:] if n_train+n_val < n_samples else np.array([])
        else:
            self.vhm0_x_train = None
            self.vhm0_x_val = None
            self.vhm0_x_test = None
        
        # Log split information (includes stratified distribution logging)
        self.data_splitter.log_split_info(split_data)
        
        # Apply sample weighting (regional and/or wave height bin weighting)
        if self.predict_bias and actual_wave_heights is not None:
            # Use actual wave heights for weighting when predicting bias
            # The data splitter already provides the properly split actual wave heights
            actual_wave_heights_train = split_data.get('actual_wave_heights_train', None)
            if actual_wave_heights_train is not None:
                self.sample_weights = self.sample_weighting.apply_weights(
                    actual_wave_heights_train, self.regions_train
                )
                logger.info("Applied sample weights based on actual wave heights (bias prediction mode)")
            else:
                # Fallback to y_train if actual_wave_heights_train is not available
                self.sample_weights = self.sample_weighting.apply_weights(
                    self.y_train, self.regions_train
                )
                logger.warning("actual_wave_heights_train not found in split_data, using y_train for weighting")
        else:
            # Use y_train for weighting (standard mode)
            self.sample_weights = self.sample_weighting.apply_weights(
                self.y_train, self.regions_train
            )
            logger.info("Applied sample weights based on target values (standard mode)")
        
        self._log_memory_usage("after splitting data")
    
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
        
        vhm0_y_train, vhm0_pred_train = reconstruct_vhm0_values(
            predict_bias=self.predict_bias, 
            predict_bias_log_space=self.predict_bias_log_space, 
            vhm0_x=self.vhm0_x_train, 
            y_true=self.y_train, 
            y_pred=train_pred)

        train_metrics = evaluate_model(vhm0_pred_train, vhm0_y_train)
        regional_train_metrics = self._calculate_regional_metrics(vhm0_y_train, vhm0_pred_train, self.regions_train)
        if self.vhm0_x_train is not None:
            baseline_regional_train_metrics = self._calculate_regional_metrics(vhm0_y_train, self.vhm0_x_train, self.regions_train)
        else:
            logger.warning("vhm0_x_train not available. Skipping baseline regional training metrics.")
            baseline_regional_train_metrics = {}
        sea_bin_train_metrics = self._calculate_sea_bin_metrics(vhm0_y_train, vhm0_pred_train)
        
        # 🚀 MEMORY OPTIMIZATION: Delete train predictions immediately
        del train_pred
        import gc; gc.collect()

        # Calculate validation metrics
        if len(self.X_val) > 0:
            val_pred = self.model.predict(self.X_val)

            vhm0_y_val, vhm0_pred_val = reconstruct_vhm0_values(
                predict_bias=self.predict_bias,
                predict_bias_log_space=self.predict_bias_log_space,
                vhm0_x=self.vhm0_x_val,
                y_true=self.y_val,
                y_pred=val_pred,
            )

            val_metrics = evaluate_model(vhm0_pred_val, vhm0_y_val)

            logger.info("Calculating regional validation metrics...")
            regional_val_metrics = self._calculate_regional_metrics(
                vhm0_y_val, vhm0_pred_val, self.regions_val
            )

            logger.info("Calculating baseline regional validation metrics...")
            if self.vhm0_x_val is not None and len(self.vhm0_x_val) > 0:
                baseline_regional_val_metrics = self._calculate_regional_metrics(vhm0_y_val, self.vhm0_x_val, self.regions_val)
            else:
                logger.warning("vhm0_x_val not available. Skipping baseline regional validation metrics.")
                baseline_regional_val_metrics = {}

            sea_bin_val_metrics = self._calculate_sea_bin_metrics(vhm0_y_val, vhm0_pred_val)

            # Memory optimization
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
            'baseline_regional_train_metrics': baseline_regional_train_metrics,
            'sea_bin_train_metrics': sea_bin_train_metrics,
            'val_metrics': val_metrics,
            'regional_val_metrics': regional_val_metrics,
            'baseline_regional_val_metrics': baseline_regional_val_metrics if len(self.X_val) > 0 else {},
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
            test_pred = self.model.predict(self.X_test)
        
        if len(self.X_test) > 0:
            test_pred = self.model.predict(self.X_test)

            vhm0_y_test, vhm0_pred_test = reconstruct_vhm0_values(
                predict_bias=self.predict_bias,
                predict_bias_log_space=self.predict_bias_log_space,
                vhm0_x=self.vhm0_x_test,
                y_true=self.y_test,
                y_pred=test_pred,
            )

            test_metrics = evaluate_model(vhm0_pred_test, vhm0_y_test)

            logger.info("Calculating regional test metrics...")
            regional_test_metrics = self._calculate_regional_metrics(
                vhm0_y_test, vhm0_pred_test, self.regions_test
            )

            logger.info("Calculating baseline regional test metrics...")
            if self.vhm0_x_test is not None and len(self.vhm0_x_test) > 0:
                baseline_regional_test_metrics = self._calculate_regional_metrics(
                    vhm0_y_test, self.vhm0_x_test, self.regions_test
                )
                baseline_sea_bin_test_metrics = self._calculate_sea_bin_metrics(
                    vhm0_y_test, self.vhm0_x_test
                )
            else:
                logger.warning("vhm0_x_test not available. Skipping baseline regional test metrics.")
                baseline_regional_test_metrics = {}
                baseline_sea_bin_test_metrics = {}

            sea_bin_test_metrics = self._calculate_sea_bin_metrics(vhm0_y_test, vhm0_pred_test)

            # 🚀 MEMORY OPTIMIZATION
            del test_pred
            gc.collect()

        else:
            test_metrics = {'rmse': 0.0, 'mae': 0.0, 'bias': 0.0, 'pearson': 0.0, 'snr': 0.0, 'snr_db': 0.0}
            regional_test_metrics = {}
            baseline_regional_test_metrics = {}
            baseline_sea_bin_test_metrics = {}
            sea_bin_test_metrics = {}
        
        # Store current test metrics as class attribute
        self.current_test_metrics = test_metrics
        
        # Store regional and sea-bin metrics as attributes for DiagnosticPlotter
        self.regional_test_metrics = regional_test_metrics
        self.baseline_regional_test_metrics = baseline_regional_test_metrics
        self.baseline_sea_bin_test_metrics = baseline_sea_bin_test_metrics
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
            'baseline_regional_test_metrics': baseline_regional_test_metrics,
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
        
        # Save preprocessing components using PreprocessingManager's save method
        self.preprocessing_manager.save_preprocessing(save_path)
        
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
        
        # Load preprocessing components using PreprocessingManager's load method
        self.preprocessing_manager.load_preprocessing(load_path)
        
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
