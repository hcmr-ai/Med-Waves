"""
Refactored Model Evaluation Pipeline

This script uses the same infrastructure as training (DataLoader, FeatureEngineer, 
PreprocessingManager, DiagnosticPlotter) to ensure consistency between training and evaluation.
"""

import logging
import yaml
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import polars as pl
import joblib
from datetime import datetime
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_engineering.data_loader import DataLoader
from src.data_engineering.feature_engineer import FeatureEngineer
from src.commons.preprocessing_manager import PreprocessingManager
from src.evaluation.diagnostic_plotter import DiagnosticPlotter
from src.evaluation.metrics import evaluate_model
from src.commons.region_mapping import RegionMapper
from src.commons.aws.s3_results_saver import S3ResultsSaver
from src.commons.aws.s3_model_loader import S3ModelLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_date_from_filename(filename: str) -> Dict[str, Any]:
    """Extract date information from filename (standalone function for both worker and class)."""
    try:
        # Extract date from filename like WAVEAN2023010100.parquet
        if "WAVEAN" in filename:
            date_str = filename.split("WAVEAN")[1].split(".")[0]
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                # Determine season
                if month in [12, 1, 2]:
                    season = "winter"
                elif month in [3, 4, 5]:
                    season = "spring"
                elif month in [6, 7, 8]:
                    season = "summer"
                else:  # 9, 10, 11
                    season = "autumn"
                
                return {
                    "year": year,
                    "month": month,
                    "day": day,
                    "season": season
                }
    except Exception:
        pass
    
    return {"year": None, "month": None, "day": None, "season": "unknown"}


def _evaluate_single_file_worker(args: Tuple[str, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Simple worker function for parallel file evaluation.
    Similar to the approach used in training pipeline.
    
    Args:
        args: Tuple of (file_path, model_path, config)
        
    Returns:
        Evaluation results or None if failed
    """
    file_path, model_path, config = args
    
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
        from src.data_engineering.data_loader import DataLoader
        from src.data_engineering.feature_engineer import FeatureEngineer
        from src.commons.preprocessing_manager import PreprocessingManager
        from src.commons.aws.s3_model_loader import S3ModelLoader
        from src.evaluation.metrics import evaluate_model
        import joblib
        from pathlib import Path
        import numpy as np
        import pandas as pd
        
        logger = logging.getLogger(__name__)
        
        # Create temporary evaluator instance for this worker
        temp_evaluator = RobustModelEvaluator(config)
        temp_evaluator.load_model(model_path)
        
        # Evaluate the file
        result = temp_evaluator.evaluate_file(file_path)
        
        # Clean up
        temp_evaluator.cleanup()
        
        return result
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error evaluating file {file_path}: {e}")
        import traceback
        logging.getLogger(__name__).error(f"Traceback: {traceback.format_exc()}")
        return None


class RobustModelEvaluator:
    """Robust model evaluator using training infrastructure."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator with configuration."""
        self.config = config
        self.feature_config = config.get("feature_block", {})
        
        # Initialize components using training infrastructure
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.preprocessing_manager = PreprocessingManager(config)
        self.diagnostic_plotter = DiagnosticPlotter(config)
        
        # Configure DataLoader with parallel processing settings
        self._configure_dataloader_parallel_processing()
        
        # Initialize S3 components
        self.s3_results_saver = S3ResultsSaver(config)
        self.s3_model_loader = S3ModelLoader(config)
        
        # Model and scaler
        self.model = None
        self.scaler = None
        self.regional_scaler = None
        self.feature_names = None
        
        # Results storage
        self.file_results = {}
        self.aggregated_results = {}
        
        # Output directory
        self.output_dir = Path(self.config.get("output", {}).get("output_dir", "evaluation_results_refactored"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        # (self.output_dir / "spatial_maps").mkdir(exist_ok=True)
        (self.output_dir / "diagnostic_plots").mkdir(exist_ok=True)
        (self.output_dir / "diagnostic_plots" / "spatial_maps").mkdir(exist_ok=True)
        (self.output_dir / "diagnostic_plots" / "spatial_maps" / "seasonal").mkdir(exist_ok=True)
        (self.output_dir / "diagnostic_plots" / "spatial_maps" / "aggregated").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)

        self.target_column = config.get("data", {}).get("target_column", "vhm0_y")
        
        # Parallel processing configuration
        self.max_workers = config.get("evaluation", {}).get("max_workers", 1)
        self.use_parallel = self.max_workers > 1
        
        logger.info(f"Initialized RobustModelEvaluator with output directory: {self.output_dir}")
        logger.info(f"Parallel processing: {'enabled' if self.use_parallel else 'disabled'} (max_workers={self.max_workers})")
    
    @property
    def predict_bias(self) -> bool:
        """Check if we're predicting bias instead of vhm0_y directly."""
        return self.feature_config.get("predict_bias", False)
    
    def _configure_dataloader_parallel_processing(self) -> None:
        """Configure DataLoader with parallel processing settings from evaluation config."""
        # Get parallel processing settings from evaluation config
        evaluation_config = self.config.get("evaluation", {})
        max_workers = evaluation_config.get("max_workers", 1)
        
        # Update DataLoader config to use the same parallel processing settings
        if "data" not in self.config:
            self.config["data"] = {}
        
        # Configure DataLoader parallel processing
        self.config["data"]["parallel_loading"] = max_workers > 1
        self.config["data"]["max_workers"] = max_workers
        
        logger.info(f"DataLoader parallel processing: {'enabled' if max_workers > 1 else 'disabled'} (max_workers={max_workers})")
    
    def cleanup(self):
        """Clean up resources, especially S3 temporary files."""
        if self.s3_model_loader:
            self.s3_model_loader.cleanup()
    
    def load_model(self, model_path: str) -> None:
        """Load model and scaler from the specified path (local or S3)."""
        logger.info(f"Loading model from: {model_path}")
        
        # Check if S3 model loading is enabled and model path is S3
        if (self.s3_model_loader and self.s3_model_loader.enabled and 
            self.s3_model_loader._is_s3_path(model_path)):
            logger.info("Loading model from S3...")
            components = self.s3_model_loader.load_model(model_path)
            
            # Assign components
            self.model = components.get('model')
            self.scaler = components.get('scaler')
            self.regional_scaler = components.get('regional_scaler')
            self.feature_names = components.get('feature_names')  # Note: S3ModelLoader uses 'feature_columns'
            
            # Log loaded components
            if self.model:
                logger.info(f"Loaded model from S3: {type(self.model)}")
            else:
                raise FileNotFoundError("Model not found in S3 components")
                
            if self.scaler:
                logger.info(f"Loaded scaler from S3: {type(self.scaler)}")
            else:
                logger.warning("Scaler not found in S3 components")
                
            if self.regional_scaler:
                logger.info(f"Loaded regional scaler from S3: {type(self.regional_scaler)}")
            else:
                logger.info("No regional scaler found in S3 - using standard scaler")
                
            if self.feature_names:
                logger.info(f"Loaded feature names from S3: {len(self.feature_names)} features")
                logger.info(f"Model expects features: {self.feature_names}")
            else:
                logger.warning("Feature names not found in S3 components")
        else:
            # Local file loading
            model_path = Path(model_path)
            
            # Load model
            model_file = model_path / "model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                logger.info(f"Loaded model from {model_file}")
                logger.info(f"Model type: {type(self.model)}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Load scaler
            scaler_file = model_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info(f"Loaded scaler from {scaler_file}")
                logger.info(f"Scaler type: {type(self.scaler)}")
            else:
                logger.warning(f"Scaler file not found: {scaler_file}")
            
            # Load regional scaler if available
            regional_scaler_file = model_path / "regional_scaler.pkl"
            if regional_scaler_file.exists():
                self.regional_scaler = joblib.load(regional_scaler_file)
                logger.info(f"Loaded regional scaler from {regional_scaler_file}")
            else:
                logger.info("No regional scaler found - using standard scaler")
            
            # Load feature names
            feature_names_file = model_path / "feature_names.pkl"
            if feature_names_file.exists():
                self.feature_names = joblib.load(feature_names_file)
                logger.info(f"Loaded feature names from {feature_names_file}")
                logger.info(f"Model expects {len(self.feature_names)} features: {self.feature_names}")
            else:
                logger.warning(f"Feature names file not found: {feature_names_file}")
    
    def get_data_files(self, data_path: str) -> List[str]:
        """Get list of data files to evaluate using training pipeline function (local or S3)."""
        logger.info(f"Getting data files from: {data_path}")
        
        # Use the training pipeline's get_data_files function
        from src.pipelines.training.train_full_dataset import get_data_files as training_get_data_files
        
        # Get file pattern from config
        file_pattern = self.config.get("data", {}).get("file_pattern", "*.parquet")
        
        # Get files using training pipeline function
        files = training_get_data_files(data_path, file_pattern, self.config)
        
        # Get year from config, default to 2023
        year = self.config.get("evaluation", {}).get("year", "2023")
        
        # Filter by year if specified
        if year:
            year_files = [f for f in files if f"WAVEAN{year}" in f]
            logger.info(f"Found {len(year_files)} parquet files for year {year}")
            files = year_files
        else:
            logger.info(f"Found {len(files)} parquet files")
        
        # Apply max_files limit if specified in config
        max_files = self.config.get("evaluation", {}).get("max_files")
        if max_files is not None and max_files > 0:
            files = files[:max_files]
            logger.info(f"Limited to {len(files)} files (max_files={max_files})")
        
        return files
    
    def load_and_preprocess_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int, np.ndarray]:
        """
        Load and preprocess a single file using training infrastructure.
        
        Returns:
            Tuple of (X, y, regions, coords, metadata_df, vhm0_x_index, X_raw)
        """
        logger.info(f"Loading and preprocessing file: {Path(file_path).name}")
        
        try:
            # Use DataLoader to load the file
            combined_df, successful_files = self.data_loader.load_data([file_path])
            
            if not successful_files:
                logger.warning(f"Failed to load {file_path}")
                return None, None, None, None, None, None
            
            # Use FeatureEngineer to prepare features
            X_raw, y, regions, coords = self.feature_engineer.prepare_features(combined_df, self.target_column)
            
            # Now select only the features that the model expects using saved feature names
            if self.feature_names is None:
                raise ValueError("Feature names not loaded from model directory")
            
            # Get the feature names from FeatureEngineer
            feature_engineer_names = self.feature_engineer.get_feature_names()
            
            # Create a mapping from feature names to indices
            feature_name_to_idx = {name: idx for idx, name in enumerate(feature_engineer_names)}
            
            # Select features in the exact order the model expects
            model_feature_indices = []
            missing_features = []
            for feature_name in self.feature_names:
                if feature_name in feature_name_to_idx:
                    model_feature_indices.append(feature_name_to_idx[feature_name])
                else:
                    missing_features.append(feature_name)
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            # Extract only the features the model expects
            # X_raw = X_raw[:, model_feature_indices]
            logger.info(f"Using {len(model_feature_indices)} features: {self.feature_names}")
            
            # Find vhm0_x index in the model feature names
            vhm0_x_index = None
            if 'vhm0_x' in self.feature_names:
                vhm0_x_index = self.feature_names.index('vhm0_x')
            
            # Apply preprocessing using PreprocessingManager
            if self.regional_scaler is not None and regions is not None:
                X = self.regional_scaler.transform_with_regions(X_raw, regions)
                logger.info("Applied regional scaling")
            elif self.scaler is not None:
                X = self.scaler.transform(X_raw)
                logger.info("Applied standard scaling")
            else:
                logger.warning("No scaler available - using raw features!")
                X = X_raw
            
            metadata_df = pl.DataFrame({
                'lat': coords[:, 0],
                'lon': coords[:, 1],
                'region': regions
            })
            
            logger.info(f"Processed {len(X)} samples with {X.shape[1]} features")
            return X, y, regions, coords, metadata_df, vhm0_x_index, X_raw
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None, None, None, None, None, None, None
    
    def _extract_date_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract date information from filename."""
        return _extract_date_from_filename(filename)
    
    def _calculate_regional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, regions: np.ndarray) -> Dict[str, Any]:
        """Calculate regional metrics for the mock trainer."""
        regional_metrics = {}
        
        if regions is None or len(regions) == 0:
            return regional_metrics
        
        unique_regions = np.unique(regions)
        for region_id in unique_regions:
            mask = regions == region_id
            if np.sum(mask) > 0:
                region_y_true = y_true[mask]
                region_y_pred = y_pred[mask]
                region_metrics = evaluate_model(region_y_pred, region_y_true)
                region_name = RegionMapper.get_display_name(region_id)
                regional_metrics[region_name] = region_metrics
        
        return regional_metrics
    
    def _calculate_sea_bin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate sea-bin metrics for the mock trainer."""
        # Define sea-bin ranges (same as in training)
        # Load sea bins from config
        sea_bins = {}
        if self.config.get("feature_block", {}).get("sea_bin_metrics", {}).get("enabled", False):
            bins_config = self.config["feature_block"]["sea_bin_metrics"]["bins"]
            for bin_config in bins_config:
                min_val = bin_config["min"]
                max_val = bin_config["max"] if bin_config["max"] != float('inf') else float('inf')
                name = str(min_val) + "-" + str(max_val) + "m"
                sea_bins[name] = (min_val, max_val)
        else:
            # Fallback to default sea bins if not configured
            sea_bins = {
                "calm": (0.0, 0.5),
                "slight": (0.5, 1.25),
                "moderate": (1.25, 2.5),
                "rough": (2.5, 4.0),
                "very_rough": (4.0, 6.0),
                "high": (6.0, 9.0),
                "phenomenal": (9.0, float('inf'))
            }
        
        sea_bin_metrics = {}
        
        for bin_name, (min_val, max_val) in sea_bins.items():
            if max_val == float('inf'):
                mask = y_true >= min_val
            else:
                mask = (y_true >= min_val) & (y_true < max_val)
            
            if np.sum(mask) > 0:
                bin_y_true = y_true[mask]
                bin_y_pred = y_pred[mask]
                bin_metrics = evaluate_model(bin_y_pred, bin_y_true)
                bin_metrics['count'] = len(bin_y_true)
                bin_metrics['percentage'] = (len(bin_y_true) / len(y_true)) * 100
                bin_metrics['description'] = f"{bin_name.title()} seas ({min_val}-{max_val if max_val != float('inf') else '∞'}m)"
                sea_bin_metrics[bin_name] = bin_metrics
        
        return sea_bin_metrics
    
    def create_all_spatial_maps(self) -> None:
        """Create all spatial maps (seasonal and aggregated) by calculating spatial metrics once."""
        logger.info("Creating all spatial maps (seasonal and aggregated)...")
        
        # Collect all data from all files
        all_data = []
        for file_name, results in self.file_results.items():
            metadata = results["metadata"]
            y_true = results["predictions"]["y_true"]
            y_pred = results["predictions"]["y_pred"]
            vhm0_x = results["vhm0_x"]
            date_info = results["date_info"]
            
            # Create dataframe for this file using Polars
            file_df = metadata.with_columns([
                pl.Series("y_true", y_true),
                pl.Series("y_pred", y_pred),
                pl.Series("vhm0_x", vhm0_x),
                pl.lit(file_name).alias("file_name")
            ])
            
            # Add season information
            if date_info.get("month") is not None:
                month = date_info["month"]
                season = self._get_season_from_month(month)
                file_df = file_df.with_columns(pl.lit(season).alias("season"))
            else:
                file_df = file_df.with_columns(pl.lit("unknown").alias("season"))
            
            all_data.append(file_df)
        
        if not all_data:
            logger.warning("No data available for spatial maps")
            return
        
        # Combine all data using Polars for better performance
        combined_df = pl.concat(all_data, how="vertical_relaxed")
        logger.info(f"Combined data from {len(all_data)} files: {len(combined_df)} total samples")
        
        # Calculate spatial metrics once for both model and baseline using optimized function
        logger.info("Calculating spatial metrics for model and baseline...")
        
        # Create spatial dataframes for model and baseline using Polars
        model_spatial_df = combined_df.select(["lat", "lon", "y_true", "y_pred"]).with_columns(
            (pl.col("y_pred") - pl.col("y_true")).alias("residual")
        )
        
        baseline_spatial_df = combined_df.select(["lat", "lon", "y_true", "vhm0_x"]).with_columns(
            (pl.col("vhm0_x") - pl.col("y_true")).alias("residual")
        ).rename({"vhm0_x": "y_pred"})
        
        # Calculate spatial metrics using optimized function
        from src.evaluation.metrics import evaluate_model_spatial
        model_spatial_metrics = evaluate_model_spatial(model_spatial_df)
        baseline_spatial_metrics = evaluate_model_spatial(baseline_spatial_df)
        
        logger.info(f"Calculated spatial metrics for {len(model_spatial_metrics)} locations")
        
        # Create seasonal maps and get seasonal metrics
        seasonal_metrics = self._create_seasonal_maps_from_metrics(combined_df, model_spatial_metrics, baseline_spatial_metrics)
        
        # Create aggregated maps
        self._create_aggregated_maps_from_metrics(model_spatial_metrics, baseline_spatial_metrics)
        
        # Save spatial metrics to files (reuse already calculated seasonal metrics)
        self._save_spatial_metrics(model_spatial_metrics, baseline_spatial_metrics, seasonal_metrics)
        
        logger.info("✅ Created all spatial maps and saved spatial metrics")
    
    def _create_seasonal_maps_from_metrics(self, combined_df, model_spatial_metrics, baseline_spatial_metrics) -> Dict[str, Dict[str, Any]]:
        """Create seasonal maps from pre-calculated spatial metrics and return seasonal metrics for saving."""
        logger.info("Creating seasonal spatial maps...")
        
        # Create seasonal maps
        output_dir = self.output_dir / "diagnostic_plots" / "spatial_maps" / "seasonal"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we're working with Polars DataFrames
        is_polars = isinstance(combined_df, pl.DataFrame)
        
        if is_polars:
            seasons = combined_df["season"].unique().to_list()
        else:
            seasons = combined_df["season"].unique()
        logger.info(f"Found seasons: {seasons}")
        
        # Store seasonal metrics for later saving
        seasonal_metrics = {}
        
        for season in seasons:
            if season == "unknown":
                continue
                
            logger.info(f"Creating maps for {season} season...")
            if is_polars:
                season_data = combined_df.filter(pl.col("season") == season)
                # Filter spatial metrics to only include locations present in this season
                season_coords = season_data.select(["lat", "lon"]).unique()
                model_season_metrics = model_spatial_metrics.join(season_coords, on=["lat", "lon"], how="inner")
                baseline_season_metrics = baseline_spatial_metrics.join(season_coords, on=["lat", "lon"], how="inner")
            else:
                season_data = combined_df[combined_df["season"] == season]
                # Filter spatial metrics to only include locations present in this season
                season_coords = season_data[["lat", "lon"]].drop_duplicates()
                model_season_metrics = model_spatial_metrics.merge(season_coords, on=["lat", "lon"], how="inner")
                baseline_season_metrics = baseline_spatial_metrics.merge(season_coords, on=["lat", "lon"], how="inner")
            
            # Store for saving later
            seasonal_metrics[season] = {
                "model": model_season_metrics,
                "baseline": baseline_season_metrics
            }
            
            # Create maps for each metric
            metrics_to_plot = self.config.get("plots", {}).get("metrics_to_plot", ["bias", "mae", "rmse", "diff"])
            
            for metric in metrics_to_plot:
                if metric in model_season_metrics.columns:
                    # Calculate consistent color scale for this metric across model and baseline
                    if is_polars:
                        model_values = model_season_metrics[metric].drop_nulls()
                        baseline_values = baseline_season_metrics[metric].drop_nulls()
                    else:
                        model_values = model_season_metrics[metric].dropna()
                        baseline_values = baseline_season_metrics[metric].dropna()
                    
                    if len(model_values) > 0 and len(baseline_values) > 0:
                        # Use the same color scale for both model and baseline
                        vmin = min(model_values.min(), baseline_values.min())
                        vmax = max(model_values.max(), baseline_values.max())
                        
                        # Model seasonal map
                        self._create_single_spatial_map(
                            model_season_metrics, 
                            metric, 
                            f"{season}_model_{metric}_map.png",
                            f"{metric.upper()} - {season.title()} (Model)",
                            output_dir,
                            vmin=vmin,
                            vmax=vmax
                        )
                        
                        # Baseline seasonal map
                        self._create_single_spatial_map(
                            baseline_season_metrics, 
                            metric, 
                            f"{season}_baseline_{metric}_map.png",
                            f"{metric.upper()} - {season.title()} (Baseline)",
                            output_dir,
                            vmin=vmin,
                            vmax=vmax
                        )
        
        return seasonal_metrics
    
    def _create_aggregated_maps_from_metrics(self, model_spatial_metrics, baseline_spatial_metrics) -> None:
        """Create aggregated maps from pre-calculated spatial metrics."""
        logger.info("Creating overall aggregated spatial maps...")
        
        # Create aggregated maps
        output_dir = self.output_dir / "diagnostic_plots" / "spatial_maps" / "aggregated"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create maps for each metric
        metrics_to_plot = self.config.get("plots", {}).get("metrics_to_plot", ["bias", "mae", "rmse", "diff"])
        
        # Check if we're working with Polars DataFrames
        is_polars = isinstance(model_spatial_metrics, pl.DataFrame)
        
        for metric in metrics_to_plot:
            if metric in model_spatial_metrics.columns:
                # Calculate consistent color scale for this metric across model and baseline
                if is_polars:
                    model_values = model_spatial_metrics[metric].drop_nulls()
                    baseline_values = baseline_spatial_metrics[metric].drop_nulls()
                else:
                    model_values = model_spatial_metrics[metric].dropna()
                    baseline_values = baseline_spatial_metrics[metric].dropna()
                
                if len(model_values) > 0 and len(baseline_values) > 0:
                    # Use the same color scale for both model and baseline
                    if is_polars:
                        vmin = min(model_values.min(), baseline_values.min())
                        vmax = max(model_values.max(), baseline_values.max())
                    else:
                        vmin = min(model_values.min(), baseline_values.min())
                        vmax = max(model_values.max(), baseline_values.max())
                    
                    # Model aggregated map
                    self._create_single_spatial_map(
                        model_spatial_metrics, 
                        metric, 
                        f"overall_model_{metric}_map.png",
                        f"{metric.upper()} - Overall Model Performance",
                        output_dir,
                        vmin=vmin,
                        vmax=vmax
                    )
                    
                    # Baseline aggregated map
                    self._create_single_spatial_map(
                        baseline_spatial_metrics, 
                        metric, 
                        f"overall_baseline_{metric}_map.png",
                        f"{metric.upper()} - Overall Baseline Performance",
                        output_dir,
                        vmin=vmin,
                        vmax=vmax
                    )
      
    def create_seasonal_diagnostic_plots(self) -> None:
        """Create diagnostic plots aggregated by season."""
        logger.info("Creating seasonal diagnostic plots...")
        
        # Collect all data from all files
        all_data = []
        for file_name, results in self.file_results.items():
            y_true = results["predictions"]["y_true"]
            y_pred = results["predictions"]["y_pred"]
            vhm0_x = results["vhm0_x"]
            regions = results["regions"]
            coords = results["coords"]
            date_info = results["date_info"]
            
            # Add season information
            if date_info.get("month") is not None:
                month = date_info["month"]
                season = self._get_season_from_month(month)
                
                # Create dataframe for this file using Polars
                file_df = pl.DataFrame({
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'vhm0_x': vhm0_x,
                    'regions': regions,
                    'lat': coords[:, 0],
                    'lon': coords[:, 1],
                    'season': [season] * len(y_true),
                    'file_name': [file_name] * len(y_true)
                })
                
                all_data.append(file_df)
        
        if not all_data:
            logger.warning("No data available for seasonal diagnostic plots")
            return
        
        # Combine all data using Polars
        combined_df = pl.concat(all_data, how="vertical_relaxed")
        logger.info(f"Combined data from {len(all_data)} files: {len(combined_df)} total samples")
        
        # Create seasonal diagnostic plots
        seasons = combined_df["season"].unique().to_list()
        logger.info(f"Found seasons: {seasons}")
        
        for season in seasons:
            logger.info(f"Creating diagnostic plots for {season} season...")
            season_data = combined_df.filter(pl.col("season") == season)
            
            # Create seasonal output directory
            season_output_dir = self.output_dir / "diagnostic_plots" / "seasonal" / season
            season_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock trainer for this season
            mock_trainer = self._create_seasonal_mock_trainer(season_data, season, self.config)
            
            # Create diagnostic plots for this season
            season_config = self.config.copy()
            season_config["diagnostics"]["plots_save_path"] = str(season_output_dir)
            season_plotter = DiagnosticPlotter(season_config)
            if isinstance(season_data, pl.DataFrame):
                season_plotter.create_diagnostic_plots(mock_trainer, season_data["y_pred"].to_numpy())
            else:
                season_plotter.create_diagnostic_plots(mock_trainer, season_data["y_pred"].values)
            
            logger.info(f"✅ Created diagnostic plots for {season} season")
    
    def create_aggregated_diagnostic_plots(self) -> None:
        """Create diagnostic plots aggregated across all data."""
        logger.info("Creating overall aggregated diagnostic plots...")
        
        # Collect all data from all files
        all_data = []
        for file_name, results in self.file_results.items():
            y_true = results["predictions"]["y_true"]
            y_pred = results["predictions"]["y_pred"]
            vhm0_x = results["vhm0_x"]
            regions = results["regions"]
            coords = results["coords"]
            
            # Create dataframe for this file using Polars
            file_df = pl.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred,
                'vhm0_x': vhm0_x,
                'regions': regions,
                'lat': coords[:, 0],
                'lon': coords[:, 1],
                'file_name': [file_name] * len(y_true)
            })
            
            all_data.append(file_df)
        
        if not all_data:
            logger.warning("No data available for aggregated diagnostic plots")
            return
        
        # Combine all data using Polars
        combined_df = pl.concat(all_data, how="vertical_relaxed")
        logger.info(f"Combined data from {len(all_data)} files: {len(combined_df)} total samples")
        
        # Create aggregated output directory
        aggregated_output_dir = self.output_dir / "diagnostic_plots" / "aggregated"
        aggregated_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock trainer for aggregated data
        mock_trainer = self._create_aggregated_mock_trainer(combined_df, self.config)
        
        # Create diagnostic plots for aggregated data
        aggregated_config = self.config.copy()
        aggregated_config["diagnostics"]["plots_save_path"] = str(aggregated_output_dir)
        aggregated_plotter = DiagnosticPlotter(aggregated_config)
        if isinstance(combined_df, pl.DataFrame):
            aggregated_plotter.create_diagnostic_plots(mock_trainer, combined_df["y_pred"].to_numpy())
        else:
            aggregated_plotter.create_diagnostic_plots(mock_trainer, combined_df["y_pred"].values)
        
        logger.info("✅ Created overall aggregated diagnostic plots")
    
    def _create_seasonal_mock_trainer(self, season_data: pd.DataFrame, season: str, config: dict):
        """Create a mock trainer object for seasonal diagnostic plots."""
        # Calculate metrics for this season
        if isinstance(season_data, pl.DataFrame):
            y_true = season_data["y_true"].to_numpy()
            y_pred = season_data["y_pred"].to_numpy()
            regions = season_data["regions"].to_numpy()
        else:
            y_true = season_data["y_true"].values
            y_pred = season_data["y_pred"].values
            regions = season_data["regions"].values
        
        # Calculate overall metrics
        from src.evaluation.metrics import evaluate_model
        metrics = evaluate_model(y_pred, y_true)
        
        # Calculate regional metrics
        regional_metrics = self._calculate_regional_metrics(y_true, y_pred, regions)
        if isinstance(season_data, pl.DataFrame):
            baseline_regional_metrics = self._calculate_regional_metrics(y_true, season_data["vhm0_x"].to_numpy(), regions)
        else:
            baseline_regional_metrics = self._calculate_regional_metrics(y_true, season_data["vhm0_x"].values, regions)
        
        # Calculate sea-bin metrics
        sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, y_pred)
        
        # Calculate baseline sea-bin metrics
        if isinstance(season_data, pl.DataFrame):
            baseline_sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, season_data["vhm0_x"].to_numpy())
        else:
            baseline_sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, season_data["vhm0_x"].values)
        
        # Create mock trainer
        class SeasonalMockTrainer:
            def __init__(self, season_data, metrics, regional_metrics, baseline_regional_metrics, sea_bin_metrics, baseline_sea_bin_metrics, season):
                self.X_train = None
                self.y_train = None
                self.regions_train = None
                self.coords_train = None
                self.metadata_train = None
                self.feature_names = None
                self.training_history = None
                self.regional_metrics = regional_metrics
                self.baseline_regional_metrics = baseline_regional_metrics
                self.sea_bin_metrics = sea_bin_metrics
                self.train_metrics = metrics
                self.val_metrics = None
                self.current_train_metrics = metrics
                self.current_val_metrics = None
                self.current_test_metrics = metrics
                self.regional_test_metrics = regional_metrics
                self.baseline_regional_test_metrics = baseline_regional_metrics
                self.sea_bin_test_metrics = sea_bin_metrics
                self.baseline_sea_bin_test_metrics = baseline_sea_bin_metrics
                if isinstance(season_data, pl.DataFrame):
                    self.y_test = season_data["y_true"].to_numpy()
                    self.regions_test = season_data["regions"].to_numpy()
                    self.coords_test = season_data.select(["lat", "lon"]).to_numpy()
                    self.metadata_test = season_data
                    self.vhm0_x_test = season_data["vhm0_x"].to_numpy()  # Add baseline data
                else:
                    self.y_test = season_data["y_true"].values
                    self.regions_test = season_data["regions"].values
                    self.coords_test = season_data[["lat", "lon"]].values
                    self.metadata_test = season_data
                    self.vhm0_x_test = season_data["vhm0_x"].values  # Add baseline data
                self.season = season
                self.config = config  # Add config attribute
        
        return SeasonalMockTrainer(season_data, metrics, regional_metrics, baseline_regional_metrics, sea_bin_metrics, baseline_sea_bin_metrics, season)
    
    def _create_aggregated_mock_trainer(self, combined_df: pd.DataFrame, config: dict):
        """Create a mock trainer object for aggregated diagnostic plots."""
        # Calculate metrics for all data
        if isinstance(combined_df, pl.DataFrame):
            y_true = combined_df["y_true"].to_numpy()
            y_pred = combined_df["y_pred"].to_numpy()
            regions = combined_df["regions"].to_numpy()
        else:
            y_true = combined_df["y_true"].values
            y_pred = combined_df["y_pred"].values
            regions = combined_df["regions"].values
        
        # Calculate overall metrics
        metrics = evaluate_model(y_pred, y_true)
        
        # Calculate regional metrics
        regional_metrics = self._calculate_regional_metrics(y_true, y_pred, regions)
        if isinstance(combined_df, pl.DataFrame):
            baseline_regional_metrics = self._calculate_regional_metrics(y_true, combined_df["vhm0_x"].to_numpy(), regions)
        else:
            baseline_regional_metrics = self._calculate_regional_metrics(y_true, combined_df["vhm0_x"].values, regions)
        
        # Calculate sea-bin metrics
        sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, y_pred)
        
        # Calculate baseline sea-bin metrics
        if isinstance(combined_df, pl.DataFrame):
            baseline_sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, combined_df["vhm0_x"].to_numpy())
        else:
            baseline_sea_bin_metrics = self._calculate_sea_bin_metrics(y_true, combined_df["vhm0_x"].values)
        
        # Create mock trainer
        class AggregatedMockTrainer:
            def __init__(self, combined_df, metrics, regional_metrics, baseline_regional_metrics, sea_bin_metrics, baseline_sea_bin_metrics, config):
                self.X_train = None
                self.y_train = None
                self.regions_train = None
                self.coords_train = None
                self.metadata_train = None
                self.feature_names = None
                self.training_history = None
                self.regional_metrics = regional_metrics
                self.baseline_regional_metrics = baseline_regional_metrics
                self.sea_bin_metrics = sea_bin_metrics
                self.train_metrics = metrics
                self.val_metrics = None
                self.current_train_metrics = metrics
                self.current_val_metrics = None
                self.current_test_metrics = metrics
                self.regional_test_metrics = regional_metrics
                self.baseline_regional_test_metrics = baseline_regional_metrics
                self.sea_bin_test_metrics = sea_bin_metrics
                self.baseline_sea_bin_test_metrics = baseline_sea_bin_metrics
                if isinstance(combined_df, pl.DataFrame):
                    self.y_test = combined_df["y_true"].to_numpy()
                    self.regions_test = combined_df["regions"].to_numpy()
                    self.coords_test = combined_df.select(["lat", "lon"]).to_numpy()
                    self.metadata_test = combined_df
                    self.vhm0_x_test = combined_df["vhm0_x"].to_numpy()  # Add baseline data
                else:
                    self.y_test = combined_df["y_true"].values
                    self.regions_test = combined_df["regions"].values
                    self.coords_test = combined_df[["lat", "lon"]].values
                    self.metadata_test = combined_df
                    self.vhm0_x_test = combined_df["vhm0_x"].values  # Add baseline data
                self.config = config
        
        return AggregatedMockTrainer(combined_df, metrics, regional_metrics, baseline_regional_metrics, sea_bin_metrics, baseline_sea_bin_metrics, config)
    
    def _get_season_from_month(self, month: int) -> str:
        """Get season name from month number."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:  # 9, 10, 11
            return "autumn"
    
    def _create_single_spatial_map(self, spatial_metrics, metric: str, filename: str, title: str, output_dir: Path, vmin: float = None, vmax: float = None) -> None:
        """Create a single spatial map."""
        try:
            from src.analytics.plots.spatial_plots import plot_spatial_feature_map
            
            # Determine coordinate columns
            if "latitude" in spatial_metrics.columns and "longitude" in spatial_metrics.columns:
                coord_cols = ["latitude", "longitude"]
            elif "lat" in spatial_metrics.columns and "lon" in spatial_metrics.columns:
                coord_cols = ["lat", "lon"]
            else:
                logger.warning(f"No coordinate columns found for {filename}")
                return
            
            # Check if we're working with Polars DataFrames
            is_polars = isinstance(spatial_metrics, pl.DataFrame)
            
            # Prepare data for plotting
            if is_polars:
                plot_df = spatial_metrics.select(coord_cols + [metric]).drop_nulls()
            else:
                plot_df = spatial_metrics[coord_cols + [metric]].copy()
                plot_df = plot_df.dropna()
            
            if len(plot_df) == 0:
                logger.warning(f"No data for {metric} in {filename}")
                return
            
            # Convert to pandas for plotting if needed
            if is_polars:
                plot_df = plot_df.to_pandas()
            
            # Create spatial map
            save_path = output_dir / filename
            colorbar_label = f"{metric.upper()}"
            
            plot_spatial_feature_map(
                df_pd=plot_df,
                feature_col=metric,
                save_path=str(save_path),
                title=title,
                colorbar_label=colorbar_label,
                s=self.config.get("evaluation", {}).get("plots", {}).get("marker_size", 8),
                alpha=self.config.get("evaluation", {}).get("plots", {}).get("alpha", 0.85),
                cmap=self.config.get("evaluation", {}).get("plots", {}).get("colormap", "viridis"),
                vmin=vmin,
                vmax=vmax,
            )
            logger.info(f"Saved {metric} map: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating {metric} map {filename}: {e}")
    
    def evaluate_file(self, file_path: str) -> Dict[str, Any]:
        """Evaluate a single file using training infrastructure."""
        logger.info(f"Evaluating file: {Path(file_path).name}")
        
        # Load and preprocess data
        X, y, regions, coords, metadata_df, vhm0_x_index, X_raw = self.load_and_preprocess_file(file_path)
        
        if X is None:
            return None
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Get vhm0_x values if available
        vhm0_x = None
        if vhm0_x_index is not None:
            vhm0_x = X_raw[:, vhm0_x_index]
        
        # Compute overall metrics
        if self.predict_bias:
            if vhm0_x is None:
                logger.error("vhm0_x not available for bias prediction evaluation")
                return None
            vhm0_y_true = vhm0_x - y  # Reconstruct true vhm0 values from bias
            vhm0_pred = vhm0_x - y_pred  # Apply bias correction to predictions
            metrics = evaluate_model(vhm0_pred, vhm0_y_true)
        else:
            vhm0_y_true = y  # y already contains true vhm0 values
            metrics = evaluate_model(y_pred, y)
        
        baseline_metrics = None
        if vhm0_x_index is not None:
            # Baseline comparison: vhm0_x vs true vhm0 values
            baseline_metrics = evaluate_model(vhm0_x, vhm0_y_true)
            logger.info(f"Baseline metrics (vhm0_x vs vhm0_y): RMSE={baseline_metrics['rmse']:.4f}, MAE={baseline_metrics['mae']:.4f}, Bias={baseline_metrics['bias']:.4f}")
        else:
            logger.warning("vhm0_x column not found in features - skipping baseline metrics")
        
        # Extract date information
        file_name = Path(file_path).stem
        date_info = self._extract_date_from_filename(file_name)
        
        # Store results
        results = {
            "file_name": file_name,
            "file_path": file_path,
            "n_samples": len(X),
            "date_info": date_info,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "predictions": {
                "y_true": vhm0_y_true,  # Always store true vhm0 values
                "y_pred": y_pred if not self.predict_bias else vhm0_pred,
                "residuals": y_pred - vhm0_y_true
            },
            "metadata": metadata_df,
            "regions": regions,
            "coords": coords,
            "vhm0_x": vhm0_x if vhm0_x is not None else X_raw  # Include raw features for spatial maps
        }
        
        logger.info(f"✅ Evaluated {file_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, Bias={metrics['bias']:.4f}")
        return results

    def run_evaluation(self, data_path: str, model_path: str) -> None:
        """Run the complete evaluation pipeline."""
        logger.info("Starting refactored model evaluation...")
        
        # Load model
        self.load_model(model_path)
        
        # Get data files
        data_files = self.get_data_files(data_path)
        logger.info(f"Found {len(data_files)} files to evaluate")
        
        # Evaluate files (parallel or sequential)
        if self.use_parallel:
            self._evaluate_files_parallel(data_files)
        else:
            self._evaluate_files_sequential(data_files)
        
        # Create spatial maps (seasonal and aggregated) - calculate spatial metrics once
        self.create_all_spatial_maps()
        
        # Create seasonal diagnostic plots (after all files processed)
        self.create_seasonal_diagnostic_plots()
        
        # Create overall aggregated diagnostic plots
        self.create_aggregated_diagnostic_plots()
        
        # Compute aggregated results
        self.compute_aggregated_results()
        
        # Save results
        self.save_results()
        
        # Upload results to S3 if enabled
        self.save_results_to_s3()
        
        logger.info("✅ Evaluation completed successfully!")
    
    def _evaluate_files_parallel(self, data_files: List[str]) -> None:
        """Evaluate files in parallel using ProcessPoolExecutor - simple approach like training pipeline."""
        logger.info(f"Evaluating {len(data_files)} files in parallel with {self.max_workers} workers...")
        
        successful_files = 0
        failed_files = 0
        
        # Get model path from config
        model_path = self.config["data"]["model_path"]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(_evaluate_single_file_worker, (file_path, model_path, self.config)): file_path 
                for file_path in data_files
            }
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_file), total=len(data_files), desc="Evaluating files"):
                result = future.result()
                
                if result is not None:
                    self.file_results[result["file_name"]] = result
                    successful_files += 1
                    
                    # Log progress for first few files
                    if successful_files <= 3:
                        logger.info(f"File {successful_files}: {result['file_name']} evaluated successfully")
                else:
                    failed_files += 1
                    file_path = future_to_file[future]
                    logger.warning(f"Failed to evaluate file: {file_path}")
        
        logger.info(f"✅ Parallel evaluation completed: {successful_files} successful, {failed_files} failed")
    
    def _evaluate_files_sequential(self, data_files: List[str]) -> None:
        """Evaluate files sequentially (original method)."""
        logger.info(f"Evaluating {len(data_files)} files sequentially...")
        
        for file_path in tqdm(data_files, desc="Evaluating files"):
            try:
                results = self.evaluate_file(file_path)
                if results:
                    self.file_results[results["file_name"]] = results
                    
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
        
        logger.info(f"✅ Sequential evaluation completed: {len(self.file_results)} files processed")
  
    def compute_aggregated_results(self) -> None:
        """Compute aggregated results across all files."""
        logger.info("Computing aggregated results...")
        
        if not self.file_results:
            return
        
        # Aggregate metrics
        all_metrics = []
        all_baseline_metrics = []
        total_samples = 0
        
        for results in self.file_results.values():
            if results and "metrics" in results:
                all_metrics.append(results["metrics"])
                total_samples += results["n_samples"]
                
                if "baseline_metrics" in results and results["baseline_metrics"] is not None:
                    all_baseline_metrics.append(results["baseline_metrics"])
        
        if not all_metrics:
            return
        
        # Compute metrics directly on combined data (more accurate than file-weighted)
        mean_metrics = self._compute_combined_metrics(self.file_results, 'model')
        
        # Compute baseline metrics directly on combined data
        mean_baseline_metrics = None
        if all_baseline_metrics:
            mean_baseline_metrics = self._compute_combined_metrics(self.file_results, 'baseline')
        
        # Compute standard deviation across files for file-level variability
        metrics_df = pd.DataFrame(all_metrics)
        metrics_std = metrics_df.std().to_dict()
        
        self.aggregated_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_files_processed": len(self.file_results),
            "total_samples": total_samples,
            "mean_metrics": mean_metrics,
            "mean_baseline_metrics": mean_baseline_metrics,
            "metrics_std": metrics_std,
            "file_summary": {
                file_name: {
                    "n_samples": results["n_samples"],
                    "rmse": results["metrics"]["rmse"],
                    "mae": results["metrics"]["mae"],
                    "bias": results["metrics"]["bias"],
                    "pearson": results["metrics"]["pearson"],
                    "baseline_rmse": results.get("baseline_metrics", {}).get("rmse", None),
                    "baseline_mae": results.get("baseline_metrics", {}).get("mae", None),
                    "baseline_bias": results.get("baseline_metrics", {}).get("bias", None)
                }
                for file_name, results in self.file_results.items()
                if results and "metrics" in results
            }
        }
        
        logger.info("Aggregated results computed")
    
    def _compute_combined_metrics(self, file_results: Dict, metric_type: str) -> Dict[str, float]:
        """
        Compute metrics directly on combined y_true and y_pred from all files.
        
        This is the most accurate approach because it calculates metrics on the actual
        combined dataset rather than aggregating pre-computed file-level metrics.
        
        Args:
            file_results: Dictionary containing file results with predictions
            metric_type: 'model' or 'baseline' to determine which predictions to use
            
        Returns:
            Dictionary of metrics computed on combined data
        """
        if not file_results:
            return {}
        
        # Collect all y_true and y_pred from all files
        all_y_true = []
        all_y_pred = []
        
        for _, results in file_results.items():
            if results and "predictions" in results:
                y_true = results["predictions"]["y_true"]
                all_y_true.append(y_true)
                
                if metric_type == 'model':
                    y_pred = results["predictions"]["y_pred"]
                elif metric_type == 'baseline':
                    y_pred = results["vhm0_x"]  # Baseline predictions
                else:
                    raise ValueError(f"Unknown metric_type: {metric_type}")
                
                all_y_pred.append(y_pred)
        
        if not all_y_true or not all_y_pred:
            return {}
        
        # Combine all arrays
        combined_y_true = np.concatenate(all_y_true)
        combined_y_pred = np.concatenate(all_y_pred)
        
        # Compute metrics on the combined data
        from src.evaluation.metrics import evaluate_model
        return evaluate_model(combined_y_pred, combined_y_true)
    
    def _save_spatial_metrics(self, model_spatial_metrics, baseline_spatial_metrics, seasonal_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Save spatial metrics (model, baseline, and seasonal) to CSV files."""
        logger.info("Saving spatial metrics to files...")
        
        # Create spatial metrics directory
        spatial_metrics_dir = self.output_dir / "metrics"
        spatial_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save overall model spatial metrics
        model_file = spatial_metrics_dir / "model_spatial_metrics.csv"
        if isinstance(model_spatial_metrics, pl.DataFrame):
            model_spatial_metrics.write_csv(model_file)
        else:
            model_spatial_metrics.to_csv(model_file, index=False)
        logger.info(f"Saved model spatial metrics: {model_file}")
        
        # Save overall baseline spatial metrics
        baseline_file = spatial_metrics_dir / "baseline_spatial_metrics.csv"
        if isinstance(baseline_spatial_metrics, pl.DataFrame):
            baseline_spatial_metrics.write_csv(baseline_file)
        else:
            baseline_spatial_metrics.to_csv(baseline_file, index=False)
        logger.info(f"Saved baseline spatial metrics: {baseline_file}")
        
        # Save seasonal spatial metrics (reuse already calculated metrics)
        self._save_seasonal_spatial_metrics(seasonal_metrics, spatial_metrics_dir)
        
        logger.info("✅ All spatial metrics saved to files")
    
    def _save_seasonal_spatial_metrics(self, seasonal_metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
        """Save seasonal spatial metrics to separate CSV files using pre-calculated metrics."""
        logger.info("Saving seasonal spatial metrics...")
        
        # Create seasonal subdirectory
        seasonal_dir = output_dir / "seasonal"
        seasonal_dir.mkdir(parents=True, exist_ok=True)
        
        seasons = list(seasonal_metrics.keys())
        logger.info(f"Found seasons: {seasons}")
        
        for season in seasons:
            if season == "unknown":
                continue
                
            logger.info(f"Saving spatial metrics for {season} season...")
            
            # Get pre-calculated seasonal metrics
            model_season_metrics = seasonal_metrics[season]["model"]
            baseline_season_metrics = seasonal_metrics[season]["baseline"]
            
            # Save model seasonal metrics
            model_season_file = seasonal_dir / f"{season}_model_spatial_metrics.csv"
            if isinstance(model_season_metrics, pl.DataFrame):
                model_season_metrics.write_csv(model_season_file)
            else:
                model_season_metrics.to_csv(model_season_file, index=False)
            logger.info(f"Saved {season} model spatial metrics: {model_season_file}")
            
            # Save baseline seasonal metrics
            baseline_season_file = seasonal_dir / f"{season}_baseline_spatial_metrics.csv"
            if isinstance(baseline_season_metrics, pl.DataFrame):
                baseline_season_metrics.write_csv(baseline_season_file)
            else:
                baseline_season_metrics.to_csv(baseline_season_file, index=False)
            logger.info(f"Saved {season} baseline spatial metrics: {baseline_season_file}")
    
    def save_results_to_s3(self) -> None:
        """Save all evaluation results to S3 using the existing S3ResultsSaver."""
        if not self.s3_results_saver.enabled:
            logger.info("S3 saving is disabled in config")
            return
        
        logger.info("Starting S3 upload of evaluation results...")
        
        # Prepare results dictionary for S3ResultsSaver
        results = {
            "aggregated_results": self.aggregated_results,
            "file_results": self.file_results,
            "evaluation_summary": {
                "total_files_processed": len(self.file_results),
                "total_samples": sum(result.get("n_samples", 0) for result in self.file_results.values()),
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }
        
        # Save results JSON
        logger.info("Uploading results JSON to S3...")
        self.s3_results_saver.save_results_json(results)
        
        # Save diagnostic plots
        plots_dir = str(self.output_dir / "diagnostic_plots")
        if Path(plots_dir).exists():
            logger.info("Uploading diagnostic plots to S3...")
            self.s3_results_saver.save_diagnostic_plots(plots_dir)
        
        # Save metrics files
        metrics_dir = str(self.output_dir / "metrics")
        if Path(metrics_dir).exists():
            logger.info("Uploading metrics files to S3...")
            # Use the same method as diagnostic plots for metrics
            self.s3_results_saver.save_diagnostic_plots(metrics_dir)
        
        logger.info(f"✅ All evaluation results uploaded to S3: {self.s3_results_saver.prefix}")
    
    def save_results(self) -> None:
        """Save evaluation results to files."""
        logger.info("Saving evaluation results...")
        
        # Save aggregated results
        aggregated_file = self.output_dir / "metrics/aggregated_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_aggregated = {}
        for key, value in self.aggregated_results.items():
            if key == "mean_metrics":
                serializable_aggregated[key] = {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in value.items()}
            elif key == "mean_baseline_metrics" and value is not None:
                serializable_aggregated[key] = {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in value.items()}
            elif key == "metrics_std":
                serializable_aggregated[key] = {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in value.items()}
            elif key == "file_summary":
                # Convert file summary metrics to native types
                serializable_aggregated[key] = {}
                for file_name, file_data in value.items():
                    serializable_aggregated[key][file_name] = {}
                    for metric_key, metric_value in file_data.items():
                        if metric_value is not None and hasattr(metric_value, 'item'):
                            serializable_aggregated[key][file_name][metric_key] = metric_value.item()
                        else:
                            serializable_aggregated[key][file_name][metric_key] = metric_value
            else:
                serializable_aggregated[key] = value
        
        with open(aggregated_file, 'w') as f:
            json.dump(serializable_aggregated, f, indent=2)
        
        # Save file results
        file_results_file = self.output_dir / "metrics/file_results.json"
        serializable_file_results = {}
        for file_name, results in self.file_results.items():
            if results:
                serializable_file_results[file_name] = {
                    "file_name": results["file_name"],
                    "file_path": results["file_path"],
                    "n_samples": results["n_samples"],
                    "metrics": {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in results["metrics"].items()},
                    "baseline_metrics": {k: (v.item() if hasattr(v, 'item') else float(v)) for k, v in results.get("baseline_metrics", {}).items()} if results.get("baseline_metrics") else None
                }
        
        with open(file_results_file, 'w') as f:
            json.dump(serializable_file_results, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Refactored Model Evaluation Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create evaluator and run evaluation
    evaluator = RobustModelEvaluator(config)
    try:
        evaluator.run_evaluation(
            data_path=config["data"]["data_path"],
            model_path=config["data"]["model_path"]
        )   
        
        # Print summary
        if evaluator.aggregated_results:
            results = evaluator.aggregated_results
            print(f"\n📊 EVALUATION SUMMARY")
            print(f"📁 Files processed: {results['total_files_processed']}")
            print(f"📈 Total samples: {results['total_samples']:,}")
            print(f"🎯 Model RMSE: {results['mean_metrics']['rmse']:.4f}")
            print(f"🎯 Model MAE: {results['mean_metrics']['mae']:.4f}")
            print(f"🎯 Model Bias: {results['mean_metrics']['bias']:.4f}")
            print(f"🎯 Model Pearson: {results['mean_metrics']['pearson']:.4f}")
            
            if results.get('mean_baseline_metrics'):
                print(f"📊 Baseline RMSE: {results['mean_baseline_metrics']['rmse']:.4f}")
                print(f"📊 Baseline MAE: {results['mean_baseline_metrics']['mae']:.4f}")
                print(f"📊 Baseline Bias: {results['mean_baseline_metrics']['bias']:.4f}")
                print(f"📊 Baseline Pearson: {results['mean_baseline_metrics']['pearson']:.4f}")
                rmse_improvement = results['mean_baseline_metrics']['rmse'] - results['mean_metrics']['rmse']
                mae_improvement = results['mean_baseline_metrics']['mae'] - results['mean_metrics']['mae']
                print(f"🎯 RMSE Improvement: {rmse_improvement:.4f}")
                print(f"🎯 MAE Improvement: {mae_improvement:.4f}")
    finally:
        # Clean up resources
        evaluator.cleanup()


if __name__ == "__main__":
    main()
