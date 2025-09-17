#!/usr/bin/env python3
"""
Comprehensive evaluation script for model evaluation with spatial maps.

This script loads a trained model and evaluates it on data for any specified year,
creating spatial maps for each metric and file. It leverages existing
spatial plotting functionality from the codebase.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import yaml
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import gc

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm

# Add src to path - more robust approach
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Also add the project root to path
project_root = os.path.join(current_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipelines.training.train_full_dataset import get_data_files
from src.analytics.plots.spatial_plots import plot_spatial_feature_map
from src.evaluation.metrics import evaluate_model
from src.commons.aws.s3_model_loader import S3ModelLoader
from src.commons.aws.s3_results_saver import S3ResultsSaver

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation of trained models with spatial analysis.
    
    Features:
    - Load trained models and preprocessing components
    - Process data files for any specified year with memory optimization
    - Compute spatial metrics per file and location
    - Generate spatial maps for each metric
    - Aggregate results across files
    - Save results and plots
    """
    
    def __init__(self, model_path: str, output_dir: str, config: Dict[str, Any] = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the saved model directory
            output_dir: Directory to save evaluation results
            config: Configuration dictionary (optional)
        """
        # Handle S3 paths properly - don't convert to Path if it's an S3 URL
        if model_path.startswith('s3://'):
            self.model_path = model_path
        else:
            self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.metrics_to_plot = self.config.get('evaluation', {}).get('plots', {}).get('metrics_to_plot', ["bias", "mae", "rmse", "pearson", "diff", "var_true", "var_pred", "snr", "snr_db"])

        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "spatial_maps").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        # (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Initialize model components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.dimension_reducer = None
        self.feature_columns = None
        
        # Results storage
        self.file_results = {}
        self.spatial_metrics = {}
        self.aggregated_results = {}
        
        # S3 support
        self.s3_model_loader = None
        self.s3_results_saver = None
        self._initialize_s3_support()
        
        # Parallel processing configuration
        parallel_config = self.config.get('parallel_processing', {})
        self.parallel_enabled = parallel_config.get('enabled', True)
        self.n_workers = parallel_config.get('n_workers', min(cpu_count(), 8))
        self.batch_size = parallel_config.get('batch_size', 10)
        self.use_multiprocessing = parallel_config.get('use_multiprocessing', True)
        self.max_workers = parallel_config.get('max_workers', self.n_workers)
        
        logger.info(f"ModelEvaluator initialized")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.parallel_enabled:
            logger.info(f"Parallel processing: {self.n_workers} workers, batch size: {self.batch_size}")
        else:
            logger.info("Parallel processing: disabled")
    
    def _initialize_s3_support(self):
        """Initialize S3 support for model loading and results saving."""
        try:
            # Initialize S3 model loader
            self.s3_model_loader = S3ModelLoader(self.config)
            if self.s3_model_loader.enabled:
                logger.info("S3 model loading enabled")
            
            # Initialize S3 results saver
            self.s3_results_saver = S3ResultsSaver(self.config)
            if self.s3_results_saver.enabled:
                logger.info("S3 results saving enabled")
                
        except Exception as e:
            logger.warning(f"Failed to initialize S3 support: {e}")
            self.s3_model_loader = None
            self.s3_results_saver = None
    
    @staticmethod
    def _evaluate_file_worker(args: Tuple[str, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Worker function for parallel file evaluation.
        This is a static method to avoid pickling issues.
        
        Args:
            args: Tuple of (file_path, model_path, config)
            
        Returns:
            Evaluation results or None if failed
        """
        file_path, model_path, config = args
        
        try:
            # Create a temporary evaluator instance for this worker
            temp_evaluator = ModelEvaluator(model_path, "/tmp", config)
            temp_evaluator.load_model()
            
            # Evaluate the file
            result = temp_evaluator.evaluate_file(file_path)
            
            # Clean up
            del temp_evaluator
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in worker processing {file_path}: {e}")
            return None
    
    def load_model(self) -> None:
        """Load the trained model and preprocessing components."""
        logger.info("Loading trained model and components...")
        
        # Check if S3 model loading is enabled and model path is S3
        if (self.s3_model_loader and self.s3_model_loader.enabled and 
            self.s3_model_loader._is_s3_path(str(self.model_path))):
            logger.info("Loading model from S3...")
            components = self.s3_model_loader.load_model(self.model_path)
            
            # Assign components
            self.model = components.get('model')
            self.scaler = components.get('scaler')
            self.feature_selector = components.get('feature_selector')
            self.dimension_reducer = components.get('dimension_reducer')
            self.feature_columns = components.get('feature_columns')
            self.training_history = components.get('training_history')
            
            if self.model is None:
                raise FileNotFoundError(f"Model not found in S3: {self.model_path}")
            
            logger.info("✅ Model and components loaded from S3 successfully")
            return
        
        # Local model loading (existing logic)
        if isinstance(self.model_path, str):
            # This is an S3 path, should have been handled above
            raise FileNotFoundError(f"S3 model loading failed for: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Load model
        model_file = self.model_path / "model.pkl"
        if model_file.exists():
            self.model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load scaler
        scaler_file = self.model_path / "scaler.pkl"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Loaded scaler from {scaler_file}")
        
        # Load feature selector
        selector_file = self.model_path / "feature_selector.pkl"
        if selector_file.exists():
            self.feature_selector = joblib.load(selector_file)
            logger.info(f"Loaded feature selector from {selector_file}")
        
        # Load dimension reducer
        reducer_file = self.model_path / "dimension_reducer.pkl"
        if reducer_file.exists():
            self.dimension_reducer = joblib.load(reducer_file)
            logger.info(f"Loaded dimension reducer from {reducer_file}")
        
        # Load feature columns
        features_file = self.model_path / "feature_columns.pkl"
        if features_file.exists():
            self.feature_columns = joblib.load(features_file)
            logger.info(f"Loaded feature columns: {len(self.feature_columns)} features")
        
        # Load training history if available
        history_file = self.model_path / "training_history.pkl"
        if history_file.exists():
            self.training_history = joblib.load(history_file)
            logger.info(f"Loaded training history")
        
        logger.info("✅ Model and components loaded successfully")
    
    def evaluate_files_parallel(self, data_files: List[str]) -> Dict[str, Any]:
        """
        Evaluate multiple files in parallel using batch processing.
        
        Args:
            data_files: List of file paths to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"🚀 Starting parallel evaluation of {len(data_files)} files")
        logger.info(f"⚙️ Using {self.n_workers} workers with batch size {self.batch_size}")
        
        # Prepare arguments for workers
        worker_args = [
            (file_path, str(self.model_path), self.config)
            for file_path in data_files
        ]
        
        # Process files in batches
        all_results = {}
        successful_evaluations = 0
        
        # Create batches
        batches = [
            worker_args[i:i + self.batch_size]
            for i in range(0, len(worker_args), self.batch_size)
        ]
        
        logger.info(f"📦 Processing {len(batches)} batches")
        
        # Create global progress bar for all files
        with tqdm(total=len(data_files), desc="📊 Parallel Evaluation", 
                 unit="file", ncols=120,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as global_pbar:
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)")
                
                # Choose executor based on configuration
                if self.use_multiprocessing:
                    executor = ProcessPoolExecutor(max_workers=self.max_workers)
                else:
                    executor = ThreadPoolExecutor(max_workers=self.max_workers)
                
                try:
                    # Submit batch jobs
                    future_to_file = {
                        executor.submit(self._evaluate_file_worker, args): args[0]
                        for args in batch
                    }
                    
                    # Collect results with batch progress bar
                    with tqdm(total=len(batch), desc=f"Batch {batch_idx + 1}", 
                             unit="file", leave=False, ncols=80) as batch_pbar:
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            file_name = Path(file_path).name
                            try:
                                result = future.result()
                                if result is not None:
                                    all_results[file_path] = result
                                    successful_evaluations += 1
                                    
                                    # Log success with metrics
                                    rmse = result.get("metrics", {}).get("rmse", "N/A")
                                    mae = result.get("metrics", {}).get("mae", "N/A")
                                    logger.info(f"✅ Evaluated {file_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
                                    
                                    batch_pbar.set_postfix({
                                        'Success': successful_evaluations,
                                        'Current': file_name[:15] + "..." if len(file_name) > 15 else file_name
                                    })
                                else:
                                    logger.warning(f"❌ Failed to evaluate {file_name}")
                            except Exception as e:
                                logger.error(f"❌ Error evaluating {file_name}: {e}")
                            finally:
                                batch_pbar.update(1)
                                global_pbar.update(1)
                                global_pbar.set_postfix({
                                    'Success': successful_evaluations,
                                    'Batch': f"{batch_idx + 1}/{len(batches)}"
                                })
                
                finally:
                    executor.shutdown(wait=True)
                
                # Memory cleanup after each batch
                gc.collect()
                logger.info(f"✅ Batch {batch_idx + 1} completed. Memory cleaned up.")
        
        logger.info(f"Parallel evaluation completed: {successful_evaluations}/{len(data_files)} files successful")
        
        # Store results
        self.file_results = all_results
        
        return {
            "total_files": len(data_files),
            "successful_files": successful_evaluations,
            "failed_files": len(data_files) - successful_evaluations,
            "file_results": all_results
        }
    
    def _get_data_files(self, data_path: str, year: str = None, file_pattern: str = None) -> List[str]:
        """
        Get data files from the specified path for a given year.
        Uses the trainer's get_data_files function directly.
        
        Args:
            data_path: Path to data (local or S3)
            year: Year to filter files (e.g., "2023", "2022")
            file_pattern: Custom file pattern (overrides year-based pattern)
            
        Returns:
            List of file paths
        """
        logger.info(f"Getting data files from: {data_path}")
        if year:
            logger.info(f"Filtering for year: {year}")
        
        pattern = file_pattern or self.config.get("data", {}).get("file_pattern", "*.parquet")
        files = get_data_files(data_path, pattern)
        
        if year:
            files = [f for f in files if year in f]
            logger.info(f"After year filtering: {len(files)} files")
        
        logger.info(f"Found {len(files)} data files")
        return files
    
    def load_and_preprocess_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load and preprocess a single file for evaluation using the same logic as trainer.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (X, y, metadata_df) or (None, None, None) if failed
        """
        logger.info(f"Loading file: {Path(file_path).name}")
        
        try:
            # Use the same feature extraction as trainer
            from src.features.helpers import extract_features_from_parquet
            
            # Load data using the trainer's method
            df = extract_features_from_parquet(file_path, use_dask=False)
            
            if df.is_empty():
                logger.warning(f"Empty dataframe from {file_path}")
                return None, None, None
            
            logger.info(f"Loaded DataFrame shape: {df.shape}, columns: {df.columns}")
            
            # Prepare features using the same logic as trainer
            X_raw, y_raw = self._prepare_features_from_dataframe(df)
            
            if X_raw.size == 0:
                logger.warning(f"No valid samples after preprocessing in {file_path}")
                return None, None, None
            
            # Extract metadata for spatial analysis (lat, lon)
            metadata_cols = []
            if "lat" in df.columns:
                metadata_cols.append("lat")
            elif "latitude" in df.columns:
                metadata_cols.append("latitude")
            if "lon" in df.columns:
                metadata_cols.append("lon")
            elif "longitude" in df.columns:
                metadata_cols.append("longitude")
            
            if metadata_cols:
                metadata_df = df.select(metadata_cols).to_pandas()
                # Use whatever column names are available (lat/lon or latitude/longitude)
            else:
                logger.warning(f"No spatial metadata found in {file_path}")
                # Create dummy metadata with standard names
                metadata_df = pd.DataFrame({
                    "latitude": np.zeros(len(X_raw)),
                    "longitude": np.zeros(len(X_raw))
                })
            
            # Apply preprocessing (same as trainer)
            X = self._apply_preprocessing(X_raw)
            
            logger.info(f"Processed {len(X)} samples from {Path(file_path).name}")
            return X, y_raw, metadata_df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None, None, None
    
    def _prepare_features_from_dataframe(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from dataframe using the same logic as trainer.
        
        Args:
            df: Polars DataFrame with features and target
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Get available features (same logic as trainer)
        available_features = df.columns
        
        # Filter out target and non-feature columns (same as trainer)
        feature_cols = [col for col in available_features
                       if col not in ["vhm0_y", "corrected_VTM02", "time", "lat", "lon", "latitude", "longitude"] 
                       and not col.startswith("_")]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Extract features and target
        X_raw = df.select(feature_cols).to_numpy()
        y_raw = df["vhm0_y"].to_numpy()
        
        # Remove rows with NaN values (same as trainer)
        valid_mask = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y_raw))
        X_raw = X_raw[valid_mask]
        y_raw = y_raw[valid_mask]
        
        logger.info(f"After removing NaN: X: {X_raw.shape}, y: {y_raw.shape}")
        
        return X_raw, y_raw
    
    def _apply_preprocessing(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformations (same as trainer).
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        # Apply preprocessing in the same order as trainer
        if self.scaler:
            X = self.scaler.transform(X)
        
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        
        if self.dimension_reducer:
            X = self.dimension_reducer.transform(X)
        
        return X
    
    def evaluate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Evaluate the model on a single file and compute comprehensive metrics.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating file: {Path(file_path).name}")
        
        # Load and preprocess data
        X, y, metadata_df = self.load_and_preprocess_file(file_path)
        
        if X is None:
            return None
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Compute overall metrics (includes all metrics: rmse, mae, bias, diff, pearson, var_true, var_pred, snr, snr_db)
        metrics = evaluate_model(y_pred, y)
        
        # Compute comprehensive spatial metrics
        spatial_df = metadata_df.copy()
        spatial_df["y_true"] = y
        spatial_df["y_pred"] = y_pred
        spatial_df["residual"] = y_pred - y
        
        # Convert to polars for spatial metrics computation
        spatial_pl = pl.from_pandas(spatial_df)
        spatial_metrics_df = self._compute_comprehensive_spatial_metrics(spatial_pl)
        
        # Extract date information for temporal analysis
        file_name = Path(file_path).stem
        date_info = self._extract_date_from_filename(file_name)
        
        # Store results
        results = {
            "file_name": file_name,
            "file_path": file_path,
            "n_samples": len(X),
            "date_info": date_info,
            "metrics": metrics,
            "spatial_metrics": spatial_metrics_df.to_pandas(),
            "predictions": {
                "y_true": y,
                "y_pred": y_pred,
                "residuals": y_pred - y
            },
            "metadata": metadata_df
        }
        
        # Clean up memory
        del X, y, y_pred, spatial_df, spatial_pl, spatial_metrics_df
        import gc; gc.collect()
        
        logger.info(f"✅ Evaluated {file_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, Bias={metrics['bias']:.4f}")
        return results
    
    def create_spatial_maps(self, results: Dict[str, Any]) -> None:
        """
        Create spatial maps for a single file's results.
        
        Args:
            results: Results dictionary from evaluate_file
        """
        file_name = results["file_name"]
        spatial_metrics = results["spatial_metrics"]
        # metadata = results["metadata"]
        
        logger.info(f"Creating spatial maps for {file_name}")
        
        
        # Determine which coordinate columns to use (same logic as spatial metrics)
        if "latitude" in spatial_metrics.columns and "longitude" in spatial_metrics.columns:
            coord_cols = ["latitude", "longitude"]
        elif "lat" in spatial_metrics.columns and "lon" in spatial_metrics.columns:
            coord_cols = ["lat", "lon"]
        else:
            logger.warning(f"No spatial coordinate columns found in {file_name}")
            return
        
        for metric in self.metrics_to_plot:
            if metric not in spatial_metrics.columns:
                continue
            
            # Prepare data for plotting
            plot_df = spatial_metrics[coord_cols + [metric]].copy()
            plot_df = plot_df.dropna()
            
            if len(plot_df) == 0:
                logger.warning(f"No data for {metric} in {file_name}")
                continue
            
            # Create spatial map
            save_path = self.output_dir / "spatial_maps" / f"{file_name}_{metric}_map.png"
            title = f"{metric.upper()} - {file_name}"
            colorbar_label = f"{metric.upper()}"
            
            try:
                plot_spatial_feature_map(
                    df_pd=plot_df,
                    feature_col=metric,
                    save_path=str(save_path),
                    title=title,
                    colorbar_label=colorbar_label,
                    s=self.config.get("evaluation", {}).get("plots", {}).get("marker_size", 8),
                    alpha=self.config.get("evaluation", {}).get("plots", {}).get("alpha", 0.85),
                    cmap=self.config.get("evaluation", {}).get("plots", {}).get("colormap", "viridis"),
                )
                logger.info(f"Saved {metric} map: {save_path}")
            except Exception as e:
                logger.error(f"Error creating {metric} map for {file_name}: {e}")
    
    def create_spatial_maps_parallel(self) -> None:
        """
        Create spatial maps for all files in parallel.
        """
        if not self.file_results:
            logger.warning("No file results available for spatial map creation")
            return
        
        # Prepare results for parallel processing
        results_to_process = [
            (file_path, results) 
            for file_path, results in self.file_results.items() 
            if results is not None
        ]
        
        if not results_to_process:
            logger.warning("No valid results to process for spatial maps")
            return
        
        logger.info(f"🗺️ Creating spatial maps for {len(results_to_process)} files in parallel...")
        
        # Use ThreadPoolExecutor for I/O-bound spatial map creation
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all spatial map creation tasks
            future_to_file = {
                executor.submit(self._create_spatial_maps_worker, results): file_path
                for file_path, results in results_to_process
            }
            
            # Process results with progress bar
            with tqdm(total=len(results_to_process), desc="🗺️ Creating Spatial Maps", 
                     unit="file", ncols=100,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                successful_maps = 0
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    file_name = Path(file_path).name
                    
                    try:
                        success = future.result()
                        if success:
                            successful_maps += 1
                            logger.info(f"✅ Created spatial maps for {file_name}")
                        else:
                            logger.warning(f"❌ Failed to create spatial maps for {file_name}")
                    except Exception as e:
                        logger.error(f"❌ Error creating spatial maps for {file_name}: {e}")
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': successful_maps,
                            'Current': file_name[:20] + "..." if len(file_name) > 20 else file_name
                        })
        
        logger.info(f"✅ Spatial map creation completed: {successful_maps}/{len(results_to_process)} files successful")
    
    def _create_spatial_maps_worker(self, results: Dict[str, Any]) -> bool:
        """
        Worker function for creating spatial maps for a single file.
        
        Args:
            results: Results dictionary from evaluate_file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.create_spatial_maps(results)
            return True
        except Exception as e:
            logger.error(f"Error in spatial maps worker: {e}")
            return False
    
    def create_aggregated_spatial_maps(self) -> None:
        """Create aggregated spatial maps across all files."""
        logger.info("Creating aggregated spatial maps...")
        
        if not self.file_results:
            logger.warning("No file results available for aggregation")
            return
        
        # Combine all spatial metrics
        all_spatial_metrics = []
        for results in self.file_results.values():
            if results and "spatial_metrics" in results:
                spatial_df = results["spatial_metrics"].copy()
                spatial_df["file_name"] = results["file_name"]
                all_spatial_metrics.append(spatial_df)
        
        if not all_spatial_metrics:
            logger.warning("No spatial metrics available for aggregation")
            return
        
        # Concatenate all spatial metrics
        combined_spatial = pd.concat(all_spatial_metrics, ignore_index=True)
        
        # Determine which coordinate columns to use
        if "latitude" in combined_spatial.columns and "longitude" in combined_spatial.columns:
            coord_cols = ["latitude", "longitude"]
        elif "lat" in combined_spatial.columns and "lon" in combined_spatial.columns:
            coord_cols = ["lat", "lon"]
        else:
            logger.warning("No spatial coordinate columns found in aggregated data")
            return
        
        # Aggregate by location (mean across files)
        aggregated = combined_spatial.groupby(coord_cols).agg({
            "bias": "mean",
            "mae": "mean", 
            "rmse": "mean",
            "diff": "mean",
        }).reset_index()
        
        
        for metric in self.metrics_to_plot:
            if metric not in aggregated.columns:
                continue
            
            # Prepare data for plotting
            plot_df = aggregated[coord_cols + [metric]].copy()
            plot_df = plot_df.dropna()
            
            if len(plot_df) == 0:
                continue
            
            # Create spatial map
            save_path = self.output_dir / "spatial_maps" / f"aggregated_{metric}_map.png"
            title = f"Aggregated {metric.upper()} - All 2023 Files"
            colorbar_label = f"Mean {metric.upper()}"
            
            try:
                plot_spatial_feature_map(
                    df_pd=plot_df,
                    feature_col=metric,
                    save_path=str(save_path),
                    title=title,
                    colorbar_label=colorbar_label,
                    s=self.config.get("evaluation", {}).get("plots", {}).get("marker_size", 8),
                    alpha=self.config.get("evaluation", {}).get("plots", {}).get("alpha", 0.85),
                    cmap=self.config.get("evaluation", {}).get("plots", {}).get("colormap", "viridis")
                )
                logger.info(f"Saved aggregated {metric} map: {save_path}")
            except Exception as e:
                logger.error(f"Error creating aggregated {metric} map: {e}")
    
    def save_results(self) -> None:
        """Save all evaluation results."""
        logger.info("Saving evaluation results...")
        
        # Save file-level results
        results_file = self.output_dir / "file_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for file_name, results in self.file_results.items():
                if results:
                    serializable_results[file_name] = {
                        "file_name": results["file_name"],
                        "file_path": results["file_path"],
                        "n_samples": results["n_samples"],
                        "metrics": results["metrics"]
                    }
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save aggregated results
        if self.aggregated_results:
            aggregated_file = self.output_dir / "aggregated_results.json"
            with open(aggregated_file, 'w') as f:
                json.dump(self.aggregated_results, f, indent=2, default=str)
        
        # Save spatial metrics as CSV
        for file_path, results in self.file_results.items():
            if results and "spatial_metrics" in results:
                spatial_metrics = results["spatial_metrics"]
                if spatial_metrics is not None and not spatial_metrics.empty:
                    # Create metrics directory if it doesn't exist
                    metrics_dir = self.output_dir / "metrics"
                    metrics_dir.mkdir(exist_ok=True)
                    
                    # Extract just the filename from the full path
                    file_name = Path(file_path).stem
                    spatial_file = metrics_dir / f"{file_name}_spatial_metrics.csv"
                    spatial_metrics.to_csv(spatial_file, index=False)
                    logger.info(f"Saved spatial metrics: {spatial_file} ({len(spatial_metrics)} rows)")
                else:
                    logger.warning(f"Spatial metrics is empty for {file_path}")
            else:
                logger.warning(f"No spatial metrics found for {file_path}")
        
        logger.info(f"Results saved to {self.output_dir}")
        
        # Save to S3 if enabled
        if self.s3_results_saver and self.s3_results_saver.enabled:
            logger.info("Uploading results to S3...")
            experiment_name = self.config.get('experiment_name', f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Upload the entire output directory
            upload_results = self.s3_results_saver.upload_directory(str(self.output_dir), experiment_name=experiment_name)
            
            if upload_results:
                successful_uploads = sum(1 for success in upload_results.values() if success)
                total_files = len(upload_results)
                logger.info(f"Uploaded {successful_uploads}/{total_files} files to S3")
            else:
                logger.warning("No files uploaded to S3")
    
    def cleanup(self):
        """Clean up resources, especially S3 temporary files."""
        if self.s3_model_loader:
            self.s3_model_loader.cleanup()
    
    def run_evaluation(self, data_path: str, year: str = None, file_pattern: str = None, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete evaluation on data for the specified year.
        
        Args:
            data_path: Path to data files
            year: Year to evaluate (e.g., "2023", "2022")
            file_pattern: Custom file pattern (overrides year-based pattern)
            max_files: Maximum number of files to process (for testing)
            
        Returns:
            Dictionary with evaluation results
        """
        year_str = f" for year {year}" if year else ""
        logger.info(f"🚀 Starting model evaluation{year_str}")
        logger.info("=" * 60)
        
        # Load model
        self.load_model()
        
        # Get data files
        data_files = self._get_data_files(data_path, year, file_pattern)
        
        if max_files:
            data_files = data_files[:max_files]
            logger.info(f"Processing first {max_files} files for testing")
        
        logger.info(f"Processing {len(data_files)} files...")
        
        # Use parallel processing if enabled and configured
        if self.parallel_enabled and self.n_workers > 1:
            logger.info(f"Using parallel processing with {self.n_workers} workers")
            parallel_results = self.evaluate_files_parallel(data_files)
            successful_evaluations = parallel_results["successful_files"]
            
            # Create spatial maps for each file in parallel
            save_plots = self.config.get("output", {}).get("save_plots", True)
            if save_plots:
                logger.info("🗺️ Creating spatial maps for all files in parallel...")
                self.create_spatial_maps_parallel()
            else:
                logger.info("Skipping spatial maps (save_plots disabled in config)")
        else:
            logger.info("Using sequential processing")
            # Process each file sequentially with global progress bar
            successful_evaluations = 0
            
            # Create global progress bar for all files
            with tqdm(total=len(data_files), desc="📊 Evaluating files", 
                     unit="file", ncols=100, 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as global_pbar:
                
                for i, file_path in enumerate(data_files, 1):
                    file_name = Path(file_path).name
                    logger.info(f"🔄 Processing file {i}/{len(data_files)}: {file_name}")
                    
                    try:
                        # Evaluate file
                        results = self.evaluate_file(file_path)
                        
                        if results:
                            self.file_results[results["file_name"]] = results
                            successful_evaluations += 1
                            
                            # Log success with metrics
                            rmse = results.get("metrics", {}).get("rmse", "N/A")
                            mae = results.get("metrics", {}).get("mae", "N/A")
                            logger.info(f"✅ Evaluated {file_name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
                            
                            # Create spatial maps for this file
                            save_plots = self.config.get("output", {}).get("save_plots", True)
                            if save_plots:
                                logger.info(f"🗺️ Creating spatial maps for {file_name}")
                                self.create_spatial_maps(results)
                        else:
                            logger.warning(f"❌ Failed to evaluate {file_name}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error evaluating {file_name}: {e}")
                    
                    # Update global progress bar
                    global_pbar.update(1)
                    global_pbar.set_postfix({
                        'Success': successful_evaluations,
                        'Current': file_name[:20] + "..." if len(file_name) > 20 else file_name
                    })
        
        logger.info(f"Successfully evaluated {successful_evaluations}/{len(data_files)} files")
        
        # Create aggregated spatial maps
        save_plots = self.config.get("output", {}).get("save_plots", True)
        if save_plots:
            self.create_aggregated_spatial_maps()
        else:
            logger.info("Skipping aggregated spatial maps (save_plots disabled in config)")
        
        # Compute temporal analysis if enabled
        temporal_enabled = self.config.get("evaluation", {}).get("temporal_analysis", {}).get("enabled", True)
        if temporal_enabled:
            logger.info("Computing temporal analysis...")
            self.compute_temporal_analysis()
        else:
            logger.info("Skipping temporal analysis (disabled in config)")
        
        # Compute aggregated results
        self.compute_aggregated_results()
        
        # Save results
        self.save_results()
        
        logger.info("✅ Model evaluation completed!")
        return self.aggregated_results
    
    def compute_aggregated_results(self) -> None:
        """Compute aggregated results across all files."""
        logger.info("Computing aggregated results...")
        
        if not self.file_results:
            return
        
        # Aggregate metrics across files
        all_metrics = []
        total_samples = 0
        
        for results in self.file_results.values():
            if results and "metrics" in results:
                all_metrics.append(results["metrics"])
                total_samples += results["n_samples"]
        
        if not all_metrics:
            return
        
        # Compute mean metrics
        metrics_df = pd.DataFrame(all_metrics)
        mean_metrics = metrics_df.mean().to_dict()
        
        # Store aggregated results
        self.aggregated_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_files_processed": len(self.file_results),
            "total_samples": total_samples,
            "mean_metrics": mean_metrics,
            "metrics_std": metrics_df.std().to_dict(),
            "file_summary": {
                file_name: {
                    "n_samples": results["n_samples"],
                    "rmse": results["metrics"]["rmse"],
                    "mae": results["metrics"]["mae"],
                    "bias": results["metrics"]["bias"],
                    "pearson": results["metrics"]["pearson"]
                }
                for file_name, results in self.file_results.items()
                if results and "metrics" in results
            }
        }
        
        logger.info("Aggregated results computed")
    
    def compute_temporal_analysis(self) -> None:
        """Compute temporal analysis (daily, monthly, seasonal metrics)."""
        logger.info("Computing temporal analysis...")
        
        if not self.file_results:
            logger.warning("No file results available for temporal analysis")
            return
        
        # Collect all temporal data
        temporal_data = []
        for results in self.file_results.values():
            if results and "date_info" in results and "metrics" in results:
                date_info = results["date_info"]
                metrics = results["metrics"]
                
                if date_info.get("date") is not None:
                    temporal_data.append({
                        "date": date_info["date"],
                        "year": date_info["year"],
                        "month": date_info["month"],
                        "day": date_info["day"],
                        "day_of_year": date_info["day_of_year"],
                        "weekday": date_info["weekday"],
                        "month_name": date_info["month_name"],
                        "season": date_info["season"],
                        "file_name": results["file_name"],
                        "n_samples": results["n_samples"],
                        **metrics
                    })
        
        if not temporal_data:
            logger.warning("No temporal data available")
            return
        
        # Convert to DataFrame
        temporal_df = pd.DataFrame(temporal_data)
        temporal_df = temporal_df.sort_values("date")
        
        # Save temporal data
        temporal_file = self.output_dir / "temporal_analysis.csv"
        temporal_df.to_csv(temporal_file, index=False)
        logger.info(f"Saved temporal analysis to {temporal_file}")
        
        # Compute daily aggregated metrics
        daily_metrics = temporal_df.groupby("date").agg({
            "rmse": "mean",
            "mae": "mean", 
            "bias": "mean",
            "diff": "mean",
            "pearson": "mean",
            "var_true": "mean",
            "var_pred": "mean",
            "snr": "mean",
            "snr_db": "mean",
            "n_samples": "sum"
        }).reset_index()
        
        daily_file = self.output_dir / "daily_metrics.csv"
        daily_metrics.to_csv(daily_file, index=False)
        logger.info(f"Saved daily metrics to {daily_file}")
        
        # Compute monthly aggregated metrics
        monthly_metrics = temporal_df.groupby(["year", "month", "month_name"]).agg({
            "rmse": "mean",
            "mae": "mean",
            "bias": "mean", 
            "diff": "mean",
            "pearson": "mean",
            "var_true": "mean",
            "var_pred": "mean",
            "snr": "mean",
            "snr_db": "mean",
            "n_samples": "sum"
        }).reset_index()
        
        monthly_file = self.output_dir / "monthly_metrics.csv"
        monthly_metrics.to_csv(monthly_file, index=False)
        logger.info(f"Saved monthly metrics to {monthly_file}")
        
        # Compute seasonal aggregated metrics
        seasonal_metrics = temporal_df.groupby(["year", "season"]).agg({
            "rmse": "mean",
            "mae": "mean",
            "bias": "mean",
            "diff": "mean", 
            "pearson": "mean",
            "var_true": "mean",
            "var_pred": "mean",
            "snr": "mean",
            "snr_db": "mean",
            "n_samples": "sum"
        }).reset_index()
        
        seasonal_file = self.output_dir / "seasonal_metrics.csv"
        seasonal_metrics.to_csv(seasonal_file, index=False)
        logger.info(f"Saved seasonal metrics to {seasonal_file}")
        
        # Create temporal plots
        save_plots = self.config.get("output", {}).get("save_plots", True)
        if save_plots:
            self.create_temporal_plots(temporal_df, daily_metrics, monthly_metrics, seasonal_metrics)
        else:
            logger.info("Skipping temporal plots (save_plots disabled in config)")
        
        # Store in aggregated results
        self.aggregated_results["temporal_analysis"] = {
            "total_days": len(daily_metrics),
            "total_months": len(monthly_metrics),
            "total_seasons": len(seasonal_metrics),
            "date_range": {
                "start": str(temporal_df["date"].min()),
                "end": str(temporal_df["date"].max())
            }
        }
        
        logger.info("Temporal analysis completed")
    
    def create_temporal_plots(self, temporal_df: pd.DataFrame, daily_metrics: pd.DataFrame, 
                            monthly_metrics: pd.DataFrame, seasonal_metrics: pd.DataFrame) -> None:
        """Create temporal trend plots."""
        logger.info("Creating temporal plots...")
        
        plots_dir = self.output_dir / "temporal_plots"
        plots_dir.mkdir(exist_ok=True)
        
        
        # for metric in self.metrics_to_plot:
        #     if metric not in daily_metrics.columns:
        #         continue
                
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(daily_metrics["date"], daily_metrics[metric], linewidth=1, alpha=0.7)
        #     plt.title(f"Daily {metric.upper()} Trends")
        #     plt.xlabel("Date")
        #     plt.ylabel(metric.upper())
        #     plt.xticks(rotation=45)
        #     plt.grid(True, alpha=0.3)
        #     plt.tight_layout()
            
        #     save_path = plots_dir / f"daily_{metric}_trends.png"
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     plt.close()
        #     logger.info(f"Saved daily {metric} trend plot: {save_path}")
        
        # Monthly trends
        for metric in self.metrics_to_plot:
            if metric not in monthly_metrics.columns:
                continue
                
            plt.figure(figsize=(12, 6))
            monthly_metrics_sorted = monthly_metrics.sort_values(["year", "month"])
            x_labels = [f"{row['year']}-{row['month']:02d}" for _, row in monthly_metrics_sorted.iterrows()]
            plt.plot(range(len(monthly_metrics_sorted)), monthly_metrics_sorted[metric], 
                    marker='o', linewidth=2, markersize=4)
            plt.title(f"Monthly {metric.upper()} Trends")
            plt.xlabel("Month")
            plt.ylabel(metric.upper())
            plt.xticks(range(len(x_labels)), x_labels, rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = plots_dir / f"monthly_{metric}_trends.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved monthly {metric} trend plot: {save_path}")
        
        # Seasonal comparison
        for metric in self.metrics_to_plot:
            if metric not in seasonal_metrics.columns:
                continue
                
            plt.figure(figsize=(10, 6))
            seasonal_pivot = seasonal_metrics.pivot(index="season", columns="year", values=metric)
            seasonal_pivot.plot(kind='bar', ax=plt.gca())
            plt.title(f"Seasonal {metric.upper()} Comparison")
            plt.xlabel("Season")
            plt.ylabel(metric.upper())
            plt.legend(title="Year")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_path = plots_dir / f"seasonal_{metric}_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved seasonal {metric} comparison plot: {save_path}")
        
        logger.info("Temporal plots created")
    
    def _compute_comprehensive_spatial_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute comprehensive spatial metrics using Polars expressions (vectorized).
        
        Args:
            df: Polars DataFrame with lat/lon or latitude/longitude, y_true, y_pred columns
            
        Returns:
            Polars DataFrame with comprehensive spatial metrics
        """
        # Determine which coordinate columns to use
        if "latitude" in df.columns and "longitude" in df.columns:
            coord_cols = ["latitude", "longitude"]
        elif "lat" in df.columns and "lon" in df.columns:
            coord_cols = ["lat", "lon"]
        else:
            logger.warning("No spatial coordinate columns found")
            return pl.DataFrame()
        
        # Use Polars expressions for vectorized computation
        try:
            spatial_metrics = df.group_by(coord_cols).agg([
                # Sample count
                pl.len().alias("n_samples"),
                
                # Basic metrics
                pl.col("y_true").mean().alias("y_true_mean"),
                pl.col("y_pred").mean().alias("y_pred_mean"),
                pl.col("y_true").var().alias("y_true_var"),
                pl.col("y_pred").var().alias("y_pred_var"),
                
                # RMSE: sqrt(mean((y_true - y_pred)^2))
                ((pl.col("y_true") - pl.col("y_pred")) ** 2).mean().sqrt().alias("rmse"),
                
                # MAE: mean(|y_true - y_pred|)
                (pl.col("y_true") - pl.col("y_pred")).abs().mean().alias("mae"),
                
                # Bias: mean(y_pred - y_true)
                (pl.col("y_pred") - pl.col("y_true")).mean().alias("bias"),
                
                # Diff: mean(y_pred - y_true) (same as bias)
                (pl.col("y_pred") - pl.col("y_true")).mean().alias("diff"),
                
                # Pearson correlation
                pl.corr("y_true", "y_pred").alias("pearson"),
                
                # Variance metrics
                pl.col("y_true").var().alias("var_true"),
                pl.col("y_pred").var().alias("var_pred"),
                
                # SNR and SNR_db will be calculated after aggregation
            ])
            
            # Calculate SNR and SNR_db using a simpler approach
            # Convert to pandas for complex calculations, then back to polars
            spatial_metrics_pd = spatial_metrics.to_pandas()
            
            # Calculate SNR: var_true / var_pred (simplified)
            spatial_metrics_pd['snr'] = spatial_metrics_pd['var_true'] / spatial_metrics_pd['var_pred']
            spatial_metrics_pd['snr'] = spatial_metrics_pd['snr'].replace([np.inf, -np.inf], float('inf'))
            spatial_metrics_pd['snr'] = spatial_metrics_pd['snr'].fillna(float('inf'))
            
            # Calculate SNR_db: 10 * log10(SNR)
            spatial_metrics_pd['snr_db'] = 10 * np.log10(spatial_metrics_pd['snr'])
            spatial_metrics_pd['snr_db'] = spatial_metrics_pd['snr_db'].replace([np.inf, -np.inf], float('inf'))
            spatial_metrics_pd['snr_db'] = spatial_metrics_pd['snr_db'].fillna(float('inf'))
            
            # Convert back to polars
            spatial_metrics = pl.from_pandas(spatial_metrics_pd)
            
            return spatial_metrics
            
        except Exception as e:
            logger.error(f"Error computing spatial metrics: {e}")
            # Fallback to empty DataFrame
            schema = {
                "rmse": [],
                "mae": [],
                "bias": [],
                "diff": [],
                "pearson": [],
                "var_true": [],
                "var_pred": [],
                "snr": [],
                "snr_db": [],
                "n_samples": []
            }
            for col in coord_cols:
                schema[col] = []
            return pl.DataFrame(schema)
    
    def _extract_date_from_filename(self, filename: str) -> Dict[str, Any]: #src.data_engineering.split
        """
        Extract date information from filename.
        
        Args:
            filename: Filename (e.g., "WAVEAN20230101")
            
        Returns:
            Dictionary with date information
        """
        import re
        from datetime import datetime
        
        # Try to extract date from filename
        # Pattern: WAVEAN followed by 8 digits (YYYYMMDD)
        date_pattern = r'WAVEAN(\d{8})'
        match = re.search(date_pattern, filename)
        
        if match:
            date_str = match.group(1)
            try:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                return {
                    "date": date_obj.date(),
                    "year": date_obj.year,
                    "month": date_obj.month,
                    "day": date_obj.day,
                    "day_of_year": date_obj.timetuple().tm_yday,
                    "weekday": date_obj.weekday(),
                    "month_name": date_obj.strftime('%B'),
                    "season": self._get_season(date_obj.month)
                }
            except ValueError:
                pass
        
        # Fallback: try to extract year from filename
        year_pattern = r'(\d{4})'
        year_match = re.search(year_pattern, filename)
        if year_match:
            year = int(year_match.group(1))
            return {
                "date": None,
                "year": year,
                "month": None,
                "day": None,
                "day_of_year": None,
                "weekday": None,
                "month_name": None,
                "season": None
            }
        
        return {
            "date": None,
            "year": None,
            "month": None,
            "day": None,
            "day_of_year": None,
            "weekday": None,
            "month_name": None,
            "season": None
        }
    
    def _get_season(self, month: int) -> str:
        """Get season from month number."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "autumn"
        else:
            return "unknown"


def main():
    """Main function to run the model evaluation."""
    parser = argparse.ArgumentParser(description="Model evaluation with spatial analysis")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Validate required config values
    model_path = config.get("data", {}).get("model_path")
    output_dir = config.get("output", {}).get("output_dir")
    data_path = config.get("data", {}).get("data_path")
    
    if not model_path:
        logger.error("model_path not specified in config")
        sys.exit(1)
    if not output_dir:
        logger.error("output_dir not specified in config")
        sys.exit(1)
    if not data_path:
        logger.error("data_path not specified in config")
        sys.exit(1)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        output_dir=output_dir,
        config=config
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            data_path=data_path,
            year=config.get("evaluation", {}).get("year"),
            file_pattern=config.get("data", {}).get("file_pattern"),
            max_files=config.get("evaluation", {}).get("max_files")
        )
        
        print("\n" + "="*60)
        print("🎉 MODEL EVALUATION COMPLETED!")
        print("="*60)
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Files processed: {results['total_files_processed']}")
        print(f"📈 Total samples: {results['total_samples']}")
        print(f"📉 Mean RMSE: {results['mean_metrics']['rmse']:.4f}")
        print(f"📉 Mean MAE: {results['mean_metrics']['mae']:.4f}")
        print(f"📉 Mean Bias: {results['mean_metrics']['bias']:.4f}")
        print(f"📉 Mean Pearson: {results['mean_metrics']['pearson']:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        # Clean up resources
        evaluator.cleanup()


if __name__ == "__main__":
    main()

