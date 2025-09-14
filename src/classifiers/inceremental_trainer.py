import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from comet_ml import Experiment
from tqdm import tqdm
from datetime import datetime
from src.evaluation.helpers import TrainingMetricsStorage

# Set up logger
logger = logging.getLogger(__name__)

from src.evaluation.metrics import evaluate_model
from src.evaluation.visuals import plot_residual_distribution
from src.data_engineering.split import (
    stratified_sample_by_location,
    random_sample_within_file,
    temporal_sample_within_file
)
from src.features.helpers import (
    extract_features_from_file,
    extract_features_from_parquet,
)

from src.classifiers.incremental_xgb import XGBIncremental


class FeatureSelector:
    """Custom feature selector that preserves indices for streaming application."""

    def __init__(self, selector, feature_names=None):
        self.selector = selector
        self.feature_names = feature_names
        self.selected_indices_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        self.selected_indices_ = self.selector.get_support(indices=True)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise ValueError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_indices_
        return self.selector.get_support()


class IncrementalTrainer:
    def __init__(self, config: Dict[str, Any]):
        # Support both old and new config formats
        if "data_dir" in config:
            self.data_dir = config["data_dir"]
        else:
            # Fallback to old format
            self.data_dir_x = config.get("data_dir_x", "")
            self.data_dir_y = config.get("data_dir_y", "")
            self.data_dir = self.data_dir_x  # Use x_dir as default

        self.batch_size = config["batch_size"]
        self.use_dask = config["use_dask"]
        self.save_model = config["save_model"]
        self.save_path = config["save_path"]
        self.log_batch_metrics = config["log_batch_metrics"]

        # Training diagnostics configuration
        self.diagnostics_config = config.get("training_diagnostics", {})
        self.diagnostics_enabled = self.diagnostics_config.get("enabled", False)

        # Feature configuration
        self.feature_config = config.get("feature_block", {})
        self.base_features = self.feature_config.get("base_features", ["vhm0_x", "wspd", "lat", "lon"])
        self.use_poly = self.feature_config.get("use_poly", False)
        self.poly_degree = self.feature_config.get("poly_degree", 2)
        self.poly_scope = self.feature_config.get("poly_scope", "subset")
        self.use_selector = self.feature_config.get("use_selector", False)
        self.selector_type = self.feature_config.get("selector_type", "elasticnet")
        self.selector_alpha = self.feature_config.get("selector_alpha", 1e-3)
        self.selector_l1_ratio = self.feature_config.get("selector_l1_ratio", 0.1)
        self.warmup_days = self.feature_config.get("warmup_days", 7)
        self.use_dimred = self.feature_config.get("use_dimred", False)
        self.dimred_type = self.feature_config.get("dimred_type", "ipca")
        self.dimred_components = self.feature_config.get("dimred_components", 64)
        self.model_type = self.feature_config.get("model_type", "sgd")

        # Initialize components
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        if self.model_type == "sgd":
            self.model = SGDRegressor(
                loss="huber", epsilon=1.0,
                penalty="elasticnet", alpha=1e-4, l1_ratio=0.1,
                learning_rate="invscaling", eta0=0.01, power_t=0.25,
                random_state=42
            )
        elif self.model_type == "xgb":
            self.model = XGBIncremental(
                rounds_per_batch=50, # trees added per streamed batch (increased)
                max_depth=6,         # increased depth for more complex patterns
                learning_rate=0.1,   # increased learning rate
                subsample=0.9,       # increased subsample
                colsample_bytree=0.9, # increased column sampling
                tree_method="hist",
                max_bin=256,
                min_child_weight=1,  # reduced from 10 to allow more splits
                reg_alpha=0.01,      # reduced L1 regularization
                reg_lambda=0.1       # reduced L2 regularization
            )
        self.scaler = StandardScaler()
        self.poly = None
        self.selector = None
        self.dimred = None

        # State tracking
        self.warmup_completed = False
        self.experiment = None
        self.feature_counts = {}

        self._setup_model_tracker(config)
        self._setup_transforms()

    def _setup_model_tracker(self, config: Dict[str, Any]):
        if config.get("use_comet", False):
            # Generate run name based on config
            run_name = self._generate_run_name()
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{run_name}"
            experiment = Experiment(
                api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                project_name=config["comet_project"],
                workspace=config["comet_workspace"],
                experiment_name=run_name
            )
            experiment.log_parameters(config)
            experiment.set_name(run_name)
            self.experiment = experiment

    def _generate_run_name(self) -> str:
        """Generate run name based on feature configuration."""
        parts = [self.model_type]

        if self.use_poly:
            parts.append(f"poly{self.poly_degree}")
            if self.poly_scope == "subset":
                parts.append("subset")

        if self.use_selector:
            parts.append(f"{self.selector_type}{self.selector_alpha}")
            if self.selector_type == "elasticnet":
                parts.append(f"l1{self.selector_l1_ratio}")

        if self.use_dimred:
            parts.append(f"{self.dimred_type}{self.dimred_components}")

        return "_".join(parts)

    def _setup_transforms(self):
        """Initialize transform components based on configuration."""
        if self.use_poly:
            from sklearn.preprocessing import PolynomialFeatures
            self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)

        if self.use_selector:
            from sklearn.linear_model import ElasticNet, Lasso
            if self.selector_type == "lasso":
                base_selector = Lasso(alpha=self.selector_alpha, random_state=42)
            else:  # elasticnet
                base_selector = ElasticNet(
                    alpha=self.selector_alpha,
                    l1_ratio=self.selector_l1_ratio,
                    random_state=42
                )
            from sklearn.feature_selection import SelectFromModel

            self.selector = FeatureSelector(SelectFromModel(base_selector))

        if self.use_dimred:
            from sklearn.decomposition import IncrementalPCA, TruncatedSVD
            if self.dimred_type == "ipca":
                self.dimred = IncrementalPCA(n_components=self.dimred_components)
            elif self.dimred_type == "tsvd":
                self.dimred = TruncatedSVD(n_components=self.dimred_components, random_state=42)
            elif self.dimred_type == "rproj":
                from sklearn.random_projection import GaussianRandomProjection
                self.dimred = GaussianRandomProjection(n_components=self.dimred_components, random_state=42)

    def _get_all_features(self, df: pl.DataFrame) -> List[str]:
        """Get all available features from dataframe."""
        available_features = df.columns
        # Filter out target and non-feature columns
        feature_cols = [col for col in available_features
                       if col not in ["vhm0_y", "vhm0_x", "corrected_VTM02","time", "lat", "lon"] and not col.startswith("_")]
        return feature_cols

    def _apply_sampling(self, df: pl.DataFrame, is_warmup: bool = False) -> pl.DataFrame:
        """Apply sampling strategy to reduce file size."""
        # Get sampling configuration
        max_samples = self.feature_config.get("max_samples_per_file", None)
        sampling_strategy = self.feature_config.get("sampling_strategy", "none")
        sampling_seed = self.feature_config.get("sampling_seed", 42)
        
        # Skip sampling if not configured or during warmup
        if max_samples is None or sampling_strategy == "none" or is_warmup:
            return df
        
        logger.info(f"Applying {sampling_strategy} sampling: max {max_samples} samples")
        
        if sampling_strategy == "per_location":
            samples_per_location = self.feature_config.get("samples_per_location", 20)
            return stratified_sample_by_location(
                df, 
                max_samples_per_file=max_samples,
                samples_per_location=samples_per_location,
                seed=sampling_seed,
                location_cols=["lat", "lon"]
            )
        elif sampling_strategy == "temporal":
            samples_per_hour = self.feature_config.get("samples_per_hour", 100)
            return temporal_sample_within_file(
                df,
                max_samples_per_file=max_samples,
                samples_per_hour=samples_per_hour,
                seed=sampling_seed
            )
        elif sampling_strategy == "random":
            return random_sample_within_file(
                df,
                max_samples_per_file=max_samples,
                seed=sampling_seed
            )
        else:
            logger.warning(f"Unknown sampling strategy: {sampling_strategy}")
            return df

    def _prepare_features(self, df: pl.DataFrame, is_warmup: bool = False, apply_sampling: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features based on configuration."""
        # Apply sampling if configured
        if apply_sampling:
            df = self._apply_sampling(df, is_warmup)
        
        # Get base features
        if self.poly_scope == "subset" and self.use_poly:
            # Use only base features for polynomial expansion
            feature_cols = self.base_features
        else:
            # Use all available features
            feature_cols = self._get_all_features(df)

        X_raw = df.select(feature_cols).to_numpy()
        # y_raw = df["vhm0_y"].to_numpy()
        # bias target = observed - simulated
        y_raw = (df["vhm0_y"] - df["vhm0_x"]).to_numpy()
        logger.info(f"y_raw shape: {y_raw.shape}")
        logger.info(f"X_raw shape: {X_raw.shape}")
        logger.info(f"X_raw columns: {feature_cols}")

        # Remove NaN values
        mask = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(y_raw)
        X = X_raw[mask]
        y = y_raw[mask]

        return X, y

    def _apply_transforms(self, X: np.ndarray, is_warmup: bool = False) -> np.ndarray:
        """Apply feature transforms in sequence."""
        X_transformed = X.copy()

        # Track feature counts
        if is_warmup:
            self.feature_counts["input"] = X.shape[1]

        # Polynomial features
        if self.poly is not None:
            if is_warmup:
                X_transformed = self.poly.fit_transform(X_transformed)
                self.feature_counts["after_poly"] = X_transformed.shape[1]
            else:
                X_transformed = self.poly.transform(X_transformed)

        # Scaling
        if is_warmup:
            X_transformed = self.scaler.fit_transform(X_transformed)
        else:
            X_transformed = self.scaler.transform(X_transformed)

        # Feature selection
        if self.selector is not None:
            if is_warmup:
                # needs y; stash it earlier in warmup (see next diff)
                X_transformed = self.selector.fit(X_transformed, self._warmup_y).transform(X_transformed)
            else:
                X_transformed = self.selector.transform(X_transformed)

        # Dimension reduction
        if self.dimred is not None:
            if is_warmup:
                if self.dimred_type == "ipca":
                    # For IncrementalPCA, we need to call partial_fit multiple times
                    # This will be handled in the warmup loop
                    pass
                else:
                    X_transformed = self.dimred.fit_transform(X_transformed)
                self.feature_counts["after_dimred"] = X_transformed.shape[1]
            else:
                X_transformed = self.dimred.transform(X_transformed)

        return X_transformed

    def _warmup_stage(self, x_train_files: List[str], y_train_files: List[str]):
        """Warmup stage: fit transforms on first warmup_days of data."""
        logger.info(f"Starting warmup stage with {self.warmup_days} days...")
        logger.info(f"Processing {min(self.warmup_days, len(x_train_files))} files for warmup")

        # Collect warmup data
        MAX_WARMUP_ROWS = 1_000_000  # cap ~1M rows to keep RAM sane
        rows_seen = 0
        X_accum, y_accum = [], []
        for i in range(min(self.warmup_days, len(x_train_files))):
            df = extract_features_from_parquet(x_train_files[i], use_dask=self.use_dask) \
                if x_train_files[i] == y_train_files[i] and x_train_files[i].endswith('.parquet') \
                    else extract_features_from_file(x_train_files[i], y_train_files[i], use_dask=self.use_dask)
            X, y = self._prepare_features(df, is_warmup=True)
            if X.size == 0: 
                continue
            take = min(MAX_WARMUP_ROWS - rows_seen, X.shape[0])
            if take <= 0:
                break
            X_accum.append(X[:take]); y_accum.append(y[:take])
            rows_seen += take
        # Combine warmup data
        X_warmup = np.vstack(X_accum)
        self._warmup_y = np.concatenate(y_accum)

        logger.info(f"Combined warmup data shape: {X_warmup.shape}")
        logger.info(f"Warmup data statistics - min: {X_warmup.min():.4f}, max: {X_warmup.max():.4f}, mean: {X_warmup.mean():.4f}")

        # Fit poly & scaler on the warmup slice
        logger.info("Applying and fitting transforms...")
        X_transformed = self._apply_transforms(X_warmup, is_warmup=True)
        logger.info(f"Transformed warmup data shape: {X_transformed.shape}")

        # IPCA supports streaming; do a single fit here and transform
        if self.dimred is not None and self.dimred_type == "ipca":
            logger.info("Fitting IncrementalPCA on warmup data...")
            # Fit IncrementalPCA on warmup data
            self.dimred.fit(X_transformed)
            X_transformed = self.dimred.transform(X_transformed)
            self.feature_counts["after_dimred"] = X_transformed.shape[1]
            logger.info(f"After IncrementalPCA: {X_transformed.shape}")

        self.warmup_completed = True

        # Log feature report
        self._log_feature_report()

        logger.info("Warmup stage completed successfully!")

    def _log_feature_report(self):
        """Log feature transformation report to Comet."""
        if not self.experiment:
            return

        report = {
            "input_features": self.feature_counts.get("input", 0),
            "after_poly": self.feature_counts.get("after_poly", self.feature_counts.get("input", 0)),
            "after_selection": self.feature_counts.get("after_selection", self.feature_counts.get("after_poly", self.feature_counts.get("input", 0))),
            "after_dimred": self.feature_counts.get("after_dimred", self.feature_counts.get("after_selection", self.feature_counts.get("after_poly", self.feature_counts.get("input", 0))))
        }

        for key, value in report.items():
            self.experiment.log_metric(f"feature_count_{key}", value)

        # Log explained variance for dimension reduction
        if self.dimred is not None and hasattr(self.dimred, 'explained_variance_ratio_'):
            explained_var = self.dimred.explained_variance_ratio_.sum()
            self.experiment.log_metric("explained_variance_ratio", explained_var)

        # Log selector coefficients if available
        if self.selector is not None and hasattr(self.selector.selector, 'estimator_'):
            coeffs = self.selector.selector.estimator_.coef_
            self._log_selector_coefficients(coeffs)

    def _log_selector_coefficients(self, coefficients: np.ndarray):
        """Log feature selector coefficients as a bar chart."""
        if not self.experiment:
            return

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(coefficients)), coefficients)
        plt.title("Feature Selector Coefficients")
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        plt.tight_layout()

        self.experiment.log_figure(figure=plt, figure_name="selector_coefficients")
        plt.close()

    def _create_validation_split(self, x_train_files: List[str], y_train_files: List[str]):
        """Create validation set from training data using temporal split."""
        validation_split = self.diagnostics_config.get("validation_split", 0.2)
        
        # Use temporal split (last portion of files chronologically)
        split_idx = int(len(x_train_files) * (1 - validation_split))
        
        x_val_files = x_train_files[split_idx:]
        y_val_files = y_train_files[split_idx:]
        x_train_files = x_train_files[:split_idx]
        y_train_files = y_train_files[:split_idx]
        logger.info(f"x_train_files: {x_train_files}")
        logger.info(f"y_train_files: {y_train_files}")
        logger.info(f"x_val_files: {x_val_files}")
        logger.info(f"y_val_files: {y_val_files}")

        return x_train_files, y_train_files, x_val_files, y_val_files

    def _quick_validation(self, val_files: List[str], max_files: int = 5) -> Dict[str, float]:
        """Ultra-fast validation using minimal data."""
        max_files = self.diagnostics_config.get("max_validation_files", max_files)
        # samples_per_file = self.diagnostics_config.get("quick_validation_samples", 1000)
        
        sample_files = val_files[:max_files]
        total_samples = 0
        total_rmse = 0.0
        total_mae = 0.0
        all_y_true = []
        all_y_pred = []
        
        for file_path in sample_files:
            try:
                # Load and process file
                df = extract_features_from_parquet(file_path, use_dask=self.use_dask)
                X, y = self._prepare_features(df, is_warmup=False, apply_sampling=True)
                
                if len(X) == 0:
                    continue
                
                # Sample for speed
                # if len(X) > samples_per_file:
                #     indices = np.random.choice(len(X), samples_per_file, replace=False)
                #     X, y = X[indices], y[indices]
                
                # Transform and predict
                X_transformed = self._apply_transforms(X, is_warmup=False)
                y_pred = self.model.predict(X_transformed)
                
                # Accumulate metrics
                total_samples += len(y)
                total_rmse += np.sum((y - y_pred) ** 2)
                total_mae += np.sum(np.abs(y - y_pred))
                
                # Collect all predictions and true values for Pearson correlation
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)
                
            except Exception as e:
                logger.warning(f"Error processing validation file {file_path}: {e}")
                continue
        
        if total_samples == 0:
            return {"rmse": float('inf'), "mae": float('inf'), "pearson": 0.0, "samples": 0}
        
        # Calculate Pearson correlation using numpy
        if len(all_y_true) > 1 and len(all_y_pred) > 1:
            # Convert to numpy arrays for correlation calculation
            y_true_array = np.array(all_y_true)
            y_pred_array = np.array(all_y_pred)
            
            # Calculate Pearson correlation coefficient
            correlation_matrix = np.corrcoef(y_true_array, y_pred_array)
            pearson = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        else:
            pearson = 0.0
        
        return {
            "rmse": np.sqrt(total_rmse / total_samples),
            "mae": total_mae / total_samples,
            "pearson": pearson,
            "samples": total_samples
        }

    def _track_feature_importance(self, batch_idx: int) -> Optional[Dict[str, Any]]:
        """Track feature importance efficiently."""
        if not self.diagnostics_config.get("track_feature_importance", True):
            return None
            
        frequency = self.diagnostics_config.get("feature_importance_frequency", 10)
        if batch_idx % frequency != 0:
            return None
            
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        try:
            importance = self.model.feature_importances_
            return {
                'batch': batch_idx,
                'importance': importance.tolist(),  # Convert to list for JSON serialization
                'top_features': np.argsort(importance)[-10:].tolist(),  # Top 10 features
                'max_importance': float(np.max(importance)),
                'mean_importance': float(np.mean(importance))
            }
        except Exception as e:
            logger.warning(f"Error tracking feature importance: {e}")
            return None

    def train(self, x_train_files: List[str], y_train_files: List[str]):
        """Main training loop with warmup and streaming."""
        logger.info(f"Starting training with {len(x_train_files)} training files")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Create validation split if diagnostics enabled
        x_val_files, y_val_files = [], []
        if self.diagnostics_enabled:
            x_train_files, y_train_files, x_val_files, y_val_files = \
                self._create_validation_split(x_train_files, y_train_files)
            logger.info(f"Created validation split: {len(x_val_files)} validation files, {len(x_train_files)} training files")
        
        # Warmup stage
        if not self.warmup_completed:
            self._warmup_stage(x_train_files, y_train_files)

        # Training stage
        logger.info("Starting training stage...")
        total_batches = (len(x_train_files) + self.batch_size - 1) // self.batch_size
        logger.info(f"Total batches to process: {total_batches}")

        # Initialize diagnostics if enabled
        metrics_storage = None
        if self.diagnostics_enabled:
            metrics_save_path = self.diagnostics_config.get("metrics_save_path", "training_metrics")
            max_history = self.diagnostics_config.get("max_metrics_history", 1000)
            metrics_storage = TrainingMetricsStorage(metrics_save_path, max_history)
            logger.info(f"Training diagnostics enabled. Metrics will be saved to: {metrics_save_path}")

        for batch_idx in tqdm(range(0, len(x_train_files), self.batch_size), desc="Batches"):
            start = batch_idx
            end   = min(batch_idx + self.batch_size, len(x_train_files))
            X_parts, y_parts = [], []

            for j in range(start, end):
                # load one file (x==y parquet or paired)
                if x_train_files[j] == y_train_files[j] and x_train_files[j].endswith(".parquet"):
                    df = extract_features_from_parquet(x_train_files[j], use_dask=self.use_dask)
                else:
                    df = extract_features_from_file(x_train_files[j], y_train_files[j], use_dask=self.use_dask)

                X_j, y_j = self._prepare_features(df, is_warmup=False)
                if X_j.size == 0:
                    continue

                # transform NOW to avoid holding large polars frames
                X_j = self._apply_transforms(X_j, is_warmup=False)
                X_parts.append(X_j)
                y_parts.append(y_j)

            if not X_parts:
                continue

            # true batch = concat transformed chunks
            X_batch = np.vstack(X_parts)
            y_batch = np.concatenate(y_parts)

            self.model.partial_fit(X_batch, y_batch)
            
            # Enhanced batch metrics and diagnostics
            if self.log_batch_metrics or self.diagnostics_enabled:
                # we already have transformed X; predict directly
                y_pred = self.model.predict(X_batch)
                metrics = evaluate_model(y_pred, y_batch)
                
                if self.log_batch_metrics:
                    logger.info(f"Batch {(batch_idx // self.batch_size) + 1}/{total_batches} Metrics: " +
                                ", ".join([f"{k.upper()}={v:.5f}" for k, v in metrics.items()]))
                    if self.experiment:
                        for k, v in metrics.items():
                            self.experiment.log_metric(f"train_{k.upper()}", v, step=batch_idx)
            
            # Training diagnostics tracking
            if self.diagnostics_enabled and metrics_storage:
                diagnostic_frequency = self.diagnostics_config.get("diagnostic_frequency", 10)
                
                if batch_idx % diagnostic_frequency == 0:
                    # Save batch metrics
                    batch_metrics = {
                        'batch': batch_idx,
                        'samples': len(y_batch),
                        **metrics
                    }
                    metrics_storage.save_batch_metrics(batch_metrics)
                    
                    # Quick validation
                    if x_val_files:
                        val_metrics = self._quick_validation(x_val_files)
                        val_metrics['batch'] = batch_idx
                        metrics_storage.save_validation_metrics(val_metrics)
                        logger.info(f"Validation RMSE: {val_metrics['rmse']:.5f}, MAE: {val_metrics['mae']:.5f}, Pearson: {val_metrics['pearson']:.5f}")
                        if self.experiment:
                            for k, v in val_metrics.items():
                                self.experiment.log_metric(f"validation_{k.upper()}", v, step=batch_idx)
                    
                    # Feature importance tracking
                    importance_data = self._track_feature_importance(batch_idx)
                    if importance_data:
                        metrics_storage.save_feature_importance(importance_data)
                        logger.info(f"Top feature importance: {importance_data['max_importance']:.5f}")
                
                # Generate diagnostic plots
                plot_frequency = self.diagnostics_config.get("plot_frequency", 50)
                if batch_idx % plot_frequency == 0 and batch_idx > 0:
                    self._create_training_diagnostics(metrics_storage)

            if self.save_model:
                self._save_artifacts()
            
            del X_parts, y_parts, X_batch, y_batch

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict on a batch of data."""
        X_transformed = self._apply_transforms(X, is_warmup=False)
        return self.model.predict(X_transformed)

    def evaluate(self, x_test_files: List[str], y_test_files: List[str]):
        """Evaluate model on test data with both per-file and aggregate metrics."""
        logger.info("Starting evaluation...")
        
        # Initialize running statistics for aggregate metrics
        n_total = 0
        sum_y_true = 0.0
        sum_y_pred = 0.0
        sum_y_true_sq = 0.0
        sum_y_pred_sq = 0.0
        sum_y_true_y_pred = 0.0
        sum_abs_error = 0.0
        
        per_file_metrics = []
        
        for i, (x_f, y_f) in enumerate(zip(x_test_files, y_test_files, strict=False)):
            # Load and prepare data
            df = extract_features_from_parquet(x_f, use_dask=self.use_dask) \
                if x_f == y_f and x_f.endswith('.parquet') \
                    else extract_features_from_file(x_f, y_f, use_dask=self.use_dask)
            X, y = self._prepare_features(df, is_warmup=False, apply_sampling=False)
            if len(X) == 0:
                continue
                
            # Predict
            y_pred = self._predict_batch(X)
            
            # Calculate per-file metrics using evaluate_model
            file_metrics = evaluate_model(y_pred, y)
            file_name = os.path.basename(x_f)
            
            # Store per-file results
            per_file_metrics.append({
                'file': file_name,
                'samples': len(y),
                **file_metrics  # Unpack rmse, mae, bias, pearson
            })
            
            # Update running statistics for aggregate metrics
            n_total += len(y)
            sum_y_true += np.sum(y)
            sum_y_pred += np.sum(y_pred)
            sum_y_true_sq += np.sum(y**2)
            sum_y_pred_sq += np.sum(y_pred**2)
            sum_y_true_y_pred += np.sum(y * y_pred)
            sum_abs_error += np.sum(np.abs(y - y_pred))
            
            # Log per-file metrics
            logger.info(f"File {i+1}/{len(x_test_files)} ({file_name}): "
                       f"RMSE={file_metrics['rmse']:.5f}, MAE={file_metrics['mae']:.5f}, "
                       f"Bias={file_metrics['bias']:.5f}, Pearson={file_metrics['pearson']:.5f}, "
                       f"Samples={len(y)}")
        
        if n_total == 0:
            logger.info("No valid test data found!")
            return
        
        # Calculate aggregate metrics from running statistics
        mean_y_true = sum_y_true / n_total
        mean_y_pred = sum_y_pred / n_total
        
        # Calculate aggregate RMSE
        mse = (sum_y_true_sq - 2 * sum_y_true_y_pred + sum_y_pred_sq) / n_total
        aggregate_rmse = np.sqrt(mse)
        
        # Calculate aggregate MAE
        aggregate_mae = sum_abs_error / n_total
        
        # Calculate aggregate bias
        aggregate_bias = mean_y_pred - mean_y_true
        
        # Calculate aggregate Pearson correlation
        numerator = sum_y_true_y_pred - n_total * mean_y_true * mean_y_pred
        denominator = np.sqrt((sum_y_true_sq - n_total * mean_y_true**2) * 
                             (sum_y_pred_sq - n_total * mean_y_pred**2))
        aggregate_pearson = numerator / denominator if denominator != 0 else 0.0
        
        aggregate_metrics = {
            "rmse": aggregate_rmse,
            "mae": aggregate_mae,
            "bias": aggregate_bias,
            "pearson": aggregate_pearson
        }
        
        # Log aggregate results
        logger.info("\n" + "="*60)
        logger.info("AGGREGATE EVALUATION RESULTS:")
        logger.info("="*60)
        for k, v in aggregate_metrics.items():
            logger.info(f"{k.upper()}: {v:.6f}")

        # Log per-file statistics
        logger.info("\n" + "="*60)
        logger.info("PER-FILE STATISTICS:")
        logger.info("="*60)
        
        metrics_to_analyze = ['rmse', 'mae', 'bias', 'pearson']
        for metric in metrics_to_analyze:
            values = [m[metric] for m in per_file_metrics]
            logger.info(f"{metric.upper()} - Mean: {np.mean(values):.6f}, "
                    f"Std: {np.std(values):.6f}, "
                    f"Min: {np.min(values):.6f}, Max: {np.max(values):.6f}")
        
        # Log worst performing files (by RMSE)
        worst_rmse_files = sorted(per_file_metrics, key=lambda x: x['rmse'], reverse=True)[:3]
        logger.info(f"\nWorst RMSE files:")
        for i, file_metric in enumerate(worst_rmse_files, 1):
            logger.info(f"  {i}. {file_metric['file']}: RMSE={file_metric['rmse']:.6f}")
        
        # Log to Comet ML
        if self.experiment:
            # Aggregate metrics
            for k, v in aggregate_metrics.items():
                self.experiment.log_metric(f"eval_{k.upper()}", v)
            
            # Per-file statistics
            for metric in metrics_to_analyze:
                values = [m[metric] for m in per_file_metrics]
                self.experiment.log_metric(f"eval_{metric}_std", np.std(values))
            
            # Log per-file metrics as a table
            df_metrics = pd.DataFrame(per_file_metrics)
            self.experiment.log_table("per_file_metrics.csv", df_metrics)
        
        # Generate visualizations using sampled data
        self._create_diagnostic_plots_sampled(per_file_metrics, x_test_files, y_test_files)
        
        return {
            'aggregate': aggregate_metrics,
            'per_file': per_file_metrics,
            'statistics': {
                metric: {
                    'mean': np.mean([m[metric] for m in per_file_metrics]),
                    'std': np.std([m[metric] for m in per_file_metrics])
                } for metric in metrics_to_analyze
            }
        }

    def _save_artifacts(self):
        """Save all pipeline artifacts."""
        if not self.save_model:
            return

        base_path = os.path.dirname(self.save_path)
        os.makedirs(base_path, exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, os.path.join(base_path, "scaler.joblib"))

        # Save polynomial features if used
        if self.poly is not None:
            joblib.dump(self.poly, os.path.join(base_path, "poly.joblib"))

        # Save selector if used
        if self.selector is not None:
            joblib.dump(self.selector, os.path.join(base_path, "selector.joblib"))

        # Save dimension reduction if used
        if self.dimred is not None:
            joblib.dump(self.dimred, os.path.join(base_path, "dimred.joblib"))

        # Save model
        if hasattr(self.model, "save"):
            self.model.save(self.save_path if self.save_path.endswith(".json") else self.save_path + ".json")
        else:
            joblib.dump(self.model, self.save_path)

    def _create_diagnostic_plots_sampled(self, per_file_metrics, x_test_files, y_test_files):
        """Create diagnostic plots using sampled data from a few files."""

        # Sample a few files for plotting (e.g., best, worst, and middle performance)
        sorted_files = sorted(per_file_metrics, key=lambda x: x['rmse'])
        sample_files = [
            sorted_files[0],  # Best
            sorted_files[len(sorted_files)//2],  # Middle
            sorted_files[-1]   # Worst
        ]
        
        logger.info(f"Creating diagnostic plots using sample files: {[f['file'] for f in sample_files]}")
        
        # Collect sample data
        sample_y_true, sample_y_pred = [], []
        
        for file_metric in sample_files:
            file_name = file_metric['file']
            # Find the corresponding file path
            file_path = next((f for f in x_test_files if f.endswith(file_name)), None)
            if file_path is None:
                continue
                
            # Load and process the file
            df = extract_features_from_parquet(file_path, use_dask=self.use_dask)
            X, y = self._prepare_features(df, is_warmup=False, apply_sampling=True)
            if len(X) == 0:
                continue
                
            y_pred = self._predict_batch(X)
            
            # Sample from this file (max 10k points per file)
            max_points_per_file = 250000
            if len(y) > max_points_per_file:
                indices = np.random.choice(len(y), max_points_per_file, replace=False)
                sample_y_true.extend(y[indices])
                sample_y_pred.extend(y_pred[indices])
            else:
                sample_y_true.extend(y)
                sample_y_pred.extend(y_pred)
        
        if not sample_y_true:
            logger.warning("No sample data available for diagnostic plots")
            return
        
        sample_y_true = np.array(sample_y_true)
        sample_y_pred = np.array(sample_y_pred)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Diagnostic Plots (Sampled Data)', fontsize=16)
        
        # 1. Predictions vs Actual (Scatter Plot)
        ax1 = axes[0, 0]
        ax1.scatter(sample_y_true, sample_y_pred, alpha=0.3, s=0.5, color='blue')
        
        # Add perfect prediction line
        min_val = min(sample_y_true.min(), sample_y_pred.min())
        max_val = max(sample_y_true.max(), sample_y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add trend line
        z = np.polyfit(sample_y_true, sample_y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(sample_y_true, p(sample_y_true), "g--", alpha=0.8, linewidth=2, label=f'Trend Line (slope={z[0]:.3f})')
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predictions vs Actual Values')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax2 = axes[0, 1]
        residuals_plot = sample_y_pred - sample_y_true
        ax2.scatter(sample_y_pred, residuals_plot, alpha=0.3, s=0.5, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals (Predicted - Actual)')
        ax2.set_title('Residuals vs Predicted Values')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution Comparison
        ax3 = axes[1, 0]
        ax3.hist(sample_y_true, bins=50, alpha=0.7, label='Actual', density=True, color='blue')
        ax3.hist(sample_y_pred, bins=50, alpha=0.7, label='Predicted', density=True, color='red')
        ax3.set_xlabel('Values')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics using sampled data
        stats_text = f"""
        STATISTICS SUMMARY (Sampled):
        
        Actual Values:
        • Mean: {sample_y_true.mean():.6f}
        • Std:  {sample_y_true.std():.6f}
        • Min:  {sample_y_true.min():.6f}
        • Max:  {sample_y_true.max():.6f}
        
        Predicted Values:
        • Mean: {sample_y_pred.mean():.6f}
        • Std:  {sample_y_pred.std():.6f}
        • Min:  {sample_y_pred.min():.6f}
        • Max:  {sample_y_pred.max():.6f}
        
        Model Performance:
        • RMSE: {np.sqrt(np.mean((sample_y_true - sample_y_pred)**2)):.6f}
        • MAE:  {np.mean(np.abs(sample_y_true - sample_y_pred)):.6f}
        • Bias: {np.mean(sample_y_pred - sample_y_true):.6f}
        • Pearson: {np.corrcoef(sample_y_true, sample_y_pred)[0,1]:.6f}
        
        Sample Size: {len(sample_y_true):,}
        Files Used: {len(sample_files)}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Log to Comet ML
        if self.experiment:
            self.experiment.log_figure(figure=fig, figure_name="diagnostic_plots")
        
        # Also save locally
        plt.savefig('diagnostic_plots.png', dpi=300, bbox_inches='tight')
        logger.info("Diagnostic plots saved as 'diagnostic_plots.png'")
        
        plt.close()

    def _create_training_diagnostics(self, metrics_storage: TrainingMetricsStorage):
        """Create comprehensive training diagnostic plots."""
        try:
            # Load recent metrics from disk
            recent_metrics = metrics_storage.load_recent_metrics(100)  # Last 100 batches
            
            if recent_metrics.empty:
                logger.warning("No training metrics available for diagnostics")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Training Diagnostics', fontsize=16)
            
            # 1. Learning Curves
            self._plot_learning_curves(axes[0, 0], recent_metrics)
            
            # 2. Feature Importance Evolution (if available)
            self._plot_feature_importance_evolution(axes[0, 1], metrics_storage)
            
            # 3. Training Speed Analysis
            self._plot_training_speed(axes[0, 2], recent_metrics)
            
            # 4. Batch Metrics Distribution
            self._plot_batch_metrics_distribution(axes[1, 0], recent_metrics)
            
            # 5. Validation vs Training (if validation data available)
            self._plot_validation_comparison(axes[1, 1], metrics_storage)
            
            # 6. Model Performance Summary
            self._plot_performance_summary(axes[1, 2], recent_metrics)
            
            plt.tight_layout()
            plt.savefig('training_diagnostics.png', dpi=300, bbox_inches='tight')
            
            if self.experiment:
                self.experiment.log_figure(figure=fig, figure_name="training_diagnostics")
            
            logger.info("Training diagnostic plots saved as 'training_diagnostics.png'")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating training diagnostics: {e}")

    def _plot_learning_curves(self, ax, metrics_df):
        """Plot learning curves for training metrics."""
        if 'batch' not in metrics_df.columns or 'rmse' not in metrics_df.columns:
            ax.text(0.5, 0.5, 'No learning curve data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves')
            return
        
        ax.plot(metrics_df['batch'], metrics_df['rmse'], label='RMSE', alpha=0.7)
        ax.plot(metrics_df['batch'], metrics_df['mae'], label='MAE', alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Error')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance_evolution(self, ax, metrics_storage):
        """Plot feature importance evolution over time."""
        # This would require loading feature importance data from disk
        # For now, show a placeholder
        ax.text(0.5, 0.5, 'Feature Importance Evolution\n(Implementation pending)', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance Evolution')

    def _plot_training_speed(self, ax, metrics_df):
        """Plot training speed analysis."""
        if 'batch' not in metrics_df.columns or 'samples' not in metrics_df.columns:
            ax.text(0.5, 0.5, 'No training speed data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Speed')
            return
        
        # Calculate samples per batch over time
        ax.plot(metrics_df['batch'], metrics_df['samples'], alpha=0.7)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Samples per Batch')
        ax.set_title('Training Speed (Samples/Batch)')
        ax.grid(True, alpha=0.3)

    def _plot_batch_metrics_distribution(self, ax, metrics_df):
        """Plot distribution of batch metrics."""
        if 'rmse' not in metrics_df.columns:
            ax.text(0.5, 0.5, 'No batch metrics data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Batch Metrics Distribution')
            return
        
        ax.hist(metrics_df['rmse'], bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('RMSE')
        ax.set_ylabel('Frequency')
        ax.set_title('Batch RMSE Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_validation_comparison(self, ax, metrics_storage):
        """Plot validation vs training comparison."""
        try:
            # Load validation metrics from disk
            validation_metrics = self._load_validation_metrics(metrics_storage)
            
            if validation_metrics.empty:
                ax.text(0.5, 0.5, 'No validation data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validation vs Training')
                return
            
            # Load training metrics for comparison
            training_metrics = metrics_storage.load_recent_metrics(100)
            
            if training_metrics.empty:
                ax.text(0.5, 0.5, 'No training data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validation vs Training')
                return
            
            # Plot RMSE comparison
            ax.plot(training_metrics['batch'], training_metrics['rmse'], 
                   label='Training RMSE', alpha=0.7, marker='o', linewidth=2)
            ax.plot(validation_metrics['batch'], validation_metrics['rmse'], 
                   label='Validation RMSE', alpha=0.7, marker='s', linewidth=2)
            
            ax.set_xlabel('Batch')
            ax.set_ylabel('RMSE')
            ax.set_title('Training vs Validation RMSE')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading validation data:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validation vs Training')

    def _load_validation_metrics(self, metrics_storage):
        """Load validation metrics from storage."""
        return metrics_storage.load_validation_metrics(100)

    def _plot_performance_summary(self, ax, metrics_df):
        """Plot performance summary statistics."""
        if metrics_df.empty:
            ax.text(0.5, 0.5, 'No performance data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Summary')
            return
        
        # Calculate summary statistics
        summary_text = f"""
        PERFORMANCE SUMMARY:
        
        Total Batches: {len(metrics_df)}
        Avg RMSE: {metrics_df['rmse'].mean():.5f}
        Min RMSE: {metrics_df['rmse'].min():.5f}
        Max RMSE: {metrics_df['rmse'].max():.5f}
        
        Avg MAE: {metrics_df['mae'].mean():.5f}
        Min MAE: {metrics_df['mae'].min():.5f}
        Max MAE: {metrics_df['mae'].max():.5f}
        
        Total Samples: {metrics_df['samples'].sum():,}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.axis('off')
        ax.set_title('Performance Summary')
