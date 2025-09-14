import logging
import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from comet_ml import Experiment
from tqdm import tqdm
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

from src.evaluation.metrics import evaluate_model
from src.evaluation.visuals import plot_residual_distribution
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
                rounds_per_batch=25, # trees added per streamed batch
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                max_bin=256,
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
        parts = ["sgd"]

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

    def _prepare_features(self, df: pl.DataFrame, is_warmup: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features based on configuration."""
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

    def train(self, x_train_files: List[str], y_train_files: List[str]):
        """Main training loop with warmup and streaming."""
        logger.info(f"Starting training with {len(x_train_files)} training files")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Warmup stage
        if not self.warmup_completed:
            self._warmup_stage(x_train_files, y_train_files)

        # Training stage
        logger.info("Starting training stage...")
        total_batches = (len(x_train_files) + self.batch_size - 1) // self.batch_size
        logger.info(f"Total batches to process: {total_batches}")

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
            if self.log_batch_metrics:
                # we already have transformed X; predict directly
                y_pred = self.model.predict(X_batch)
                metrics = evaluate_model(y_pred, y_batch)
                logger.info(f"Batch {(batch_idx // self.batch_size) + 1}/{total_batches} Metrics: " +
                            ", ".join([f"{k.upper()}={v:.5f}" for k, v in metrics.items()]))
                if self.experiment:
                    for k, v in metrics.items():
                        self.experiment.log_metric(f"train_{k.upper()}", v, step=batch_idx)

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
            X, y = self._prepare_features(df, is_warmup=False)
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
            # if self.experiment:
            #     self.experiment.log_metric(f"eval_{k.upper()}", v)
        
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
            import pandas as pd
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
            X, y = self._prepare_features(df, is_warmup=False)
            if len(X) == 0:
                continue
                
            y_pred = self._predict_batch(X)
            
            # Sample from this file (max 10k points per file)
            max_points_per_file = 10000
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
