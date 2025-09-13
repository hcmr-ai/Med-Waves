import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from comet_ml import Experiment
from tqdm import tqdm

from src.evaluation.metrics import evaluate_model
from src.evaluation.visuals import plot_residual_distribution
from src.features.helpers import (
    extract_features_from_file,
    extract_features_from_parquet,
)


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

        # Initialize components
        from sklearn.linear_model import SGDRegressor
        from sklearn.preprocessing import StandardScaler

        self.model = SGDRegressor()
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

            experiment = Experiment(
                api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                project_name=config["comet_project"],
                workspace=config["comet_workspace"],
                experiment_name=run_name
            )
            experiment.log_parameters(config)
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
                       if col not in ["vhm0_y", "time", "lat", "lon"] and not col.startswith("_")]
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
        y_raw = df["vhm0_y"].to_numpy()

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
                X_transformed = self.selector.fit(X_transformed).transform(X_transformed)
                self.feature_counts["after_selection"] = X_transformed.shape[1]
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
        print(f"Starting warmup stage with {self.warmup_days} days...")

        warmup_data = []
        warmup_targets = []

        # Collect warmup data
        for i in range(min(self.warmup_days, len(x_train_files))):
            # Check if we're using parquet files (same file for x and y)
            if x_train_files[i] == y_train_files[i] and x_train_files[i].endswith('.parquet'):
                df = extract_features_from_parquet(
                    x_train_files[i], use_dask=self.use_dask
                )
            else:
                df = extract_features_from_file(
                    x_train_files[i], y_train_files[i], use_dask=self.use_dask
                )
            X, y = self._prepare_features(df, is_warmup=True)
            warmup_data.append(X)
            warmup_targets.append(y)

        # Combine warmup data
        X_warmup = np.vstack(warmup_data)
        # y_warmup = np.concatenate(warmup_targets)

        print(f"Warmup data shape: {X_warmup.shape}")

        # Apply transforms and fit them
        X_transformed = self._apply_transforms(X_warmup, is_warmup=True)

        # Special handling for IncrementalPCA
        if self.dimred is not None and self.dimred_type == "ipca":
            # Fit IncrementalPCA on warmup data
            self.dimred.fit(X_transformed)
            X_transformed = self.dimred.transform(X_transformed)
            self.feature_counts["after_dimred"] = X_transformed.shape[1]

        self.warmup_completed = True

        # Log feature report
        self._log_feature_report()

        print("Warmup stage completed!")

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
        # Warmup stage
        if not self.warmup_completed:
            self._warmup_stage(x_train_files, y_train_files)

        # Training stage
        print("Starting training stage...")
        for i in tqdm(range(0, len(x_train_files), self.batch_size)):
            batch_dfs = []
            for j in range(i, min(i + self.batch_size, len(x_train_files))):
                # Check if we're using parquet files (same file for x and y)
                if x_train_files[j] == y_train_files[j] and x_train_files[j].endswith('.parquet'):
                    df = extract_features_from_parquet(
                        x_train_files[j], use_dask=self.use_dask
                    )
                else:
                    df = extract_features_from_file(
                        x_train_files[j], y_train_files[j], use_dask=self.use_dask
                    )
                batch_dfs.append(df)

            batch = pl.concat(batch_dfs)
            X, y = self._prepare_features(batch, is_warmup=False)

            if len(X) == 0:
                continue

            # Apply frozen transforms
            X_transformed = self._apply_transforms(X, is_warmup=False)

            # Train model
            self.model.partial_fit(X_transformed, y)

            if self.log_batch_metrics:
                y_pred = self._predict_batch(X)
                metrics = evaluate_model(y_pred, y)
                print(f"Batch {i // self.batch_size + 1} Metrics:")
                for k, v in metrics.items():
                    print(f"{k.upper()}: {v}")
                    if self.experiment:
                        self.experiment.log_metric(f"train_{k.upper()}", v, step=i)

            if self.save_model:
                self._save_artifacts()

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict on a batch of data."""
        X_transformed = self._apply_transforms(X, is_warmup=False)
        return self.model.predict(X_transformed)

    def evaluate(self, x_test_files: List[str], y_test_files: List[str]):
        """Evaluate model on test data."""
        print("Starting evaluation...")
        test_dfs = []
        for x_f, y_f in zip(x_test_files, y_test_files, strict=False):
            # Check if we're using parquet files (same file for x and y)
            if x_f == y_f and x_f.endswith('.parquet'):
                df = extract_features_from_parquet(x_f, use_dask=self.use_dask)
            else:
                df = extract_features_from_file(x_f, y_f, use_dask=self.use_dask)
            test_dfs.append(df)

        test_batch = pl.concat(test_dfs)
        X, y = self._prepare_features(test_batch, is_warmup=False)

        if len(X) == 0:
            print("No valid test data found!")
            return

        y_pred = self._predict_batch(X)

        metrics = evaluate_model(y_pred, y)

        print("\nFinal Evaluation:")
        for k, v in metrics.items():
            print(f"{k.upper()}: {v}")
            if self.experiment:
                self.experiment.log_metric(f"eval_{k.upper()}", v)

        fig = plot_residual_distribution(y, y_pred, "residual_distribution")

        if self.experiment:
            self.experiment.log_figure(figure=fig, figure_name="residual_distribution")

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
        joblib.dump(self.model, self.save_path)
