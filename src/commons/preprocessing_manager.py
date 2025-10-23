"""
PreprocessingManager for Wave Height Bias Correction Research

This module provides the PreprocessingManager class for handling all preprocessing
and scaling operations in a modular and reusable way.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.commons.preprocessing import RegionalScaler

logger = logging.getLogger(__name__)


class PreprocessingManager:
    """Handles all preprocessing and scaling operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize PreprocessingManager with configuration."""
        self.config = config
        self.feature_config = config.get("feature_block", {})
        self.logger = logging.getLogger(__name__)

        # Initialize scalers
        self.scaler = None
        self.regional_scaler = None

        # Initialize preprocessing components
        self._initialize_preprocessing()

    def _initialize_preprocessing(self):
        """Initialize preprocessing components."""
        # Validate configuration first
        self._validate_config()

        # Scaler
        scaler_type = self.feature_config.get("scaler", "standard")
        use_regional_scaling = self.feature_config.get("regional_scaling", {}).get(
            "enabled", False
        )

        # Check if scaler is null first
        if scaler_type is None or scaler_type == "null":
            if use_regional_scaling:
                self.logger.warning(
                    "Cannot use regional scaling with null scaler. Disabling regional scaling."
                )
                use_regional_scaling = False

            self.logger.info("No scaling applied - using raw features")
            self.scaler = None
            self.regional_scaler = None
        elif use_regional_scaling:
            self.logger.info("Using regional scaling for geographic regions")
            self.regional_scaler = RegionalScaler(
                base_scaler=scaler_type, region_column="atlantic_region"
            )
            # No need for regular scaler when using regional scaling
            self.scaler = None
        else:
            self.logger.info(f"Using {scaler_type} scaling")
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "robust":
                self.scaler = RobustScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.logger.warning(
                    f"Unknown scaler type '{scaler_type}', using StandardScaler"
                )
                self.scaler = StandardScaler()
            # No regional scaler when using standard scaling
            self.regional_scaler = None

    def _validate_config(self):
        """Validate preprocessing configuration."""
        scaler_type = self.feature_config.get("scaler", "standard")
        use_regional_scaling = self.feature_config.get("regional_scaling", {}).get(
            "enabled", False
        )

        # Check for configuration conflicts
        if (scaler_type is None or scaler_type == "null") and use_regional_scaling:
            self.logger.warning(
                "Configuration conflict: scaler is 'null' but regional_scaling is enabled."
            )
            self.logger.warning("Regional scaling requires a base scaler. Consider:")
            self.logger.warning(
                "  - Set scaler to 'standard', 'robust', or 'minmax' for regional scaling"
            )
            self.logger.warning(
                "  - Set regional_scaling.enabled to false to disable regional scaling"
            )

    def fit_preprocessing(
        self, X_train: np.ndarray, regions_train: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit preprocessing components on training data.

        Args:
            X_train: Training feature matrix
            regions_train: Training region information (optional)
        """
        if self.regional_scaler is not None and regions_train is not None:
            self.logger.info("Fitting regional scaler on training data...")
            self.regional_scaler.fit_with_regions(X_train, regions_train)
        elif self.scaler is not None:
            self.logger.info("Fitting scaler on training data...")
            self.scaler.fit(X_train)
        else:
            self.logger.info("No preprocessing to fit - using raw features")

    def transform_data(
        self, X: np.ndarray, regions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transform data using fitted preprocessing components.

        Args:
            X: Feature matrix to transform
            regions: Region information (optional)

        Returns:
            Transformed feature matrix
        """
        if self.regional_scaler is not None and regions is not None:
            return self.regional_scaler.transform_with_regions(X, regions)
        elif self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X

    def preprocess_splits(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        regions_train: Optional[np.ndarray] = None,
        regions_val: Optional[np.ndarray] = None,
        regions_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing to train/validation/test splits.

        Args:
            X_train: Training feature matrix
            X_val: Validation feature matrix
            X_test: Test feature matrix
            regions_train: Training region information (optional)
            regions_val: Validation region information (optional)
            regions_test: Test region information (optional)

        Returns:
            Tuple of (transformed_X_train, transformed_X_val, transformed_X_test)
        """
        self.logger.info("Preprocessing data splits...")

        # Fit preprocessing on training data
        self.fit_preprocessing(X_train, regions_train)

        # Transform all splits
        X_train_transformed = self.transform_data(X_train, regions_train)

        X_val_transformed = X_val
        if len(X_val) > 0:
            X_val_transformed = self.transform_data(X_val, regions_val)
        else:
            self.logger.warning("Skipping validation set preprocessing (empty set)")

        X_test_transformed = X_test
        if len(X_test) > 0:
            X_test_transformed = self.transform_data(X_test, regions_test)
        else:
            self.logger.warning("Skipping test set preprocessing (empty set)")

        return X_train_transformed, X_val_transformed, X_test_transformed

    def preprocess_splits_memory_efficient(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        regions_train: Optional[np.ndarray] = None,
        regions_val: Optional[np.ndarray] = None,
        regions_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply preprocessing to splits with memory optimization.

        Args:
            X_train: Training feature matrix
            X_val: Validation feature matrix
            X_test: Test feature matrix
            regions_train: Training region information (optional)
            regions_val: Validation region information (optional)
            regions_test: Test region information (optional)

        Returns:
            Tuple of (transformed_X_train, transformed_X_val, transformed_X_test)
        """
        self.logger.info("Preprocessing data splits (memory optimized)...")
        if X_train.dtype == np.float64:
            self.logger.info(
                "Converting features from float64 to float32 (50% memory reduction)"
            )
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            X_test = X_test.astype(np.float32)

        # Fit preprocessing on training data
        self.fit_preprocessing(X_train, regions_train)

        # Transform training data
        X_train_transformed = self.transform_data(X_train, regions_train)

        # Transform validation data with memory optimization
        X_val_transformed = X_val
        if len(X_val) > 0:
            X_val_transformed = self.transform_data(X_val, regions_val)
        else:
            self.logger.warning("Skipping validation set preprocessing (empty set)")

        # Transform test data with memory optimization
        X_test_transformed = X_test
        if len(X_test) > 0:
            X_test_transformed = self.transform_data(X_test, regions_test)
        else:
            self.logger.warning("Skipping test set preprocessing (empty set)")

        return X_train_transformed, X_val_transformed, X_test_transformed

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about current preprocessing configuration.

        Returns:
            Dictionary with preprocessing information
        """
        info = {
            "scaler_type": self.feature_config.get("scaler", "standard"),
            "regional_scaling_enabled": self.feature_config.get(
                "regional_scaling", {}
            ).get("enabled", False),
            "has_scaler": self.scaler is not None,
            "has_regional_scaler": self.regional_scaler is not None,
        }

        if self.scaler is not None:
            info["scaler_class"] = self.scaler.__class__.__name__

        if self.regional_scaler is not None:
            info["regional_scaler_class"] = self.regional_scaler.__class__.__name__
            info["regional_scaler_base"] = self.regional_scaler.base_scaler

        return info

    def log_preprocessing_info(self):
        """Log information about current preprocessing configuration."""
        info = self.get_preprocessing_info()

        self.logger.info("Preprocessing Configuration:")
        self.logger.info(f"  Scaler type: {info['scaler_type']}")
        self.logger.info(
            f"  Regional scaling: {'Enabled' if info['regional_scaling_enabled'] else 'Disabled'}"
        )

        if info["has_scaler"]:
            self.logger.info(f"  Scaler: {info['scaler_class']}")

        if info["has_regional_scaler"]:
            self.logger.info(
                f"  Regional scaler: {info['regional_scaler_class']} (base: {info['regional_scaler_base']})"
            )

        if not info["has_scaler"] and not info["has_regional_scaler"]:
            self.logger.info("  No preprocessing applied - using raw features")

    def save_preprocessing(self, save_path: Path) -> None:
        """
        Save preprocessing components to disk.

        Args:
            save_path: Path to save preprocessing components
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save scaler
        if self.scaler is not None:
            scaler_path = save_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")

        # Save regional scaler
        if self.regional_scaler is not None:
            regional_scaler_path = save_path / "regional_scaler.pkl"
            joblib.dump(self.regional_scaler, regional_scaler_path)
            self.logger.info(f"Saved regional scaler to {regional_scaler_path}")

    def load_preprocessing(self, load_path: Path) -> None:
        """
        Load preprocessing components from disk.

        Args:
            load_path: Path to load preprocessing components from
        """
        load_path = Path(load_path)

        # Load scaler
        scaler_path = load_path / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")
        else:
            self.scaler = None

        # Load regional scaler
        regional_scaler_path = load_path / "regional_scaler.pkl"
        if regional_scaler_path.exists():
            self.regional_scaler = joblib.load(regional_scaler_path)
            self.logger.info(f"Loaded regional scaler from {regional_scaler_path}")
        else:
            self.regional_scaler = None

    def inverse_transform(
        self, X: np.ndarray, regions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Inverse transform data back to original scale.

        Args:
            X: Transformed feature matrix
            regions: Region information (optional)

        Returns:
            Inverse transformed feature matrix
        """
        if self.regional_scaler is not None and regions is not None:
            return self.regional_scaler.inverse_transform_with_regions(X, regions)
        elif self.scaler is not None:
            return self.scaler.inverse_transform(X)
        else:
            return X
