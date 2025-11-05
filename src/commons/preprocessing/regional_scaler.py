"""
Regional Scaler for geographic region-based preprocessing.

This module provides functionality to apply different scaling to different
geographic regions, which is useful for wave prediction models where
Atlantic and Mediterranean conditions have different characteristics.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class RegionalScaler(BaseEstimator, TransformerMixin):
    """
    Regional scaler that applies different scaling to different geographic regions.
    This helps preserve regional differences in wave physics while normalizing features.

    Example:
        >>> scaler = RegionalScaler(base_scaler="standard")
        >>> scaler.fit_with_regions(X_train, regions_train)
        >>> X_scaled = scaler.transform_with_regions(X_test, regions_test)
    """

    def __init__(self, base_scaler="standard", region_column="atlantic_region"):
        """
        Initialize regional scaler.

        Args:
            base_scaler: Type of scaler to use ("standard", "robust", "minmax")
            region_column: Column name for regional classification (for compatibility)
        """
        self.base_scaler = base_scaler
        self.region_column = region_column
        self.scalers = {}
        self.region_masks = {}

    def _get_scaler(self):
        """Get the base scaler instance."""
        if self.base_scaler == "standard":
            return StandardScaler()
        elif self.base_scaler == "robust":
            return RobustScaler()
        elif self.base_scaler == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

    def fit(self, X, y=None):
        """
        Fit regional scalers on training data.

        Args:
            X: Feature matrix with region information
            y: Target vector (unused)
        """
        # Extract region information
        if isinstance(X, np.ndarray):
            # If X is numpy array, assume region info is in the last columns
            # This is a simplified approach - in practice, you'd need to pass region info separately
            raise ValueError(
                "RegionalScaler requires region information. Use fit_with_regions() method."
            )

        return self

    def fit_with_regions(self, X, regions):
        """
        Fit regional scalers with explicit region information.

        Args:
            X: Feature matrix
            regions: Array of region labels (e.g., ['atlantic', 'mediterranean', ...])

        Returns:
            self: Fitted scaler
        """
        unique_regions = np.unique(regions)
        logger.info(f"Fitting regional scalers for regions: {unique_regions}")

        for region in unique_regions:
            region_mask = regions == region
            region_data = X[region_mask]

            if len(region_data) > 0:
                scaler = self._get_scaler()
                scaler.fit(region_data)
                self.scalers[region] = scaler
                self.region_masks[region] = region_mask
                logger.info(f"Fitted scaler for {region}: {len(region_data)} samples")
            else:
                logger.warning(f"No data found for region: {region}")

        return self

    def transform(self, X, regions=None):
        """
        Transform data using regional scalers.

        Args:
            X: Feature matrix
            regions: Array of region labels (if None, assumes X contains region info)
        """
        if regions is None:
            raise ValueError(
                "RegionalScaler requires region information for transform. Use transform_with_regions() method."
            )

        X_transformed = X.copy()

        for region, scaler in self.scalers.items():
            region_mask = regions == region
            if np.any(region_mask):
                region_data = X[region_mask]
                X_transformed[region_mask] = scaler.transform(region_data)

        return X_transformed

    def transform_with_regions(self, X, regions):
        """Transform data with explicit region information."""
        return self.transform(X, regions)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step (requires region info)."""
        raise NotImplementedError(
            "Use fit_with_regions() and transform_with_regions() for RegionalScaler"
        )

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"base_scaler": self.base_scaler, "region_column": self.region_column}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
