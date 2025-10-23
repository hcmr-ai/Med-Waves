"""
FeatureEngineer for Wave Height Bias Correction Research

This module provides the FeatureEngineer class for handling feature engineering operations
including geographic context, lag features, regional classification, and feature preparation.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl

from src.classifiers.build_cluster_map import GridClusterMapper, PointClusterMapper
from src.commons.region_mapping import RegionMapper


class FeatureEngineer:
    """Handles feature engineering operations including geographic context, lag features, and regional classification."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize FeatureEngineer with configuration."""
        self.feature_config = config.get("feature_block", {})
        self.logger = logging.getLogger(__name__)
        self.feature_names = None

    def prepare_features(
        self, df: pl.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Prepare features, target, and regions from dataframe.

        Args:
            df: Input dataframe
            target_column: Name of the target column

        Returns:
            Tuple of (X, y, regions, coords) where:
                X: Feature matrix
                y: Target vector
                regions: Region information for regional scaling
                coords: Coordinate array (lat, lon)
        """
        # Get available features
        available_features = df.columns

        # Remove NaN values at the beginning to avoid issues with lag features
        self.logger.info(f"Removing NaN values from {df.shape[0]} rows...")
        df = df.drop_nulls()
        self.logger.info(f"After removing NaN values: {df.shape[0]} rows remaining")

        # Start with base features (exclude target and non-feature columns)
        feature_cols = [
            col
            for col in available_features
            if col
            not in [target_column] + self.feature_config.get("features_to_exclude", [])
            and not col.startswith("_")
        ]

        # Add geographic context features if enabled
        df, feature_cols = self._add_geographic_context(df, feature_cols)

        # Handle lag features (created during data loading)
        feature_cols = self._handle_lag_features(df, feature_cols)

        # Create regional classification
        df, regions_raw = self._create_regional_classification(df)

        # Apply regional training filter if enabled
        df = self._apply_regional_filtering(df)

        # Add basin feature if enabled
        df, feature_cols = self._add_basin_feature(df, feature_cols)

        # add cluster_id feature if available
        df, cluster_ids, unique_clusters_ids, feature_cols = (
            self._add_cluster_id_feature_extended(df, feature_cols)
        )

        # Log regional scaling and weighting status
        self._log_regional_config()

        # Store feature names for later use
        self.feature_names = feature_cols
        self.logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

        # Extract features and target
        X_raw = df.select(feature_cols)

        # Check if we should predict bias instead of vhm0_y directly
        predict_bias = self.feature_config.get("predict_bias", False)
        predict_bias_log_space = self.feature_config.get(
            "predict_bias_log_space", False
        )

        if predict_bias:
            if "vhm0_x" in df.columns and target_column in df.columns:
                if predict_bias_log_space:
                    # Multiplicative log-space bias: z = log(y_obs) - log(max(vhm0_x, eps))
                    eps = float(self.feature_config.get("log_space_epsilon", 1e-6))
                    y_raw = np.log(np.maximum(df[target_column], eps)) - np.log(
                        np.maximum(df["vhm0_x"], eps)
                    )
                    self.logger.info(
                        f"Using log-space bias as target: log(vhm0_y) - log(max(vhm0_x, {eps}))"
                    )
                else:
                    # Standard additive bias: y_obs - vhm0_x
                    y_raw = df[target_column] - df["vhm0_x"]
                    self.logger.info("Using additive bias as target: vhm0_y - vhm0_x")
            else:
                self.logger.error(
                    "Cannot predict bias: vhm0_x or target column not found"
                )
                raise ValueError(
                    "vhm0_x and target column required for bias prediction"
                )
        else:
            # Standard approach: predict vhm0_y directly
            y_raw = df[target_column]

        # Extract regions from the filtered dataframe (not the original regions_raw)
        regions_filtered = df["region"]

        # Extract coordinates for spatial plotting
        coords_raw = df.select(["lat", "lon"])

        self.logger.info(
            f"Final dataset - X: {X_raw.shape}, y: {y_raw.shape}, regions: {regions_filtered.shape}, coords: {coords_raw.shape}"
        )

        return (
            X_raw,
            y_raw,
            regions_filtered,
            coords_raw,
            unique_clusters_ids,
            cluster_ids,
        )

    def prepare_metadata(
        self, df: pl.DataFrame, target_column: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Prepare metadata such as years, months, raw vhm0_x, and actual wave heights."""
        # create years and months to use for year splitting
        years, months = self._create_years_months(df)

        # Extract raw input data for baseline metrics
        if "vhm0_x" in df.columns:
            vhm0_x_raw = df["vhm0_x"]
        else:
            self.logger.warning(
                "vhm0_x column not found in data. Baseline metrics will be empty."
            )
            vhm0_x_raw = None

        # Extract actual wave heights for proper binning (needed when predict_bias=True)
        if target_column in df.columns:
            actual_wave_heights = df[target_column]
        else:
            self.logger.warning(
                f"{target_column} column not found in data. Wave height binning may be incorrect."
            )
            actual_wave_heights = None

        return years, months, vhm0_x_raw, actual_wave_heights

    def _add_geographic_context(
        self, df: pl.DataFrame, feature_cols: List[str]
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Add geographic context features if enabled."""
        use_geo_context = self.feature_config.get("use_geo_context", {}).get(
            "enabled", False
        )
        if use_geo_context:
            self.logger.info("Adding geographic context features...")
            # Add geographic features to the dataframe
            df = df.with_columns(
                [
                    # Distance from key geographic points
                    ((pl.col("lon") - (-5.5)) ** 2 + (pl.col("lat") - 36.0) ** 2)
                    .sqrt()
                    .alias("dist_from_gibraltar"),
                    ((pl.col("lon") - 0) ** 2 + (pl.col("lat") - 40) ** 2)
                    .sqrt()
                    .alias("dist_from_center"),
                    # Fetch proxy (longitude-based)
                    (pl.col("lon") + 10).alias("fetch_proxy"),
                    # Bathymetry proxy (longitude-based)
                    (pl.col("lon") < -5).alias("deep_water_proxy"),
                ]
            )

            # Add these to feature columns
            geo_features = [
                "dist_from_gibraltar",
                "dist_from_center",
                "fetch_proxy",
                "deep_water_proxy",
            ]
            feature_cols.extend(geo_features)
            self.logger.info(f"Added geographic features: {geo_features}")

        return df, feature_cols

    def _handle_lag_features(
        self, df: pl.DataFrame, feature_cols: List[str]
    ) -> List[str]:
        """Handle lag features that were created during data loading."""
        use_lag_features = self.feature_config.get("lag_features", {}).get(
            "enabled", False
        )
        if use_lag_features:
            self.logger.info(
                "Lag features were created during data loading to preserve temporal sequences"
            )
            lag_config = self.feature_config.get("lag_features", {}).get("lags", {})

            # Add existing lag features to feature columns
            lag_features_found = []
            for variable, lags in lag_config.items():
                for lag in lags:
                    lag_col_name = f"{variable}_lag_{lag}h"
                    if lag_col_name in df.columns:
                        feature_cols.append(lag_col_name)
                        lag_features_found.append(lag_col_name)

            self.logger.info(
                f"Found {len(lag_features_found)} lag features: {lag_features_found}"
            )

            # Remove time column from feature_cols if it exists
            if "time" in feature_cols:
                feature_cols.remove("time")
                self.logger.info("Removed 'time' column from features")

        return feature_cols

    def _create_regional_classification(
        self, df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, np.ndarray]:
        """Create regional classification for monitoring and analysis."""
        self.logger.info("Creating regional classification for monitoring...")

        # Add regional classification to dataframe
        df = df.with_columns(
            [
                (pl.col("lon") < -5).alias("atlantic_region"),
                (pl.col("lon") > 30).alias("eastern_med_region"),
            ]
        )

        # Create combined region column (using integer IDs for performance)
        df = df.with_columns(
            [
                pl.when(pl.col("lon") < -5)
                .then(pl.lit(0))  # atlantic
                .when(pl.col("lon") > 30)
                .then(pl.lit(2))  # eastern_med
                .otherwise(pl.lit(1))  # mediterranean
                .alias("region")
            ]
        )

        # Extract region information
        regions_raw = df["region"]
        unique_regions = regions_raw.unique()
        region_names = [RegionMapper.get_display_name(rid) for rid in unique_regions]
        self.logger.info(
            f"Created regional classification: {region_names} (IDs: {unique_regions})"
        )

        return df, regions_raw

    def _apply_regional_filtering(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply regional training filter if enabled."""
        use_regional_training = self.feature_config.get("regional_training", {}).get(
            "enabled", False
        )
        if use_regional_training:
            training_regions = self.feature_config.get("regional_training", {}).get(
                "training_regions", [0]
            )  # Default to atlantic (0)
            self.logger.info(
                f"Regional training enabled - filtering to regions: {training_regions}"
            )

            # Filter data to only include specified regions
            df = df.filter(pl.col("region").is_in(training_regions))
            self.logger.info(f"After regional filtering: {df.shape[0]} rows remaining")

            # Log regional distribution after filtering
            region_counts = df["region"].value_counts().sort("region")
            self.logger.info("Regional distribution after filtering:")
            for row in region_counts.iter_rows(named=True):
                region_id = row["region"]
                count = row["count"]
                region_name = RegionMapper.get_display_name(region_id)
                self.logger.info(f"  {region_name}: {count:,} samples")

        return df

    def _add_basin_feature(
        self, df: pl.DataFrame, feature_cols: List[str]
    ) -> Tuple[pl.DataFrame, List[str]]:
        """Add basin feature if geographic context is enabled."""
        use_geo_basin = self.feature_config.get("use_geo_context", {}).get(
            "include_basin", True
        )
        if use_geo_basin:
            # Add basin feature as categorical indicator
            self.logger.info("Adding basin categorical feature...")
            df = df.with_columns(
                [
                    pl.when(pl.col("lon") < -5)
                    .then(pl.lit(0))  # Atlantic basin
                    .when(pl.col("lon") > 30)
                    .then(pl.lit(2))  # Eastern Mediterranean basin
                    .otherwise(pl.lit(1))  # Mediterranean basin
                    .alias("basin")
                ]
            )
            feature_cols.append("basin")
            self.logger.info("Added basin categorical feature to model features")

            # Log basin distribution
            basin_counts = df["basin"].value_counts().sort("basin")
            basin_names = {0: "Atlantic", 1: "Mediterranean", 2: "Eastern Med"}
            self.logger.info("Basin distribution:")
            for row in basin_counts.iter_rows(named=True):
                basin_id = row["basin"]
                count = row["count"]
                basin_name = basin_names.get(basin_id, f"Unknown({basin_id})")
                self.logger.info(f"  {basin_name} (ID: {basin_id}): {count:,} samples")

        return df, feature_cols

    def _add_cluster_id_feature(
        self, df: pl.DataFrame, feature_cols: List[str]
    ) -> Tuple[pl.DataFrame, np.ndarray, List[str], List[str]]:
        add_cluster_ids = self.feature_config.get("cluster_training", {}).get(
            "enabled", False
        )
        lat_step = self.feature_config.get("cluster_training", {}).get("lat_step", 1.0)
        lon_step = self.feature_config.get("cluster_training", {}).get("lon_step", 1.0)
        self.logger.info(f"✅ Detected grid resolution: {lat_step}° × {lon_step}°")

        if not add_cluster_ids:
            self.logger.info("Cluster ID feature not added to model features")
            return df, None, [], feature_cols

        mapper = GridClusterMapper(lat_step=lat_step, lon_step=lon_step)
        cluster_ids = mapper.get_cluster_id(df["lat"], df["lon"])

        df = df.with_columns(pl.Series("cluster_id", cluster_ids))
        feature_cols.append("cluster_id")

        unique_cluster_ids = pl.Series(cluster_ids).unique().to_list()
        unique_cluster_ids.sort()
        self.logger.info(
            f"Added cluster_id feature. Found {len(unique_cluster_ids)} unique clusters"
        )

        return df, cluster_ids, unique_cluster_ids, feature_cols
    
    def _add_cluster_id_feature_extended(
        self, df: pl.DataFrame, feature_cols: List[str]
    ) -> Tuple[pl.DataFrame, np.ndarray, List[str], List[str]]:
        """
        Add cluster ID feature to dataframe.
        
        Supports two modes:
        - grid: Groups points into rectangular cells (faster, fewer clusters)
        - point: One cluster per unique point (more clusters, higher resolution)
        
        Config structure:
            feature_block:
            cluster_training:
                enabled: true
                mode: "grid"  # or "point"
                
                # Grid mode parameters:
                lat_step: 0.5
                lon_step: 0.5
                
                # Point mode parameters:
                precision: 4  # decimal places
        """
        cluster_config = self.feature_config.get("cluster_training", {})
        
        # Check if enabled
        if not cluster_config.get("enabled", False):
            self.logger.info("Cluster ID feature not added to model features")
            return df, None, [], feature_cols
        
        # Get mode (default to grid for backwards compatibility)
        mode = cluster_config.get("mode", "grid")
        
        # Create appropriate mapper
        if mode == "grid":
            lat_step = cluster_config.get("lat_step", 1.0)
            lon_step = cluster_config.get("lon_step", 1.0)
            
            self.logger.info(f"✅ Using GRID mode: {lat_step}° × {lon_step}° cells")
            
            mapper = GridClusterMapper(lat_step=lat_step, lon_step=lon_step)
            cluster_ids = mapper.get_cluster_id(df["lat"], df["lon"])
            
        elif mode == "point":
            precision = cluster_config.get("precision", 4)
            
            self.logger.info(f"✅ Using POINT mode: 1 cluster per unique point (precision={precision})")
            
            mapper = PointClusterMapper(precision=precision)
            cluster_ids = mapper.get_cluster_id_np(df["lat"], df["lon"])
            
        else:
            raise ValueError(f"Unknown cluster mode: {mode}. Use 'grid' or 'point'")
        
        # Add to dataframe
        df = df.with_columns(cluster_ids.alias("cluster_id"))
        feature_cols.append("cluster_id")
        
        # Get unique cluster IDs
        unique_cluster_ids = cluster_ids.unique().to_list()
        unique_cluster_ids.sort()
        
        # Log statistics
        n_clusters = len(unique_cluster_ids)
        n_samples = len(df)
        samples_per_cluster = n_samples / n_clusters
        
        self.logger.info(f"Added cluster_id feature:")
        self.logger.info(f"   Total clusters: {n_clusters:,}")
        self.logger.info(f"   Total samples: {n_samples:,}")
        self.logger.info(f"   Avg samples/cluster: {samples_per_cluster:.1f}")
        
        # Warn about small clusters
        cluster_sizes = (
            df.group_by("cluster_id")
            .agg(pl.count().alias("n_samples"))
            .sort("n_samples")
        )
        
        min_samples = cluster_sizes["n_samples"].min()
        max_samples = cluster_sizes["n_samples"].max()
        median_samples = cluster_sizes["n_samples"].median()
        
        self.logger.info(f"   Sample distribution: min={min_samples}, median={median_samples:.0f}, max={max_samples}")
        
        # Warning for very small clusters
        small_threshold = 100
        n_small = (cluster_sizes["n_samples"] < small_threshold).sum()
        if n_small > 0:
            pct_small = n_small / n_clusters * 100
            self.logger.warning(
                f"   ⚠️  {n_small} clusters ({pct_small:.1f}%) have < {small_threshold} samples"
            )
            if mode == "point":
                self.logger.warning(
                    f"   Consider using grid mode or merging small clusters"
                )
        
        return df, cluster_ids, unique_cluster_ids, feature_cols

    def _create_years_months(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create year and month features from time column."""
        if "time" in df.columns:
            self.logger.info("Creating year and month features from 'time' column")
            df = df.with_columns(
                [
                    pl.col("time").dt.year().alias("year"),
                    pl.col("time").dt.month().alias("month"),
                ]
            )
            years = df["time"].dt.year()
            months = df["time"].dt.month()
        else:
            self.logger.warning(
                "'time' column not found - cannot create year and month features"
            )
            years = None
            months = None

        return years, months

    def _log_regional_config(self):
        """Log regional scaling and weighting configuration."""
        use_regional_scaling = self.feature_config.get("regional_scaling", {}).get(
            "enabled", False
        )
        use_regional_weighting = self.feature_config.get("regional_weighting", {}).get(
            "enabled", False
        )

        if use_regional_scaling and use_regional_weighting:
            self.logger.info("Regional scaling and weighting both enabled")
        elif use_regional_scaling:
            self.logger.info("Regional scaling enabled, weighting disabled")
        elif use_regional_weighting:
            self.logger.info("Regional weighting enabled, scaling disabled")
        else:
            self.logger.info(
                "Regional scaling and weighting disabled - using standard scaling with regional monitoring"
            )

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names used in the last feature preparation."""
        return self.feature_names
