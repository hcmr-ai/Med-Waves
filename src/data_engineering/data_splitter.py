"""
DataSplitter for Wave Height Bias Correction Research

This module provides the DataSplitter class for handling data splitting strategies
in a modular and reusable way.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """Handles data splitting strategies for train/validation/test sets."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize DataSplitter with configuration."""
        self.config = config
        self.data_config = config.get("data", {})
        self.logger = logging.getLogger(__name__)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regions: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        file_paths: Optional[List[str]] = None,
        actual_wave_heights: Optional[np.ndarray] = None,
        years: Optional[np.ndarray] = None,
        months: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Split data into train/validation/test sets based on configuration.

        Args:
            X: Feature matrix
            y: Target vector (may be bias values if predict_bias=true)
            regions: Regional classification array (optional)
            coords: Coordinate array (lat, lon) (optional)
            file_paths: List of file paths (required for year-based splitting)
            actual_wave_heights: Actual wave heights for proper binning (optional)

        Returns:
            Dictionary containing all split data
        """
        split_config = self.data_config.get("split", {})
        split_type = split_config.get("type", "random")
        test_size = split_config.get("test_size", 0.2)
        val_size = split_config.get("val_size", 0.2)
        random_state = split_config.get("random_state", 42)

        self.logger.info(f"Splitting data using {split_type} strategy...")

        if split_type == "year_based":
            return self._split_by_years(
                X,
                y,
                regions,
                coords,
                file_paths,
                split_config,
                actual_wave_heights,
                years,
                months,
                cluster_ids,
            )
        elif split_type == "random":
            return self._split_random(
                X,
                y,
                regions,
                coords,
                test_size,
                val_size,
                random_state,
                actual_wave_heights,
            )
        elif split_type == "temporal":
            return self._split_temporal(
                X, y, regions, coords, test_size, val_size, actual_wave_heights
            )
        elif split_type == "stratified":
            return self._split_stratified(
                X,
                y,
                regions,
                coords,
                test_size,
                val_size,
                random_state,
                split_config,
                actual_wave_heights,
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def _split_random(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regions: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        actual_wave_heights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Random split: train -> val+test, then val -> val/test."""
        # Split train -> val+test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Split val from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )

        # Split regions if available
        regions_train, regions_val, regions_test = None, None, None
        if regions is not None:
            regions_temp, regions_test = train_test_split(
                regions, test_size=test_size, random_state=random_state
            )
            regions_train, regions_val = train_test_split(
                regions_temp, test_size=val_size_adjusted, random_state=random_state
            )

        # Split coordinates if available
        coords_train, coords_val, coords_test = None, None, None
        if coords is not None:
            coords_temp, coords_test = train_test_split(
                coords, test_size=test_size, random_state=random_state
            )
            coords_train, coords_val = train_test_split(
                coords_temp, test_size=val_size_adjusted, random_state=random_state
            )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "regions_train": regions_train,
            "regions_val": regions_val,
            "regions_test": regions_test,
            "coords_train": coords_train,
            "coords_val": coords_val,
            "coords_test": coords_test,
        }

    def _split_temporal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regions: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Temporal split based on time (assumes data is sorted by time)."""
        n_samples = len(X)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - val_size))

        # Split features and targets
        X_train = X[:val_start]
        X_val = X[val_start:test_start]
        X_test = X[test_start:]

        y_train = y[:val_start]
        y_val = y[val_start:test_start]
        y_test = y[test_start:]

        # Split regions if available
        regions_train, regions_val, regions_test = None, None, None
        if regions is not None:
            regions_train = regions[:val_start]
            regions_val = regions[val_start:test_start]
            regions_test = regions[test_start:]

        # Split coordinates if available
        coords_train, coords_val, coords_test = None, None, None
        if coords is not None:
            coords_train = coords[:val_start]
            coords_val = coords[val_start:test_start]
            coords_test = coords[test_start:]

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "regions_train": regions_train,
            "regions_val": regions_val,
            "regions_test": regions_test,
            "coords_train": coords_train,
            "coords_val": coords_val,
            "coords_test": coords_test,
        }

    def _split_stratified(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regions: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        split_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Stratified split (requires categorical target - convert to bins)."""
        n_bins = split_config.get("n_bins", 10) if split_config else 10
        y_binned = pd.cut(y, bins=n_bins, labels=False)

        # Split train -> val+test
        X_temp, X_test, y_temp, y_test, _, y_test_binned = train_test_split(
            X,
            y,
            y_binned,
            test_size=test_size,
            random_state=random_state,
            stratify=y_binned,
        )

        # Split val from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, _, y_val_binned = train_test_split(
            X_temp,
            y_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp,
        )

        # Split regions if available
        regions_train, regions_val, regions_test = None, None, None
        if regions is not None:
            regions_temp, regions_test = train_test_split(
                regions,
                test_size=test_size,
                random_state=random_state,
                stratify=y_binned,
            )
            regions_train, regions_val = train_test_split(
                regions_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_val_binned,
            )

        # Split coordinates if available
        coords_train, coords_val, coords_test = None, None, None
        if coords is not None:
            coords_temp, coords_test = train_test_split(
                coords,
                test_size=test_size,
                random_state=random_state,
                stratify=y_binned,
            )
            coords_train, coords_val = train_test_split(
                coords_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_val_binned,
            )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "regions_train": regions_train,
            "regions_val": regions_val,
            "regions_test": regions_test,
            "coords_train": coords_train,
            "coords_val": coords_val,
            "coords_test": coords_test,
        }

    def _split_by_years(
        self,
        X: pl.DataFrame,
        y: pl.DataFrame,
        regions: Optional[pl.DataFrame] = None,
        coords: Optional[pl.DataFrame] = None,
        file_paths: Optional[List[str]] = None,
        split_config: Dict[str, Any] = None,
        actual_wave_heights: Optional[pl.DataFrame] = None,
        years: Optional[pl.DataFrame] = None,
        months: Optional[pl.DataFrame] = None,
        cluster_ids: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Split data by years: 2017-2022 for train/val, 2023 for test.

        Args:
            X: Feature matrix
            y: Target vector
            regions: Regional classification array (optional)
            coords: Coordinate array (optional)
            file_paths: List of file paths (required)
            split_config: Split configuration

        Returns:
            Dictionary containing all split data
        """
        if file_paths is None:
            raise ValueError("file_paths required for year_based splitting")

        # Import the date extraction function
        from src.data_engineering.split import extract_date_from_filename

        # Get configuration
        train_end_year = split_config.get("train_end_year", 2022)
        test_start_year = split_config.get("test_start_year", 2023)
        val_months = split_config.get("val_months", [])
        eval_months = split_config.get(
            "eval_months", list(range(1, 13))
        )  # Default: All months

        self.logger.info(
            f"Year-based split: Train up to {train_end_year}, Val months {val_months} of {train_end_year}, Test months {eval_months} from {test_start_year}"
        )

        # First, categorize files by their dates
        train_files = []
        val_files = []
        test_files = []

        for file_path in file_paths:
            try:
                # Extract date from filename
                date = extract_date_from_filename(file_path)
                year = date.year
                month = date.month

                # Debug: log first few files
                if len(train_files) + len(val_files) + len(test_files) <= 5:
                    self.logger.info(
                        f"File: {file_path} -> Year: {year}, Month: {month}"
                    )

                if year >= test_start_year and month in eval_months:
                    test_files.append(file_path)
                elif year <= train_end_year:
                    if year < train_end_year:
                        train_files.append(file_path)
                    elif year == train_end_year and month in val_months:
                        val_files.append(file_path)
                    else:
                        train_files.append(file_path)

            except Exception as e:
                self.logger.warning(f"Could not extract date from {file_path}: {e}")
                continue

        # Log file distribution
        self.logger.info(f"Total files processed: {len(file_paths)}")
        self.logger.info(
            f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}"
        )
        self.logger.info(f"Total samples: {X.height}")

        # Since we can't easily map back to original file indices after regional filtering,
        # we'll use a simple proportional split based on the number of files
        total_files = len(train_files) + len(val_files) + len(test_files)

        if total_files == 0:
            raise ValueError("No valid files found for splitting")

        # Create masks
        if years is not None and months is not None:
            train_mask = (years < train_end_year) | (
                (years == train_end_year) & ~years.is_in(val_months).not_()
            )
            val_mask = (years == train_end_year) & months.is_in(val_months)
            test_mask = (years >= test_start_year) & months.is_in(eval_months)
            n_samples = X.height
            train_ratio = train_mask.sum() / n_samples
            val_ratio = val_mask.sum() / n_samples
            test_ratio = test_mask.sum() / n_samples
        else:
            # Calculate proportional splits
            train_ratio = len(train_files) / total_files
            val_ratio = len(val_files) / total_files
            test_ratio = len(test_files) / total_files
            # Create indices for splitting
            n_samples = X.height
            train_end_idx = int(n_samples * train_ratio)
            val_end_idx = int(n_samples * (train_ratio + val_ratio))
            train_mask = np.zeros(n_samples, dtype=bool)
            val_mask = np.zeros(n_samples, dtype=bool)
            test_mask = np.zeros(n_samples, dtype=bool)

            train_mask[:train_end_idx] = True
            if val_ratio > 0:
                val_mask[train_end_idx:val_end_idx] = True
            if test_ratio > 0:
                test_mask[val_end_idx:] = True

        # Create splits
        X_train = X.filter(train_mask)
        X_val = X.filter(val_mask)
        X_test = X.filter(test_mask)
        cluster_ids_train = (
            cluster_ids.filter(train_mask) if cluster_ids is not None else None
        )
        cluster_ids_val = (
            cluster_ids.filter(val_mask) if cluster_ids is not None else None
        )
        cluster_ids_test = (
            cluster_ids.filter(test_mask) if cluster_ids is not None else None
        )

        y_train = y.filter(train_mask)
        y_val = y.filter(val_mask)
        y_test = y.filter(test_mask)

        # Split regions if available
        regions_train, regions_val, regions_test = None, None, None
        if regions is not None:
            regions_train = regions.filter(train_mask)
            regions_val = regions.filter(val_mask)
            regions_test = regions.filter(test_mask)

        # Split coordinates if available
        coords_train, coords_val, coords_test = None, None, None
        if coords is not None:
            coords_train = coords.filter(train_mask)
            coords_val = coords.filter(val_mask)
            coords_test = coords.filter(test_mask)

        # Split actual wave heights if available
        actual_wave_heights_train, actual_wave_heights_val, actual_wave_heights_test = (
            None,
            None,
            None,
        )
        if actual_wave_heights is not None:
            actual_wave_heights_train = actual_wave_heights.filter(train_mask)
            actual_wave_heights_val = actual_wave_heights.filter(val_mask)
            actual_wave_heights_test = actual_wave_heights.filter(test_mask)

        # Log split statistics
        train_years_months = []
        val_years_months = []
        test_years_months = []

        for file_path in train_files:
            try:
                date = extract_date_from_filename(file_path)
                train_years_months.append((date.year, date.month))
            except FileNotFoundError:
                continue

        for file_path in val_files:
            try:
                date = extract_date_from_filename(file_path)
                val_years_months.append((date.year, date.month))
            except FileNotFoundError:
                continue

        for file_path in test_files:
            try:
                date = extract_date_from_filename(file_path)
                test_years_months.append((date.year, date.month))
            except FileNotFoundError:
                continue

        # Group by year for cleaner logging
        train_years = sorted(set(year for year, month in train_years_months))
        val_years = sorted(set(year for year, month in val_years_months))
        test_years = sorted(set(year for year, month in test_years_months))

        self.logger.info("Year-based split completed:")
        self.logger.info(
            f"  Train: {X_train.height} samples ({train_ratio:.1%}), years: {train_years}"
        )
        self.logger.info(
            f"  Val: {X_val.height} samples ({val_ratio:.1%}), months {val_months} of {val_years}"
        )
        self.logger.info(
            f"  Test: {X_test.height} samples ({test_ratio:.1%}), months {eval_months} of {test_years}"
        )

        # Check for empty splits
        if X_val.height == 0:
            self.logger.warning(
                "Validation set is empty! Check your val_months configuration."
            )
        if X_test.height == 0:
            self.logger.warning(
                "Test set is empty! Check your eval_months and test_start_year configuration."
            )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "regions_train": regions_train,
            "regions_val": regions_val,
            "regions_test": regions_test,
            "coords_train": coords_train,
            "coords_val": coords_val,
            "coords_test": coords_test,
            "actual_wave_heights_train": actual_wave_heights_train,
            "actual_wave_heights_val": actual_wave_heights_val,
            "actual_wave_heights_test": actual_wave_heights_test,
            "train_indices": train_mask.to_numpy().nonzero()[0],
            "val_indices": val_mask.to_numpy().nonzero()[0],
            "test_indices": test_mask.to_numpy().nonzero()[0],
            "cluster_ids_train": cluster_ids_train,
            "cluster_ids_val": cluster_ids_val,
            "cluster_ids_test": cluster_ids_test,
        }

    def get_split_info(self, split_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about the data splits.

        Args:
            split_data: Dictionary containing split data

        Returns:
            Dictionary with split information
        """
        info = {}

        # Basic split sizes
        info["train_size"] = len(split_data["X_train"])
        info["val_size"] = len(split_data["X_val"])
        info["test_size"] = len(split_data["X_test"])
        info["total_size"] = info["train_size"] + info["val_size"] + info["test_size"]

        # Split percentages
        info["train_percent"] = (info["train_size"] / info["total_size"]) * 100
        info["val_percent"] = (info["val_size"] / info["total_size"]) * 100
        info["test_percent"] = (info["test_size"] / info["total_size"]) * 100

        # Feature information
        info["n_features"] = split_data["X_train"].shape[1]

        # Regional information (if available)
        if split_data["regions_train"] is not None:
            info["regions_available"] = True
            info["unique_regions"] = len(np.unique(split_data["regions_train"]))
        else:
            info["regions_available"] = False

        # Coordinate information (if available)
        if split_data["coords_train"] is not None:
            info["coords_available"] = True
        else:
            info["coords_available"] = False

        return info

    def log_split_info(self, split_data: Dict[str, Any]):
        """
        Log information about the data splits.

        Args:
            split_data: Dictionary containing split data
        """
        info = self.get_split_info(split_data)

        self.logger.info("Data Split Information:")
        self.logger.info(f"  Total samples: {info['total_size']:,}")
        self.logger.info(
            f"  Train: {info['train_size']:,} ({info['train_percent']:.1f}%)"
        )
        self.logger.info(f"  Val:   {info['val_size']:,} ({info['val_percent']:.1f}%)")
        self.logger.info(
            f"  Test:  {info['test_size']:,} ({info['test_percent']:.1f}%)"
        )
        self.logger.info(f"  Features: {info['n_features']}")

        if info["regions_available"]:
            self.logger.info(f"  Regions: {info['unique_regions']} unique regions")

        if info["coords_available"]:
            self.logger.info("  Coordinates: Available for spatial analysis")

        # Log stratified distribution after split
        self._log_stratified_distribution_after_split(split_data)

    def _log_stratified_distribution_after_split(
        self, split_data: Dict[str, Any]
    ) -> None:
        """Log the stratified distribution after data split to show actual training data."""
        self.logger.info("=" * 80)
        self.logger.info("STRATIFIED DISTRIBUTION AFTER DATA SPLIT")
        self.logger.info("=" * 80)

        # Get wave height bins from configuration
        wave_bins = self._get_wave_height_bins_from_config()

        # Check if we're predicting bias - if so, we need to use actual wave heights for binning
        feature_config = self.config.get("feature_block", {})
        predict_bias = feature_config.get("predict_bias", False)

        if predict_bias:
            self.logger.info(
                "NOTE: Model is predicting bias, but wave height bins are based on actual wave heights"
            )
            self.logger.info(
                "Wave height distribution shows actual wave heights, not bias values"
            )

        # Log training data distribution
        if len(split_data["y_train"]) > 0:
            self.logger.info(f"TRAINING DATA: {len(split_data['y_train']):,} samples")
            # Use actual wave heights if available, otherwise use y data
            wave_data = split_data.get(
                "actual_wave_heights_train", split_data["y_train"]
            )
            self._log_wave_height_distribution_for_split(
                wave_data, "Training", wave_bins
            )

        # Log validation data distribution
        if len(split_data["y_val"]) > 0:
            self.logger.info(f"VALIDATION DATA: {len(split_data['y_val']):,} samples")
            # Use actual wave heights if available, otherwise use y data
            wave_data = split_data.get("actual_wave_heights_val", split_data["y_val"])
            self._log_wave_height_distribution_for_split(
                wave_data, "Validation", wave_bins
            )

        # Log test data distribution
        if len(split_data["y_test"]) > 0:
            self.logger.info(f"TEST DATA: {len(split_data['y_test']):,} samples")
            # Use actual wave heights if available, otherwise use y data
            wave_data = split_data.get("actual_wave_heights_test", split_data["y_test"])
            self._log_wave_height_distribution_for_split(wave_data, "Test", wave_bins)

        self.logger.info("=" * 80)

    def _get_wave_height_bins_from_config(self) -> Dict[str, tuple]:
        """Get wave height bins from configuration."""
        # Get stratification bins from feature config
        feature_config = self.config.get("feature_block", {})
        stratification_bins = feature_config.get("per_point_stratification_bins", {})

        if not stratification_bins:
            # Fallback to default bins if not configured
            self.logger.warning(
                "No per_point_stratification_bins found in config, using default bins"
            )
            return {
                "calm": (0.0, 1.0),
                "moderate": (1.0, 3.0),
                "rough": (3.0, 6.0),
                "high": (6.0, 9.0),
                "extreme": (9.0, float("inf")),
            }

        # Convert config bins to the format expected by logging
        wave_bins = {}
        for bin_name, bin_config in stratification_bins.items():
            min_val, max_val = bin_config["range"]

            # Convert to float in case they come from YAML as strings
            min_val = float(min_val)
            if str(max_val) in [".inf", "float('inf')"] or max_val == float("inf"):
                max_val = float("inf")
            else:
                max_val = float(max_val)

            wave_bins[bin_name] = (min_val, max_val)

        return wave_bins

    def _log_wave_height_distribution_for_split(
        self, y_data: pl.Series, split_name: str, wave_bins: Dict[str, tuple]
    ) -> None:
        """
        Log wave height distribution for a specific data split.

        OPTIMIZATIONS:
        - Single pass through data using vectorized operations
        - Pre-compute statistics once
        - Eliminate redundant conversions
        - Use numpy for numerical operations (faster than Polars for math)
        """
        if len(y_data) == 0:
            return

        y_numpy = y_data.to_numpy()
        total_samples = len(y_numpy)

        min_wave = y_numpy.min()
        max_wave = y_numpy.max()

        self.logger.info(f"  {split_name} wave height distribution:")
        self.logger.info(
            f"    Actual wave height range: {min_wave:.2f}m - {max_wave:.2f}m"
        )

        bin_counts = {}
        bin_masks = {}

        for bin_name, (min_val, max_val) in wave_bins.items():
            if max_val == float("inf"):
                mask = y_numpy >= min_val
            else:
                mask = (y_numpy >= min_val) & (y_numpy < max_val)

            count = mask.sum()
            bin_counts[bin_name] = count
            bin_masks[bin_name] = mask

        total_counted = sum(bin_counts.values())

        for bin_name, (min_val, max_val) in wave_bins.items():
            count = bin_counts[bin_name]
            percentage = (count / total_samples) * 100

            range_str = (
                f"{min_val}+m" if max_val == float("inf") else f"{min_val}-{max_val}m"
            )
            self.logger.info(
                f"    {bin_name:12} ({range_str:>8}): {count:8,} samples ({percentage:5.1f}%)"
            )

        missing_samples = total_samples - total_counted
        if missing_samples > 0:
            self.logger.warning(
                f"    DEBUG: {missing_samples:,} samples not accounted for in any bin! "
                f"(Total: {total_samples:,}, Counted: {total_counted:,})"
            )

        extreme_count = bin_counts.get("extreme", 0)
        if extreme_count > 0:
            extreme_mask = bin_masks.get("extreme")
            if extreme_mask is not None and extreme_mask.any():
                extreme_data = y_numpy[extreme_mask]

                self.logger.info(
                    f"    Extreme wave statistics:\n"
                    f"      Maximum wave height: {extreme_data.max():.2f}m\n"
                    f"      Mean extreme wave: {extreme_data.mean():.2f}m"
                )

        del y_numpy, bin_masks
        import gc

        gc.collect()
