from typing import List, Union, Optional, Dict, Any

import numpy as np
import s3fs
from pathlib import Path
from logging import getLogger
import yaml

logger = getLogger(__name__)


class SeasonHelper:
    """Helper class for mapping timestamps/months to seasons.

    Seasons are defined as:
        - Winter: December, January, February (12, 1, 2)
        - Spring: March, April, May (3, 4, 5)
        - Summer: June, July, August (6, 7, 8)
        - Autumn: September, October, November (9, 10, 11)
    """

    # Class constant for season-month mapping
    SEASON_MONTHS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11]
    }

    @staticmethod
    def get_season_from_month(month: int) -> str:
        """Get season from month number (1-12).

        Args:
            month: Month number (1-12)

        Returns:
            Season name as string ('winter', 'spring', 'summer', 'autumn')
        """
        for season, months in SeasonHelper.SEASON_MONTHS.items():
            if month in months:
                return season
        return "unknown"

    @staticmethod
    def get_seasons_from_timestamps(timestamps: np.ndarray) -> List[str]:
        """Extract seasons from numpy datetime64 timestamps.

        Args:
            timestamps: Numpy array of datetime64 timestamps

        Returns:
            List of season names corresponding to each timestamp
        """
        months = timestamps.astype('datetime64[M]').astype(int) % 12 + 1
        return [SeasonHelper.get_season_from_month(month) for month in months]

    @staticmethod
    def get_seasons_from_months(months: Union[List[int], np.ndarray]) -> List[str]:
        """Get seasons from month numbers.

        Args:
            months: List or array of month numbers (1-12)

        Returns:
            List of season names corresponding to each month
        """
        if isinstance(months, np.ndarray):
            months = months.tolist()
        return [SeasonHelper.get_season_from_month(month) for month in months]

    @staticmethod
    def count_seasons(timestamps: np.ndarray) -> dict:
        """Count occurrences of each season in timestamps.

        Args:
            timestamps: Numpy array of datetime64 timestamps

        Returns:
            Dictionary with season counts: {'winter': count, 'spring': count, ...}
        """
        seasons = SeasonHelper.get_seasons_from_timestamps(timestamps)
        counts = {season: 0 for season in SeasonHelper.SEASON_MONTHS.keys()}
        for season in seasons:
            if season in counts:
                counts[season] += 1
        return counts


def get_file_list(
    data_path: str, file_pattern: str, max_files: Optional[int] = None
) -> list:
    """Get list of files from S3 or local path.

    Returns files in sorted order for reproducibility.
    """
    if data_path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        # Search in both the directory itself and subdirectories
        data_path_clean = data_path.rstrip("/")
        pattern = f"{data_path_clean}/{file_pattern}"
        print(f"Searching S3 with pattern: {pattern}")
        files = fs.glob(pattern)
        # Also add files from subdirectories
        pattern_recursive = f"{data_path_clean}/**/{file_pattern}"
        files_recursive = fs.glob(pattern_recursive)
        # Combine and deduplicate, then sort for reproducibility
        files = sorted(list(set(files + files_recursive)))
        # Ensure s3:// prefix
        files = [f if f.startswith("s3://") else f"s3://{f}" for f in files]
    else:
        files = list(Path(data_path).glob(f"**/{file_pattern}"))
        files = [str(f) for f in files]
        # Sort for reproducibility
        files = sorted(files)

    if max_files:
        files = files[:max_files]

    return files


def split_files_by_year(
    files: list,
    train_year: int | list = 2021,
    val_year: int | list = 2022,
    test_year: int | list = 2023,
    val_months: list = None,
    test_months: list = None,
) -> tuple:
    """Split files into train/val/test based on year lists and validation months.

    Assumptions:
    - `train_year`, `val_year`, `test_year` are lists of years.
    - A file goes to validation only if (year in `val_year`) AND (month in `val_months`).
    - Otherwise it goes to train/test based on year membership.
    """
    train_files = []
    val_files = []
    test_files = []

    # Normalize inputs to sets of years for easy membership checks
    def _to_year_set(y):
        if isinstance(y, (list, tuple, set)):
            return set(int(v) for v in y)
        return {int(y)}

    train_years = _to_year_set(train_year)
    val_years = _to_year_set(val_year)
    test_years = _to_year_set(test_year)
    test_months_set = set(int(m) for m in test_months) if test_months else set()
    val_months_set = set(int(m) for m in val_months) if val_months else set()

    def _parse_year_month(name: str) -> tuple[int | None, int | None]:
        """Parse year and month from filename assuming pattern like WAVEANYYYYMM...
        Returns (year, month) where either can be None if not parsed.
        """
        try:
            marker = "WAVEAN"
            idx = name.find(marker)
            if idx != -1 and len(name) >= idx + 6 + 6:  # WAVEAN + YYYYMM
                year_str = name[idx + 6 : idx + 10]
                month_str = name[idx + 10 : idx + 12]
                year_val = int(year_str)
                month_val = int(month_str)
                if 1 <= month_val <= 12:
                    return year_val, month_val
                return year_val, None
        except Exception:
            pass

        # Fallback: find first 4-digit year and optional following month
        import re

        match = re.search(r"(20\d{2})(?:[^\d]?([01]?\d))?", name)
        if match:
            try:
                y = int(match.group(1))
                m = match.group(2)
                m_val = int(m) if m is not None else None
                if m_val is not None and not (1 <= m_val <= 12):
                    m_val = None
                return y, m_val
            except Exception:
                return None, None
        return None, None

    for file_path in files:
        filename = Path(file_path).name
        year, month = _parse_year_month(filename)

        if year is None:
            logger.warning(f"Skipping file {filename} - could not parse year")
            continue

        # Validation: require both year and month match (simple rule)
        if year in val_years and month in val_months_set:
            val_files.append(file_path)
            continue

        # Test split
        if year in test_years and month in test_months_set:
            test_files.append(file_path)
            continue

        # Train split
        if year in train_years:
            train_files.append(file_path)
            continue

        # Not in any target years
        logger.warning(f"Skipping file {filename} - year {year} not in target years")

    return train_files, val_files, test_files


class DNNConfig:
    """Configuration class for DNN training"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        if config_path:
            self._load_config(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "data": {
                "data_path": "s3://medwav-dev-data/parquet/hourly/",
                "file_pattern": "*.parquet",
                "train_year": 2021,
                "val_year": 2022,
                "test_year": 2023,
                "patch_size": [256, 256],
                "max_files": None,
                "random_seed": 42,
                "excluded_columns": ["time", "latitude", "longitude", "timestamp"],
                "target_columns": {"vhm0": "corrected_VHM0"},
                "predict_bias": False,
            },
            "model": {
                "in_channels": 14,
                "learning_rate": 1e-4,
                "loss_type": "weighted_mse",
                "filters": [64, 128, 256, 512, 1024],
                "weight_decay": 0,
            },
            "training": {
                "batch_size": 8,
                "max_epochs": 20,
                "num_workers": 4,
                "pin_memory": True,
                "accelerator": "gpu",
                "devices": 1,
                "precision": 16,
                "log_every_n_steps": 10,
                "early_stopping_patience": 5,
                "save_top_k": 3,
                "monitor": "val_loss",
                "mode": "min",
            },
            "checkpoint": {
                "resume_from_checkpoint": None,
                "checkpoint_dir": "checkpoints",
                "save_last": True,
            },
            "logging": {
                "log_dir": "logs",
                "experiment_name": "dnn_wave_correction",
                "use_comet": True,
            },
        }

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Deep merge user config with defaults
        self._deep_update(self.config, user_config)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value