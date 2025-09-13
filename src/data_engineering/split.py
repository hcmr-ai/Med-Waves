import glob
import os
import re
from datetime import datetime
from typing import List, Tuple


def pair_input_target_files(
    input_dir: str, target_dir: str
) -> Tuple[List[str], List[str]]:
    """
    Pair input (X) and target (Y) NetCDF files by matching filenames (e.g., by date).
    Returns two lists: x_files, y_files
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    target_files = sorted(glob.glob(os.path.join(target_dir, "*.nc")))

    # Match on filename (e.g., WAVEAN20170101.nc)
    input_map = {os.path.basename(f): f for f in input_files}
    target_map = {os.path.basename(f): f for f in target_files}

    common_keys = sorted(set(input_map.keys()) & set(target_map.keys()))
    x_files = [input_map[k] for k in common_keys]
    y_files = [target_map[k] for k in common_keys]

    if not x_files or not y_files:
        raise ValueError("No matching file pairs found.")

    return x_files, y_files


def extract_date_from_filename(filename: str) -> datetime:
    """
    Extract date from filename like WAVEAN20231231.parquet or WAVEAN20231231.nc
    """
    basename = os.path.basename(filename)
    # Match pattern like WAVEAN20231231
    match = re.search(r'WAVEAN(\d{8})', basename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    else:
        raise ValueError(f"Could not extract date from filename: {filename}")


def time_based_split(
    x_files: List[str],
    y_files: List[str],
    train_end_year: int = 2022,
    test_start_year: int = 2023,
    debug_mode: bool = False,
    debug_train_days: int = 1,
    debug_test_days: int = 1
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split files based on time periods for proper time series evaluation.

    Args:
        x_files: List of input file paths
        y_files: List of target file paths
        train_end_year: Last year to include in training (inclusive)
        test_start_year: First year to include in testing (inclusive)
        debug_mode: If True, use only a few days for quick testing
        debug_train_days: Number of days to use for training in debug mode
        debug_test_days: Number of days to use for testing in debug mode

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    assert len(x_files) == len(y_files), "Mismatched X and Y file counts"

    # Extract dates and create file-date pairs
    x_with_dates = [(f, extract_date_from_filename(f)) for f in x_files]
    y_with_dates = [(f, extract_date_from_filename(f)) for f in y_files]

    # Sort by date
    x_with_dates.sort(key=lambda x: x[1])
    y_with_dates.sort(key=lambda x: x[1])

    if debug_mode:
        # Debug mode: use only a few days for quick testing
        print(f"Debug mode: Using {debug_train_days} train days and {debug_test_days} test days")

        # Take first few days from training period
        train_files = [f for f, date in x_with_dates if date.year <= train_end_year]
        x_train = train_files[:debug_train_days]
        y_train = train_files[:debug_train_days]  # Same files for parquet

        # Take first few days from testing period
        test_files = [f for f, date in x_with_dates if date.year >= test_start_year]
        x_test = test_files[:debug_test_days]
        y_test = test_files[:debug_test_days]  # Same files for parquet
    else:
        # Normal mode: split based on year
        x_train = [f for f, date in x_with_dates if date.year <= train_end_year]
        y_train = [f for f, date in y_with_dates if date.year <= train_end_year]
        x_test = [f for f, date in x_with_dates if date.year >= test_start_year]
        y_test = [f for f, date in y_with_dates if date.year >= test_start_year]

    print("Time-based split:")
    print(f"  Training: {len(x_train)} files (up to {train_end_year})")
    print(f"  Testing: {len(x_test)} files (from {test_start_year})")

    if x_train and x_test:
        train_start_date = extract_date_from_filename(x_train[0])
        train_end_date = extract_date_from_filename(x_train[-1])
        test_start_date = extract_date_from_filename(x_test[0])
        test_end_date = extract_date_from_filename(x_test[-1])

        print(f"  Train period: {train_start_date.strftime('%Y-%m-%d')} to {train_end_date.strftime('%Y-%m-%d')}")
        print(f"  Test period: {test_start_date.strftime('%Y-%m-%d')} to {test_end_date.strftime('%Y-%m-%d')}")

    return x_train, y_train, x_test, y_test


def holdout_split(
    x_files: List[str], y_files: List[str], train_ratio: float = 0.9
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Simple train/test split by slicing files chronologically.
    WARNING: This is not appropriate for time series data!
    Use time_based_split instead for proper temporal evaluation.
    """
    assert len(x_files) == len(y_files), "Mismatched X and Y file counts"
    split_idx = int(train_ratio * len(x_files))
    return (
        x_files[:split_idx],
        y_files[:split_idx],
        x_files[split_idx:],
        y_files[split_idx:],
    )


def time_series_kfold_split(x_files, y_files, n_splits=5):
    """
    Generator that yields (x_train, y_train, x_val, y_val) for time-series k-fold CV.
    Each fold uses earlier data for training and a following chunk for validation.
    """
    fold_size = len(x_files) // n_splits
    for i in range(n_splits):
        start_val = i * fold_size
        end_val = (i + 1) * fold_size
        x_train = x_files[:start_val]
        y_train = y_files[:start_val]
        x_val = x_files[start_val:end_val]
        y_val = y_files[start_val:end_val]
        yield x_train, y_train, x_val, y_val


def expanding_window_split(x_files, y_files, initial_size, step):
    """
    Generator that yields expanding train set and fixed-size validation set.
    """
    for start in range(initial_size, len(x_files) - step, step):
        x_train = x_files[:start]
        y_train = y_files[:start]
        x_val = x_files[start : start + step]
        y_val = y_files[start : start + step]
        yield x_train, y_train, x_val, y_val
