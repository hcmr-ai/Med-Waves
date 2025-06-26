import glob
import os
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


def holdout_split(
    x_files: List[str], y_files: List[str], train_ratio: float = 0.9
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Simple train/test split by slicing files chronologically.
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
