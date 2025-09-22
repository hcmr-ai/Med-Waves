import glob
import os
import re
from datetime import datetime
from typing import List, Tuple
import polars as pl


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


def stratified_sample_by_location(
    df: pl.DataFrame, 
    max_samples_per_file: int = 10000,
    samples_per_location: int = 20,
    location_cols: List[str] = ["latitude", "longitude"],
    seed: int = 42
) -> pl.DataFrame:
    """
    Optimized stratified sampling by location for large datasets.
    
    Args:
        df: Input DataFrame
        max_samples_per_file: Maximum total samples to keep
        samples_per_location: Number of samples per location
        location_cols: Columns that define a location
        seed: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame with balanced representation across locations
    """
    if len(df) <= max_samples_per_file:
        return df
    
    # Count unique locations
    unique_locations = df.group_by(location_cols).count()
    n_locations = len(unique_locations)
    print(f"n_locations: {n_locations}")
    
    # Calculate samples per location
    if n_locations * samples_per_location > max_samples_per_file:
        # Too many locations, reduce samples per location
        samples_per_location = max_samples_per_file // n_locations
        samples_per_location = max(1, samples_per_location)  # At least 1 per location
    
    # For very large numbers of locations, use a more efficient approach
    if n_locations > 100000:  # Extremely large datasets
        return _ultra_fast_stratified_sample(df, location_cols, samples_per_location, max_samples_per_file, seed)
    elif n_locations > 50000:  # Large datasets
        return _fast_stratified_sample(df, location_cols, samples_per_location, max_samples_per_file, seed)
    
    # Original approach for smaller datasets
    return _standard_stratified_sample(df, location_cols, samples_per_location, seed)


def _fast_stratified_sample(
    df: pl.DataFrame,
    location_cols: List[str],
    samples_per_location: int,
    max_samples_per_file: int,
    seed: int
) -> pl.DataFrame:
    """
    Fast stratified sampling for large numbers of locations using window functions.
    """
    # Add row numbers within each location group
    df_with_rownum = df.with_row_index("_global_idx").with_columns(
        pl.concat_str([pl.col(col) for col in location_cols], separator="_").alias("_location_id")
    ).with_columns(
        pl.int_range(pl.len()).over("_location_id").alias("_local_idx")
    )
    
    # Sample by taking rows where local_idx < samples_per_location
    # This gives us up to samples_per_location from each location
    sampled_df = df_with_rownum.filter(
        pl.col("_local_idx") < samples_per_location
    ).drop(["_global_idx", "_location_id", "_local_idx"])
    
    # If we still have too many samples, randomly sample from the result
    if len(sampled_df) > max_samples_per_file:
        sampled_df = sampled_df.sample(n=max_samples_per_file, seed=seed)
    
    return sampled_df


def _ultra_fast_stratified_sample(
    df: pl.DataFrame,
    location_cols: List[str],
    samples_per_location: int,
    max_samples_per_file: int,
    seed: int
) -> pl.DataFrame:
    """
    Ultra-fast stratified sampling for extremely large datasets using spatial grid sampling.
    """
    # For extremely large datasets, use spatial grid sampling instead of exact location sampling
    # This trades some precision for massive speed gains
    
    # Create spatial grid bins
    df_with_grid = df.with_columns([
        (pl.col("lat") * 100).round().cast(pl.Int32).alias("_lat_bin"),
        (pl.col("lon") * 100).round().cast(pl.Int32).alias("_lon_bin")
    ]).with_columns(
        pl.concat_str([pl.col("_lat_bin"), pl.col("_lon_bin")], separator="_").alias("_grid_id")
    )
    
    # Sample from each grid cell
    df_with_rownum = df_with_grid.with_columns(
        pl.int_range(pl.len()).over("_grid_id").alias("_local_idx")
    )
    
    # Take samples from each grid cell
    sampled_df = df_with_rownum.filter(
        pl.col("_local_idx") < samples_per_location
    ).drop(["_lat_bin", "_lon_bin", "_grid_id", "_local_idx"])
    
    # If we still have too many samples, randomly sample from the result
    if len(sampled_df) > max_samples_per_file:
        sampled_df = sampled_df.sample(n=max_samples_per_file, seed=seed)
    
    return sampled_df


def _standard_stratified_sample(
    df: pl.DataFrame,
    location_cols: List[str],
    samples_per_location: int,
    seed: int
) -> pl.DataFrame:
    """
    Standard stratified sampling for smaller datasets.
    """
    # Create a location identifier column
    df_with_location_id = df.with_row_index("_row_index").with_columns(
        pl.concat_str([pl.col(col) for col in location_cols], separator="_").alias("_location_id")
    )
    
    # Sample from each location using a more reliable approach
    sampled_dfs = []
    for location_id in df_with_location_id["_location_id"].unique():
        location_data = df_with_location_id.filter(pl.col("_location_id") == location_id)
        if len(location_data) > samples_per_location:
            sampled_location = location_data.sample(n=samples_per_location, seed=seed)
        else:
            sampled_location = location_data
        sampled_dfs.append(sampled_location)
    
    # Combine all sampled data
    if sampled_dfs:
        sampled_df = pl.concat(sampled_dfs).drop(["_row_index", "_location_id"])
    else:
        sampled_df = df
    
    return sampled_df


def random_sample_within_file(
    df: pl.DataFrame,
    max_samples_per_file: int = 10000,
    seed: int = 42
) -> pl.DataFrame:
    """
    Simple random sampling within a file.
    
    Args:
        df: Input DataFrame
        max_samples_per_file: Maximum samples to keep
        seed: Random seed for reproducibility
        
    Returns:
        Randomly sampled DataFrame
    """
    if len(df) <= max_samples_per_file:
        return df
    
    return df.sample(n=max_samples_per_file, seed=seed)


def temporal_sample_within_file(
    df: pl.DataFrame,
    max_samples_per_file: int = 10000,
    time_col: str = "time",
    samples_per_hour: int = 100,
    seed: int = 42
) -> pl.DataFrame:
    """
    Temporal sampling to ensure diversity across time periods.
    
    Args:
        df: Input DataFrame
        max_samples_per_file: Maximum samples to keep
        time_col: Name of the time column
        samples_per_hour: Number of samples per hour
        seed: Random seed for reproducibility
        
    Returns:
        Temporally sampled DataFrame
    """
    if len(df) <= max_samples_per_file:
        return df
    
    # Extract hour from time column
    df_with_hour = df.with_columns(
        pl.col(time_col).dt.hour().alias("hour")
    )
    
    # Sample from each hour using a more reliable approach
    sampled_dfs = []
    for hour in df_with_hour["hour"].unique():
        hour_data = df_with_hour.filter(pl.col("hour") == hour)
        if len(hour_data) > samples_per_hour:
            sampled_hour = hour_data.sample(n=samples_per_hour, seed=seed)
        else:
            sampled_hour = hour_data
        sampled_dfs.append(sampled_hour)
    
    # Combine all sampled data
    if sampled_dfs:
        sampled_df = pl.concat(sampled_dfs).drop("hour")
    else:
        sampled_df = df
    
    # If still too many samples, randomly sample from the result
    if len(sampled_df) > max_samples_per_file:
        sampled_df = sampled_df.sample(n=max_samples_per_file, seed=seed)
    
    return sampled_df
