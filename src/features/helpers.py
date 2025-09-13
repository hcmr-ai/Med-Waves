import numpy as np
import polars as pl
import xarray as xr


def extract_features_from_file(x_path, y_path, use_dask=False):
    xds = xr.open_dataset(x_path, chunks="auto" if use_dask else None)
    yds = xr.open_dataset(y_path, chunks="auto" if use_dask else None)

    x_vhm0 = xds["VHM0"].values
    wspd = xds["WSPD"].values
    y_vhm0 = yds["VHM0"].values

    # All should have the same shape: (time, lat, lon)
    assert x_vhm0.shape == wspd.shape == y_vhm0.shape

    # Create coordinate grids
    time_len, lat_len, lon_len = x_vhm0.shape
    time = np.repeat(xds.time.values, lat_len * lon_len)
    lat = np.tile(np.repeat(xds.latitude.values, lon_len), time_len)
    lon = np.tile(np.tile(xds.longitude.values, lat_len), time_len)

    df = pl.DataFrame(
        {
            "vhm0_x": x_vhm0.flatten(),
            "wspd": wspd.flatten(),
            "vhm0_y": y_vhm0.flatten(),
            "lat": lat,
            "lon": lon,
            "time": time,
        }
    ).drop_nulls()

    return df


def extract_features_from_parquet(parquet_path, use_dask=False):
    """
    Extract features from a single parquet file that contains both input and target data.

    Args:
        parquet_path: Path to the parquet file
        use_dask: Whether to use dask for lazy loading (not used for parquet)

    Returns:
        pl.DataFrame with features and target
    """
    # Load parquet file
    df = pl.read_parquet(parquet_path)

    # The parquet file should contain both input features and target
    # We need to identify which columns are features vs target
    # Based on your data_loader.py, it looks like you have:
    # - VHM0 (original), corrected_VHM0 (target)
    # - VTM02 (original), corrected_VTM02 (target)
    # - Other features like WSPD, lat, lon, etc.

    # Create the feature matrix
    feature_df = pl.DataFrame()

    # Add base features
    if "VHM0" in df.columns:
        feature_df = feature_df.with_columns(pl.col("VHM0").alias("vhm0_x"))
    if "WSPD" in df.columns:
        feature_df = feature_df.with_columns(pl.col("WSPD").alias("wspd"))
    if "latitude" in df.columns:
        feature_df = feature_df.with_columns(pl.col("latitude").alias("lat"))
    elif "lat" in df.columns:
        feature_df = feature_df.with_columns(pl.col("lat"))
    if "longitude" in df.columns:
        feature_df = feature_df.with_columns(pl.col("longitude").alias("lon"))
    elif "lon" in df.columns:
        feature_df = feature_df.with_columns(pl.col("lon"))

    # Add target
    if "corrected_VHM0" in df.columns:
        feature_df = feature_df.with_columns(pl.col("corrected_VHM0").alias("vhm0_y"))
    elif "VHM0" in df.columns:
        # If no corrected version, use original as target (for testing)
        feature_df = feature_df.with_columns(pl.col("VHM0").alias("vhm0_y"))

    # Add additional features if available
    additional_features = ["VTM02", "corrected_VTM02", "VMDR", "WDIR", "U10", "V10"]
    for feat in additional_features:
        if feat in df.columns:
            feature_df = feature_df.with_columns(pl.col(feat))

    # Add temporal features if available
    temporal_features = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "sin_month", "cos_month"]
    for feat in temporal_features:
        if feat in df.columns:
            feature_df = feature_df.with_columns(pl.col(feat))

    # Add normalized coordinates if available
    if "lat_norm" in df.columns:
        feature_df = feature_df.with_columns(pl.col("lat_norm"))
    if "lon_norm" in df.columns:
        feature_df = feature_df.with_columns(pl.col("lon_norm"))

    # Add wave direction features if available
    if "wave_dir_sin" in df.columns:
        feature_df = feature_df.with_columns(pl.col("wave_dir_sin"))
    if "wave_dir_cos" in df.columns:
        feature_df = feature_df.with_columns(pl.col("wave_dir_cos"))

    # Remove rows with NaN values
    feature_df = feature_df.drop_nulls()

    return feature_df
