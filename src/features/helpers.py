import logging
import numpy as np
import polars as pl
import xarray as xr

# Set up logger
logger = logging.getLogger(__name__)


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
        parquet_path: Path to the parquet file (can be local path or S3 URL)
        use_dask: Whether to use dask for lazy loading (not used for parquet)

    Returns:
        pl.DataFrame with features and target
    """
    logger.info(f"Loading parquet file: {parquet_path}")
    
    # Load parquet file - handle S3 URLs
    if parquet_path.startswith('s3://'):
        logger.info("Loading from S3...")
        # Handle S3 URLs
        import boto3
        from io import BytesIO
        
        # Parse S3 URL
        s3_url = parquet_path[5:]  # Remove 's3://'
        bucket, key = s3_url.split('/', 1)
        logger.info(f"S3 bucket: {bucket}, key: {key}")
        
        # Download from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        parquet_data = response['Body'].read()
        logger.info(f"Downloaded {len(parquet_data)} bytes from S3")
        
        # Read with polars
        df = pl.read_parquet(BytesIO(parquet_data))
    else:
        logger.info("Loading local file...")
        # Handle local files
        df = pl.read_parquet(parquet_path)
    
    logger.info(f"Loaded DataFrame shape: {df.shape}")
    logger.info(f"Available columns: {df.columns}")

    # The parquet file should contain both input features and target
    # We need to identify which columns are features vs target
    # Based on your data_loader.py, it looks like you have:
    # - VHM0 (original), corrected_VHM0 (target)
    # - VTM02 (original), corrected_VTM02 (target)
    # - Other features like WSPD, lat, lon, etc.

    # Build feature selection list
    feature_columns = []
    
    # Add base features
    if "VHM0" in df.columns:
        feature_columns.append(pl.col("VHM0").alias("vhm0_x"))
    if "WSPD" in df.columns:
        feature_columns.append(pl.col("WSPD").alias("wspd"))
    if "latitude" in df.columns:
        feature_columns.append(pl.col("latitude").alias("lat"))
    elif "lat" in df.columns:
        feature_columns.append(pl.col("lat"))
    if "longitude" in df.columns:
        feature_columns.append(pl.col("longitude").alias("lon"))
    elif "lon" in df.columns:
        feature_columns.append(pl.col("lon"))

    # Add target
    if "corrected_VHM0" in df.columns:
        feature_columns.append(pl.col("corrected_VHM0").alias("vhm0_y"))
    elif "VHM0" in df.columns:
        # If no corrected version, use original as target (for testing)
        feature_columns.append(pl.col("VHM0").alias("vhm0_y"))

    # Add additional features if available
    additional_features = ["VTM02", "corrected_VTM02", "VMDR", "WDIR", "U10", "V10"]
    for feat in additional_features:
        if feat in df.columns:
            feature_columns.append(pl.col(feat))

    # Add temporal features if available
    temporal_features = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "sin_month", "cos_month"]
    for feat in temporal_features:
        if feat in df.columns:
            feature_columns.append(pl.col(feat))

    # Add normalized coordinates if available
    if "lat_norm" in df.columns:
        feature_columns.append(pl.col("lat_norm"))
    if "lon_norm" in df.columns:
        feature_columns.append(pl.col("lon_norm"))

    # Add wave direction features if available
    if "wave_dir_sin" in df.columns:
        feature_columns.append(pl.col("wave_dir_sin"))
    if "wave_dir_cos" in df.columns:
        feature_columns.append(pl.col("wave_dir_cos"))

    # Create the feature matrix by selecting columns
    logger.info(f"Building feature matrix with {len(feature_columns)} columns")
    if feature_columns:
        feature_df = df.select(feature_columns)
        logger.info(f"Feature DataFrame shape: {feature_df.shape}")
        logger.info(f"Feature columns: {feature_df.columns}")
    else:
        logger.warning("No feature columns found! Creating empty DataFrame")
        feature_df = pl.DataFrame()

    # Remove rows with NaN values
    original_rows = len(feature_df)
    feature_df = feature_df.drop_nulls()
    final_rows = len(feature_df)
    logger.info(f"Removed {original_rows - final_rows} rows with NaN values")
    logger.info(f"Final feature DataFrame shape: {feature_df.shape}")

    return feature_df
