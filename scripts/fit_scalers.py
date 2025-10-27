import os
import boto3
import polars as pl
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import s3fs
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer

S3_BUCKET = "medwav-dev-data"
S3_PREFIX = "scalers/"
LOCAL_TMP = "data/scalers/"
DATA_PATH = "s3://medwav-dev-data/parquet/hourly/year=2021"# "/Users/deeplab/Documents/projects/hcmr/data/hourly/"

FEATURES = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos']
def load_all_data(parquet_dir: str, features: list[str] = None) -> np.ndarray:
    """Load and stack all parquet files into numpy"""
    # lf = pl.scan_parquet(parquet_dir + "/*.parquet", storage_options={
    #     "AWS_REGION": "eu-central-1",
    #     "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
    #     "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
    # })
    # if features:
    #     lf = lf.select(features)
    # df = lf.collect()
    # arr = df.to_numpy()
    # return arr
    fs = s3fs.S3FileSystem()
    files = fs.glob(f"{parquet_dir}/*.parquet")
    print(f"Found {len(files)} parquet files")

    dfs = []
    for f in tqdm(files, desc="Loading parquet files"):
        with fs.open(f, "rb") as fh:
            df = pl.read_parquet(fh)
            print(f"\nFile: {f}")
            print(f"  Shape: {df.shape}")
            df = df.drop_nulls(subset=["VHM0", "corrected_VHM0"])
            if features:
                df = df.select(features)
            dfs.append(df)
            print(f"  After drop nulls: {df.shape}")

    # Concatenate vertically
    full_df = pl.concat(dfs, how="vertical")

    return full_df.to_numpy().astype(np.float32)

def save_to_s3(local_path, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    print(f"✓ Uploaded to s3://{bucket}/{key}")

if __name__ == "__main__":
    X = load_all_data(DATA_PATH, features=FEATURES)
    print("Loaded data:", X.shape)  # e.g. (N, 1)

    # Reshape for normalizer: (N, H, W, C)
    if X.ndim == 2:   # (N, C)
        X = X.reshape(-1, 1, 1, X.shape[-1])

    # Define configs
    configs = {
        "BU24h_zscore": dict(mode="zscore"),
        "BU48h_quantile": dict(mode="quantile"),
        "BU72h_quantile": dict(mode="quantile"),
    }

    for name, cfg in tqdm(configs.items(), desc="Fitting scalers"):
        # Fit normalizer
        normalizer = WaveNormalizer(**cfg)
        normalizer.fit(X)

        # Save locally
        local_path = os.path.join(LOCAL_TMP, f"{name}.pkl")
        os.makedirs(LOCAL_TMP, exist_ok=True)
        normalizer.save(local_path)

        # Upload to S3
        s3_key = f"{S3_PREFIX}{name}.pkl"
        normalizer.save_to_s3(local_path, S3_BUCKET, s3_key)
