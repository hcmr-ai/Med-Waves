import os
import boto3
import polars as pl
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import s3fs
import glob
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer

S3_BUCKET = "medwav-dev-data"
S3_PREFIX = "scalers/"
LOCAL_TMP = "data/scalers/"
USE_S3 = True

if USE_S3:
    DATA_PATHS = ["s3://medwav-dev-data/parquet/hourly/year=2021", "s3://medwav-dev-data/parquet/hourly/year=2022"]
else:
    DATA_PATHS = ["/Users/deeplab/Documents/projects/hcmr/data/hourly/"]

FEATURES = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos', 'corrected_VHM0']
def load_all_data(parquet_dir: str, features: list[str] = None) -> np.ndarray:
    """Load and stack all parquet files into numpy"""
    if USE_S3:
        fs = s3fs.S3FileSystem()
        files = fs.glob(f"{parquet_dir}/*.parquet")
    else:
        files = glob.glob(f"{parquet_dir}/*.parquet")
    print(f"Found {len(files)} parquet files")

    dfs = []
    for f in tqdm(files, desc="Loading parquet files"):
        if USE_S3:
            with fs.open(f, "rb") as fh:
                df = pl.read_parquet(fh)
                print(f"\nFile: {f}")
                print(f"  Shape: {df.shape}")
                df = df.drop_nulls(subset=["VHM0", "corrected_VHM0"])
                if features:
                    df = df.select(features)
                dfs.append(df)
                print(f"  After drop nulls: {df.shape}")
        else:
            with open(os.path.join(parquet_dir, f), "rb") as fh:
                df = pl.read_parquet(fh)
                print(f"\nFile: {f}")
                print(f"  Shape: {df.shape}")
                df = df.drop_nulls(subset=["VHM0", "corrected_VHM0"])
                if features:
                    df = df.select(features)
                print(df.head())
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

    X_parts = [load_all_data(p, features=FEATURES) for p in DATA_PATHS]
    X = np.concatenate(X_parts, axis=0)
    print("Loaded data:", X.shape)  # e.g. (N, C)

    # Reshape for normalizer: (N, H, W, C)
    if X.ndim == 2:   # (N, C)
        X = X.reshape(-1, 1, 1, X.shape[-1])

    # Define configs
    configs = {
        "BU24h_zscore_target": dict(mode="zscore"),
        # "BU48h_quantile_target": dict(mode="quantile"),
        # "BU72h_quantile_target": dict(mode="quantile"),
    }

    for name, cfg in tqdm(configs.items(), desc="Fitting scalers"):
        # Fit normalizer
        normalizer = WaveNormalizer(**cfg)
        
        # Validation: Check feature order matches data shape
        print(f"\n{'='*80}")
        print(f"Fitting normalizer: {name}")
        print(f"{'='*80}")
        print(f"Data shape: {X.shape}")
        print(f"Number of features: {len(FEATURES)}")
        print(f"Number of channels in data: {X.shape[-1]}")
        
        if len(FEATURES) != X.shape[-1]:
            raise ValueError(
                f"Mismatch: FEATURES has {len(FEATURES)} items but data has {X.shape[-1]} channels"
            )
        
        # Fit the normalizer
        normalizer.fit(X, feature_order=FEATURES, target_feature_name="corrected_VHM0")
        
        # Validation: Verify target feature name was stored correctly
        print(f"\nNormalizer metadata:")
        print(f"  Feature order length: {len(normalizer.feature_order_) if normalizer.feature_order_ else 'None'}")
        print(f"  Target feature name: {normalizer.target_feature_name_}")
        print(f"  Number of stats channels: {len(normalizer.stats_)}")
        
        # Validate target feature lookup
        if normalizer.feature_order_ and normalizer.target_feature_name_:
            try:
                target_idx = normalizer.feature_order_.index(normalizer.target_feature_name_)
                print(f"  Target '{normalizer.target_feature_name_}' found at index: {target_idx}")
                print(f"  Stats available at index {target_idx}: {target_idx in normalizer.stats_}")
                if target_idx in normalizer.stats_:
                    target_stats = normalizer.stats_[target_idx]
                    if isinstance(target_stats, tuple):
                        mean, std = target_stats
                        print(f"  Target stats (tuple): mean={mean:.6f}, std={std:.6f}")
                    else:
                        print(f"  Target stats type: {type(target_stats)}")
            except ValueError:
                print(f"  WARNING: Target '{normalizer.target_feature_name_}' not found in feature_order!")
        
        # Print first and last few features for verification
        if normalizer.feature_order_:
            print(f"\n  First 3 features: {normalizer.feature_order_[:3]}")
            print(f"  Last 3 features: {normalizer.feature_order_[-3:]}")
        
        print(f"{'='*80}\n")

        # Save locally
        local_path = os.path.join(LOCAL_TMP, f"{name}_with_corrected.pkl")
        os.makedirs(LOCAL_TMP, exist_ok=True)
        normalizer.save(local_path)
        print(f"✓ Saved normalizer to {local_path}")

        # Upload to S3 if USE_S3 is True
        if USE_S3:
            s3_key = f"{S3_PREFIX}{name}_with_corrected.pkl"
            normalizer.save_to_s3(local_path, S3_BUCKET, s3_key)
            print(f"✓ Uploaded to s3://{S3_BUCKET}/{s3_key}")
        else:
            print(f"Skipping upload to S3 as USE_S3 is False")
