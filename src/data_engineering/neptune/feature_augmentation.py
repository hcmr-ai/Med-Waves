import io
import time
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import polars as pl
from botocore.config import Config
from tqdm import tqdm


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    wind_dir_rad = df["WDIR"] * np.pi / 180
    wave_dir_rad = df["VMDR"] * np.pi / 180
    delta = wind_dir_rad - wave_dir_rad

    df = df.with_columns(
        [
            (df["WSPD"] * wind_dir_rad.sin()).alias("U10"),
            (df["WSPD"] * wind_dir_rad.cos()).alias("V10"),
            wind_dir_rad.sin().alias("wind_dir_sin"),
            wind_dir_rad.cos().alias("wind_dir_cos"),
            wave_dir_rad.sin().alias("wave_dir_sin"),
            wave_dir_rad.cos().alias("wave_dir_cos"),
            delta.cos().alias("cos_delta"),
            delta.sin().alias("sin_delta"),
            pl.col("time").cast(pl.Datetime).alias("timestamp"),
        ]
    )

    # Wind-wave alignment projections
    df = df.with_columns(
        [
            (df["WSPD"] * df["cos_delta"]).alias("alongwind"),
            (df["WSPD"] * df["sin_delta"]).alias("crosswind"),
        ]
    )

    # Temporal differences (assumes rows are time-ordered)
    df = df.with_columns(
        [
            pl.col("VHM0").diff(1).fill_null(0).alias("dVHM0"),
            pl.col("WSPD").diff(1).fill_null(0).alias("dWSPD"),
        ]
    )

    # Spatial gradient proxy: rolling std over a 3-step window
    df = df.with_columns(
        pl.col("VHM0").rolling_std(window_size=3, min_periods=1).alias("grad_mag"),
    )

    # Storm regime features (VHM0 is the degraded/raw model output)
    df = df.with_columns(
        [
            (1 + pl.col("VHM0")).log().alias("storm_regime"),
            (1 / (1 + (-(pl.col("VHM0") - 3.0) / 0.5).exp())).alias(
                "storm_regime_sig"
            ),
        ]
    )

    # Cyclic time encodings
    df = df.with_columns(
        [
            (2 * np.pi * df["timestamp"].dt.hour() / 24).sin().alias("sin_hour"),
            (2 * np.pi * df["timestamp"].dt.hour() / 24).cos().alias("cos_hour"),
            (2 * np.pi * df["timestamp"].dt.month() / 12).sin().alias("sin_month"),
            (2 * np.pi * df["timestamp"].dt.month() / 12).cos().alias("cos_month"),
            (2 * np.pi * df["timestamp"].dt.ordinal_day() / 365.0)
            .sin()
            .alias("sin_doy"),
            (2 * np.pi * df["timestamp"].dt.ordinal_day() / 365.0)
            .cos()
            .alias("cos_doy"),
        ]
    )

    # Normalize lat/lon
    lat_norm = (df["latitude"] - df["latitude"].min()) / (
        df["latitude"].max() - df["latitude"].min()
    )
    lon_norm = (df["longitude"] - df["longitude"].min()) / (
        df["longitude"].max() - df["longitude"].min()
    )

    df = df.with_columns([lat_norm.alias("lat_norm"), lon_norm.alias("lon_norm")])

    return df


def add_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    wind_dir_rad = pl.col("WDIR") * np.pi / 180
    wave_dir_rad = pl.col("VMDR") * np.pi / 180
    delta = wind_dir_rad - wave_dir_rad

    df = df.with_columns(
        [
            (pl.col("WSPD") * wind_dir_rad.sin()).alias("U10"),
            (pl.col("WSPD") * wind_dir_rad.cos()).alias("V10"),
            wind_dir_rad.sin().alias("wind_dir_sin"),
            wind_dir_rad.cos().alias("wind_dir_cos"),
            wave_dir_rad.sin().alias("wave_dir_sin"),
            wave_dir_rad.cos().alias("wave_dir_cos"),
            delta.cos().alias("cos_delta"),
            delta.sin().alias("sin_delta"),
            pl.col("time").cast(pl.Datetime).alias("timestamp"),
        ]
    )

    # Wind-wave alignment projections
    df = df.with_columns(
        [
            (pl.col("WSPD") * pl.col("cos_delta")).alias("alongwind"),
            (pl.col("WSPD") * pl.col("sin_delta")).alias("crosswind"),
        ]
    )

    # Temporal differences (assumes rows are time-ordered)
    df = df.with_columns(
        [
            pl.col("VHM0").diff(1).fill_null(0).alias("dVHM0"),
            pl.col("WSPD").diff(1).fill_null(0).alias("dWSPD"),
        ]
    )

    # Spatial gradient proxy: rolling std over a 3-step window
    df = df.with_columns(
        pl.col("VHM0").rolling_std(window_size=3, min_periods=1).alias("grad_mag"),
    )

    # Storm regime features (VHM0 is the degraded/raw model output)
    df = df.with_columns(
        [
            (1 + pl.col("VHM0")).log().alias("storm_regime"),
            (1 / (1 + (-(pl.col("VHM0") - 3.0) / 0.5).exp())).alias(
                "storm_regime_sig"
            ),
        ]
    )

    # Cyclic time encodings
    df = df.with_columns(
        [
            (2 * np.pi * pl.col("timestamp").dt.hour() / 24).sin().alias("sin_hour"),
            (2 * np.pi * pl.col("timestamp").dt.hour() / 24).cos().alias("cos_hour"),
            (2 * np.pi * pl.col("timestamp").dt.ordinal_day() / 365.0)
            .sin()
            .alias("sin_doy"),
            (2 * np.pi * pl.col("timestamp").dt.ordinal_day() / 365.0)
            .cos()
            .alias("cos_doy"),
            (2 * np.pi * pl.col("timestamp").dt.month() / 12.0)
            .sin()
            .alias("sin_month"),
            (2 * np.pi * pl.col("timestamp").dt.month() / 12.0)
            .cos()
            .alias("cos_month"),
        ]
    )

    # Normalize lat/lon lazily (min/max per file)
    lat_norm = (pl.col("latitude") - pl.col("latitude").min()) / (
        pl.col("latitude").max() - pl.col("latitude").min()
    )
    lon_norm = (pl.col("longitude") - pl.col("longitude").min()) / (
        pl.col("longitude").max() - pl.col("longitude").min()
    )

    df = df.with_columns([lat_norm.alias("lat_norm"), lon_norm.alias("lon_norm")])

    float32_feats = [
        "sin_hour",
        "cos_hour",
        "sin_doy",
        "cos_doy",
        "sin_month",
        "cos_month",
    ]
    df = df.with_columns([pl.col(f).cast(pl.Float32) for f in float32_feats])

    return df


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """'s3://bucket/prefix/path' -> ('bucket', 'prefix/path')"""
    stripped = uri.removeprefix("s3://")
    bucket, _, prefix = stripped.partition("/")
    return bucket, prefix.rstrip("/")


def _make_s3_client():
    return boto3.client(
        "s3",
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def _upload_df_to_s3(df: pl.DataFrame, s3_client, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def process_all(
    degraded_dir: str,
    corrected_dir: str,
    output_dir: str,
    dry_run: bool = False,
    s3_uri: Optional[str] = None,
):
    degraded_dir = Path(degraded_dir)
    corrected_dir = Path(corrected_dir)
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    s3_client, s3_bucket, s3_prefix = None, None, None
    if s3_uri:
        s3_bucket, s3_prefix = _parse_s3_uri(s3_uri)
        s3_client = _make_s3_client()

    files = sorted(degraded_dir.glob("*.parquet"))
    print(f"Found {len(files)} files to process...")

    for file in tqdm(files, desc="Processing files", unit="file"):
        df_cor_path = corrected_dir / file.name

        if not df_cor_path.exists():
            print(f"⚠️ Skipping {file.name} – corrected file not found.")
            continue

        if not dry_run:
            df_deg = pl.scan_parquet(file)
            df_cor = pl.scan_parquet(df_cor_path)

            df_cor_labels = df_cor.select(
                [
                    pl.col("VHM0").alias("corrected_VHM0"),
                    pl.col("VTM02").alias("corrected_VTM02"),
                ]
            )
            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")
            df_out = add_features_lazy(df_combined).collect()

            df_out.write_parquet(output_path / file.name)
            if s3_client:
                key = f"{s3_prefix}/{file.name}"
                _upload_df_to_s3(df_out, s3_client, s3_bucket, key)

        if dry_run:
            dest = f"s3://{s3_bucket}/{s3_prefix}/{file.name}" if s3_uri else str(output_path / file.name)
            print(f"ℹ️ Dry-run: would save to {dest}")

    print("✅ Dry-run complete." if dry_run else "✅ All files processed.")


def process_all_lazy(
    degraded_dir: str,
    corrected_dir: str,
    output_dir: str,
    dry_run: bool = False,
    s3_uri: Optional[str] = None,
):
    degraded_dir = Path(degraded_dir)
    corrected_dir = Path(corrected_dir)
    output_path = Path(output_dir)
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    s3_client, s3_bucket, s3_prefix = None, None, None
    if s3_uri:
        s3_bucket, s3_prefix = _parse_s3_uri(s3_uri)
        s3_client = _make_s3_client()

    files = sorted(degraded_dir.glob("*.parquet"))
    print(f"Found {len(files)} files to process...")

    for file in tqdm(files, desc="Processing files", unit="file"):
        file_name = file.name
        start_time = time.time()
        tqdm.write(f"🔄 Processing {file_name}...")

        corrected_file = corrected_dir / file_name
        if not corrected_file.exists():
            tqdm.write(f"⚠️  Skipping {file_name} – corrected file not found.")
            continue

        try:
            df_deg = pl.scan_parquet(str(file))
            df_cor = pl.scan_parquet(str(corrected_file))

            df_cor_labels = df_cor.select(
                [
                    pl.col("VHM0").alias("corrected_VHM0"),
                    pl.col("VTM02").alias("corrected_VTM02"),
                ]
            )

            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")
            df_aug = add_features_lazy(df_combined)

            if dry_run:
                dest = f"s3://{s3_bucket}/{s3_prefix}/{file_name}" if s3_uri else str(output_path / file_name)
                tqdm.write(f"ℹ️ Dry-run: would write {dest}")
            else:
                df_out = df_aug.collect()
                df_out.write_parquet(output_path / file_name)
                if s3_client:
                    key = f"{s3_prefix}/{file_name}"
                    _upload_df_to_s3(df_out, s3_client, s3_bucket, key)

            duration = time.time() - start_time
            tqdm.write(f"✅ Finished {file_name} in {duration:.2f}s")

        except Exception as e:
            tqdm.write(f"❌ Error processing {file_name}: {e}")

    print("🏁 All files processed." if not dry_run else "✅ Dry-run complete.")


# --- Run ---
if __name__ == "__main__":
    process_all_lazy(
        degraded_dir="/data/tsolis/AI_project/parquet/without_reduced/hourly",
        corrected_dir="/data/tsolis/AI_project/parquet/with_reduced/hourly",
        output_dir="/data/tsolis/AI_project/parquet/augmented_with_labels/hourly",
        s3_uri="s3://medwav-dev-data/parquet/hourly_extra_features/",
        dry_run=False,
    )
