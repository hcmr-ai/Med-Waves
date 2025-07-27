import time
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    wind_dir_rad = df['WDIR'] * np.pi / 180
    wave_dir_rad = df['VMDR'] * np.pi / 180

    df = df.with_columns([
        (df['WSPD'] * wind_dir_rad.sin()).alias('U10'),
        (df['WSPD'] * wind_dir_rad.cos()).alias('V10'),
        wave_dir_rad.sin().alias('wave_dir_sin'),
        wave_dir_rad.cos().alias('wave_dir_cos'),
        pl.col("time").cast(pl.Datetime).alias("timestamp")
    ])

    # Time encodings
    df = df.with_columns([
        (2 * np.pi * df["timestamp"].dt.hour() / 24).sin().alias("sin_hour"),
        (2 * np.pi * df["timestamp"].dt.hour() / 24).cos().alias("cos_hour"),
        (2 * np.pi * df["timestamp"].dt.month() / 12).sin().alias("sin_month"),
        (2 * np.pi * df["timestamp"].dt.month() / 12).cos().alias("cos_month"),
        (2 * np.pi * df["timestamp"].dt.ordinal_day() / 365.0).sin().alias("sin_doy"),
        (2 * np.pi * df["timestamp"].dt.ordinal_day() / 365.0).cos().alias("cos_doy"),
    ])

    # Normalize lat/lon
    lat_norm = (df["latitude"] - df["latitude"].min()) / (df["latitude"].max() - df["latitude"].min())
    lon_norm = (df["longitude"] - df["longitude"].min()) / (df["longitude"].max() - df["longitude"].min())

    df = df.with_columns([
        lat_norm.alias("lat_norm"),
        lon_norm.alias("lon_norm")
    ])

    return df

def add_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    # Wind & wave directions (radians)
    wind_dir_rad = pl.col('WDIR') * np.pi / 180
    wave_dir_rad = pl.col('VMDR') * np.pi / 180

    df = df.with_columns([
        (pl.col('WSPD') * wind_dir_rad.sin()).alias('U10'),
        (pl.col('WSPD') * wind_dir_rad.cos()).alias('V10'),
        wave_dir_rad.sin().alias('wave_dir_sin'),
        wave_dir_rad.cos().alias('wave_dir_cos'),
        pl.col("time").cast(pl.Datetime).alias("timestamp")
    ])

    # Cyclic time encodings
    df = df.with_columns([
        (2 * np.pi * pl.col("timestamp").dt.hour() / 24).sin().alias("sin_hour"),
        (2 * np.pi * pl.col("timestamp").dt.hour() / 24).cos().alias("cos_hour"),
        (2 * np.pi * pl.col("timestamp").dt.ordinal_day() / 365.0).sin().alias("sin_doy"),
        (2 * np.pi * pl.col("timestamp").dt.ordinal_day() / 365.0).cos().alias("cos_doy"),
        (2 * np.pi * pl.col("timestamp").dt.month() / 12.0).sin().alias("sin_month"),
        (2 * np.pi * pl.col("timestamp").dt.month() / 12.0).cos().alias("cos_month"),
    ])

    # Normalize lat/lon lazily (min/max per file)
    lat_norm = (pl.col("latitude") - pl.col("latitude").min()) / (pl.col("latitude").max() - pl.col("latitude").min())
    lon_norm = (pl.col("longitude") - pl.col("longitude").min()) / (pl.col("longitude").max() - pl.col("longitude").min())

    df = df.with_columns([
        lat_norm.alias("lat_norm"),
        lon_norm.alias("lon_norm")
    ])

    float32_feats = ["sin_hour", "cos_hour", "sin_doy", "cos_doy", "sin_month", "cos_month"]
    df = df.with_columns([pl.col(f).cast(pl.Float32) for f in float32_feats])

    return df

def process_all(degraded_dir: str, corrected_dir: str, output_dir: str, dry_run: bool = False):
    degraded_dir = Path(degraded_dir)
    corrected_dir = Path(corrected_dir)
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(degraded_dir.glob("*.parquet"))
    print(f"Found {len(files)} files to process...")

    # for i, file in enumerate(files):
    for file in tqdm(files, desc="Processing files", unit="file"):
        # print(f"[{i+1}/{len(files)}] Processing {file.name}...")

        df_cor_path = corrected_dir / file.name

        if not df_cor_path.exists():
            print(f"⚠️ Skipping {file.name} – corrected file not found.")
            continue

        if not dry_run:
            df_deg = pl.scan_parquet(file)
            df_cor = pl.scan_parquet(df_cor_path)

            # df_deg = df_deg.with_columns([
            #     df_cor["VHM0"].alias("corrected_VHM0"),
            #     df_cor["VTM02"].alias("corrected_VTM02"),
            # ])
            # lazy
            df_cor_labels = df_cor.select([
                pl.col("VHM0").alias("corrected_VHM0"),
                pl.col("VTM02").alias("corrected_VTM02")
            ])
            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")

            df_aug = add_features_lazy(df_combined)

        if dry_run:
            print(f"ℹ️ Dry-run: would save to {output_dir / file.name}")
        else:
            df_out = df_aug.collect()
            df_out.write_parquet(output_dir / file.name)

    print("✅ Dry-run complete." if dry_run else "✅ All files processed.")


def process_all_lazy(degraded_dir: str, corrected_dir: str, output_dir: str, dry_run: bool = False):
    degraded_dir = Path(degraded_dir)
    corrected_dir = Path(corrected_dir)
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

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

            df_cor_labels = df_cor.select([
                pl.col("VHM0").alias("corrected_VHM0"),
                pl.col("VTM02").alias("corrected_VTM02")
            ])

            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")
            df_aug = add_features_lazy(df_combined)

            if dry_run:
                tqdm.write(f"ℹ️ Dry-run: would write {output_dir / file_name}")
            else:
                df_out = df_aug.collect()
                df_out.write_parquet(output_dir / file_name)

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
        dry_run=False
    )
