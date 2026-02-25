import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import boto3
import fsspec
import numpy as np
import polars as pl
import xarray as xr
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


def _read_netcdf_as_polars(path: str) -> pl.DataFrame:
    """Read a NetCDF file (local or S3) into a Polars DataFrame."""
    is_s3 = path.startswith("s3://")
    engines = ("h5netcdf", "netcdf4", "scipy")
    last_err: Exception | None = None
    for engine in engines:
        try:
            if is_s3:
                with fsspec.open(path, "rb") as fobj:
                    with xr.open_dataset(fobj, engine=engine) as ds:
                        return pl.DataFrame(ds.to_dataframe().reset_index())
            else:
                with xr.open_dataset(path, engine=engine) as ds:
                    return pl.DataFrame(ds.to_dataframe().reset_index())
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Failed to open NetCDF {path}. "
        f"Tried engines {engines}. Last error: {last_err}"
    )


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

            # Extract year from the first timestamp for partitioning
            year = df_out["timestamp"][0].year
            df_out.write_parquet(output_path / file.name)
            if s3_client:
                key = f"{s3_prefix}/year={year}/{file.name}"
                _upload_df_to_s3(df_out, s3_client, s3_bucket, key)

        if dry_run:
            dest = f"s3://{s3_bucket}/{s3_prefix}/year=YYYY/{file.name}" if s3_uri else str(output_path / file.name)
            print(f"ℹ️ Dry-run: would save to {dest}")

    print("✅ Dry-run complete." if dry_run else "✅ All files processed.")


def process_all_lazy(
    degraded_dir: str,
    corrected_dir: str,
    output_dir: str,
    dry_run: bool = False,
    concurrency: int = 1,
):
    """Augment features, reading/writing from local paths or S3 URIs."""
    is_s3_degraded = degraded_dir.startswith("s3://")
    is_s3_corrected = corrected_dir.startswith("s3://")
    is_s3_output = output_dir.startswith("s3://")

    fs = (
        fsspec.filesystem("s3")
        if (is_s3_degraded or is_s3_corrected or is_s3_output)
        else None
    )

    if not dry_run and not is_s3_output:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _list_files(directory: str, is_s3: bool) -> tuple[list[str], list[str]]:
        if is_s3:
            pq = sorted(
                f.split("/")[-1]
                for f in fs.glob(directory.rstrip("/") + "/*.parquet")  # type: ignore[union-attr]
            )
            nc = sorted(
                f.split("/")[-1]
                for f in fs.glob(directory.rstrip("/") + "/*.nc")  # type: ignore[union-attr]
            )
        else:
            loc = Path(directory)
            pq = sorted(f.name for f in loc.glob("*.parquet"))
            nc = sorted(f.name for f in loc.glob("*.nc"))
        return pq, nc

    deg_pq, deg_nc = _list_files(degraded_dir, is_s3_degraded)
    deg_mode = "parquet" if deg_pq else ("netcdf" if deg_nc else None)
    if deg_mode is None:
        print("No parquet or netcdf files found in degraded directory.")
        return
    deg_files = deg_pq if deg_mode == "parquet" else deg_nc

    cor_pq, cor_nc = _list_files(corrected_dir, is_s3_corrected)
    cor_mode = "parquet" if cor_pq else ("netcdf" if cor_nc else None)
    if cor_mode is None:
        print("No parquet or netcdf files found in corrected directory.")
        return
    cor_by_stem: dict[str, str] = {
        name.rsplit(".", 1)[0]: name for name in (cor_pq if cor_mode == "parquet" else cor_nc)
    }

    print(
        f"Found {len(deg_files)} {deg_mode} degraded / "
        f"{len(cor_by_stem)} {cor_mode} corrected files."
    )

    def _process_one(file_name: str) -> tuple[str, bool, str, float]:
        start = time.time()
        try:
            stem = file_name.rsplit(".", 1)[0]

            deg_path = (
                degraded_dir.rstrip("/") + f"/{file_name}"
                if is_s3_degraded
                else str(Path(degraded_dir) / file_name)
            )

            cor_name = cor_by_stem.get(stem)
            if cor_name is None:
                return file_name, False, "corrected file not found", time.time() - start

            cor_path = (
                corrected_dir.rstrip("/") + f"/{cor_name}"
                if is_s3_corrected
                else str(Path(corrected_dir) / cor_name)
            )

            out_name = stem + ".parquet"

            if is_s3_output:
                target = output_dir.rstrip("/") + f"/{out_name}"
                if fs.exists(target):  # type: ignore[union-attr]
                    return file_name, False, "exists", time.time() - start
            else:
                if (Path(output_dir) / out_name).exists():
                    return file_name, False, "exists", time.time() - start

            if dry_run:
                return file_name, True, "dry-run", time.time() - start

            if deg_mode == "parquet":
                df_deg = pl.scan_parquet(deg_path)
            else:
                df_deg = _read_netcdf_as_polars(deg_path).lazy()

            if cor_mode == "parquet":
                df_cor = pl.scan_parquet(cor_path)
            else:
                df_cor = _read_netcdf_as_polars(cor_path).lazy()

            df_cor_labels = df_cor.select(
                [
                    pl.col("VHM0").alias("corrected_VHM0"),
                    pl.col("VTM02").alias("corrected_VTM02"),
                ]
            )
            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")
            df_out = add_features_lazy(df_combined).collect()

            if is_s3_output:
                target = output_dir.rstrip("/") + f"/{out_name}"
                with fsspec.open(target, "wb") as f:
                    df_out.write_parquet(f)
            else:
                df_out.write_parquet(Path(output_dir) / out_name)

            return file_name, True, "ok", time.time() - start
        except Exception as e:
            return file_name, False, str(e), time.time() - start

    def _report(fname: str, ok: bool, msg: str, dur: float) -> None:
        if ok and msg == "ok":
            tqdm.write(f"✅ Finished {fname} in {dur:.2f}s")
        elif ok and msg == "dry-run":
            out = fname.rsplit(".", 1)[0] + ".parquet" if fname.endswith(".nc") else fname
            target = (
                output_dir.rstrip("/") + f"/{out}"
                if is_s3_output
                else str(Path(output_dir) / out)
            )
            tqdm.write(f"ℹ️  Dry-run: would write {target}")
        elif msg == "corrected file not found":
            tqdm.write(f"⚠️  Skipping {fname} – corrected file not found.")
        elif msg == "exists":
            tqdm.write(f"⏭️  Skipping {fname} – output already exists.")
        else:
            tqdm.write(f"❌ Error processing {fname}: {msg}")

    if concurrency <= 1:
        for name in tqdm(deg_files, desc="Processing files", unit="file"):
            tqdm.write(f"🔄 Processing {name}...")
            _report(*_process_one(name))
    else:
        with tqdm(total=len(deg_files), desc="Processing files", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_process_one, n): n for n in deg_files}
                for fut in as_completed(futures):
                    _report(*fut.result())
                    pbar.update(1)

    print("🏁 All files processed." if not dry_run else "✅ Dry-run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Feature augmentation with S3 I/O support"
    )
    parser.add_argument(
        "--degraded-dir",
        default="s3://medwav-dev-data/raw/without_reduced/year=2017",
        # /data/tsolis/AI_project/parquet/without_reduced/hourly/
    )
    parser.add_argument(
        "--corrected-dir",
        default="s3://medwav-dev-data/raw/with_reduced/year=2017",
        # /data/tsolis/AI_project/parquet/with_reduced/hourly/
    )
    parser.add_argument(
        "--output-dir",
        default="s3://medwav-dev-data/parquet/hourly_extra_features/year=2017",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    process_all_lazy(
        degraded_dir=args.degraded_dir,
        corrected_dir=args.corrected_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        concurrency=max(1, args.concurrency),
    )


if __name__ == "__main__":
    main()
