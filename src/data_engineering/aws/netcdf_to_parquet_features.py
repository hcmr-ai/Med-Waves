import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import xarray as xr
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import time

import fsspec

from src.data_engineering.feature_augmentation import add_features_lazy


def convert_netcdf_to_parquet_hourly(netcdf_file_path) -> pl.DataFrame:
    """
    Convert a NetCDF file into a flat hourly Parquet file using Polars.

    Parameters
    ----------
    netcdf_file_path : Path
        Path to the input NetCDF file.
    output_file_path : Path
        Path to the output Parquet file where the hourly data will be stored.

    Notes
    -----
    - Assumes the NetCDF contains hourly measurements.
    - Converts all available variables and metadata into a flat DataFrame.
    """
    # Prefer explicit engines to avoid ambiguous-backend errors
    engines_to_try = ("h5netcdf", "netcdf4", "scipy")
    last_err: Exception | None = None
    df = None
    is_s3 = isinstance(netcdf_file_path, str) and netcdf_file_path.startswith("s3://")
    for engine in engines_to_try:
        try:
            if is_s3:
                # Open a fresh file-like for each attempt
                with fsspec.open(netcdf_file_path, "rb") as fobj:
                    with xr.open_dataset(fobj, engine=engine) as ds:
                        df = ds.to_dataframe().reset_index()
            else:
                with xr.open_dataset(netcdf_file_path, engine=engine) as ds:
                    df = ds.to_dataframe().reset_index()
            break
        except Exception as e:  # try next engine
            last_err = e
            df = None
            continue
    if df is None:
        raise RuntimeError(
            "Failed to open NetCDF. Please install one of: 'h5netcdf' (recommended) + 'h5py', or 'netCDF4'.\n"
            f"Tried engines: {engines_to_try}. Last error: {last_err}"
        )
    pl_df = pl.DataFrame(df)

    return pl_df

def process_all_lazy(degraded_dir: str, corrected_dir: str, output_dir: str, dry_run: bool = False, concurrency: int = 1):
    is_s3_degraded = degraded_dir.startswith("s3://")
    is_s3_corrected = corrected_dir.startswith("s3://")
    is_s3_output = output_dir.startswith("s3://")

    fs = fsspec.filesystem("s3") if (is_s3_degraded or is_s3_corrected or is_s3_output) else None

    # Prepare output directory only for local paths
    if not dry_run and not is_s3_output:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # List degraded files, supporting parquet and netcdf
    if is_s3_degraded:
        files_parquet = sorted(fs.glob(degraded_dir.rstrip("/") + "/*.parquet"))  # type: ignore[union-attr]
        files_netcdf = sorted(fs.glob(degraded_dir.rstrip("/") + "/*.nc"))  # type: ignore[union-attr]
        names_parquet = [Path(f).name for f in files_parquet]
        names_netcdf = [Path(f).name for f in files_netcdf]
    else:
        local_degraded = Path(degraded_dir)
        files_parquet = sorted(local_degraded.glob("*.parquet"))
        files_netcdf = sorted(local_degraded.glob("*.nc"))
        names_parquet = [f.name for f in files_parquet]
        names_netcdf = [f.name for f in files_netcdf]

    mode = "parquet" if len(names_parquet) > 0 else ("netcdf" if len(names_netcdf) > 0 else None)
    if mode is None:
        print("Found 0 files to process under degraded directory.")
        return

    file_names = names_parquet if mode == "parquet" else names_netcdf
    print(f"Found {len(file_names)} {mode} files to process...")

    def _process_one(file_name: str) -> tuple[str, bool, str, float]:
        start_time = time.time()
        try:
            # Build paths for current file
            if is_s3_degraded:
                degraded_path = degraded_dir.rstrip("/") + f"/{file_name}"
            else:
                degraded_path = str(Path(degraded_dir) / file_name)

            if is_s3_corrected:
                corrected_path = corrected_dir.rstrip("/") + f"/{file_name}"
                corrected_exists = fs.exists(corrected_path)  # type: ignore[union-attr]
            else:
                corrected_path = str(Path(corrected_dir) / file_name)
                corrected_exists = Path(corrected_path).exists()

            if not corrected_exists:
                return file_name, False, "corrected file not found", time.time() - start_time

            # Read inputs based on mode
            if mode == "parquet":
                df_deg = pl.scan_parquet(degraded_path)
                df_cor = pl.scan_parquet(corrected_path)
                out_file_name = file_name
            else:
                df_deg_pl = convert_netcdf_to_parquet_hourly(degraded_path)
                df_cor_pl = convert_netcdf_to_parquet_hourly(corrected_path)
                df_deg = df_deg_pl.lazy()
                df_cor = df_cor_pl.lazy()
                out_file_name = file_name[:-3] + ".parquet" if file_name.endswith(".nc") else (file_name + ".parquet")

            df_cor_labels = df_cor.select([
                pl.col("VHM0").alias("corrected_VHM0"),
                pl.col("VTM02").alias("corrected_VTM02")
            ])

            df_combined = pl.concat([df_deg, df_cor_labels], how="horizontal")
            df_aug = add_features_lazy(df_combined)

            if dry_run:
                # Just report target path
                return file_name, True, "dry-run", time.time() - start_time
            else:
                df_out = df_aug.collect()
                if is_s3_output:
                    target_path = output_dir.rstrip("/") + f"/{out_file_name}"
                    with fsspec.open(target_path, "wb") as f:
                        df_out.write_parquet(f)
                else:
                    df_out.write_parquet(Path(output_dir) / out_file_name)

            return file_name, True, "ok", time.time() - start_time
        except Exception as e:
            return file_name, False, str(e), time.time() - start_time

    # Execute sequentially or in parallel
    if concurrency <= 1:
        for name in tqdm(file_names, desc="Processing files", unit="file"):
            tqdm.write(f"🔄 Processing {name}...")
            fname, ok, msg, dur = _process_one(name)
            if ok:
                tqdm.write(f"✅ Finished {fname} in {dur:.2f}s")
            else:
                if msg == "dry-run":
                    tqdm.write(f"ℹ️ Dry-run: would write {(output_dir.rstrip('/') + '/' + (fname[:-3] + '.parquet' if fname.endswith('.nc') else fname)) if is_s3_output else str(Path(output_dir) / (fname[:-3] + '.parquet' if fname.endswith('.nc') else fname))}")
                elif msg == "corrected file not found":
                    tqdm.write(f"⚠️  Skipping {fname} – corrected file not found.")
                else:
                    tqdm.write(f"❌ Error processing {fname}: {msg}")
    else:
        with tqdm(total=len(file_names), desc="Processing files", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = {ex.submit(_process_one, name): name for name in file_names}
                for fut in as_completed(futures):
                    fname, ok, msg, dur = fut.result()
                    if ok and msg == "ok":
                        tqdm.write(f"✅ Finished {fname} in {dur:.2f}s")
                    elif ok and msg == "dry-run":
                        out_name = fname[:-3] + ".parquet" if fname.endswith(".nc") else fname
                        target_desc = (output_dir.rstrip("/") + f"/{out_name}") if is_s3_output else str(Path(output_dir) / out_name)
                        tqdm.write(f"ℹ️ Dry-run: would write {target_desc}")
                    elif msg == "corrected file not found":
                        tqdm.write(f"⚠️  Skipping {fname} – corrected file not found.")
                    else:
                        tqdm.write(f"❌ Error processing {fname}: {msg}")
                    pbar.update(1)

    print("🏁 All files processed." if not dry_run else "✅ Dry-run complete.")


def main():
    """
    Run feature augmentation joining degraded and corrected parquet files.

    Supports both local paths and S3 URIs (s3://bucket/prefix).
    """
    parser = argparse.ArgumentParser(description="Augment parquet features with labels")
    parser.add_argument("--degraded-dir", required=False, default="s3://medwav-dev-data/raw/without_reduced/year=2020")
    parser.add_argument("--corrected-dir", required=False, default="s3://medwav-dev-data/raw/with_reduced/year=2020")
    parser.add_argument("--output-dir", required=False, default="s3://medwav-dev-data/parquet/hourly/year=2020")
    parser.add_argument("--dry-run", action="store_true", help="Don't write outputs")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of parallel workers")
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
