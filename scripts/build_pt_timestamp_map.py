"""
Build a pt_stem × hour_idx → timestamp mapping from Parquet files.

Each .pt file is produced by preprocessing one Parquet file
(WAVEAN{date}.parquet → WAVEAN{date}.pt).  Inside the .pt tensor,
axis-0 is sorted unique timestamps: tensor[hour_idx] corresponds to
unique_times[hour_idx] from the source Parquet.

This script reads only the "time" column from every Parquet file and
writes a CSV that evaluate_bunet.py can consume via --timestamps-csv.

Output CSV columns
------------------
  pt_stem   : Parquet/pt filename without extension  (e.g. WAVEAN20230115)
  hour_idx  : 0-based position in sorted unique timestamps within that file
  timestamp : ISO-8601 UTC string  (e.g. 2023-01-15T03:00:00+00:00)

Usage
-----
  # Local Parquet directory
  poetry run python scripts/build_pt_timestamp_map.py \\
      --parquet-dir /mnt/blobstorage/parquet/hourly_extra_features \\
      --output-csv  /mnt/blobstorage/diagnostics/pt_timestamp_map.csv

  # S3
  poetry run python scripts/build_pt_timestamp_map.py \\
      --parquet-dir s3://medwav-dev-data/parquet/hourly_extra_features \\
      --output-csv  /mnt/blobstorage/diagnostics/pt_timestamp_map.csv

  # Restrict to specific years
  poetry run python scripts/build_pt_timestamp_map.py \\
      --parquet-dir /mnt/blobstorage/parquet/hourly_extra_features \\
      --output-csv  /mnt/blobstorage/diagnostics/pt_timestamp_map.csv \\
      --years 2022 2023
"""

import argparse
import csv
import glob
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_parquet_files(parquet_dir: str, years: list[int] | None) -> list[str]:
    is_s3 = parquet_dir.startswith("s3://")

    if is_s3:
        import fsspec
        fs = fsspec.filesystem("s3")
        raw = parquet_dir.replace("s3://", "").rstrip("/")
        patterns = (
            [f"{raw}/**/WAVEAN*.parquet", f"{raw}/WAVEAN*.parquet"]
            if not years
            else [f"{raw}/**/WAVEAN{y}*.parquet" for y in years]
               + [f"{raw}/WAVEAN{y}*.parquet" for y in years]
        )
        files = sorted({f"s3://{p}" for pat in patterns for p in fs.glob(pat)})
    else:
        root = parquet_dir.rstrip("/")
        patterns = (
            [f"{root}/**/WAVEAN*.parquet", f"{root}/WAVEAN*.parquet"]
            if not years
            else [f"{root}/**/WAVEAN{y}*.parquet" for y in years]
               + [f"{root}/WAVEAN{y}*.parquet" for y in years]
        )
        seen: set[str] = set()
        files = []
        for pat in patterns:
            for f in glob.glob(pat, recursive=True):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        files.sort()

    if not files:
        raise FileNotFoundError(
            f"No WAVEAN*.parquet files found under '{parquet_dir}'"
            + (f" for years {years}" if years else "")
        )
    return files


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _extract_timestamps(path: str) -> tuple[str, list]:
    """Return (pt_stem, sorted_unique_datetimes) for one Parquet file.

    Reads only the 'time' column — fast even for large files.
    Mirrors exactly what the preprocessing pipeline does:
        unique_times = np.unique(table.column('time').to_numpy())
    so hour_idx 0 … T-1 matches tensor[0] … tensor[T-1].
    """
    is_s3 = path.startswith("s3://")

    if is_s3:
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(path, "rb") as fh:
            table = pq.read_table(fh, columns=["time"])
    else:
        table = pq.read_table(path, columns=["time"])

    time_col = table.column("time")

    # Convert to numpy datetime64; cast if stored as int/timestamp-micros/etc.
    time_np = time_col.to_numpy()
    if not np.issubdtype(time_np.dtype, np.datetime64):
        time_np = time_np.astype("datetime64[ns]")

    unique_times = np.unique(time_np)  # sorted, same as preprocessing

    # Convert each numpy datetime64 → Python datetime (UTC)
    datetimes = [
        ts.astype("datetime64[ms]").astype("O").replace(tzinfo=timezone.utc)
        for ts in unique_times
    ]

    pt_stem = Path(path).stem.replace(".parquet", "")  # handles bare stem too
    return pt_stem, datetimes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_map(parquet_dir: str, output_csv: str, years: list[int] | None) -> None:
    files = _discover_parquet_files(parquet_dir, years)
    print(f"Found {len(files)} Parquet file(s) to process.")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pt_stem", "hour_idx", "timestamp"])

        for i, path in enumerate(files):
            try:
                pt_stem, datetimes = _extract_timestamps(path)
            except Exception as exc:
                print(f"  [WARN] Skipping {path}: {exc}", file=sys.stderr)
                continue

            for hour_idx, dt in enumerate(datetimes):
                writer.writerow([pt_stem, hour_idx, dt.isoformat()])
                total_rows += 1

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  [{i + 1}/{len(files)}] {pt_stem}  ({len(datetimes)} hours)")

    print(f"\nDone. {total_rows:,} rows written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pt_stem × hour_idx → timestamp CSV from Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet-dir",
        required=True,
        help="Root directory containing WAVEAN*.parquet files (local or s3://).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Restrict to these years (e.g. --years 2022 2023). Default: all years.",
    )
    args = parser.parse_args()
    build_map(args.parquet_dir, args.output_csv, args.years)


if __name__ == "__main__":
    main()
