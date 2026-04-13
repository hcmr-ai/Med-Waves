"""
Load one file per region and select 30 random grid points.

For each region (atlantic / mediterranean) the script:
  1. Discovers one representative file from the data source.
  2. Loads it (Parquet or .pt).
  3. Keeps sea grid cells only (finite VHM0), matching CachedWaveDataset sea mask.
  4. Applies the Gibraltar/Biscay region mask.
  5. Randomly samples N unique grid points (lat/lon pairs).
  6. Saves results as CSV files and a cartopy map PNG.

Default output directory: /mnt/blobstorage/diagnostics/sampled_grid_points/

  Outputs
  -------
  sampled_grid_points_atlantic.csv
  sampled_grid_points_mediterranean.csv
  sampled_grid_points_all.csv
  sampled_grid_points_map.png

Region definitions (aligned with CachedWaveDataset and train_edcdf_regional):
  atlantic:      longitude < -5.5  OR  (latitude > 43 AND longitude < 0)
  mediterranean: longitude >= -5.5 AND NOT (latitude > 43 AND longitude < 0)

Usage:
    # Parquet (local)
    poetry run python -m src.pipelines.diagnostics.sample_grid_points \\
        --data-path /mnt/blobstorage/parquet/hourly/year=2022 \\
        --format parquet

    # PyTorch tensors (.pt)
    poetry run python -m src.pipelines.diagnostics.sample_grid_points \\
        --data-path /mnt/local_datasets/preprocessed_extended_subsampled_step_5 \\
        --format pt

    # S3 Parquet
    poetry run python -m src.pipelines.diagnostics.sample_grid_points \\
        --data-path s3://medwav-dev-data/parquet/hourly_extra_features/year=2022 \\
        --format parquet

    # Custom number of points and output directory
    poetry run python -m src.pipelines.diagnostics.sample_grid_points \\
        --data-path /mnt/blobstorage/parquet/hourly/year=2022 \\
        --format parquet \\
        --n-points 30 \\
        --output-path /mnt/blobstorage/diagnostics/sampled_grid_points \\
        --seed 42
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Region constants (mirrors train_edcdf_regional.py and CachedWaveDataset)
# ---------------------------------------------------------------------------
GIBRALTAR_LON: float = -5.5
BISCAY_LAT: float = 43.0
BISCAY_LON: float = 0.0

REGIONS = ["atlantic", "mediterranean"]


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _discover_files(data_path: str, fmt: str) -> List[str]:
    """Return sorted list of data files under *data_path*."""
    ext = ".parquet" if fmt == "parquet" else ".pt"
    is_s3 = data_path.startswith("s3://")

    if is_s3:
        import fsspec
        fs = fsspec.filesystem("s3")
        raw = data_path.replace("s3://", "")
        files = sorted(f"s3://{p}" for p in fs.glob(f"{raw}/**/*{ext}") + fs.glob(f"{raw}/*{ext}"))
    else:
        root = Path(data_path)
        files = sorted(str(p) for p in root.rglob(f"*{ext}"))

    if not files:
        raise FileNotFoundError(
            f"No {ext} files found under '{data_path}'. "
            "Check --data-path and --format."
        )
    return files


def _pick_one_file(files: List[str]) -> str:
    """Return the first file (files are sorted, so deterministic)."""
    return files[0]


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------

def _load_coords_parquet(
    path: str, sea_column: str = "VHM0"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique (lat, lon) for sea grid cells only.

    Sea is defined as finite *sea_column* (same convention as CachedWaveDataset:
    land cells use NaN for wave fields).
    """
    import polars as pl

    df = pl.read_parquet(path, columns=["latitude", "longitude", sea_column])
    df = df.filter(pl.col(sea_column).is_finite())
    if len(df) == 0:
        raise ValueError(
            f"No sea rows in {path!r} after filtering on finite {sea_column!r}."
        )
    df_grid = df.unique(subset=["latitude", "longitude"]).sort(["latitude", "longitude"])
    return (
        df_grid["latitude"].to_numpy(),
        df_grid["longitude"].to_numpy(),
    )


def _load_coords_pt(path: str, sea_column: str = "VHM0") -> Tuple[np.ndarray, np.ndarray]:
    """Return unique (lat, lon) for sea grid cells from a .pt tensor file.

    The .pt file is expected to be a dict with keys 'tensor' and 'feature_cols'.
    Tensor shape: (H, W, C) for hourly or (T, H, W, C) for daily.
    Sea pixels are those with finite *sea_column* (e.g. VHM0), matching
    ``sea_mask = (~torch.isnan(vhm0))`` in CachedWaveDataset.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    tensor: torch.Tensor = data["tensor"]
    feature_cols: List[str] = data["feature_cols"]

    if sea_column not in feature_cols:
        raise KeyError(
            f"{sea_column!r} not found in feature_cols (needed for sea mask): {feature_cols}"
        )
    if "latitude" not in feature_cols or "longitude" not in feature_cols:
        raise KeyError(
            f"'latitude' and/or 'longitude' not found in feature_cols: {feature_cols}"
        )

    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    sea_idx = feature_cols.index(sea_column)

    if tensor.ndim == 3:  # (H, W, C) – hourly
        lat_map = tensor[..., lat_idx].numpy()
        lon_map = tensor[..., lon_idx].numpy()
        vhm0_map = tensor[..., sea_idx].numpy()
    elif tensor.ndim == 4:  # (T, H, W, C) – daily; use first timestep
        lat_map = tensor[0, ..., lat_idx].numpy()
        lon_map = tensor[0, ..., lon_idx].numpy()
        vhm0_map = tensor[0, ..., sea_idx].numpy()
    else:
        raise ValueError(f"Unexpected tensor ndim={tensor.ndim}; expected 3 or 4.")

    # Sea = finite wave height (land is NaN in the model grid)
    valid = np.isfinite(vhm0_map)
    if not np.any(valid):
        raise ValueError(f"No sea pixels in {path!r} (no finite {sea_column}).")

    lats_grid = lat_map[valid].ravel()
    lons_grid = lon_map[valid].ravel()

    coords = np.unique(np.stack([lats_grid, lons_grid], axis=1), axis=0)
    return coords[:, 0], coords[:, 1]


def load_grid_coords(
    path: str, fmt: str, sea_column: str = "VHM0"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load unique sea (lat, lon) grid points from *path* according to *fmt*."""
    if fmt == "parquet":
        return _load_coords_parquet(path, sea_column=sea_column)
    elif fmt == "pt":
        return _load_coords_pt(path, sea_column=sea_column)
    else:
        raise ValueError(f"Unknown format '{fmt}'. Choose 'parquet' or 'pt'.")


# ---------------------------------------------------------------------------
# Region filtering
# ---------------------------------------------------------------------------

def apply_region_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    region: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the subset of (lats, lons) that belongs to *region*."""
    biscay = (lats > BISCAY_LAT) & (lons < BISCAY_LON)

    if region == "atlantic":
        mask = (lons < GIBRALTAR_LON) | biscay
    elif region == "mediterranean":
        mask = (lons >= GIBRALTAR_LON) & ~biscay
    else:
        raise ValueError(f"Unknown region '{region}'. Choose from {REGIONS}.")

    return lats[mask], lons[mask]


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def sample_points(
    lats: np.ndarray,
    lons: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Randomly sample *n* unique grid points without replacement."""
    total = len(lats)
    if total == 0:
        raise ValueError("No grid points available after region filtering.")
    if n > total:
        print(
            f"  Warning: requested {n} points but only {total} available; "
            "returning all."
        )
        n = total

    indices = rng.choice(total, size=n, replace=False)
    indices.sort()  # keep geographic ordering for readability
    return {"latitude": lats[indices], "longitude": lons[indices]}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_PATH = "/mnt/blobstorage/diagnostics/sampled_grid_points"

# Colours and markers for each region on the map
_REGION_STYLE: Dict[str, Dict] = {
    "atlantic": {"color": "#2196F3", "marker": "o", "label": "Atlantic"},
    "mediterranean": {"color": "#F44336", "marker": "s", "label": "Mediterranean"},
}


def run(
    data_path: str,
    fmt: str,
    n_points: int = 30,
    regions: Optional[List[str]] = None,
    output_path: str = DEFAULT_OUTPUT_PATH,
    seed: int = 42,
    sea_column: str = "VHM0",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load one file per region and sample *n_points* random grid points (sea only).

    Saves CSVs and a map PNG to *output_path*.
    Returns a dict mapping region name → {'latitude': array, 'longitude': array}.
    """
    if regions is None:
        regions = REGIONS

    rng = np.random.default_rng(seed)

    print(f"Discovering files in '{data_path}' (format={fmt})...")
    all_files = _discover_files(data_path, fmt)
    print(f"  Found {len(all_files)} file(s). Using one representative file.")

    source_file = _pick_one_file(all_files)
    print(f"  Source file: {source_file}")

    print(f"\nLoading sea grid coordinates (finite {sea_column})...")
    lats_all, lons_all = load_grid_coords(source_file, fmt, sea_column=sea_column)
    print(f"  Unique sea grid points in file: {len(lats_all):,}")

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for region in regions:
        print(f"\n--- Region: {region.upper()} ---")
        lats_r, lons_r = apply_region_mask(lats_all, lons_all, region)
        print(f"  Grid points after region filter: {len(lats_r):,}")

        sampled = sample_points(lats_r, lons_r, n=n_points, rng=rng)
        results[region] = sampled

        print(f"  Sampled {len(sampled['latitude'])} points:")
        for i, (lat, lon) in enumerate(
            zip(sampled["latitude"], sampled["longitude"]), start=1
        ):
            print(f"    {i:2d}. lat={lat:.4f}  lon={lon:.4f}")

    print(f"\nSaving outputs to '{output_path}'...")
    _save_csvs(results, output_path)
    _save_map(results, lats_all, lons_all, output_path)

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_csvs(
    results: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
) -> None:
    """Write one CSV per region plus a combined CSV under *output_path*."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for region, coords in results.items():
        dest = out / f"sampled_grid_points_{region}.csv"
        rows = [
            {"region": region, "latitude": float(lat), "longitude": float(lon)}
            for lat, lon in zip(coords["latitude"], coords["longitude"])
        ]
        _write_csv(rows, dest)
        print(f"  Saved {region} CSV  → {dest}")
        all_rows.extend(rows)

    combined_path = out / "sampled_grid_points_all.csv"
    _write_csv(all_rows, combined_path)
    print(f"  Saved combined CSV → {combined_path}")


def _write_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["region", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Map output
# ---------------------------------------------------------------------------

def _save_map(
    results: Dict[str, Dict[str, np.ndarray]],
    lats_all: np.ndarray,
    lons_all: np.ndarray,
    output_path: str,
) -> None:
    """Draw a cartopy map of the full grid (background) with sampled points overlaid."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    # Infer domain extent from full grid with a small padding
    pad = 1.0
    lon_min, lon_max = lons_all.min() - pad, lons_all.max() + pad
    lat_min, lat_max = lats_all.min() - pad, lats_all.max() + pad

    fig, ax = plt.subplots(
        figsize=(14, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.LAND, facecolor="#e8e0d0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#c8dcf0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#888888", zorder=1)
    ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )

    # Gibraltar boundary line (use plot, not axvline — matplotlib disallows
    # transform= on axvline; cartopy needs explicit PlateCarree transform)
    ax.plot(
        [GIBRALTAR_LON, GIBRALTAR_LON],
        [lat_min, lat_max],
        color="#FF9800",
        linewidth=1.2,
        linestyle="--",
        transform=ccrs.PlateCarree(),
        zorder=2,
        label=f"Gibraltar boundary ({GIBRALTAR_LON}°)",
    )

    # Plot sampled points per region
    for region, coords in results.items():
        style = _REGION_STYLE.get(region, {"color": "green", "marker": "^", "label": region})
        ax.scatter(
            coords["longitude"],
            coords["latitude"],
            color=style["color"],
            marker=style["marker"],
            s=60,
            linewidths=0.5,
            edgecolors="white",
            transform=ccrs.PlateCarree(),
            zorder=3,
            label=f"{style['label']} ({len(coords['latitude'])} pts)",
        )
        # Annotate each point with its index
        for idx, (lat, lon) in enumerate(
            zip(coords["latitude"], coords["longitude"]), start=1
        ):
            ax.text(
                lon + 0.15,
                lat + 0.15,
                str(idx),
                fontsize=5,
                color=style["color"],
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    ax.set_title("Sampled grid points per region", fontsize=13, pad=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    dest = Path(output_path) / "sampled_grid_points_map.png"
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved map PNG    → {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load one file per region and select N random grid points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Directory containing data files (local or s3://…). "
            "For Parquet: a year partition dir such as 'year=2022/' works too."
        ),
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["parquet", "pt"],
        default="parquet",
        help="File format to load.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=30,
        help="Number of random grid points to sample per region.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=REGIONS,
        choices=REGIONS,
        help="Regions to sample from.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Directory where CSVs and the map PNG are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sea-column",
        default="VHM0",
        help=(
            "Column/channel used as sea mask: finite values = sea, NaN = land "
            "(same as CachedWaveDataset)."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  Sample Grid Points per Region")
    print("=" * 60)
    print(f"  Data path   : {args.data_path}")
    print(f"  Format      : {args.fmt}")
    print(f"  Regions     : {args.regions}")
    print(f"  N points    : {args.n_points}")
    print(f"  Output path : {args.output_path}")
    print(f"  Sea column  : {args.sea_column}")
    print(f"  Seed        : {args.seed}")
    print("=" * 60)

    run(
        data_path=args.data_path,
        fmt=args.fmt,
        n_points=args.n_points,
        regions=args.regions,
        output_path=args.output_path,
        seed=args.seed,
        sea_column=args.sea_column,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
