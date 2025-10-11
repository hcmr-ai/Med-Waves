"""
Build Lat/Lon → Cluster Map
---------------------------
Loads one Parquet file with grid points, assigns each to a cluster,
and saves a mapping file (lat, lon, cluster_id).
"""

import polars as pl
import numpy as np
import argparse
from pathlib import Path


class GridClusterMapper:
    def __init__(self, lat_step=1.0, lon_step=1.0,
                 lat_min=25.0, lat_max=60.0,
                 lon_min=-25.0, lon_max=40.0):
        self.lat_step = lat_step
        self.lon_step = lon_step
        self.lat_min = lat_min
        self.lon_min = lon_min

        # Number of bins
        self.n_lat_bins = int(np.ceil((lat_max - lat_min) / lat_step))
        self.n_lon_bins = int(np.ceil((lon_max - lon_min) / lon_step))

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_idx = np.floor((lat - self.lat_min) / self.lat_step).astype(int)
        lon_idx = np.floor((lon - self.lon_min) / self.lon_step).astype(int)
        return lat_idx * self.n_lon_bins + lon_idx


def build_cluster_map(input_file: Path, output_file: Path,
                      lat_step=1.0, lon_step=1.0):
    print(f"📂 Loading {input_file} ...")
    df = pl.read_parquet(input_file)

    # Extract unique points
    unique_points = df.select(["lat", "lon"]).unique()
    print(f"✅ Found {unique_points.height:,} unique points")

    # Assign clusters
    mapper = GridClusterMapper(lat_step=lat_step, lon_step=lon_step)
    cluster_ids = mapper.assign(
        unique_points["lat"].to_numpy(),
        unique_points["lon"].to_numpy()
    )

    # Add cluster column
    mapped_df = unique_points.with_columns(
        pl.Series("cluster_id", cluster_ids)
    )

    # Save mapping
    mapped_df.write_parquet(output_file)
    print(f"💾 Saved cluster map to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lat/Lon → Cluster map")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to input Parquet file")
    parser.add_argument("--output", type=Path, default=Path("latlon_to_cluster.parquet"),
                        help="Path to output mapping file")
    parser.add_argument("--lat_step", type=float, default=1.0,
                        help="Latitude bin size (deg)")
    parser.add_argument("--lon_step", type=float, default=1.0,
                        help="Longitude bin size (deg)")
    args = parser.parse_args()

    build_cluster_map(args.input, args.output,
                      lat_step=args.lat_step,
                      lon_step=args.lon_step)
