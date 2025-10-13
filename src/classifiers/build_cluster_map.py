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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_cluster_rectangles(df: pl.DataFrame, lat_step=0.5, lon_step=0.5, plot_labels=False, plot_points=False):
    """
    Plot rectangular grid cells that actually appear in the dataset.
    Each cell is filled with a unique color and labeled with its cluster_id.
    """
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title("Rectangular Cluster Grid", fontsize=14, fontweight="bold")

    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")

    # Focus only on the Mediterranean / Atlantic region
    ax.set_extent([-25, 45, 25, 48], crs=ccrs.PlateCarree())

    # Deduplicate cells by cluster_id
    unique_cells = df.select(["lat", "lon", "cluster_id"]).unique(subset=["cluster_id"])

    cluster_ids = unique_cells["cluster_id"].to_numpy()
    cmap = cm.get_cmap("tab20", len(cluster_ids))  # categorical colormap

    for idx, row in enumerate(unique_cells.iter_rows(named=True)):
        lat = float(row["lat"])
        lon = float(row["lon"])
        cluster_id = int(row["cluster_id"])

        # Snap to bottom-left corner of the rectangle
        lat0 = np.floor(lat / lat_step) * lat_step
        lon0 = np.floor(lon / lon_step) * lon_step

        # Pick color
        color = cmap(idx % cmap.N)

        # Draw rectangle
        rect = Rectangle((lon0, lat0), lon_step, lat_step,
                         linewidth=0.3, edgecolor="k", facecolor=color,
                         alpha=0.5, transform=ccrs.PlateCarree())
        ax.add_patch(rect)

        # add label
        if plot_labels:
            ax.text(lon0 + lon_step/2, lat0 + lat_step/2,
                    str(cluster_id), ha="center", va="center", fontsize=5,
                    transform=ccrs.PlateCarree())

        # Extract actual points belonging to this cluster
        if plot_points:
            cluster_points = df.filter(df["cluster_id"] == cluster_id)
            ax.scatter(cluster_points["lon"], cluster_points["lat"],
                    color="black", s=2, transform=ccrs.PlateCarree(), zorder=3)

    plt.tight_layout()
    plt.savefig("rectangular_clusters.png", dpi=300)

class GridClusterMapper:
    def __init__(self, lat_step=1.0, lon_step=1.0):

        self.lat_step = lat_step
        self.lon_step = lon_step
        self.n_lat_bins = int(180 / self.lat_step)
        self.n_lon_bins = int(360 / self.lon_step)
    
    def get_cluster_id(self, lat, lon):
        """
        Assign a cluster ID based on lat/lon grid cell using polars expressions.
        """
        lat_idx = ((lat + 90.0) / self.lat_step).floor().cast(pl.Int32)
        lon_idx = ((lon + 180.0) / self.lon_step).floor().cast(pl.Int32)
        cluster_id = lat_idx * self.n_lon_bins + lon_idx
        return cluster_id

    def build_map(self, df: pl.DataFrame, lat_col="lat", lon_col="lon"):
        """
        Build cluster map for all rows in dataframe.
        Adds both raw cluster_id and a normalized cluster_plot_id for visualization.
        """
        cluster_ids = self.get_cluster_id(df[lat_col], df[lon_col])
        norm_ids = self._normalize_cluster_ids(cluster_ids)
        df_out = df.with_columns([
            pl.Series("cluster_id", cluster_ids),
            pl.Series("cluster_plot_id", norm_ids)
        ])
        # Rename columns and select only the required ones
        df_out = df_out.rename({lat_col: "lat", lon_col: "lon"})
        return df_out.select(["lat", "lon", "cluster_id", "cluster_plot_id"])

    def _normalize_cluster_ids(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Map raw cluster IDs to a compact 0..N-1 index for plotting.
        """
        unique, inv = np.unique(cluster_ids, return_inverse=True)
        return inv


def build_cluster_map(input_file: Path, output_file: Path,
                      lat_step=1.0, lon_step=1.0):
    print(f"📂 Loading {input_file} ...")
    df = pl.read_parquet(input_file)
    df = df.drop_nulls()
    print(f"✅ After dropping NAs: {df.height:,} rows")

    # Extract unique points
    unique_points = df.select(["latitude", "longitude"]).unique()
    print(f"✅ Found {unique_points.height:,} unique points")

    print(f"✅ Detected grid resolution: {lat_step}° × {lon_step}°")

    # Assign clusters
    mapper = GridClusterMapper(lat_step=lat_step, lon_step=lon_step)
    df_clusters = mapper.build_map(
        df, lat_col="latitude", lon_col="longitude"
    )

    # Save mapping
    df_clusters.write_parquet(output_file)
    print(f"💾 Saved cluster map to {output_file}")
    print(df_clusters.select("lat", "lon", "cluster_id").head(5))
    print("Unique clusters:", len(df_clusters["cluster_plot_id"].unique().to_numpy()))
    print("Unique clusters (raw):", len(df_clusters))
    plot_cluster_rectangles(df_clusters, lat_step=lat_step, lon_step=lon_step, plot_points=False, plot_labels=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lat/Lon → Cluster map")
    parser.add_argument("--input", type=Path, default=Path("~/Documents/projects/hcmr/data/hourly/WAVEAN20210101.parquet"),
                        help="Path to input Parquet file")
    parser.add_argument("--output", type=Path, default=Path("latlon_to_cluster.parquet"),
                        help="Path to output mapping file")
    parser.add_argument("--lat_step", type=float, default=0.5,
                        help="Latitude bin size (deg)")
    parser.add_argument("--lon_step", type=float, default=0.5,
                        help="Longitude bin size (deg)")
    args = parser.parse_args()

    build_cluster_map(args.input, args.output, lat_step=args.lat_step, lon_step=args.lon_step)
