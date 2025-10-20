"""
Build Lat/Lon → Cluster Map
---------------------------
Loads one Parquet file with grid points, assigns each to a cluster,
and saves a mapping file (lat, lon, cluster_id).
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle


def plot_cluster_rectangles(
    df: pl.DataFrame, lat_step=0.5, lon_step=0.5, plot_labels=False, plot_points=False
):
    """
    Plot rectangular grid cells that actually appear in the dataset.
    Each cell is filled with a unique color and labeled with its cluster_id.
    """
    _ = plt.figure(figsize=(14, 8))
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
        rect = Rectangle(
            (lon0, lat0),
            lon_step,
            lat_step,
            linewidth=0.3,
            edgecolor="k",
            facecolor=color,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)

        # add label
        if plot_labels:
            ax.text(
                lon0 + lon_step / 2,
                lat0 + lat_step / 2,
                str(cluster_id),
                ha="center",
                va="center",
                fontsize=5,
                transform=ccrs.PlateCarree(),
            )

        # Extract actual points belonging to this cluster
        if plot_points:
            cluster_points = df.filter(df["cluster_id"] == cluster_id)
            ax.scatter(
                cluster_points["lon"],
                cluster_points["lat"],
                color="black",
                s=2,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

    plt.tight_layout()
    plt.savefig("rectangular_clusters.png", dpi=300)

class PointClusterMapper:
    """Point-based clustering: One cluster per unique point."""
    
    def __init__(self, precision: int = 4):
        """
        Args:
            precision: Decimal places for lat/lon rounding
                      2 = ~1km, 3 = ~100m, 4 = ~10m, 6 = ~0.1m
        """
        self.precision = precision
        self.point_to_cluster = {}
        self.cluster_to_point = {}
        self.next_cluster_id = 0
    
    def _round_coords(self, lat: float, lon: float) -> tuple:
        """Round coordinates to specified precision."""
        return (round(lat, self.precision), round(lon, self.precision))
    
    def get_cluster_id_old(self, lat_series: pl.Series, lon_series: pl.Series) -> pl.Series:
        """
        Get cluster IDs for series of coordinates.
        
        Args:
            lat_series: Polars series of latitudes
            lon_series: Polars series of longitudes
            
        Returns:
            Polars series of cluster IDs
        """
        # Convert to numpy for processing
        lats = lat_series.to_numpy()
        lons = lon_series.to_numpy()
        
        # Assign cluster IDs
        cluster_ids = np.zeros(len(lats), dtype=np.int32)
        
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            point = self._round_coords(lat, lon)
            
            if point not in self.point_to_cluster:
                self.point_to_cluster[point] = self.next_cluster_id
                self.cluster_to_point[self.next_cluster_id] = point
                self.next_cluster_id += 1
            
            cluster_ids[i] = self.point_to_cluster[point]
        
        return pl.Series(cluster_ids)
    
    def get_cluster_id_np(self, lat_series: pl.Series, lon_series: pl.Series) -> pl.Series:
        """
        Assign cluster IDs for rounded (lat, lon) pairs using vectorized NumPy.
        """
        # Convert to numpy
        lats = lat_series.to_numpy()
        lons = lon_series.to_numpy()

        # Vectorized rounding
        rounded_lats = np.round(lats, self.precision)
        rounded_lons = np.round(lons, self.precision)

        # Stack into pairs
        coords = np.column_stack((rounded_lats, rounded_lons))

        # Find unique pairs + inverse indices
        unique_points, cluster_ids = np.unique(coords, axis=0, return_inverse=True)

        # Update mappings if needed
        self.point_to_cluster = {tuple(pt): i for i, pt in enumerate(unique_points)}
        self.cluster_to_point = {i: tuple(pt) for i, pt in enumerate(unique_points)}
        self.next_cluster_id = len(unique_points)

        return pl.Series(cluster_ids.astype(np.int32))

    def get_cluster_id(self, lat_series: pl.Series, lon_series: pl.Series) -> pl.Series:
        df = pl.DataFrame({
            "lat": lat_series.round(self.precision),
            "lon": lon_series.round(self.precision),
        })
        # factorize gives unique IDs for each unique pair
        keys = (df["lat"].cast(str) + "_" + df["lon"].cast(str))

        # Factorize keys → cluster IDs
        cluster_ids, _ = keys.factorize()
        return cluster_ids.cast(pl.Int32)

    def get_n_clusters(self) -> int:
        """Get total number of clusters created."""
        return self.next_cluster_id

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
        df_out = df.with_columns(
            [
                pl.Series("cluster_id", cluster_ids),
                pl.Series("cluster_plot_id", norm_ids),
            ]
        )
        # Rename columns and select only the required ones
        df_out = df_out.rename({lat_col: "lat", lon_col: "lon"})
        return df_out.select(["lat", "lon", "cluster_id", "cluster_plot_id"])

    def _normalize_cluster_ids(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Map raw cluster IDs to a compact 0..N-1 index for plotting.
        """
        unique, inv = np.unique(cluster_ids, return_inverse=True)
        return inv


def build_cluster_map(input_file: Path, output_file: Path, lat_step=1.0, lon_step=1.0):
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
    df_clusters = mapper.build_map(df, lat_col="latitude", lon_col="longitude")

    # Save mapping
    df_clusters.write_parquet(output_file)
    print(f"💾 Saved cluster map to {output_file}")
    print(df_clusters.select("lat", "lon", "cluster_id").head(5))
    print("Unique clusters:", len(df_clusters["cluster_plot_id"].unique().to_numpy()))
    print("Unique clusters (raw):", len(df_clusters))
    plot_cluster_rectangles(
        df_clusters,
        lat_step=lat_step,
        lon_step=lon_step,
        plot_points=False,
        plot_labels=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Lat/Lon → Cluster map")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("~/Documents/projects/hcmr/data/hourly/WAVEAN20210101.parquet"),
        help="Path to input Parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latlon_to_cluster.parquet"),
        help="Path to output mapping file",
    )
    parser.add_argument(
        "--lat_step", type=float, default=0.5, help="Latitude bin size (deg)"
    )
    parser.add_argument(
        "--lon_step", type=float, default=0.5, help="Longitude bin size (deg)"
    )
    args = parser.parse_args()

    build_cluster_map(
        args.input, args.output, lat_step=args.lat_step, lon_step=args.lon_step
    )
