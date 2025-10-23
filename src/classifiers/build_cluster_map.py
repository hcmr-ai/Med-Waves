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
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def plot_cluster_distribution_suite(df: pl.DataFrame, output_dir: str = "."):
    """
    Create a comprehensive suite of distribution plots.

    Args:
        df: DataFrame with 'cluster_id' column
        output_dir: Directory to save plots
    """
    # Calculate distribution
    cluster_counts = (
        df.group_by("cluster_id")
        .agg(pl.count().alias("point_count"))
        .sort("point_count", descending=True)
    )

    counts = cluster_counts["point_count"].to_numpy()

    print("📊 Cluster Distribution Statistics:")
    print(f"   Total clusters: {len(counts):,}")
    print(f"   Total points: {counts.sum():,}")
    print(f"   Mean points/cluster: {counts.mean():.0f}")
    print(f"   Median points/cluster: {np.median(counts):.0f}")
    print(f"   Min points: {counts.min():,}")
    print(f"   Max points: {counts.max():,}")
    print(f"   Std dev: {counts.std():.0f}")

    # Create all plots
    _ = plt.figure(figsize=(20, 12))

    # 1. Histogram with log scale
    ax1 = plt.subplot(2, 3, 1)
    plot_histogram(counts, ax1)

    # 2. CDF (Cumulative Distribution)
    ax2 = plt.subplot(2, 3, 2)
    plot_cdf(counts, ax2)

    # 3. Box plot with outliers
    ax3 = plt.subplot(2, 3, 3)
    plot_boxplot(counts, ax3)

    # 4. Top N clusters bar chart
    ax4 = plt.subplot(2, 3, 4)
    plot_top_clusters(cluster_counts, ax4, top_n=20)

    # 5. Distribution by percentiles
    ax5 = plt.subplot(2, 3, 5)
    plot_percentile_distribution(counts, ax5)

    # 6. Violin plot
    ax6 = plt.subplot(2, 3, 6)
    plot_violin(counts, ax6)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_distribution_suite.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved distribution suite to {output_dir}/cluster_distribution_suite.png")

    # Create separate spatial heatmap
    plot_spatial_heatmap(df, output_dir)


def plot_histogram(counts, ax):
    """Histogram with log scale."""
    ax.hist(counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Number of Points per Cluster', fontsize=12)
    ax.set_ylabel('Number of Clusters (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Distribution of Points per Cluster', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')

    # Add mean and median lines
    mean_val = counts.mean()
    median_val = np.median(counts)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.0f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
    ax.legend()


def plot_cdf(counts, ax):
    """Cumulative Distribution Function."""
    sorted_counts = np.sort(counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100

    ax.plot(sorted_counts, cumulative, linewidth=2, color='darkblue')
    ax.set_xlabel('Number of Points per Cluster', fontsize=12)
    ax.set_ylabel('Cumulative % of Clusters', fontsize=12)
    ax.set_title('Cumulative Distribution (CDF)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xscale('log')

    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(counts, p)
        ax.axvline(val, color='red', linestyle=':', alpha=0.5)
        ax.text(val, 5, f'P{p}\n{val:.0f}', ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_boxplot(counts, ax):
    """Box plot showing quartiles and outliers."""
    _ = ax.boxplot(counts, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     showmeans=True, meanline=True,
                     meanprops=dict(color='green', linestyle='--', linewidth=2))

    ax.set_ylabel('Number of Points per Cluster', fontsize=12)
    ax.set_title('Box Plot with Outliers', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', axis='y')

    # Add statistics text
    q1, median, q3 = np.percentile(counts, [25, 50, 75])
    iqr = q3 - q1
    stats_text = f"Q1: {q1:.0f}\nMedian: {median:.0f}\nQ3: {q3:.0f}\nIQR: {iqr:.0f}"
    ax.text(1.15, median, stats_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_top_clusters(cluster_counts, ax, top_n=20):
    """Bar chart of top N clusters."""
    top_clusters = cluster_counts.head(top_n)

    y_pos = np.arange(len(top_clusters))
    counts = top_clusters["point_count"].to_numpy()
    cluster_ids = top_clusters["cluster_id"].to_numpy()

    bars = ax.barh(y_pos, counts, color='coral', edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Cluster {cid}" for cid in cluster_ids], fontsize=8)
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_title(f'Top {top_n} Clusters by Point Count', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='x')

    # Add value labels on bars
    for i, (_, count) in enumerate(zip(bars, counts, strict=False)):
        ax.text(count + counts.max()*0.01, i, f'{count:,}',
                va='center', fontsize=8)


def plot_percentile_distribution(counts, ax):
    """Show distribution by percentile bins."""
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(counts, percentiles)

    # Create bins based on percentiles
    bin_counts = []
    bin_labels = []
    for i in range(len(percentiles) - 1):
        mask = (counts >= percentile_values[i]) & (counts < percentile_values[i+1])
        bin_counts.append(mask.sum())
        bin_labels.append(f"P{percentiles[i]}-{percentiles[i+1]}\n({percentile_values[i]:.0f}-{percentile_values[i+1]:.0f})")

    x_pos = np.arange(len(bin_labels))
    bars = ax.bar(x_pos, bin_counts, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Number of Clusters', fontsize=12)
    ax.set_title('Clusters by Percentile Bins', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='y')

    # Add value labels
    for bar, count in zip(bars, bin_counts, strict=False):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=9)


def plot_violin(counts, ax):
    """Violin plot showing distribution density."""
    parts = ax.violinplot([counts], vert=True, showmeans=True, showmedians=True)

    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)

    ax.set_ylabel('Number of Points per Cluster', fontsize=12)
    ax.set_title('Distribution Density (Violin Plot)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    ax.set_xticks([])

    # Add quartile lines
    quartiles = np.percentile(counts, [25, 50, 75])
    for q, label in zip(quartiles, ['Q1', 'Q2', 'Q3'], strict=False):
        ax.axhline(q, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(1.1, q, f'{label}: {q:.0f}', fontsize=9)


def plot_spatial_heatmap(df: pl.DataFrame, output_dir: str = "."):
    """
    Spatial heatmap showing point density per cluster on a map.
    This is the BEST visualization for understanding geographic distribution!
    """
    # Calculate points per cluster
    cluster_stats = (
        df.group_by("cluster_id")
        .agg([
            pl.count().alias("point_count"),
            pl.col("lat").first().alias("lat"),
            pl.col("lon").first().alias("lon")
        ])
    )

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3)

    # Focus on your region
    ax.set_extent([-25, 45, 25, 48], crs=ccrs.PlateCarree())

    # Extract data
    lats = cluster_stats["lat"].to_numpy()
    lons = cluster_stats["lon"].to_numpy()
    counts = cluster_stats["point_count"].to_numpy()

    # Create scatter plot with point size and color based on count
    scatter = ax.scatter(
        lons, lats,
        c=counts,
        s=counts / counts.max() * 500,  # Scale point size
        cmap='YlOrRd',
        norm=LogNorm(vmin=counts.min(), vmax=counts.max()),
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8, label='Number of Points per Cluster')
    cbar.ax.tick_params(labelsize=10)

    # Add title with statistics
    mean_count = counts.mean()
    median_count = np.median(counts)
    ax.set_title(
        f'Spatial Distribution of Points per Cluster\n'
        f'Mean: {mean_count:.0f} | Median: {median_count:.0f} | '
        f'Range: {counts.min():,} - {counts.max():,}',
        fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_cluster_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved spatial heatmap to {output_dir}/spatial_cluster_heatmap.png")

    return fig


def plot_imbalance_analysis(df: pl.DataFrame, output_dir: str = "."):
    """
    Analyze and visualize data imbalance across clusters.
    Shows which clusters are over/under-represented.
    """
    cluster_counts = (
        df.group_by("cluster_id")
        .agg(pl.count().alias("point_count"))
        .sort("cluster_id")
    )

    counts = cluster_counts["point_count"].to_numpy()
    expected = counts.mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Deviation from mean
    ax1 = axes[0, 0]
    deviation = (counts - expected) / expected * 100
    colors = ['red' if d < -50 else 'orange' if d < -20 else 'green' if d > 20 else 'yellow' if d > 50 else 'gray'
              for d in deviation]
    ax1.bar(range(len(deviation)), deviation, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=1)
    ax1.axhline(-20, color='orange', linestyle='--', alpha=0.5, label='-20%')
    ax1.axhline(20, color='green', linestyle='--', alpha=0.5, label='+20%')
    ax1.set_xlabel('Cluster Index', fontsize=12)
    ax1.set_ylabel('Deviation from Mean (%)', fontsize=12)
    ax1.set_title('Cluster Imbalance Analysis', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    # 2. Lorenz curve (inequality)
    ax2 = axes[0, 1]
    sorted_counts = np.sort(counts)
    cumsum = np.cumsum(sorted_counts)
    lorenz = cumsum / cumsum[-1]
    ax2.plot([0, len(counts)], [0, 1], 'k--', label='Perfect equality', linewidth=2)
    ax2.plot(range(len(lorenz)), lorenz, 'b-', label='Actual distribution', linewidth=2)
    ax2.fill_between(range(len(lorenz)), lorenz, np.linspace(0, 1, len(lorenz)), alpha=0.3)
    ax2.set_xlabel('Cumulative % of Clusters', fontsize=12)
    ax2.set_ylabel('Cumulative % of Points', fontsize=12)
    ax2.set_title('Lorenz Curve (Data Inequality)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Calculate Gini coefficient
    gini = 2 * (0.5 - np.trapz(lorenz, dx=1/len(lorenz)))
    ax2.text(0.05, 0.95, f'Gini Coefficient: {gini:.3f}\n(0=perfect equality, 1=perfect inequality)',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 3. Sample size categories
    ax3 = axes[1, 0]
    categories = [
        ('Tiny (<100)', counts < 100),
        ('Small (100-500)', (counts >= 100) & (counts < 500)),
        ('Medium (500-1k)', (counts >= 500) & (counts < 1000)),
        ('Large (1k-5k)', (counts >= 1000) & (counts < 5000)),
        ('Very Large (5k+)', counts >= 5000)
    ]

    cat_counts = [mask.sum() for _, mask in categories]
    cat_labels = [f"{label}\n({count} clusters)" for (label, _), count in zip(categories, cat_counts, strict=False)]

    ax3.pie(cat_counts, labels=cat_labels, autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette('Set3', len(categories)))
    ax3.set_title('Clusters by Sample Size Category', fontsize=14, fontweight='bold')

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    stats_data = [
        ['Metric', 'Value'],
        ['Total Clusters', f'{len(counts):,}'],
        ['Total Points', f'{counts.sum():,}'],
        ['Mean', f'{counts.mean():.0f}'],
        ['Median', f'{np.median(counts):.0f}'],
        ['Std Dev', f'{counts.std():.0f}'],
        ['CV (Coef. of Variation)', f'{counts.std()/counts.mean():.2f}'],
        ['Min', f'{counts.min():,}'],
        ['Max', f'{counts.max():,}'],
        ['Range', f'{counts.max() - counts.min():,}'],
        ['Q1 (25%)', f'{np.percentile(counts, 25):.0f}'],
        ['Q3 (75%)', f'{np.percentile(counts, 75):.0f}'],
        ['IQR', f'{np.percentile(counts, 75) - np.percentile(counts, 25):.0f}'],
        ['Gini Coefficient', f'{gini:.3f}'],
        ['Clusters < Mean', f'{(counts < counts.mean()).sum()} ({(counts < counts.mean()).sum()/len(counts)*100:.1f}%)'],
        ['Clusters > 2×Mean', f'{(counts > 2*counts.mean()).sum()} ({(counts > 2*counts.mean()).sum()/len(counts)*100:.1f}%)'],
    ]

    table = ax4.table(cellText=stats_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_imbalance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"💾 Saved imbalance analysis to {output_dir}/cluster_imbalance_analysis.png")


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
    plt.savefig(f"rectangular_clusters_{lat}_{lon}.png", dpi=300)


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
    # ✅ NEW: Create comprehensive visualizations
    print("\n📊 Generating distribution visualizations...")
    plot_cluster_distribution_suite(df_clusters, output_dir=".")
    plot_imbalance_analysis(df_clusters, output_dir=".")

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
