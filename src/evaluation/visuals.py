import logging

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_residual_distribution(y_true, y_pred, save_path=None):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, bins=100, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Frequency")

    if save_path:
        plt.savefig(save_path)
    return fig


def plot_spatial_error_map(df: pl.DataFrame, lat_col="lat", lon_col="lon", error_col="residual", save_path=None):
    """
    Plot spatial heatmap of residuals (requires 'lat', 'lon', and 'residual' columns).
    """
    df_pandas = df.select([lat_col, lon_col, error_col]).to_pandas()
    pivot_table = df_pandas.pivot_table(index=lat_col, columns=lon_col, values=error_col, aggfunc=np.mean)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap="coolwarm", center=0)
    plt.title("Spatial Error Map (mean residuals)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compute_spatial_metrics(df: pl.DataFrame, lat_col="lat", lon_col="lon", y_col="true", y_hat_col="pred"):
    """
    Returns a Polars DataFrame of spatial MAE, Bias, and Pearson R per (lat, lon).
    """
    df = df.with_columns([
        (pl.col(y_hat_col) - pl.col(y_col)).alias("residual"),
        pl.col(y_hat_col).alias("pred"),
        pl.col(y_col).alias("true")
    ])

    grouped = df.group_by([lat_col, lon_col]).agg([
        (pl.col("residual").mean()).alias("bias"),
        (pl.col("residual").abs().mean()).alias("mae"),
        (pl.pearson_corr("true", "pred")).alias("pearson")
    ])
    return grouped


# ============================================================================
# Evaluation Plotting Functions
# ============================================================================


def load_coordinates_from_parquet(file_path, subsample_step=None, return_timestamps=False):
    """Load latitude, longitude coordinates, and optionally timestamps from a parquet file.

    Args:
        file_path: Path to parquet file (can be S3 path with or without s3:// prefix)
        subsample_step: Subsampling step size
        return_timestamps: If True, also return timestamps array

    Returns:
        If return_timestamps is False:
            lat_grid: 2D array of latitudes (H, W)
            lon_grid: 2D array of longitudes (H, W)
        If return_timestamps is True:
            lat_grid: 2D array of latitudes (H, W)
            lon_grid: 2D array of longitudes (H, W)
            timestamps: 1D array of timestamps or 2D array (H, W) if available per pixel
    """
    import pyarrow.parquet as pq
    import s3fs

    # Check if it's an S3 path (with or without s3:// prefix)
    is_s3 = file_path.startswith("s3://") or not file_path.startswith("/")

    if is_s3:
        # Ensure s3:// prefix for s3fs
        if not file_path.startswith("s3://"):
            file_path = f"s3://{file_path}"

        # Use s3fs to open the file
        fs = s3fs.S3FileSystem()
        with fs.open(file_path, "rb") as f:
            table = pq.read_table(f)
    else:
        # Local file
        table = pq.read_table(file_path)

    # Extract coordinate columns
    lat_data = table.column("latitude").to_numpy()
    lon_data = table.column("longitude").to_numpy()

    # Extract timestamps if requested
    timestamps = None
    if return_timestamps:
        # Try to find timestamp column (check common column names)
        timestamp_columns = ["time", "timestamp", "datetime", "date"]
        for col_name in timestamp_columns:
            if col_name in table.column_names:
                timestamps = table.column(col_name).to_numpy()
                # Convert to datetime64 if not already
                if not np.issubdtype(timestamps.dtype, np.datetime64):
                    try:
                        timestamps = timestamps.astype('datetime64[ns]')
                    except (ValueError, TypeError):
                        pass
                break

        if timestamps is None:
            import logging
            logging.warning(f"No timestamp column found in {file_path}. Tried: {timestamp_columns}")

    # Get unique sorted coordinates
    unique_lats = np.unique(lat_data)
    unique_lons = np.unique(lon_data)
    if subsample_step is not None and subsample_step > 1:
        unique_lats = unique_lats[::subsample_step]
        unique_lons = unique_lons[::subsample_step]

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)

    if return_timestamps:
        return lat_grid, lon_grid, timestamps
    else:
        return lat_grid, lon_grid


def plot_spatial_rmse_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    rmse_data: np.ndarray,
    save_path: str,
    title: str,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "YlOrRd",
    geo_bounds: dict = None,
    unit: str = "m",
    norm=None,
):
    """
    Plot a spatial RMSE heatmap with coastlines and proper projection.

    Parameters
    ----------
    lat_grid : np.ndarray (H, W)
        2D array of latitudes
    lon_grid : np.ndarray (H, W)
        2D array of longitudes
    rmse_data : np.ndarray (H, W)
        2D array of RMSE values
    save_path : str
        File path to save the plot
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    cmap : str
        Colormap name
    geo_bounds : dict, optional
        Geographic bounds for filtering: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        If provided, zooms into this region and adds a rectangle
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines and geographic features
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        alpha=0.5,
    )

    # Plot the RMSE heatmap
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        rmse_data,
        cmap=cmap,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Add colorbar
    _ = plt.colorbar(
        im, ax=ax, orientation="vertical", label=f"RMSE ({unit})", pad=0.05, shrink=0.8
    )

    # Set extent based on geo_bounds or data bounds
    if geo_bounds is not None:
        # Zoom into the filtered region with a small margin
        margin_lat = (geo_bounds["lat_max"] - geo_bounds["lat_min"]) * 0.1
        margin_lon = (geo_bounds["lon_max"] - geo_bounds["lon_min"]) * 0.1
        ax.set_extent(
            [
                geo_bounds["lon_min"] - margin_lon,
                geo_bounds["lon_max"] + margin_lon,
                geo_bounds["lat_min"] - margin_lat,
                geo_bounds["lat_max"] + margin_lat,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Add rectangle showing the filtered region
        rect = mpatches.Rectangle(
            (geo_bounds["lon_min"], geo_bounds["lat_min"]),
            geo_bounds["lon_max"] - geo_bounds["lon_min"],
            geo_bounds["lat_max"] - geo_bounds["lat_min"],
            fill=False,
            edgecolor="red",
            linewidth=2.5,
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10,
            label="Filtered Region",
        )
        ax.add_patch(rect)

        # Update title to indicate filtering
        title = (
            f"{title}\n(Filtered: {geo_bounds['lat_min']:.1f}°-{geo_bounds['lat_max']:.1f}°N, "
            f"{geo_bounds['lon_min']:.1f}°-{geo_bounds['lon_max']:.1f}°E)"
        )
    else:
        # Use data bounds
        ax.set_extent(
            [
                np.nanmin(lon_grid),
                np.nanmax(lon_grid),
                np.nanmin(lat_grid),
                np.nanmax(lat_grid),
            ],
            crs=ccrs.PlateCarree(),
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {save_path}")
