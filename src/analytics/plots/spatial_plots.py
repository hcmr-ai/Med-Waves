import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def plot_spatial_feature_map(
    df_pd: pd.DataFrame,
    feature_col: str,
    save_path: str,
    title: str,
    colorbar_label: str,
    s: int = 8,
    alpha: float = 0.85,
):
    """
    Plot a spatial map of feature values across grid positions.

    Parameters
    ----------
    df_pd : pandas.DataFrame
        DataFrame with 'latitude', 'longitude', and feature_col columns.
    feature_col : str, default="VHM0"
        Column to use for coloring the points.
    save_path : str, default="outputs/eda/map.png"
        File path to save the resulting plot.
    title : str, optional
        Plot title.
    colorbar_label : str, optional
        Label for the colorbar.
    s : int or float, optional
        Marker size.
    alpha : float, optional
        Marker transparency.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title)

    sc = ax.scatter(
        df_pd["longitude"],
        df_pd["latitude"],
        c=df_pd[feature_col],
        cmap="viridis",
        s=s,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(sc, ax=ax, orientation="vertical", label=colorbar_label)
    ax.set_extent([
        df_pd["longitude"].min(),
        df_pd["longitude"].max(),
        df_pd["latitude"].min(),
        df_pd["latitude"].max()
    ], crs=ccrs.PlateCarree())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_spatial_feature_heatmap(
    df: pd.DataFrame,
    feature_col: str,
    output_dir: str,
    stat_name: str = "mean",
    cmap: str = "viridis",
    label: str = ""
) -> None:
    """
    Plot and save a spatial heatmap of a given statistic for a grid feature.

    Args:
        df (pd.DataFrame): DataFrame with 'longitude', 'latitude', and the statistic column (e.g. 'VHM0_mean').
        feature_col (str): The name of the statistic column to plot (e.g. 'VHM0_mean').
        output_dir (Path): Directory to save the output plot.
        stat_name (str, optional): Name of the statistic for labeling. Defaults to "mean".
        cmap (str, optional): Colormap for scatter plot. Defaults to "viridis".
        label (str, optional): Extra label to append to filename and title. Defaults to "".

    Returns:
        None. Saves the plot as a PNG file.
    """
    plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    c = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df[feature_col],
        cmap="viridis",
        s=10,
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )
    plt.colorbar(c, ax=ax, orientation="vertical", label=f"{stat_name.capitalize()} {feature_col} (m)")
    title = f"{stat_name.capitalize()} {feature_col} per grid cell"
    if label:
        title += f" ({label})"
    ax.set_title(title)

    plt.savefig(output_dir, bbox_inches="tight")
    plt.close()


def plot_missing_spatial_heatmap(df: pd.DataFrame, output_dir: str, label="") -> None:
    """
    Plot and save a spatial heatmap showing the percentage of missing values for each grid cell.

    Args:
        missing_grid (pd.DataFrame): Pivoted DataFrame where index is latitude, columns are longitude, values are % missing.
        output_dir (Path): Directory where the heatmap image will be saved.
        label (str, optional): Extra label for filename/title (e.g. year or season). Defaults to "".

    Returns:
        None. Saves the plot as a PNG file in the specified directory.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(df, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="% Missing")
    title = "Spatial Distribution of Missing Data (%)"
    if label:
        title += f" ({label})"
    plt.title(title)
    plt.xlabel("Grid longitude index")
    plt.ylabel("Grid latitude index")
    plt.savefig(output_dir)
    plt.close()
