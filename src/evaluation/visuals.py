import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


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
