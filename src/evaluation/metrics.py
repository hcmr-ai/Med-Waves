import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(y_pred, y):
    """
    Evaluate the model on a given dataset using standard metrics.
    """
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(np.mean(y_pred) - y)
    diff = np.mean(y_pred - y)
    pearson = pearsonr(y, y_pred)
    
    # Calculate variance metrics
    var_true = np.var(y)
    var_pred = np.var(y_pred)
    
    # Calculate Signal-to-Noise Ratio (SNR)
    # SNR = signal_power / noise_power
    # Signal: variance of actual values
    # Noise: variance of residuals (prediction errors)
    residuals = y - y_pred
    signal_power = var_true
    noise_power = np.var(residuals)
    
    # Avoid division by zero
    if noise_power > 0:
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr)  # Convert to dB
    else:
        snr = float('inf')
        snr_db = float('inf')

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "diff": diff,
        "pearson": pearson[0],
        "var_true": var_true,
        "var_pred": var_pred,
        "snr": snr,
        "snr_db": snr_db,
    }


def evaluate_model_spatial(spatial_df):
    """
    Evaluate spatial metrics using Polars for high performance on large datasets.
    Supports both Polars and pandas DataFrames.
    """
    # Check if input is Polars DataFrame
    is_polars = isinstance(spatial_df, pl.DataFrame)
    
    if is_polars:
        # Polars implementation - much faster for large datasets
        coord_cols = ["lat", "lon"] if "lat" in spatial_df.columns else ["latitude", "longitude"]
        
        # Single groupby with all aggregations at once using Polars
        spatial_metrics = spatial_df.group_by(coord_cols).agg([
            pl.col("y_true").mean().alias("y_true_mean"),
            pl.col("y_true").var().alias("y_true_var"),
            pl.col("y_true").count().alias("y_true_count"),
            pl.col("y_pred").mean().alias("y_pred_mean"),
            pl.col("y_pred").var().alias("y_pred_var"),
            pl.col("residual").mean().alias("residual_mean"),
            pl.col("residual").std().alias("residual_std")
        ])
        
        # Calculate additional metrics using Polars expressions
        spatial_metrics = spatial_metrics.with_columns([
            (pl.col("y_pred_mean") - pl.col("y_true_mean")).alias("bias"),
            pl.col("residual_mean").abs().alias("mae"),
            (pl.col("residual_std").pow(2) + pl.col("residual_mean").pow(2)).sqrt().alias("rmse"),
            (pl.col("y_pred_mean") - pl.col("y_true_mean")).alias("diff"),
            pl.col("y_true_var").alias("var_true"),
            pl.col("y_pred_var").alias("var_pred"),
            pl.col("y_true_count").alias("count")
        ])
        
        # Calculate SNR metrics
        spatial_metrics = spatial_metrics.with_columns([
            pl.when(pl.col("residual_std").pow(2) > 0)
            .then(pl.col("var_true") / pl.col("residual_std").pow(2))
            .otherwise(float('inf'))
            .alias("snr")
        ])
        
        # Calculate SNR in dB
        spatial_metrics = spatial_metrics.with_columns([
            pl.when(pl.col("snr").is_finite())
            .then(10 * pl.col("snr").log10())
            .otherwise(float('inf'))
            .alias("snr_db")
        ])
        
        # For Pearson correlation, we need to use a different approach with Polars
        # Group by coordinates and calculate correlation for each group
        pearson_correlations = []
        for group in spatial_df.group_by(coord_cols):
            group_data = group[1]
            if len(group_data) >= 2:
                try:
                    corr = pearsonr(group_data["y_true"].to_numpy(), group_data["y_pred"].to_numpy())[0]
                    pearson_correlations.append(corr)
                except:
                    pearson_correlations.append(np.nan)
            else:
                pearson_correlations.append(np.nan)
        
        # Add Pearson correlations to the result
        spatial_metrics = spatial_metrics.with_columns(
            pl.Series("pearson", pearson_correlations)
        )
        
        return spatial_metrics
        
    else:
        # Fallback to pandas implementation for compatibility
        coord_cols = ["lat", "lon"] if "lat" in spatial_df.columns else ["latitude", "longitude"]
        
        # Single groupby with all aggregations at once - much faster!
        spatial_metrics = spatial_df.groupby(coord_cols).agg({
            "y_true": ["mean", "var", "count"],
            "y_pred": ["mean", "var"],
            "residual": ["mean", "std"]
        }).reset_index()
        
        # Flatten column names
        spatial_metrics.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in spatial_metrics.columns]
        
        # Calculate metrics using vectorized operations (much faster than apply)
        spatial_metrics["bias"] = np.mean(spatial_metrics["y_pred_mean"]) - spatial_metrics["y_true_mean"]
        spatial_metrics["mae"] = np.abs(spatial_metrics["residual_mean"])
        spatial_metrics["rmse"] = np.sqrt(spatial_metrics["residual_std"]**2 + spatial_metrics["residual_mean"]**2)
        spatial_metrics["diff"] = spatial_metrics["y_pred_mean"] - spatial_metrics["y_true_mean"]
        spatial_metrics["var_true"] = spatial_metrics["y_true_var"]
        spatial_metrics["var_pred"] = spatial_metrics["y_pred_var"]
        spatial_metrics["count"] = spatial_metrics["y_true_count"]
        
        # Calculate SNR metrics using vectorized operations
        signal_power = spatial_metrics["var_true"]
        noise_power = spatial_metrics["residual_std"]**2
        spatial_metrics["snr"] = np.where(noise_power > 0, signal_power / noise_power, np.inf)
        spatial_metrics["snr_db"] = np.where(np.isfinite(spatial_metrics["snr"]), 
                                            10 * np.log10(spatial_metrics["snr"]), np.inf)
        
        # For Pearson correlation, we need to use apply but only once
        def calc_pearson_vectorized(group):
            if len(group) < 2:
                return np.nan
            try:
                return pearsonr(group["y_true"], group["y_pred"])[0]
            except:
                return np.nan
        
        spatial_metrics["pearson"] = spatial_df.groupby(coord_cols).apply(calc_pearson_vectorized, include_groups=False).values
        
        return spatial_metrics