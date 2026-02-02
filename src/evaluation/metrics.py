from typing import Dict, List

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
                except Exception:
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
            except Exception:
                return np.nan

        spatial_metrics["pearson"] = spatial_df.groupby(coord_cols).apply(calc_pearson_vectorized, include_groups=False).values

        return spatial_metrics


def compute_overall_metrics_from_accumulators(
    total_count: int,
    sum_mae: float,
    sum_mse: float,
    sum_bias: float,
    sum_baseline_mae: float,
    sum_baseline_mse: float,
    sum_baseline_bias: float,
    sum_y_true: float,
    sum_y_true_sq: float,
    sum_y_pred: float,
    sum_y_pred_sq: float,
    sum_y_true_y_pred: float,
    predict_bias: bool = False,
) -> Dict[str, float]:
    """
    Compute overall performance metrics from accumulated statistics.

    This function calculates comprehensive evaluation metrics including MAE, RMSE,
    bias, R², correlation, and baseline comparisons from pre-accumulated sums.

    Args:
        total_count: Total number of samples processed
        sum_mae: Sum of absolute errors
        sum_mse: Sum of squared errors
        sum_bias: Sum of biases (pred - true)
        sum_baseline_mae: Sum of baseline absolute errors
        sum_baseline_mse: Sum of baseline squared errors
        sum_baseline_bias: Sum of baseline biases
        sum_y_true: Sum of true values
        sum_y_true_sq: Sum of squared true values
        sum_y_pred: Sum of predicted values
        sum_y_pred_sq: Sum of squared predicted values
        sum_y_true_y_pred: Sum of (true * pred) products
        predict_bias: Whether model is in bias prediction mode

    Returns:
        Dictionary containing computed metrics
    """
    if total_count == 0:
        return {"error": "No valid data processed"}

    # Model metrics
    mae = sum_mae / total_count
    mse = sum_mse / total_count
    rmse = np.sqrt(mse)
    bias = sum_bias / total_count

    # Baseline metrics
    baseline_mae = None
    baseline_rmse = None
    baseline_bias = None
    mae_improvement = None
    rmse_improvement = None

    if sum_baseline_mae > 0:  # Check if baseline data exists
        baseline_mae = sum_baseline_mae / total_count
        baseline_mse = sum_baseline_mse / total_count
        baseline_rmse = np.sqrt(baseline_mse)
        baseline_bias = sum_baseline_bias / total_count
        mae_improvement = (
            ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0.0
        )
        rmse_improvement = (
            ((baseline_rmse - rmse) / baseline_rmse) * 100
            if baseline_rmse > 0
            else 0.0
        )

    # R² score
    mean_y_true = sum_y_true / total_count
    ss_res = sum_mse * total_count  # Already sum of squared residuals
    ss_tot = sum_y_true_sq - total_count * (mean_y_true**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Correlation
    mean_y_pred = sum_y_pred / total_count
    cov = (sum_y_true_y_pred / total_count) - (mean_y_true * mean_y_pred)
    std_y_true = np.sqrt((sum_y_true_sq / total_count) - (mean_y_true**2))
    std_y_pred = np.sqrt((sum_y_pred_sq / total_count) - (mean_y_pred**2))
    correlation = (
        cov / (std_y_true * std_y_pred) if (std_y_true * std_y_pred) > 0 else 0.0
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "mse": float(mse),
        "r2": float(r2),
        "correlation": float(correlation),
        "baseline_mae": float(baseline_mae) if baseline_mae is not None else None,
        "baseline_rmse": float(baseline_rmse)
        if baseline_rmse is not None
        else None,
        "baseline_bias": float(baseline_bias)
        if baseline_bias is not None
        else None,
        "mae_improvement_pct": float(mae_improvement)
        if mae_improvement is not None
        else None,
        "rmse_improvement_pct": float(rmse_improvement)
        if rmse_improvement is not None
        else None,
        "n_samples": int(total_count),
        "predict_bias_mode": predict_bias,
    }

    return metrics


def compute_sea_bin_metrics_from_accumulators(
    sea_bins: List[Dict],
    sea_bin_accumulators: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Compute per-bin performance metrics from accumulated statistics.

    This function calculates metrics for each sea state bin, including model
    vs baseline comparisons and percentage of samples where model performs better.

    Args:
        sea_bins: List of bin configuration dictionaries with 'name' and 'label' keys
        sea_bin_accumulators: Dictionary mapping bin names to accumulated statistics.
                             Each accumulator should contain:
                             - count: number of samples
                             - sum_mae, sum_mse, sum_bias: model error sums
                             - sum_baseline_mae, sum_baseline_mse, sum_baseline_bias: baseline error sums
                             - count_model_better, count_model_worse: comparison counts

    Returns:
        Dictionary mapping bin names to their computed metrics
    """
    sea_bin_metrics = {}

    for bin_config in sea_bins:
        bin_name = bin_config["name"]
        bin_data = sea_bin_accumulators[bin_name]
        bin_count = bin_data["count"]

        if bin_count > 0:
            # Model metrics
            mae = bin_data["sum_mae"] / bin_count
            mse = bin_data["sum_mse"] / bin_count
            rmse = np.sqrt(mse)
            bias = bin_data["sum_bias"] / bin_count

            # Baseline metrics
            baseline_mae = None
            baseline_rmse = None
            baseline_bias = None
            mae_improvement = None
            rmse_improvement = None
            pct_model_better = None

            if bin_data["sum_baseline_mae"] > 0:
                baseline_mae = bin_data["sum_baseline_mae"] / bin_count
                baseline_mse = bin_data["sum_baseline_mse"] / bin_count
                baseline_rmse = np.sqrt(baseline_mse)
                baseline_bias = bin_data["sum_baseline_bias"] / bin_count
                mae_improvement = (
                    ((baseline_mae - mae) / baseline_mae) * 100
                    if baseline_mae > 0
                    else 0.0
                )
                rmse_improvement = (
                    ((baseline_rmse - rmse) / baseline_rmse) * 100
                    if baseline_rmse > 0
                    else 0.0
                )

                # Calculate percentage of samples where model is better
                pct_model_better = (
                    bin_data["count_model_better"] / bin_count
                ) * 100
                pct_model_worse = (
                    bin_data["count_model_worse"] / bin_count
                ) * 100

            sea_bin_metrics[bin_name] = {
                "label": bin_config["label"],
                "count": int(bin_count),
                "mae": float(mae),
                "rmse": float(rmse),
                "bias": float(bias),
                "baseline_mae": float(baseline_mae)
                if baseline_mae is not None
                else None,
                "baseline_rmse": float(baseline_rmse)
                if baseline_rmse is not None
                else None,
                "baseline_bias": float(baseline_bias)
                if baseline_bias is not None
                else None,
                "mae_improvement_pct": float(mae_improvement)
                if mae_improvement is not None
                else None,
                "rmse_improvement_pct": float(rmse_improvement)
                if rmse_improvement is not None
                else None,
                "count_model_better": int(bin_data["count_model_better"]),
                "pct_model_better": float(pct_model_better)
                if pct_model_better is not None
                else None,
                "count_model_worse": int(bin_data["count_model_worse"]),
                "pct_model_worse": float(pct_model_worse)
                if pct_model_worse is not None
                else None,
            }

    return sea_bin_metrics


def compute_snr(features: np.ndarray) -> np.ndarray:
    """Compute Signal-to-Noise Ratio (SNR) from features.

    Computes SNR as: SNR = 10 * log10(mean^2 / variance) across feature channels.

    Args:
        features: Numpy array of shape (N, C) where N=samples, C=feature channels

    Returns:
        Numpy array of shape (N,) containing SNR values in dB
    """
    # SNR = 10 * log10(mean^2 / variance) across feature channels
    mean_val = features.mean(axis=1)  # Mean across feature channels
    var_val = features.var(axis=1) + 1e-8  # Variance across channels (add epsilon to avoid division by zero)

    # Compute SNR in dB
    snr = 10 * np.log10(np.abs(mean_val)**2 / var_val)

    return snr
