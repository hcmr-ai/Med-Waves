import numpy as np
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


def evaluate_bias_model(bias_pred, bias_true, vhm0_x, vhm0_y):
    """
    Evaluate a bias prediction model and convert results to vhm0 level metrics.
    
    Args:
        bias_pred: Predicted bias values (vhm0_x - vhm0_y)
        bias_true: True bias values (vhm0_x - vhm0_y)
        vhm0_x: Original model predictions
        vhm0_y: Observed values
        
    Returns:
        Dictionary with vhm0-level metrics
    """
    # Convert bias predictions back to vhm0 level
    vhm0_pred = vhm0_x - bias_pred  # Apply bias correction
    
    # Calculate vhm0-level metrics
    rmse = np.sqrt(mean_squared_error(vhm0_y, vhm0_pred))
    mae = mean_absolute_error(vhm0_y, vhm0_pred)
    bias = np.mean(np.mean(vhm0_pred) - vhm0_y)
    diff = np.mean(vhm0_pred - vhm0_y)
    pearson = pearsonr(vhm0_y, vhm0_pred)
    
    # Calculate variance metrics
    var_true = np.var(vhm0_y)
    var_pred = np.var(vhm0_pred)
    
    # Calculate Signal-to-Noise Ratio (SNR)
    residuals = vhm0_y - vhm0_pred
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
    import numpy as np
    from scipy.stats import pearsonr
    
    # Group by coordinates and compute metrics efficiently
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