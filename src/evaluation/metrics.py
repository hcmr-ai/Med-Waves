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
