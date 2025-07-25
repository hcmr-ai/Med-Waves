import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(y_pred, y):
    """
    Evaluate the model on a given dataset using standard metrics.
    """
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    pearson = pearsonr(y, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson": pearson[0],
    }
