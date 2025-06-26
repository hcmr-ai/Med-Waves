import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X, y, poly, scaler):
    """
    Evaluate the model on a given dataset using standard metrics.
    """
    X_poly = poly.transform(X)
    X_scaled = scaler.transform(X_poly)
    y_pred = model.predict(X_scaled)

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
