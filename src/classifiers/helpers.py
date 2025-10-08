import numpy as np

def reconstruct_vhm0_values(
    predict_bias: bool,
    predict_bias_log_space: bool,
    vhm0_x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs true and predicted VHM0 values from bias space
    (additive or log-space) back to linear VHM0 domain.

    Parameters
    ----------
    vhm0_x : np.ndarray
        Baseline model output (e.g., vhm0_x_train).
    y_true : np.ndarray
        True target values (bias_true if predict_bias=True).
    y_pred : np.ndarray
        Model predictions (bias_pred if predict_bias=True).

    Returns
    -------
    vhm0_y_true : np.ndarray
        Reconstructed true wave heights.
    vhm0_y_pred : np.ndarray
        Reconstructed predicted wave heights.
    """
    if predict_bias:
        if predict_bias_log_space:
            vhm0_y_true = vhm0_x * np.exp(y_true)
            vhm0_y_pred = vhm0_x * np.exp(y_pred)
        else:
            # Additive bias: y_hat = vhm0_x + bias
            vhm0_y_true = vhm0_x + y_true
            vhm0_y_pred = vhm0_x + y_pred
    else:
        # direct regression mode
        vhm0_y_true = y_true
        vhm0_y_pred = y_pred

    return vhm0_y_true, vhm0_y_pred