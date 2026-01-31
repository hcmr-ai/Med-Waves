import torch


def masked_huber_loss(y_pred, y_true, mask, delta=1.0):
    """
    Huber loss masked version - robust για outliers/extremes

    Args:
        delta: Threshold μεταξύ L1 (large errors) και L2 (small errors)
              delta=1.0 ιδανικό για SWH ~0-10m scale
    """
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])

    # Crop to common size
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    # Clean NaNs
    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)

    # Apply mask
    error = torch.abs(y_pred_clean[mask] - y_clean[mask])

    # Huber: L2 για small errors (<delta), L1 για large errors (>delta)
    less_than_delta = error < delta
    huber_loss = torch.where(
        less_than_delta,
        0.5 * (error ** 2) / delta,      # Quadratic regime
        error - 0.5 * delta               # Linear regime
    )

    return huber_loss.mean()
