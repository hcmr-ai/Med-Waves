import torch


def masked_huber_loss(y_pred, y_true, mask, delta=1.0):
    """
    Huber loss masked version - robust for outliers/extremes

    Args:
        delta: Threshold between L1 (large errors) and L2 (small errors)
              delta=1.0 ideal for SWH ~0-10m scale
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

    # Huber: L2 for small errors (<delta), L1 for large errors (>delta)
    less_than_delta = error < delta
    huber_loss = torch.where(
        less_than_delta,
        0.5 * (error**2) / delta,  # Quadratic regime
        error - 0.5 * delta,  # Linear regime
    )

    return huber_loss.mean()


def masked_classical_huber_loss(y_pred, y_true, mask, delta=1.0):
    """
    Classical (delta-scaled) Huber loss, masked version.

    Piecewise form:
      0.5 * e^2                  if |e| <= delta
      delta * (|e| - 0.5*delta)  otherwise

    This differs from SmoothL1-style Huber where the quadratic branch is divided by delta.
    """
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])

    # Crop to common size
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    if delta <= 0:
        raise ValueError(f"delta must be > 0 for classical Huber, got {delta}")

    # Clean NaNs
    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)

    # Apply mask
    error = torch.abs(y_pred_clean[mask] - y_clean[mask])
    less_than_delta = error < delta

    classical_huber = torch.where(
        less_than_delta,
        0.5 * (error**2),  # Quadratic regime
        delta * (error - 0.5 * delta),  # Linear regime with slope=delta
    )
    return classical_huber.mean()


def _huber_per_pixel(error, delta):
    """Element-wise Huber: quadratic for |e|<delta, linear beyond."""
    return torch.where(
        error < delta,
        0.5 * (error**2) / delta,
        error - 0.5 * delta,
    )


def masked_mse_huber_tail_loss(
    y_pred,
    y_true,
    mask,
    vhm0,
    tail_threshold=8.0,
    delta=0.5,
    tail_weight=5.0,
    epsilon=1e-6,
):
    """
    Hybrid loss: MSE on calm/moderate seas, Huber on extreme tails.

    MSE works well where the model already performs (0–tail_threshold m).
    Huber's linear regime prevents gradient explosion on the rare extreme
    events, giving the model a stable learning signal for the tails.

    Args:
        y_pred:         (B, C, H, W) model prediction
        y_true:         (B, C, H, W) target
        mask:           (B, C, H, W) bool mask of valid pixels
        vhm0:           (B, 1, H, W) unnormalized VHM0 in metres
        tail_threshold: VHM0 above which Huber replaces MSE (metres)
        delta:          Huber transition point (error magnitude)
        tail_weight:    Multiplier on the tail loss to compensate for rarity
        epsilon:        Numerical stability constant
    """
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]
    vhm0 = vhm0[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)
    vhm0_clean = torch.nan_to_num(vhm0, nan=0.0)

    error = torch.abs(y_pred_clean - y_clean)

    # Split pixels into bulk vs tail based on raw wave height
    is_tail = (vhm0_clean >= tail_threshold) & mask
    is_bulk = (~is_tail) & mask

    # Bulk: standard MSE
    loss_bulk = torch.tensor(0.0, device=y_pred.device)
    n_bulk = is_bulk.sum()
    if n_bulk > 0:
        loss_bulk = (error[is_bulk] ** 2).sum() / (n_bulk.float() + epsilon)

    # Tail: Huber (linear penalty for large errors, quadratic for small)
    loss_tail = torch.tensor(0.0, device=y_pred.device)
    n_tail = is_tail.sum()
    if n_tail > 0:
        loss_tail = _huber_per_pixel(error[is_tail], delta).sum() / (
            n_tail.float() + epsilon
        )

    return loss_bulk + tail_weight * loss_tail
