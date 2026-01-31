import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_switch_loss(
    y_pred,
    y_true,
    mask,
    threshold_m=None,      # 1 meter threshold in real space
    std=None,             # std used during normalization (required!)
    weight_normal=0.0,
    weight_hard=1.0,
):
    """
    Pixel-Switch Loss (Weighted MSE / SmoothL1)
    - threshold_m: threshold in meters
    - std: used to convert to normalized space
    - base_loss_fn: must support reduction="none"
    """

    # Convert threshold to normalized units

    if std is not None and threshold_m is not None:
        threshold = threshold_m / std
    elif threshold_m is not None:
        threshold = threshold_m

    # Crop shapes
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask   = mask[:, :, :min_h, :min_w]

    mask = mask.bool()
    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)
    # Compute pixelwise base loss
    loss_per_pixel = (y_pred - y_true) ** 2

    # Compute absolute error in normalized space
    error = torch.abs(y_pred - y_true)

    # Define hard vs normal pixels
    hard_mask = (error > threshold) & mask
    normal_mask = (~hard_mask) & mask

    # Apply weights only where valid
    weights = torch.zeros_like(error)
    weights[hard_mask] = weight_hard
    weights[normal_mask] = weight_normal

    # Final weighted loss
    weighted_loss = loss_per_pixel * weights
    return weighted_loss[mask].mean()

def pixel_switch_loss_stable_old(
    y_pred,
    y_true,
    mask,
    threshold_m=None,   # threshold in meters (optional)
    std=None,           # normalization std if threshold_m is in meters
    weight_normal=0.1,  # soft weight for "easy" pixels
    weight_hard=1.0,    # weight for "hard" pixels
    dynamic_quantile=0.90,  # used if threshold_m is None
    smooth_mix=0.2,     # fraction of SmoothL1 stabilizer
    epsilon=1e-6,
):
    """
    Soft Pixel-Switch Loss (stable version for fine-tuning).

    Args:
        y_pred, y_true: tensors (B, 1, H, W)
        mask: valid pixels (B, 1, H, W)
        base_loss_fn: must support reduction="none" (e.g. nn.MSELoss(reduction="none"))
        threshold_m: threshold in meters (optional)
        std: normalization std (if data normalized)
        weight_normal, weight_hard: weights for normal/hard pixels
        dynamic_quantile: used if threshold_m is None (e.g. 0.9 = 90th percentile)
        smooth_mix: fraction of SmoothL1Loss blended in for stability
    """

    # Crop dimensions to match
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w].bool()

    # Replace NaNs
    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)

    # Compute per-pixel base loss (e.g. MSE)
    loss_per_pixel = (y_pred - y_true) ** 2
    # if loss_per_pixel.ndim > 3:
    #     loss_per_pixel = loss_per_pixel.squeeze(1)

    # Compute absolute error
    error = torch.abs(y_pred - y_true)

    # Determine threshold (in normalized space)
    if threshold_m is not None and std is not None:
        threshold = threshold_m / std
    elif threshold_m is not None:
        threshold = threshold_m

    # Identify hard pixels
    hard_mask = (error > threshold) & mask
    normal_mask = (~hard_mask) & mask

    # Apply weights softly
    weights = torch.zeros_like(error)
    weights[normal_mask] = weight_normal
    weights[hard_mask] = weight_hard

    # Weighted pixel-switch component
    weighted_loss = loss_per_pixel * weights
    pixel_switch_term = weighted_loss[mask].mean()

    # Add SmoothL1 stabilizer
    smooth_l1_term = nn.SmoothL1Loss()(y_pred[mask], y_true[mask])

    # Combine them
    total_loss = (1 - smooth_mix) * pixel_switch_term + smooth_mix * smooth_l1_term

    return total_loss


def pixel_switch_loss_stable(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold_m: float | None = None,   # threshold in meters (optional)
    std: float | torch.Tensor | None = None,  # normalization std if threshold_m is in meters
    weight_normal: float = 0.1,         # weight for "easy" pixels
    weight_hard: float = 1.0,           # weight for "hard" pixels
    dynamic_quantile: float = 0.90,     # used if threshold_m is None
    smooth_mix: float = 0.2,            # fraction of SmoothL1 stabilizer
    normalize_weights: bool = True,     # keeps loss scale stable across batches
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Pixel-Switch Loss (stable fine-tuning version).

    - "Hard" pixels are defined per-sample by a threshold on |error|
      (either fixed threshold_m or masked quantile).
    - Base per-pixel loss is |diff| + diff^2 (like the paper's MAE+MSE per pixel).
    - Weights are soft (weight_normal vs weight_hard) and optionally normalized.
    - Blends a SmoothL1 term for stability.

    Shapes:
      y_pred, y_true: (B, C, H, W)  (C can be 1)
      mask:           (B, 1 or C, H, W) or (B, H, W)
    """

    # --- align spatial dims ---
    min_h = min(y_pred.shape[-2], y_true.shape[-2])
    min_w = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred = y_pred[..., :min_h, :min_w]
    y_true = y_true[..., :min_h, :min_w]
    mask = mask[..., :min_h, :min_w]

    # --- ensure mask is boolean with same shape as y_pred ---
    if mask.ndim == 3:  # (B,H,W) -> (B,1,H,W)
        mask = mask.unsqueeze(1)
    mask = mask.bool()
    if mask.shape[1] == 1 and y_pred.shape[1] != 1:
        mask = mask.expand(-1, y_pred.shape[1], -1, -1)

    # --- NaN safety ---
    y_true = torch.nan_to_num(y_true, nan=0.0)
    y_pred = torch.nan_to_num(y_pred, nan=0.0)

    # --- handle no valid pixels ---
    valid_count = mask.sum()
    if valid_count == 0:
        # return a zero loss that won't break backward()
        return (y_pred * 0.0).sum()

    diff = y_pred - y_true
    abs_err = diff.abs()

    # Base per-pixel loss: |diff| + diff^2  (paper-style per-pixel)
    base_loss = abs_err + diff.pow(2)

    # --- compute threshold (normalized space) ---
    B = y_pred.shape[0]
    thresholds = []

    # If threshold_m provided, convert to normalized units if std given
    if threshold_m is not None:
        if std is not None:
            # std can be float or tensor; assume it's compatible scalar
            thr = threshold_m / std
        else:
            thr = threshold_m
        thresholds = [torch.as_tensor(thr, device=y_pred.device, dtype=y_pred.dtype) for _ in range(B)]
        thresholds = torch.stack(thresholds, dim=0)
    else:
        # per-sample masked quantile threshold
        for b in range(B):
            mb = mask[b]
            eb = abs_err[b][mb]
            if eb.numel() == 0:
                thresholds.append(torch.zeros((), device=y_pred.device, dtype=y_pred.dtype))
            else:
                q = torch.quantile(eb, dynamic_quantile)
                thresholds.append(q)
        thresholds = torch.stack(thresholds, dim=0)

    thresholds = thresholds.detach().view(B, 1, 1, 1)

    # --- hard vs normal pixels ---
    hard_mask = (abs_err > thresholds) & mask
    normal_mask = (~hard_mask) & mask

    # --- soft weights ---
    weights = torch.zeros_like(abs_err)
    weights[normal_mask] = weight_normal
    weights[hard_mask] = weight_hard

    # normalize weights over valid pixels to stabilize loss scale
    if normalize_weights:
        w_mean = weights[mask].mean()
        weights = weights / (w_mean + epsilon)

    # Pixel-switch term
    pixel_switch_term = (base_loss * weights)[mask].mean()

    # SmoothL1 stabilizer (masked)
    smooth_term = F.smooth_l1_loss(y_pred, y_true, reduction="none")
    smooth_term = smooth_term[mask].mean()

    # Combine
    total_loss = (1.0 - smooth_mix) * pixel_switch_term + smooth_mix * smooth_term
    return total_loss
