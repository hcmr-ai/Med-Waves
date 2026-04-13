import torch
import torch.nn.functional as F


def masked_smooth_l1_loss(y_pred, y_true, mask, criterion):
    """
    SmoothL1 loss masked version - robust for outliers/extremes

    Args:
        y_pred:    (B, C, H, W) predictions
        y_true:    (B, C, H, W) targets
        mask:      (B, C, H, W) boolean mask of valid pixels
        criterion: SmoothL1Loss instance (its beta/delta controls the L1/L2 transition)
    """
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)

    return criterion(y_pred_clean[mask], y_clean[mask])


def masked_multi_bin_weighted_smooth_l1(
    y_pred,
    y_true,
    mask,
    vhm0,
    criterion,
    bin_thresholds=None,
    bin_weights=None,
    epsilon=1e-6,  # To avoid division by zero
):
    """
    Physics-aware SmoothL1 loss with wave-height bin weighting.
    Applies SmoothL1 to (normalized) y_pred/y_true but uses *unnormalized* vhm0 to weight bins.

    Args:
        y_pred: (B, C, H, W) normalized predictions
        y_true: (B, C, H, W) normalized targets
        mask:   (B, C, H, W) boolean mask of valid pixels
        vhm0:   (B, 1, H, W) unnormalized VHM0 (meters), used for binning
        bin_thresholds: sea-state edges in meters (len = N)
        bin_weights:    loss weights per bin (len = N+1)
        criterion:      SmoothL1 loss criterion
    """
    if bin_thresholds is None:
        bin_thresholds = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0]
    if bin_weights is None:
        bin_weights = [0.9, 1.0, 1.2, 1.5, 2.2, 3.0, 4.0]

    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]
    vhm0 = vhm0[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    # --- Clean NaNs ---
    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)
    vhm0_clean = torch.nan_to_num(vhm0, nan=0.0)

    # --- Build bin weights using real wave heights (meters) ---
    weights = torch.zeros_like(vhm0_clean)
    prev_t = -float("inf")
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights, strict=False):
        weights += ((vhm0_clean >= prev_t) & (vhm0_clean < t)) * w
        prev_t = t

    # --- SmoothL1Loss per-pixel, but weighted by sea state ---
    loss_per_pixel = criterion(y_pred_clean, y_clean) * weights  # (B,1,H,W)

    # Apply mask + normalize by total weight in mask
    num = loss_per_pixel[mask].sum()
    den = weights[mask].sum() + epsilon
    weighted_loss = num / den

    return weighted_loss


def masked_bin_balanced_smooth_l1(
    y_pred,
    y_true,
    mask,
    vhm0,
    bin_thresholds=None,
    bin_weights=None,
    beta=0.3,
    epsilon=1e-6,
):
    """
    Bin-balanced SmoothL1 loss using raw VHM0 for sea-state assignment.

    Unlike per-pixel weighted losses, this computes one SmoothL1 mean per non-empty
    raw-VHM0 bin, then averages those bin losses. A rare bin can therefore
    contribute as much as a common bin, which is useful when a model has
    regime-specific failures hidden by the dominant sea states.
    """
    if bin_thresholds is None:
        # Raw VHM0 bins: 0-1, 1-2, 2-3, 3-5, 5-7, 7+ m.
        bin_thresholds = [1.0, 2.0, 3.0, 5.0, 7.0]
    n_bins = len(bin_thresholds) + 1
    if bin_weights is None:
        bin_weights = [1.0] * n_bins
    if len(bin_weights) != n_bins:
        raise ValueError(
            f"bin_weights must have {n_bins} values for {len(bin_thresholds)} thresholds, "
            f"got {len(bin_weights)}"
        )

    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w].bool()
    vhm0 = vhm0[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)
    vhm0_clean = torch.nan_to_num(vhm0, nan=0.0)
    loss_per_pixel = F.smooth_l1_loss(
        y_pred_clean,
        y_clean,
        beta=beta,
        reduction="none",
    )

    bin_losses = []
    bin_loss_weights = []
    prev_t = -float("inf")
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights, strict=False):
        bin_mask = (vhm0_clean >= prev_t) & (vhm0_clean < t) & mask
        prev_t = t
        if bin_mask.any():
            bin_losses.append(loss_per_pixel[bin_mask].mean())
            bin_loss_weights.append(float(w))

    if not bin_losses:
        return torch.tensor(0.0, device=y_true.device, requires_grad=True)

    weights = torch.tensor(bin_loss_weights, device=y_true.device, dtype=y_true.dtype)
    losses = torch.stack(bin_losses)
    return (losses * weights).sum() / (weights.sum() + epsilon)


def masked_atlantic_low_bin_balanced_smooth_l1(
    y_pred,
    y_true,
    mask,
    vhm0,
    beta=0.3,
):
    """
    Atlantic-focused bin-balanced SmoothL1.

    The 2023 Atlantic evaluations show the regional TransUNet is worst in 0-2 m
    while useful in moderate/high seas. These weights keep the per-bin balancing
    but put extra pressure on calm/light bins.
    """
    return masked_bin_balanced_smooth_l1(
        y_pred=y_pred,
        y_true=y_true,
        mask=mask,
        vhm0=vhm0,
        bin_thresholds=[1.0, 2.0, 3.0, 5.0, 7.0],
        bin_weights=[3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        beta=beta,
    )
