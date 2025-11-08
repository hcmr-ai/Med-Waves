import torch


def masked_smooth_l1_loss(y_pred, y_true, mask, criterion):
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


def masked_mse_loss(y_pred, y_true, mask, epsilon=1e-6):
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]  # Resize mask to match cropped tensors

    # Use the provided mask (already filters NaN values from dataset)
    combined_mask = mask

    # Check if we have any valid pixels
    if not combined_mask.any():
        return torch.tensor(0.0, device=y_true.device)

    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)
    diff = (y_pred_clean - y_clean) ** 2
    loss = diff[combined_mask].mean()

    # Check for NaN in loss
    if torch.isnan(loss):
        print(f"Warning: NaN loss detected. y_pred stats: min={y_pred.min()}, max={y_pred.max()}, mean={y_pred.mean()}")
        print(f"y_true stats: min={y_true.min()}, max={y_true.max()}, mean={y_true.mean()}")
        print(f"mask stats: valid_pixels={combined_mask.sum()}, total_pixels={combined_mask.numel()}")
        return torch.tensor(0.0, device=y_true.device)

    return loss

def masked_weighted_mse(y_pred, y_true, mask, threshold=5.0, high_weight=1.0, epsilon=1e-6):
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]  # Resize mask to match cropped tensors

    # Use the provided mask (already filters NaN values from dataset)
    combined_mask = mask

    # Check if we have any valid pixels
    if not combined_mask.any():
        return torch.tensor(0.0, device=y_true.device)

    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)

    weights = torch.ones_like(y_clean)
    weights = torch.where(y_clean >= threshold, high_weight, weights)

    diff = (y_pred_clean - y_clean) ** 2 * weights
    weight_sum = weights[combined_mask].sum()

    if weight_sum == 0:
        return torch.tensor(0.0, device=y_true.device)

    loss = diff[combined_mask].sum() / (weight_sum + epsilon)

    # Check for NaN in loss
    if torch.isnan(loss):
        print(f"Warning: NaN loss detected. y_pred stats: min={y_pred.min()}, max={y_pred.max()}, mean={y_pred.mean()}")
        print(f"y_true stats: min={y_clean.min()}, max={y_clean.max()}, mean={y_clean.mean()}")
        print(f"mask stats: valid_pixels={combined_mask.sum()}, total_pixels={combined_mask.numel()}")
        return torch.tensor(1.0, device=y_true.device)

    return loss

def masked_multi_bin_weighted_mse(
    y_pred,
    y_true,
    mask,
    vhm0,
    bin_thresholds = None,
    bin_weights = None,
    epsilon=1e-6,
):
    """
    Weighted MSE with physics-based binning using unnormalized VHM0.
    
    Args:
        y_pred: (B, C, H, W) normalized model prediction
        y_true: (B, C, H, W) normalized target
        mask:   (B, C, H, W) bool mask of valid pixels
        vhm0:   (B, 1, H, W) unnormalized significant wave height in meters
    """
    if bin_thresholds is None or bin_weights is None:
        bin_thresholds = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0]
    if bin_weights is None:
        bin_weights = [0.9,  1.0,  1.2,  1.5,  2.2,  3.0,  4.0]
    # Crop shapes to match
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask = mask[:, :, :min_h, :min_w]
    vhm0 = vhm0[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    # Clean numeric input
    y_clean = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean = torch.nan_to_num(y_pred, nan=0.0)
    vhm0_clean = torch.nan_to_num(vhm0, nan=0.0)

    # Build weight mask based on real wave heights (in meters)
    weights = torch.zeros_like(vhm0_clean)
    prev_t = -float("inf")
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights):
        weights += ((vhm0_clean >= prev_t) & (vhm0_clean < t)) * w
        prev_t = t

    # MSE weighted by sea state
    diff = (y_pred_clean - y_clean) ** 2 * weights
    weighted_loss = diff[mask].sum() / (weights[mask].sum() + epsilon)

    return weighted_loss

def masked_multi_bin_weighted_smooth_l1(
    y_pred,
    y_true,
    mask,
    vhm0,
    criterion,
    bin_thresholds = None,
    bin_weights = None,
    epsilon=1e-6    # To avoid division by zero
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
        bin_weights = [0.9,  1.0,  1.2,  1.5,  2.2,  3.0,  4.0]
    # --- Crop dimensions if needed ---
    min_h = min(y_pred.shape[2], y_true.shape[2])
    min_w = min(y_pred.shape[3], y_true.shape[3])
    y_pred = y_pred[:, :, :min_h, :min_w]
    y_true = y_true[:, :, :min_h, :min_w]
    mask   = mask[:, :, :min_h, :min_w]
    vhm0   = vhm0[:, :, :min_h, :min_w]

    if not mask.any():
        return torch.tensor(0.0, device=y_true.device)

    # --- Clean NaNs ---
    y_clean     = torch.nan_to_num(y_true, nan=0.0)
    y_pred_clean= torch.nan_to_num(y_pred, nan=0.0)
    vhm0_clean  = torch.nan_to_num(vhm0,  nan=0.0)

    # --- Build bin weights using real wave heights (meters) ---
    weights = torch.zeros_like(vhm0_clean)
    prev_t = -float("inf")
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights):
        weights += ((vhm0_clean >= prev_t) & (vhm0_clean < t)) * w
        prev_t = t
    
    # --- SmoothL1Loss per-pixel, but weighted by sea state ---
    loss_per_pixel = criterion(y_pred_clean, y_clean) * weights     # (B,1,H,W)
    
    # Apply mask + normalize by total weight in mask
    num = loss_per_pixel[mask].sum()
    den = weights[mask].sum() + epsilon
    weighted_loss = num / den
    
    return weighted_loss
