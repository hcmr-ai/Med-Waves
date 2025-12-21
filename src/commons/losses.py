import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from src.classifiers.networks.mdn import mdn_expected_value


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
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights, strict=False):
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
    for t, w in zip(bin_thresholds + [float("inf")], bin_weights, strict=False):
        weights += ((vhm0_clean >= prev_t) & (vhm0_clean < t)) * w
        prev_t = t

    # --- SmoothL1Loss per-pixel, but weighted by sea state ---
    loss_per_pixel = criterion(y_pred_clean, y_clean) * weights     # (B,1,H,W)

    # Apply mask + normalize by total weight in mask
    num = loss_per_pixel[mask].sum()
    den = weights[mask].sum() + epsilon
    weighted_loss = num / den

    return weighted_loss

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


def masked_mse_perceptual_loss(y_pred, y_true, mask, perceptual_loss, lambda_perceptual=0.05):
    """
    Masked MSE perceptual loss.
    Args:
        y_pred: (B, 1, H, W) normalized model prediction
        y_true: (B, 1, H, W) normalized target
        mask:   (B, 1, H, W) bool mask of valid pixels
        lambda_perceptual: weight for perceptual loss
    """
    mse_loss = masked_mse_loss(y_pred, y_true, mask)
    perc_loss = perceptual_loss(y_pred, y_true)
    return mse_loss + lambda_perceptual * perc_loss
    # return mse_loss

def masked_mse_ssim_loss(y_pred, y_true, mask, ssim_loss, lambda_ssim=0.1):
    """
    Masked SSIM loss.
    Args:
        y_pred: (B, 1, H, W) normalized model prediction
        y_true: (B, 1, H, W) normalized target
        mask:   (B, 1, H, W) bool mask of valid pixels
        lambda_ssim: weight for SSIM loss
    """
    return masked_mse_loss(y_pred, y_true, mask) + lambda_ssim * ssim_loss

def masked_ssim_perceptual_loss(y_pred, y_true, mask, ssim_loss, perceptual_loss, lambda_ssim=0.1, lambda_perceptual=0.05):
    """
    Masked SSIM perceptual loss.
    Args:
        y_pred: (B, 1, H, W) normalized model prediction
        y_true: (B, 1, H, W) normalized target
        mask:   (B, 1, H, W) bool mask of valid pixels
        lambda_ssim: weight for SSIM loss
        lambda_perceptual: weight for perceptual loss
    """
    return masked_mse_loss(y_pred, y_true, mask) + lambda_ssim * ssim_loss + lambda_perceptual * perceptual_loss(y_pred, y_true)

def mdn_nll_loss(pi, mu, sigma, y, mask=None, eps=1e-6):
    """
    Numerically stable MDN NLL loss.
    pi, mu, sigma: [B, K, H, W]
    y:             [B, 1, H, W]
    mask:          [B, 1, H, W]  (1=ocean, 0=land), optional

    returns scalar NLL loss
    """
    # ===== CLEAN NaN VALUES FIRST (from land pixels) =====
    pi = torch.nan_to_num(pi, nan=1.0/pi.shape[1], posinf=1.0, neginf=0.0)  # Use uniform distribution for NaN
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=10.0, neginf=eps)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clamp sigma to prevent numerical issues
    sigma = torch.clamp(sigma, min=eps, max=100.0)
    
    # Expand y to match mixture dimension
    y_expanded = y.expand_as(mu)    # [B, K, H, W]
    
    # Compute log Gaussian density (more stable than exp then log)
    # log N(y|mu,sigma) = -0.5*log(2*pi) - log(sigma) - 0.5*((y-mu)/sigma)^2
    log_2pi = math.log(2 * math.pi)
    z_score = (y_expanded - mu) / sigma
    log_normal = -0.5 * log_2pi - torch.log(sigma) - 0.5 * z_score**2
    
    # Add log mixture weights: log(pi * N(y|mu,sigma)) = log(pi) + log(N)
    log_pi = torch.log(pi + eps)
    log_weighted = log_pi + log_normal  # [B, K, H, W]
    
    # Log-sum-exp trick for numerical stability
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    max_log_weighted = log_weighted.max(dim=1, keepdim=True)[0]
    log_sum_exp = torch.sum(torch.exp(log_weighted - max_log_weighted), dim=1)
    log_likelihood = max_log_weighted.squeeze(1) + torch.log(log_sum_exp + eps)  # [B, H, W]
    
    # Check for NaN before masking (debug)
    if torch.isnan(log_likelihood).any():
        print(f"WARNING: NaN in MDN log_likelihood BEFORE masking!")
        print(f"  pi range: [{pi.min():.4f}, {pi.max():.4f}], has_nan: {torch.isnan(pi).any()}")
        print(f"  mu range: [{mu.min():.4f}, {mu.max():.4f}], has_nan: {torch.isnan(mu).any()}")
        print(f"  sigma range: [{sigma.min():.4f}, {sigma.max():.4f}], has_nan: {torch.isnan(sigma).any()}")
        print(f"  log_sum_exp range: [{log_sum_exp.min():.4f}, {log_sum_exp.max():.4f}]")
        # Replace NaN with large negative value (high loss)
        log_likelihood = torch.nan_to_num(log_likelihood, nan=-10.0)
    
    # Apply mask
    if mask is not None:
        mask = mask.squeeze(1)
        valid_mask = mask.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=pi.device, requires_grad=True)
        log_likelihood_masked = log_likelihood[valid_mask]
    else:
        log_likelihood_masked = log_likelihood
    
    # NLL: -mean(log p(y|x))
    # Clamp log_likelihood to prevent negative loss
    log_likelihood_masked = torch.clamp(log_likelihood_masked, max=0.0)
    nll = -log_likelihood_masked.mean()
    
    # Check for NaN in output
    if torch.isnan(nll):
        print(f"WARNING: NaN in MDN NLL output!")
        return torch.tensor(0.0, device=pi.device, requires_grad=True)
    
    return nll

def masked_mse_mdn_loss(pi, mu, sigma, y, mask=None, eps=1e-6, lambda_mse=0.1, lambda_nll=1.0):
    """
    Masked MSE + MDN NLL loss.
    Args:
        pi, mu, sigma: [B, K, H, W]
        y:             [B, 1, H, W]
        mask:          [B, 1, H, W]  (1=ocean, 0=land), optional
        lambda_mse:    weight for MSE loss
        lambda_nll:    weight for NLL loss
        returns scalar loss
    """
    # Compute expected value for MSE
    y_pred = mdn_expected_value(pi, mu)
    
    # MSE term
    mse_term = masked_mse_loss(y_pred, y, mask)
    
    # NLL term
    nll_term = mdn_nll_loss(pi, mu, sigma, y, mask, eps)
    
    # Combined loss
    total_loss = lambda_mse * mse_term + lambda_nll * nll_term
    
    # Safety check
    if torch.isnan(total_loss):
        print(f"WARNING: NaN in combined MDN loss!")
        print(f"  mse_term: {mse_term.item()}, nll_term: {nll_term.item()}")
        return torch.tensor(0.0, device=pi.device, requires_grad=True)
    
    return total_loss


def adversarial_loss_G(D_fake: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    LSGAN generator loss: want D(fake) -> 1.
    
    Args:
        D_fake: Discriminator output on fake samples [B, 1, H', W']
        mask: Optional mask downsampled to match D_fake shape [B, 1, H', W']
    """
    if mask is not None:
        # Align shapes
        min_h = min(D_fake.shape[2], mask.shape[2])
        min_w = min(D_fake.shape[3], mask.shape[3])
        D_fake = D_fake[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]
        
        # Check if we have any valid pixels
        if not mask.any():
            return torch.tensor(0.0, device=D_fake.device)
        
        # Compute loss only on valid pixels
        loss_map = (D_fake - 1.0) ** 2
        return loss_map[mask].mean()
    else:
        return torch.mean((D_fake - 1.0) ** 2)


def adversarial_loss_D(D_real: torch.Tensor, D_fake: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    LSGAN discriminator loss:
      D(real) -> 1
      D(fake) -> 0
    
    Args:
        D_real: Discriminator output on real samples [B, 1, H', W']
        D_fake: Discriminator output on fake samples [B, 1, H', W']
        mask: Optional mask downsampled to match outputs [B, 1, H', W']
    """
    if mask is not None:
        # Align shapes
        min_h = min(D_real.shape[2], mask.shape[2])
        min_w = min(D_real.shape[3], mask.shape[3])
        D_real = D_real[:, :, :min_h, :min_w]
        D_fake = D_fake[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]
        
        # Check if we have any valid pixels
        if not mask.any():
            return torch.tensor(0.0, device=D_real.device)
        
        # Compute losses only on valid pixels
        loss_real_map = (D_real - 1.0) ** 2
        loss_fake_map = D_fake ** 2
        
        loss_real = loss_real_map[mask].mean()
        loss_fake = loss_fake_map[mask].mean()
    else:
        loss_real = torch.mean((D_real - 1.0) ** 2)
        loss_fake = torch.mean(D_fake ** 2)
    
    return 0.5 * (loss_real + loss_fake)
