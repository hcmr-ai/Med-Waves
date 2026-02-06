"""
MSE-based Loss Functions

Collection of Mean Squared Error (MSE) loss variants with masking support.
All functions handle NaN values and provide different weighting strategies.
"""

import torch

from src.classifiers.networks.mdn import mdn_expected_value


def masked_mse_loss(y_pred, y_true, mask, epsilon=1e-6):
    """
    Basic masked MSE loss.

    Args:
        y_pred: (B, C, H, W) model prediction
        y_true: (B, C, H, W) target
        mask: (B, C, H, W) bool mask of valid pixels
        epsilon: Small constant for numerical stability

    Returns:
        Scalar MSE loss computed only on valid (masked) pixels
    """
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
    """
    Weighted MSE loss with higher weight for values above threshold.

    Args:
        y_pred: (B, C, H, W) model prediction
        y_true: (B, C, H, W) target
        mask: (B, C, H, W) bool mask of valid pixels
        threshold: Value threshold for high weighting
        high_weight: Weight multiplier for values >= threshold
        epsilon: Small constant for numerical stability

    Returns:
        Weighted MSE loss with emphasis on high-value predictions
    """
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
    bin_thresholds=None,
    bin_weights=None,
    epsilon=1e-6,
    focus_on_low_waves=True,
):
    """
    Weighted MSE with physics-based binning using unnormalized VHM0.

    Applies different loss weights based on wave height bins, allowing the model
    to focus more on specific sea state conditions.

    Args:
        y_pred: (B, C, H, W) normalized model prediction
        y_true: (B, C, H, W) normalized target
        mask: (B, C, H, W) bool mask of valid pixels
        vhm0: (B, 1, H, W) unnormalized significant wave height in meters
        bin_thresholds: List of VHM0 thresholds in meters (default: [1, 2, 3, 4, 6, 9, 15])
        bin_weights: List of weights for each bin (default: [0.9, 1.0, 1.2, 1.5, 2.2, 3.0, 4.0])
        epsilon: Small constant for numerical stability

    Returns:
        Weighted MSE loss with bin-specific emphasis
    """
    if bin_thresholds is None:
        bin_thresholds = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0]
    if bin_weights is None:
        bin_weights = [15.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0] if focus_on_low_waves else [0.9, 1.0, 1.2, 1.5, 2.2, 3.0, 4.0]

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


def masked_mse_perceptual_loss(y_pred, y_true, mask, perceptual_loss, lambda_perceptual=0.05):
    """
    Masked MSE + perceptual loss combination.

    Combines pixel-wise MSE with perceptual features extracted from a pretrained network.

    Args:
        y_pred: (B, 1, H, W) normalized model prediction
        y_true: (B, 1, H, W) normalized target
        mask: (B, 1, H, W) bool mask of valid pixels
        perceptual_loss: PerceptualLoss module instance
        lambda_perceptual: Weight for perceptual loss component (default: 0.05)

    Returns:
        Combined MSE + perceptual loss
    """
    mse_loss = masked_mse_loss(y_pred, y_true, mask)
    perc_loss = perceptual_loss(y_pred, y_true)
    return mse_loss + lambda_perceptual * perc_loss


def masked_mse_ssim_loss(y_pred, y_true, mask, ssim_loss, lambda_ssim=0.1):
    """
    Masked MSE + SSIM loss combination.

    Combines pixel-wise MSE with structural similarity index (SSIM) for better
    perceptual quality in predictions.

    Args:
        y_pred: (B, 1, H, W) normalized model prediction
        y_true: (B, 1, H, W) normalized target
        mask: (B, 1, H, W) bool mask of valid pixels
        ssim_loss: SSIMLoss module instance
        lambda_ssim: Weight for SSIM loss component (default: 0.1)

    Returns:
        Combined MSE + SSIM loss
    """
    return masked_mse_loss(y_pred, y_true, mask) + lambda_ssim * ssim_loss


def masked_mse_mdn_loss(pi, mu, sigma, y, mask=None, eps=1e-6, lambda_mse=0.1, lambda_nll=1.0):
    """
    Masked MSE + MDN (Mixture Density Network) NLL loss.

    Combines MSE on the expected value with negative log-likelihood for the full
    mixture distribution, useful for uncertainty-aware predictions.

    Args:
        pi: (B, K, H, W) mixture weights (K components)
        mu: (B, K, H, W) mixture means
        sigma: (B, K, H, W) mixture standard deviations
        y: (B, 1, H, W) target values
        mask: (B, 1, H, W) bool mask of valid pixels (1=ocean, 0=land), optional
        eps: Small constant for numerical stability
        lambda_mse: Weight for MSE loss component (default: 0.1)
        lambda_nll: Weight for NLL loss component (default: 1.0)

    Returns:
        Combined MSE + MDN NLL loss
    """
    # Import here to avoid circular dependency
    from src.commons.losses import mdn_nll_loss

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
        print("WARNING: NaN in combined MDN loss!")
        print(f"  mse_term: {mse_term.item()}, nll_term: {nll_term.item()}")
        return torch.tensor(0.0, device=pi.device, requires_grad=True)

    return total_loss
