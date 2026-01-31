

from src.commons.loss_functions.huber_loss import masked_huber_loss
from src.commons.loss_functions.l1_loss import (
    masked_multi_bin_weighted_smooth_l1,
    masked_smooth_l1_loss,
)
from src.commons.loss_functions.mdn_nll_loss import mdn_nll_loss
from src.commons.loss_functions.mse_loss import (
    masked_mse_loss,
    masked_mse_mdn_loss,
    masked_mse_perceptual_loss,
    masked_mse_ssim_loss,
    masked_multi_bin_weighted_mse,
    masked_weighted_mse,
)
from src.commons.loss_functions.perceptual_loss import masked_ssim_perceptual_loss
from src.commons.loss_functions.pixel_switch_loss import pixel_switch_loss_stable


def compute_loss(
    loss_type: str,
    y_pred,
    y_true,
    mask,
    vhm0_for_reconstruction=None,
    pi=None,
    mu=None,
    sigma=None,
    criterion=None,
    pixel_switch_threshold_m=None,
    perceptual_loss=None,
    ssim_loss=None,
):
    """
    Unified loss computation wrapper that selects and applies the appropriate loss function.

    Args:
        loss_type: Type of loss function to use
        y_pred: Predicted values
        y_true: Ground truth values
        mask: Valid pixel mask
        vhm0_for_reconstruction: VHM0 values for reconstruction-based losses (optional)
        pi: MDN mixture weights (optional, for MDN losses)
        mu: MDN mixture means (optional, for MDN losses)
        sigma: MDN mixture standard deviations (optional, for MDN losses)
        criterion: Pre-initialized loss criterion (e.g., SmoothL1Loss) (optional)
        pixel_switch_threshold_m: Threshold for pixel switch loss (optional)
        perceptual_loss: Perceptual loss module (optional)
        ssim_loss: SSIM loss module (optional)

    Returns:
        Computed loss tensor

    Raises:
        ValueError: If an unsupported loss_type is provided
    """
    if loss_type == "mse":
        return masked_mse_loss(y_pred, y_true, mask)

    elif loss_type == "smooth_l1":
        if criterion is None:
            raise ValueError("smooth_l1 loss requires a criterion (SmoothL1Loss)")
        return masked_smooth_l1_loss(y_pred, y_true, mask, criterion)

    elif loss_type == "weighted_mse":
        return masked_weighted_mse(y_pred, y_true, mask, threshold=6.0, high_weight=5.0, epsilon=1e-6)

    elif loss_type == "multi_bin_weighted_smooth_l1":
        if criterion is None:
            raise ValueError("multi_bin_weighted_smooth_l1 requires a criterion (SmoothL1Loss)")
        if vhm0_for_reconstruction is None:
            raise ValueError("multi_bin_weighted_smooth_l1 requires vhm0_for_reconstruction")
        return masked_multi_bin_weighted_smooth_l1(y_pred, y_true, mask, vhm0_for_reconstruction, criterion)

    elif loss_type == "pixel_switch_mse":
        if pixel_switch_threshold_m is None:
            raise ValueError("pixel_switch_mse requires pixel_switch_threshold_m")
        return pixel_switch_loss_stable(y_pred, y_true, mask, threshold_m=pixel_switch_threshold_m)

    elif loss_type == "mse_perceptual":
        if perceptual_loss is None:
            raise ValueError("mse_perceptual requires a perceptual_loss module")
        return masked_mse_perceptual_loss(y_pred, y_true, mask, perceptual_loss)

    elif loss_type == "mse_ssim":
        if ssim_loss is None:
            raise ValueError("mse_ssim requires an ssim_loss module")
        return masked_mse_ssim_loss(y_pred, y_true, mask, ssim_loss=ssim_loss)

    elif loss_type == "mse_ssim_perceptual":
        if ssim_loss is None or perceptual_loss is None:
            raise ValueError("mse_ssim_perceptual requires both ssim_loss and perceptual_loss modules")
        return masked_ssim_perceptual_loss(y_pred, y_true, mask, ssim_loss, perceptual_loss)

    elif loss_type == "mse_mdn":
        if pi is None or mu is None or sigma is None:
            raise ValueError("mse_mdn requires pi, mu, and sigma from MDN")
        return masked_mse_mdn_loss(pi, mu, sigma, y_true, mask, eps=1e-9, lambda_mse=0.1, lambda_nll=1.0)

    elif loss_type == "mdn":
        if pi is None or mu is None or sigma is None:
            raise ValueError("mdn requires pi, mu, and sigma from MDN")
        return mdn_nll_loss(pi, mu, sigma, y_true, mask, eps=1e-9)

    elif loss_type == "mse_gan":
        return masked_mse_loss(y_pred, y_true, mask)

    elif loss_type == "huber":
        return masked_huber_loss(y_pred, y_true, mask)

    elif loss_type == "multi_bin_weighted_mse":
        # Default fallback
        if vhm0_for_reconstruction is None:
            raise ValueError("multi_bin_weighted_mse requires vhm0_for_reconstruction")
        return masked_multi_bin_weighted_mse(y_pred, y_true, mask, vhm0_for_reconstruction)

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}. See compute_loss() docstring for supported types.")
