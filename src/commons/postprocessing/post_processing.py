"""
Post-processing utilities for model predictions.

This module contains functions for bias correction, filtering, and other
post-processing operations on model outputs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def apply_binwise_correction(
    y_pred,
    y_true,
    vhm0,
    bins: List[float] = None,
    normalize: bool = False,
    std: float = None,
    mean: float = None,
    mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
    """
    Apply post-hoc bin-wise bias correction based on VHM0 (true or predicted).

    Args:
        y_pred: (B, 1, H, W) torch.Tensor, model predictions (normalized or real)
        y_true: (B, 1, H, W) torch.Tensor, ground truth (same units as y_pred)
        vhm0:   (B, 1, H, W) torch.Tensor, original wave height (same units as y_true)
        bins:   List of bin edges in meters for binning vhm0
        normalize: If True, assumes y_pred and y_true are normalized
        std, mean: If normalize=True, std and mean should be provided
        mask: (B, 1, H, W) optional boolean mask of valid sea pixels

    Returns:
        y_corr: (B, 1, H, W) bias-corrected predictions
        bin_biases: dict of {bin_label: bias_value}
        bin_counts: dict of {bin_label: sample_count}
    """

    # 1. Unnormalize if needed
    if bins is None:
        bins = [0, 1, 2, 3, 4, 5, 10]
    if normalize:
        assert std is not None and mean is not None, "Need mean/std to unnormalize"
        y_pred_denorm = y_pred * std + mean
        y_true_denorm = y_true * std + mean
        vhm0_denorm = vhm0 * std + mean
    else:
        y_pred_denorm, y_true_denorm, vhm0_denorm = y_pred, y_true, vhm0

    # 2. Prepare mask
    if mask is None:
        mask = ~torch.isnan(y_true_denorm)

    # 3. Compute residuals (true - pred)
    residuals = y_true_denorm - y_pred_denorm

    # 4. Compute mean bias per bin
    bin_biases = {}
    bin_counts = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_label = f"{low}-{high}m"

        in_bin = (vhm0_denorm >= low) & (vhm0_denorm < high) & mask
        if in_bin.any():
            bin_bias = residuals[in_bin].mean().item()
            bin_count = in_bin.sum().item()
        else:
            bin_bias = 0.0  # no data in this bin → no correction
            bin_count = 0
        bin_biases[bin_label] = bin_bias
        bin_counts[bin_label] = bin_count

    # 5. Apply correction
    y_corr_denorm = y_pred_denorm.clone()
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_label = f"{low}-{high}m"
        in_bin = (vhm0_denorm >= low) & (vhm0_denorm < high) & mask
        y_corr_denorm[in_bin] += bin_biases[bin_label]

    # 6. Normalize back if needed
    if normalize:
        y_corr = (y_corr_denorm - mean) / std
    else:
        y_corr = y_corr_denorm

    return y_corr, bin_biases, bin_counts


def compute_global_bin_biases(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    bins: List[float],
    predict_bias: bool,
    normalize_target: bool,
    normalizer: Optional[object],
    unit: str = "m",
    task_name: str = "vhm0",
) -> Dict[str, float]:
    """
    Compute global bin-wise correction biases from training/validation set.

    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for training/validation data
        device: Device to run computations on ('cuda' or 'cpu')
        bins: List of bin edges in meters
        predict_bias: Whether model predicts bias
        normalize_target: Whether target is normalized
        normalizer: Normalizer object with inverse_transform_torch method
        unit: Unit string for printing (default: 'm')
        task_name: Task name for multi-task models (default: 'vhm0')

    Returns:
        Dictionary mapping bin labels to computed bias values
    """
    print(
        "Computing global bin-wise correction biases from training/validation set..."
    )

    all_residuals_by_bin = {}

    # Initialize storage for each bin
    for i in range(len(bins) - 1):
        bin_label = f"{bins[i]}-{bins[i + 1]}m"
        all_residuals_by_bin[bin_label] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing biases from train/val"):
            # Unpack batch
            X, y, mask, vhm0_batch = batch
            vhm0 = vhm0_batch.to(device) if vhm0_batch is not None else None
            
            # Handle multi-task vs single-task format
            # If y is a dict (multi-task), extract the target for the task we're evaluating
            if isinstance(y, dict):
                y = y[task_name]

            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            # Get predictions
            y_pred = model(X)
            
            # Handle multi-task predictions
            # If y_pred is a dict (multi-task model), extract the prediction for the task
            if isinstance(y_pred, dict):
                y_pred = y_pred[task_name]

            # Align dimensions
            min_h = min(y_pred.shape[2], y.shape[2])
            min_w = min(y_pred.shape[3], y.shape[3])
            y_pred = y_pred[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]
            mask = mask[:, :, :min_h, :min_w]
            if vhm0 is not None:
                vhm0 = vhm0[:, :, :min_h, :min_w]

            # Unnormalize
            if normalize_target and normalizer is not None:
                y_pred = normalizer.inverse_transform_torch(y_pred)
                y = normalizer.inverse_transform_torch(y)

            # Compute residuals and bin them
            residuals = y - y_pred

            for i in range(len(bins) - 1):
                low, high = bins[i], bins[i + 1]
                bin_label = f"{low}-{high}m"
                in_bin = (vhm0 >= low) & (vhm0 < high) & mask
                if in_bin.any():
                    all_residuals_by_bin[bin_label].append(residuals[in_bin].cpu())

    # Compute global bin biases
    global_bin_biases = {}
    print("\nGlobal bin-wise correction biases (from train/val set):")
    print(f"{'Bin':<12} {'Count':<15} {'Bias ({unit})':<12}")
    print("-" * 39)

    for bin_label, residual_list in all_residuals_by_bin.items():
        if residual_list:
            all_residuals = torch.cat(residual_list)
            global_bin_biases[bin_label] = all_residuals.mean().item()
            bin_count = len(all_residuals)
            print(
                f"{bin_label:<12} {bin_count:<15,} {global_bin_biases[bin_label]:>10.4f}"
            )
        else:
            global_bin_biases[bin_label] = 0.0

    return global_bin_biases


def apply_bin_corrections(
    y_pred: torch.Tensor,
    vhm0: torch.Tensor,
    mask: torch.Tensor,
    bins: List[float],
    global_bin_biases: Dict[str, float],
) -> torch.Tensor:
    """
    Apply pre-computed global bin-wise corrections to predictions.

    Args:
        y_pred: Predictions tensor [B, 1, H, W]
        vhm0: Wave height tensor [B, 1, H, W]
        mask: Boolean mask [B, 1, H, W]
        bins: List of bin edges in meters
        global_bin_biases: Dictionary of bin-wise bias corrections

    Returns:
        Corrected predictions tensor [B, 1, H, W]
    """
    y_pred_corrected = y_pred.clone()
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_label = f"{low}-{high}m"
        in_bin = (vhm0 >= low) & (vhm0 < high) & mask
        if in_bin.any() and bin_label in global_bin_biases:
            y_pred_corrected[in_bin] += global_bin_biases[bin_label]
    return y_pred_corrected


def apply_bilateral_filter(
    predictions: torch.Tensor,
    mask: torch.Tensor,
    d: int = 5,
    sigma_color: float = 0.3,
    sigma_space: int = 5,
) -> torch.Tensor:
    """
    Apply bilateral filter to smooth extreme predictions while preserving edges.

    The bilateral filter is a non-linear, edge-preserving smoothing filter that
    combines spatial and intensity information to reduce noise while maintaining
    sharp transitions at edges.

    Args:
        predictions: [B, 1, H, W] tensor of predictions
        mask: [B, 1, H, W] boolean mask (sea pixels)
        d: Diameter of pixel neighborhood (default: 5)
        sigma_color: Filter sigma in value space, wave height diff tolerance (default: 0.3)
        sigma_space: Filter sigma in coordinate space, spatial distance (default: 5)

    Returns:
        Filtered predictions [B, 1, H, W]
    """
    import cv2

    filtered = torch.zeros_like(predictions)

    for i in range(predictions.shape[0]):
        pred_np = predictions[i, 0].cpu().numpy()
        mask_np = mask[i, 0].cpu().numpy()

        # Only filter sea pixels
        pred_filtered = pred_np.copy()

        # Apply bilateral filter (only on valid data)
        if mask_np.sum() > 0:
            pred_filtered = cv2.bilateralFilter(
                pred_np.astype(np.float32),
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )

            # Keep land pixels unchanged
            pred_filtered[~mask_np] = pred_np[~mask_np]

        filtered[i, 0] = torch.from_numpy(pred_filtered).to(predictions.device)

    return filtered
