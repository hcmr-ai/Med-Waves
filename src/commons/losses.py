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
