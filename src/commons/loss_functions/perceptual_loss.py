import torch
import torch.nn as nn
import torch.nn.functional as F

from src.commons.loss_functions.mse_loss import masked_mse_loss


class WaveFeatureExtractor(nn.Module):
    """
    Tiny CNN to extract multi-scale features from 2D wave fields.
    Designed for 1-channel targets (e.g. VHM0 or bias), but can take more.
    """
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()

        # Block 1: HxW -> HxW
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Block 2: HxW -> H/2 x W/2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Block 3: H/2 x W/2 -> H/4 x W/4
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns list of feature maps at 3 scales
        """
        f1 = self.block1(x)      # [B, C,   H,   W]
        f2 = self.block2(f1)     # [B, 2C,  H/2, W/2]
        f3 = self.block3(f2)     # [B, 4C,  H/4, W/4]
        return [f1, f2, f3]


class PerceptualLoss(nn.Module):
    """
    Perceptual / feature loss for 2D fields.
    Uses a fixed feature extractor (by default, gradients DO NOT flow into it).
    """
    def __init__(self, feature_extractor: nn.Module, layer_weights=None, detach_target=True):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.detach_target = detach_target

        # Set to eval mode (disables dropout, batchnorm updates, etc.)
        self.feature_extractor.eval()

        # default equal weight for each feature map
        self.layer_weights = layer_weights

    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W] (or [B, C, H, W] if you want)
        returns scalar perceptual loss
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            feats_pred_no_grad = self.feature_extractor(pred)

        # Re-enable gradients by creating new tensors that require grad
        feats_pred = [f.detach().requires_grad_(True) for f in feats_pred_no_grad]

        # Extract features from target - no gradients needed
        with torch.no_grad():
            feats_tgt = self.feature_extractor(target)

        if self.layer_weights is None:
            weights = [1.0] * len(feats_pred)
        else:
            weights = self.layer_weights
            assert len(weights) == len(feats_pred), "layer_weights length mismatch"

        loss = 0.0
        for w, fp, ft in zip(weights, feats_pred, feats_tgt, strict=False):
            # feature-wise MSE, normalized by number of elements
            loss_feat = F.mse_loss(fp, ft)
            loss = loss + w * loss_feat

        return loss


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
