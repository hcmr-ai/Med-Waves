import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for 2D fields.
    SSIM is computed patch-wise and averaged spatially.
    Output loss = 1 - SSIM (so lower is better).
    """

    def __init__(self, window_size=11, sigma=1.5, data_range=None):
        """
        data_range: max_value - min_value of ground truth.
                    If None, it is computed dynamically.
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

        # Create fixed Gaussian kernel
        self.window = self.create_gaussian_window(window_size, sigma)

    def create_gaussian_window(self, size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window_2d = g[:, None] * g[None, :]
        return window_2d

    def _ssim(self, x, y, window, data_range, eps=1e-6):
        # x, y: [B, 1, H, W]
        B, C, H, W = x.shape
        window = window.to(x.device).to(x.dtype)
        window = window.expand(C, 1, self.window_size, self.window_size)

        # Means
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        # Variances & Covariance
        sigma_x2 = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy

        # Stability constants (from SSIM paper)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # SSIM formula
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

        return ssim_map.mean()

    def forward(self, pred, target, mask=None):
        """
        pred, target: [B, 1, H, W] (no NaNs!)
        mask: optional [B, 1, H, W], 1=ocean, 0=land
        """

        # Compute dynamic data range if not provided
        if self.data_range is None:
            data_range = target.max().item() - target.min().item()
        else:
            data_range = self.data_range

        # Apply mask (optional)
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # Compute SSIM
        ssim_val = self._ssim(pred, target, self.window, data_range)

        # Convert to loss
        return 1 - ssim_val
