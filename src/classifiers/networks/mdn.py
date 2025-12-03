
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNHead(nn.Module):
    """
    MDN output head for 2D fields.
    Outputs mixture weights π, means μ, and std σ for each pixel.

    Given final feature map [B, C, H, W], outputs:
      π:     [B, K, H, W]
      μ:     [B, K, H, W]
      σ:     [B, K, H, W]

    where K is # mixture components.
    """

    def __init__(self, in_channels, K=3):
        super().__init__()
        self.K = K
        # One conv outputs [K π, K μ, K σ_raw]
        self.mdn_conv = nn.Conv2d(in_channels, 3 * K, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns pi, mu, sigma
        """
        mdn_out = self.mdn_conv(x)                  # [B, 3K, H, W]
        B, C, H, W = mdn_out.shape

        # Split into mixture components
        pi, mu, sigma_raw = torch.split(mdn_out, self.K, dim=1)

        # Mixture weights: softmax over K
        pi = F.softmax(pi, dim=1)                   # [B, K, H, W]

        # Sigma: softplus for stability
        sigma = F.softplus(sigma_raw) + 1e-3        # positive, stable

        return pi, mu, sigma

# Helper functions for MDN inference
def mdn_sample(pi, mu, sigma):
    """
    Sample from MDN mixture.
    """
    # Choose component per pixel
    comp = torch.distributions.Categorical(pi.permute(0,2,3,1)).sample()
    comp = comp.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    # Gather selected component parameters
    mu_sel = torch.gather(mu, 1, comp)
    sigma_sel = torch.gather(sigma, 1, comp)

    # Sample from chosen Gaussian
    eps = torch.randn_like(mu_sel)
    return mu_sel + sigma_sel * eps

def mdn_expected_value(pi, mu):
    """
    pi: [B, K, H, W]
    mu: [B, K, H, W]

    returns y_pred: [B, 1, H, W]
    """
    y_pred = (pi * mu).sum(dim=1, keepdim=True)
    return y_pred
