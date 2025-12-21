import torch
import torch.nn as nn
import lightning as pl
from src.classifiers.networks.trans_unet import TransUNetGeo

class PatchDiscriminator(nn.Module):
    """
    Lightweight 2D PatchGAN discriminator for geophysical fields.
    Input:  predicted or real fields [B, 1, H, W]
    Output: patch scores [B, 1, H/8, W/8] (approx)
    """

    def __init__(self, in_channels: int = 1, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class WaveTransUNetGAN(pl.LightningModule):
    """
    Lightning module wrapping:
      - TransUNetGeo (G)
      - PatchDiscriminator (D)

    Assumes batch format:
      X:      [B, C_in, H, W]
      y_true: [B, 1,    H, W]
      mask:   [B, 1,    H, W]  (1=ocean, 0=land)
      (extra fields in batch are ignored)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        use_mdn: bool = False,
        base_channels: int = 64,
        bottleneck_dim: int = 1024,
        patch_size: int = 16,
        num_layers: int = 8,

    ):
        super().__init__()
        self.save_hyperparameters()

        self.G = TransUNetGeo(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, bottleneck_dim=bottleneck_dim, patch_size=patch_size, num_layers=num_layers, use_mdn=use_mdn)
        self.D = PatchDiscriminator(in_channels=out_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.G(X)
