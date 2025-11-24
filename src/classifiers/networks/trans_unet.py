import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Basic Conv Blocks
# -------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock + Strided conv (downsample)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.down(x)
        return x, skip


class DualUp(nn.Module):
    """Dual upsampling: bilinear + pixelshuffle."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid * 4, 3, padding=1),
            nn.BatchNorm2d(mid * 4),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b1 = F.interpolate(b1, scale_factor=2, mode="bilinear", align_corners=False)

        b2 = self.branch2(x)
        b2 = F.pixel_shuffle(b2, 2)

        x = torch.cat([b1, b2], dim=1)
        return self.fuse(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = DualUp(in_ch, out_ch)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Align shapes (crop if necessary)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # H, W = x.shape[-2:]
            # skip = skip[..., :H, :W]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -------------------------
# Size-agnostic Transformer Branch
# -------------------------

class TransformerBranch(nn.Module):
    """
    Patch embedding on arbitrary H×W:
        Conv2d(kernel_size=patch, stride=patch)
    Produces:
        [B, emb_dim, H/patch, W/patch]
    Flatten to N tokens:
        [B, N, emb_dim]
    Transformer works on variable N.
    """
    def __init__(self,
                 in_channels,
                 emb_dim=1024,
                 patch_size=16,
                 num_layers=6,
                 num_heads=8,
                 mlp_ratio=4.0):
        super().__init__()

        self.patch = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.emb_dim = emb_dim
        self.patch_size = patch_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding → [B, emb_dim, H/ps, W/ps]
        x = self.patch(x)
        Hp, Wp = x.shape[-2:]

        # Flatten tokens → [B, Hp*Wp, emb_dim]
        x = x.flatten(2).transpose(1, 2)

        # Transformer
        x = self.encoder(x)
        x = self.norm(x)

        # Restore spatial grid
        x = x.transpose(1, 2).view(B, self.emb_dim, Hp, Wp)
        return x


# -------------------------
# Full Size-Agnostic TransUNet
# -------------------------

class TransUNetGeo(nn.Module):
    """
    TransUNet as in:
    "AI-based Correction of Wave Forecasts Using the Transformer-enhanced UNet Model"
    (Cao et al., 2025)

    - Encoder: UNet-style conv + conv-stride2 downsampling, no max-pool.
    - Parallel Transformer branch on the raw input.
    - Bottleneck fusion at 1024×4×4.
    - Decoder: dual-sampling upsampling + skip connections.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 bottleneck_dim=1024,
                 patch_size=16,   # must match CNN bottleneck size!
                 num_layers=8):
        super().__init__()

        # Encoder channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Encoder
        self.d1 = DownBlock(in_channels, c1)
        self.d2 = DownBlock(c1, c2)
        self.d3 = DownBlock(c2, c3)
        self.d4 = DownBlock(c3, c4)

        # Bottleneck
        self.bottleneck = ConvBlock(c4, bottleneck_dim)

        # Transformer on raw input
        self.transformer = TransformerBranch(
            in_channels=in_channels,
            emb_dim=bottleneck_dim,
            patch_size=patch_size,
            num_layers=num_layers,
        )

        # Fusion conv
        self.fuse = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, 1)

        # Decoder
        self.u4 = UpBlock(bottleneck_dim, c4, c4)
        self.u3 = UpBlock(c4, c3, c3)
        self.u2 = UpBlock(c3, c2, c2)
        self.u1 = UpBlock(c2, c1, c1)

        self.final = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1d, x1 = self.d1(x)  # H/2
        x2d, x2 = self.d2(x1d)  # H/4
        x3d, x3 = self.d3(x2d)  # H/8
        x4d, x4 = self.d4(x3d)  # H/16

        # CNN bottleneck
        b_cnn = self.bottleneck(x4d)

        # Transformer branch (patch_size must match H/16, W/16 or we crop)
        b_trans = self.transformer(x)

        # Align shapes
        # Ht, Wt = b_trans.shape[-2:]
        # Hc, Wc = b_cnn.shape[-2:]
        # H = min(Ht, Hc)
        # W = min(Wt, Wc)
        # b_trans = b_trans[..., :H, :W]
        # b_cnn = b_cnn[..., :H, :W]
        if b_trans.shape[-2:] != b_cnn.shape[-2:]:
            b_trans = F.interpolate(b_trans, size=b_cnn.shape[-2:], mode='bilinear', align_corners=False)

        # Fuse
        b = torch.cat([b_cnn, b_trans], 1)
        b = self.fuse(b)

        # Decode
        u4 = self.u4(b, x4)
        u3 = self.u3(u4, x3)
        u2 = self.u2(u3, x2)
        u1 = self.u1(u2, x1)

        return self.final(u1)


# -----------------------------
# Sanity check on random sizes
# -----------------------------
if __name__ == "__main__":
    model = TransUNetGeo(
        in_channels=8,
        out_channels=1,
        patch_size=8,
        base_channels=32,
    )

    for H, W in [(64, 64), (76, 261), (128,128), (64, 96)]:
        x = torch.randn(2, 8, H, W)
        y = model(x)
        print(f"Input: {x.shape} → Output: {y.shape}")
