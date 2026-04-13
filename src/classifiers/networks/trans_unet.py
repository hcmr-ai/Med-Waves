import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classifiers.networks.mdn import MDNHead

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
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
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

    def __init__(
        self,
        in_channels,
        emb_dim=1024,
        patch_size=16,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4.0,
        use_coord_pos_enc=True,
        sea_mask_channel_index=None,
    ):
        super().__init__()

        self.use_coord_pos_enc = use_coord_pos_enc
        self.sea_mask_channel_index = sea_mask_channel_index
        self.patch = nn.Conv2d(
            in_channels + (2 if use_coord_pos_enc else 0),
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.emb_dim = emb_dim
        self.patch_size = patch_size
        # Lightweight local positional bias on the patch grid.
        self.pos_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim)

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
        B, _, H, W = x.shape
        x_raw = x

        if self.use_coord_pos_enc:
            y_coords = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
            yy = y_coords.view(1, 1, H, 1).expand(B, 1, H, W)
            xx = x_coords.view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([x, yy, xx], dim=1)

        # Patch embedding → [B, emb_dim, H/ps, W/ps]
        x = self.patch(x)

        if (
            self.sea_mask_channel_index is not None
            and 0 <= self.sea_mask_channel_index < x_raw.shape[1]
        ):
            sea_mask = x_raw[:, self.sea_mask_channel_index : self.sea_mask_channel_index + 1]
            sea_mask = torch.nan_to_num(sea_mask, nan=0.0, posinf=0.0, neginf=0.0)
            sea_mask = sea_mask.clamp(0.0, 1.0)
            sea_mask = F.avg_pool2d(
                sea_mask, kernel_size=self.patch_size, stride=self.patch_size
            )
            if sea_mask.shape[-2:] != x.shape[-2:]:
                sea_mask = F.interpolate(
                    sea_mask, size=x.shape[-2:], mode="nearest"
                )
            # Keep a small floor so sparse sea tiles do not collapse token gradients.
            x = x * (0.1 + 0.9 * sea_mask)

        x = x + self.pos_conv(x)
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

    Supports multi-task learning with auxiliary tasks (e.g., predicting VHM0 and VTM02).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (deprecated for multi-task, kept for compatibility)
        auxiliary_tasks: List of task names (e.g., ['vhm0', 'vtm02']).
                        Defaults to ['vhm0'] for single-task learning.
        base_channels: Base number of channels in encoder/decoder
        bottleneck_dim: Dimension of bottleneck/transformer embeddings
        patch_size: Patch size for transformer tokenization
        num_layers: Number of transformer encoder layers
        use_mdn: Whether to use Mixture Density Network heads

    Returns:
        Single task: Returns tensor of shape [B, 1, H, W]
        Multi-task: Returns dict {'task_name': tensor} where each tensor is [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        auxiliary_tasks=None,  # List of task names: ['vhm0', 'vtm02']
        base_channels=64,
        bottleneck_dim=1024,
        patch_size=16,  # must match CNN bottleneck size!
        num_layers=8,
        num_heads=8,
        use_mdn=False,
        transformer_use_coord_pos_enc=True,
        transformer_sea_mask_channel_index=None,
    ):
        super().__init__()

        # Multi-task setup
        self.auxiliary_tasks = auxiliary_tasks or ["vhm0"]  # Default single task
        self.use_mdn = use_mdn

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
            num_heads=num_heads,
            use_coord_pos_enc=transformer_use_coord_pos_enc,
            sea_mask_channel_index=transformer_sea_mask_channel_index,
        )

        # Fusion conv
        self.fuse = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, 1)

        # Decoder
        self.u4 = UpBlock(bottleneck_dim, c4, c4)
        self.u3 = UpBlock(c4, c3, c3)
        self.u2 = UpBlock(c3, c2, c2)
        self.u1 = UpBlock(c2, c1, c1)

        # Multi-task prediction heads
        self.task_heads = nn.ModuleDict()
        for task in self.auxiliary_tasks:
            if use_mdn:
                # MDN head for each task
                self.task_heads[task] = MDNHead(c1, K=3)
            else:
                # Simple 1x1 conv for task-specific prediction
                # Efficient for related tasks like VHM0 and VTM02
                self.task_heads[task] = nn.Conv2d(c1, 1, kernel_size=1)

    def extract_features(self, x):
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
        if b_trans.shape[-2:] != b_cnn.shape[-2:]:
            b_trans = F.interpolate(
                b_trans, size=b_cnn.shape[-2:], mode="bilinear", align_corners=False
            )

        # Fuse
        b = torch.cat([b_cnn, b_trans], 1)
        b = self.fuse(b)

        # Decode
        u4 = self.u4(b, x4)
        u3 = self.u3(u4, x3)
        u2 = self.u2(u3, x2)
        u1 = self.u1(u2, x1)  # Shared features: [B, c1, H, W]
        return u1

    def forward(self, x):
        u1 = self.extract_features(x)
        # Multi-task predictions
        outputs = {}
        for task in self.auxiliary_tasks:
            outputs[task] = self.task_heads[task](u1)

        # Backward compatibility: single task returns tensor, not dict
        if len(self.auxiliary_tasks) == 1:
            return outputs[self.auxiliary_tasks[0]]

        return outputs


class MoETransUNetGeo(nn.Module):
    """
    Sea-state-aware Mixture-of-Experts wrapper around a shared TransUNet backbone.

    The model predicts multiple candidate bias fields and a spatial gate, then returns
    one mixed prediction:
        prediction = sum_k gate_weight_k * expert_output_k

    Auxiliary gate tensors are returned for diagnostics and optional regularization.
    """

    def __init__(
        self,
        in_channels,
        out_channels=1,
        auxiliary_tasks=None,
        base_channels=64,
        bottleneck_dim=1024,
        patch_size=16,
        num_layers=8,
        num_heads=8,
        num_experts=3,
        gate_temperature=1.0,
        gate_input_mode="features",
        vhm0_channel_index=0,
        expert_dropout=0.0,
        return_gate_maps=True,
        transformer_use_coord_pos_enc=True,
        transformer_sea_mask_channel_index=None,
    ):
        super().__init__()
        self.auxiliary_tasks = auxiliary_tasks or ["vhm0"]
        if len(self.auxiliary_tasks) != 1:
            raise ValueError("MoETransUNetGeo currently supports single-task training only.")
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2 for MoETransUNetGeo.")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be > 0.")
        if gate_input_mode not in {"features", "vhm0_only"}:
            raise ValueError("gate_input_mode must be 'features' or 'vhm0_only'.")

        self.num_experts = int(num_experts)
        self.gate_temperature = float(gate_temperature)
        self.gate_input_mode = gate_input_mode
        self.return_gate_maps = return_gate_maps
        self.vhm0_channel_index = int(vhm0_channel_index)

        self.backbone = TransUNetGeo(
            in_channels=in_channels,
            out_channels=out_channels,
            auxiliary_tasks=self.auxiliary_tasks,
            base_channels=base_channels,
            bottleneck_dim=bottleneck_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            use_mdn=False,
            transformer_use_coord_pos_enc=transformer_use_coord_pos_enc,
            transformer_sea_mask_channel_index=transformer_sea_mask_channel_index,
        )
        feature_channels = base_channels

        expert_layers = []
        for _ in range(self.num_experts):
            if expert_dropout > 0:
                expert_layers.append(
                    nn.Sequential(
                        nn.Dropout2d(expert_dropout),
                        nn.Conv2d(feature_channels, out_channels, kernel_size=1),
                    )
                )
            else:
                expert_layers.append(nn.Conv2d(feature_channels, out_channels, kernel_size=1))
        self.expert_heads = nn.ModuleList(expert_layers)

        if self.gate_input_mode == "features":
            self.gate_head = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_channels, self.num_experts, kernel_size=1),
            )
        else:
            self.gate_head = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.num_experts, kernel_size=1),
            )

    def forward(self, x):
        features = self.backbone.extract_features(x)
        expert_outputs = torch.stack(
            [head(features) for head in self.expert_heads], dim=1
        )  # [B, K, 1, H, W]

        if self.gate_input_mode == "vhm0_only":
            gate_input = x[:, self.vhm0_channel_index : self.vhm0_channel_index + 1]
            gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=0.0, neginf=0.0)
            if gate_input.shape[-2:] != features.shape[-2:]:
                gate_input = F.interpolate(
                    gate_input, size=features.shape[-2:], mode="bilinear", align_corners=False
                )
            gate_logits = self.gate_head(gate_input)
        else:
            gate_logits = self.gate_head(features)

        gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=1)
        prediction = (expert_outputs * gate_weights.unsqueeze(2)).sum(dim=1)

        if not self.return_gate_maps:
            return prediction

        return {
            "prediction": prediction,
            "expert_outputs": expert_outputs,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
        }


class SimpleMLPGeo(nn.Module):
    """
    Lightweight per-pixel MLP baseline for geospatial fields.

    Applies the same MLP to each pixel independently (shared weights across space),
    preserving spatial resolution while keeping model complexity low.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (kept for compatibility)
        auxiliary_tasks: Task names, e.g. ['vhm0', 'vtm02']
        hidden_dim: Hidden width of MLP layers
        num_layers: Number of Linear+ReLU blocks
        dropout: Dropout probability applied after hidden activations
        use_mdn: Whether to use MDN heads per task

    Returns:
        Single task: tensor [B, 1, H, W]
        Multi-task: dict {'task_name': tensor [B, 1, H, W]}
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        auxiliary_tasks=None,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        use_mdn=False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.auxiliary_tasks = auxiliary_tasks or ["vhm0"]
        self.use_mdn = use_mdn

        layers = []
        in_dim = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Task-specific heads over shared feature map
        self.task_heads = nn.ModuleDict()
        for task in self.auxiliary_tasks:
            if use_mdn:
                self.task_heads[task] = MDNHead(hidden_dim, K=3)
            else:
                self.task_heads[task] = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(x)  # [B, H, W, hidden_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, hidden_dim, H, W]

        outputs = {}
        for task in self.auxiliary_tasks:
            outputs[task] = self.task_heads[task](x)

        if len(self.auxiliary_tasks) == 1:
            return outputs[self.auxiliary_tasks[0]]

        return outputs


# -----------------------------
# Sanity check on random sizes
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Single-Task TransUNetGeo")
    print("=" * 60)
    model_single = TransUNetGeo(
        in_channels=8,
        out_channels=1,
        auxiliary_tasks=["vhm0"],  # Single task
        patch_size=8,
        base_channels=32,
    )

    for H, W in [(64, 64), (76, 261), (128, 128)]:
        x = torch.randn(2, 8, H, W)
        y = model_single(x)
        print(f"Input: {x.shape} → Output: {y.shape}")

    print("\n" + "=" * 60)
    print("Testing Multi-Task TransUNetGeo (VHM0 + VTM02)")
    print("=" * 60)
    model_multi = TransUNetGeo(
        in_channels=8,
        out_channels=1,
        auxiliary_tasks=["vhm0", "vtm02"],  # Multi-task
        patch_size=8,
        base_channels=32,
    )

    for H, W in [(64, 64), (76, 261)]:
        x = torch.randn(2, 8, H, W)
        outputs = model_multi(x)
        print(f"Input: {x.shape}")
        for task_name, task_output in outputs.items():
            print(f"  → {task_name}: {task_output.shape}")

    print("\n" + "=" * 60)
    print("Testing with MDN heads")
    print("=" * 60)
    model_mdn = TransUNetGeo(
        in_channels=8,
        out_channels=1,
        auxiliary_tasks=["vhm0", "vtm02"],
        patch_size=8,
        base_channels=32,
        use_mdn=True,
    )

    x = torch.randn(2, 8, 64, 64)
    outputs = model_mdn(x)
    print(f"Input: {x.shape}")
    for task_name, (pi, mu, sigma) in outputs.items():
        print(f"  → {task_name}: pi={pi.shape}, mu={mu.shape}, sigma={sigma.shape}")
