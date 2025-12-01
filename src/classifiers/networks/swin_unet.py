import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utility functions
# -------------------------

def window_partition(x, window_size):
    """
    x: [B, H, W, C]
    return: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1,
                                                             window_size,
                                                             window_size,
                                                             C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    windows: [num_windows*B, window_size, window_size, C]
    return: [B, H, W, C]
    """
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size, W // window_size,
                     window_size, window_size,
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def pad_to_swin_size(x, patch_size=4, window_size=4):
    """
    Pads input tensor to make H and W divisible by:
      - patch_size (from PatchEmbed)
      - window_size (from Swin blocks)

    Returns:
      x_pad: padded tensor
      pad: (pad_left, pad_right, pad_top, pad_bottom)
    """
    B, C, H, W = x.shape

    # Required multiples
    req = torch.lcm(torch.tensor(patch_size), torch.tensor(window_size)).item()

    H_pad = (req - H % req) % req
    W_pad = (req - W % req) % req

    pad_top = H_pad // 2
    pad_bottom = H_pad - pad_top
    pad_left = W_pad // 2
    pad_right = W_pad - pad_left

    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(x, pad):
    """
    Removes padding in reverse order.
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    H, W = x.shape[-2:]
    return x[..., pad_top:H - pad_bottom, pad_left:W - pad_right]



# -------------------------
# Swin blocks
# -------------------------

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (no shift).
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )  # [2Wh-1 * 2Ww-1, nH]

        # get pair-wise relative position index for each token inside a window
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        """
        x: [num_windows*B, N, C]  where N = window_size*window_size
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B_, nH, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size,
               self.window_size * self.window_size,
               -1)  # [N, N, nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, N, N]

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """
    Single Swin Transformer block (no shifted windows to keep it simpler).
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path == 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "Input feature has wrong size"

        # [B, H, W, C]
        x = x.view(B, H, W, C)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # [nW*B, w, w, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(self.norm1(x_windows))

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H, W, C]

        x = x.view(B, H * W, C)
        x = x + self.drop_path(x) * 0  # keep residual structure, no shift

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Downsampling: merge 2x2 patches, increasing channels by 4 and then projecting.
    """

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W

        x = x.view(B, H, W, C)

        # pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        H, W = x.shape[1], x.shape[2]

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)            # [B, H/2*W/2, 4C]
        x = self.norm(x)
        x = self.reduction(x)               # [B, H/2*W/2, 2C]
        return x


class PatchExpand(nn.Module):
    """
    Upsampling: expand spatial resolution by factor 2 using linear + pixel shuffle.
    """

    def __init__(self, input_resolution, dim, expand_dim=True):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand_dim = expand_dim

        out_dim = dim // 2 if expand_dim else dim
        self.expand = nn.Linear(dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W

        x = self.expand(x)  # [B, L, 4*C']
        x = x.view(B, H, W, 2, 2, -1)  # [B,H,W,2,2,C']
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B,H,2,W,2,C']
        x = x.view(B, H * 2, W * 2, -1)               # [B,2H,2W,C']

        H, W = H * 2, W * 2
        x = x.view(B, H * W, -1)
        x = self.norm(x)
        return x, (H, W)


class PatchEmbed(nn.Module):
    """
    Simple patch embedding: conv with stride=patch_size (usually 4).
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,C,H,W]
        x = self.proj(x)  # [B,embed_dim,H/ps,W/ps]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B,H*W,C]
        x = self.norm(x)
        return x, (H, W)


class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


# -------------------------
# Swin-UNet
# -------------------------

class SwinUNet(nn.Module):
    """
    Swin-UNet for 2D fields.

    Args:
        img_size:   tuple or int (e.g. 64 or (64,64))
        in_chans:   input channels (e.g. VHM0, U10, V10, WSPD,...)
        num_classes: output channels (e.g. 1 for SWH/bias)
    """

    def __init__(self,
                 img_size=64,
                 in_chans=2,
                 num_classes=1,
                 embed_dim=64,
                 depths=(2, 2, 2, 2),
                 num_heads=(2, 4, 8, 8),
                 window_size=4,
                 mlp_ratio=4.):
        super().__init__()

        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.depths = depths

        # 1) Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size=4,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.patches_resolution = patches_resolution

        # Encoder layers (Swin stages + patch merging)
        self.layer1 = self._make_layer(embed_dim,
                                       patches_resolution,
                                       depth=depths[0],
                                       num_heads=num_heads[0])
        self.down1 = PatchMerging(patches_resolution, embed_dim)

        res2 = (patches_resolution[0] // 2, patches_resolution[1] // 2)
        dim2 = embed_dim * 2
        self.layer2 = self._make_layer(dim2, res2,
                                       depth=depths[1],
                                       num_heads=num_heads[1])
        self.down2 = PatchMerging(res2, dim2)

        res3 = (res2[0] // 2, res2[1] // 2)
        dim3 = embed_dim * 4
        self.layer3 = self._make_layer(dim3, res3,
                                       depth=depths[2],
                                       num_heads=num_heads[2])
        self.down3 = PatchMerging(res3, dim3)

        res4 = (res3[0] // 2, res3[1] // 2)
        dim4 = embed_dim * 8
        self.layer4 = self._make_layer(dim4, res4,
                                       depth=depths[3],
                                       num_heads=num_heads[3])

        # Decoder: patch expand + Swin layers
        self.up3 = PatchExpand(res4, dim4, expand_dim=True)  # -> dim3
        self.dec_layer3 = self._make_layer(dim3, res3,
                                           depth=1,
                                           num_heads=num_heads[2])

        self.up2 = PatchExpand(res3, dim3, expand_dim=True)  # -> dim2
        self.dec_layer2 = self._make_layer(dim2, res2,
                                           depth=1,
                                           num_heads=num_heads[1])

        self.up1 = PatchExpand(res2, dim2, expand_dim=True)  # -> dim1
        self.dec_layer1 = self._make_layer(embed_dim, patches_resolution,
                                           depth=1,
                                           num_heads=num_heads[0])

        # Final upsample to full resolution if needed (4x from patch embed)
        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def _make_layer(self, dim, input_resolution, depth, num_heads):
        layers = []
        for _ in range(depth):
            layers.append(
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    mlp_ratio=4.,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, in_chans, H, W] with H,W matching img_size
        """
        B, C, H, W = x.shape
        # assert (H, W) == self.img_size, "For now, SwinUNet assumes fixed img_size."

        # Patch embed
        x, (H_p, W_p) = self.patch_embed(x)  # [B, H_p*W_p, embed_dim]
        # Encoder
        x1 = self.layer1(x)       # [B, H_p*W_p, dim1]
        x2 = self.down1(x1)       # [B, (H_p/2)*(W_p/2), dim2]

        x2 = self.layer2(x2)
        x3 = self.down2(x2)       # dim3

        x3 = self.layer3(x3)
        x4 = self.down3(x3)       # dim4

        x4 = self.layer4(x4)      # bottleneck (no further downsample)

        # Decoder
        # up from bottleneck to res3
        x_up3, res3 = self.up3(x4)  # -> dim3, res3
        x_up3 = x_up3 + x3  # skip connection (same dim)
        x_up3 = self.dec_layer3(x_up3)

        # up from res3 to res2
        x_up2, res2 = self.up2(x_up3)
        x_up2 = x_up2 + x2
        x_up2 = self.dec_layer2(x_up2)

        # up from res2 to res1
        x_up1, res1 = self.up1(x_up2)
        x_up1 = x_up1 + x1
        x_up1 = self.dec_layer1(x_up1)

        # reshape to feature map at patch resolution
        H1, W1 = res1
        x_up1 = x_up1.view(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()  # [B,dim1,H1,W1]

        # upsample back to original resolution (factor 4)
        x_up = F.interpolate(x_up1, size=(H, W), mode="bilinear", align_corners=False)

        out = self.final_conv(x_up)  # [B,num_classes,H,W]
        return out

class SwinUNetAgnostic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SwinUNet(**kwargs)

        # Store patch and window sizes for padding logic
        self.patch_size = 4                     # PatchEmbed uses stride=4
        self.window_size = kwargs.get("window_size", 4)

    def forward(self, x):
        # 1) Pad
        x_pad, pad = pad_to_swin_size(
            x,
            patch_size=self.patch_size,
            window_size=self.window_size
        )

        # 2) Run SwinUNet on padded input
        out_pad = self.model(x_pad)

        # 3) Unpad output to match *original* size
        out = unpad(out_pad, pad)
        return out

# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    img_size = 64
    in_chans = n_input_features = 6
    # Create model for 64×64 or arbitrary sizes
    model = SwinUNetAgnostic(
        img_size=(64, 64),      # dummy, we ignore this
        in_chans=in_chans,
        num_classes=1,
        embed_dim=64,
        depths=(2,2,2,2),
        num_heads=(2,4,8,8),
        window_size=4,
    )

    # Example: full-field 76×261
    x = torch.randn(2, in_chans, 76, 261)

    # Runs automatically with padding + unpadding
    y = model(x)

    print("Input:", x.shape)
    print("Output:", y.shape)   # guaranteed → [2,1,76,261]

