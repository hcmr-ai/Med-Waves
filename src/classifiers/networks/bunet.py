import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classifiers.networks.mdn import MDNHead


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class GeoConv(nn.Module):
    """Conv block with reflection padding in latitude and circular padding in longitude."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lat_pad = nn.ReflectionPad2d((0, 0, 1, 1))  # pad north/south only
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.lat_pad(x)  # reflect latitudes
        x = self.conv(x)  # circular wrap on longitudes
        x = self.bn(x)
        return self.relu(x)


class BU_Net_Geo(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2, add_vhm0_residual=False, vhm0_channel_index=0
    ):
        """
        BU-Net with geophysical padding.
        Input: (batch, in_channels, H=380, W=1307)
        Output: (batch, out_channels, H, W)
        """
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        self.dropout = dropout
        self.add_vhm0_residual = add_vhm0_residual
        self.vhm0_channel_index = vhm0_channel_index
        self.encoder_dropouts = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for f in filters:
            self.encoders.append(nn.Sequential(GeoConv(prev_c, f), GeoConv(f, f)))
            self.pools.append(nn.MaxPool2d(2))
            self.encoder_dropouts.append(nn.Dropout2d(dropout))
            prev_c = f

        # Bottleneck
        self.bottleneck = nn.Sequential(
            GeoConv(filters[-1], filters[-1] * 2),
            GeoConv(filters[-1] * 2, filters[-1] * 2),
            nn.Dropout2d(dropout),
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_filters = list(reversed(filters))
        prev_c = filters[-1] * 2
        for f in rev_filters:
            self.upconvs.append(nn.ConvTranspose2d(prev_c, f, kernel_size=2, stride=2))
            self.decoders.append(nn.Sequential(GeoConv(prev_c, f), GeoConv(f, f)))
            self.decoder_dropouts.append(nn.Dropout2d(dropout))
            prev_c = f

        # Final 1×1 conv to SWH correction
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Store input dimensions for output matching
        # input_h, input_w = x.shape[2], x.shape[3]
        if self.add_vhm0_residual:
            vhm0_input = x[:, self.vhm0_channel_index:self.vhm0_channel_index+1, :, :]

        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools, strict=False):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(
            self.upconvs, self.decoders, reversed(enc_feats), strict=False
        ):
            x = up(x)
            # Align in case of odd/non-divisible dims
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.final_conv(x)
        if self.add_vhm0_residual:
            # Ensure vhm0 matches correction dimensions (in case of interpolation)
            if vhm0_input.shape[2:] != x.shape[2:]:
                vhm0_input = F.interpolate(
                    vhm0_input, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            x = x + vhm0_input
        else:
            x = x

        # Ensure output matches input dimensions
        # if x.shape[2] != input_h or x.shape[3] != input_w:
        #     x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)

        return x

class BU_Net_Geo_Nick_Shallow(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2, add_vhm0_residual=False, vhm0_channel_index=0
    ):
        """
        BU-Net matching notebook architecture exactly.
        Input: (batch, in_channels, H, W)
        Output: (batch, out_channels, H, W)
        """
        super().__init__()
        # Notebook uses fixed [32, 64] filters, ignore filters parameter
        self.add_vhm0_residual = add_vhm0_residual
        self.vhm0_channel_index = vhm0_channel_index

        # Reflection padding: ((2, 2), (1, 1)) -> (left, right, top, bottom)
        self.reflection_pad = nn.ReflectionPad2d((1, 1, 2, 2))

        # Encoder
        # Block 1: 32 filters
        self.enc1_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(32)
        self.enc1_pool = nn.MaxPool2d(2, 2)

        # Block 2: 64 filters
        self.enc2_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(64)
        self.enc2_pool = nn.MaxPool2d(2, 2)

        # Bottleneck: 128 filters
        self.bottleneck_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(128)

        # Decoder
        # Block 1: Upsample + concat with enc2 + 64 filters
        self.dec1_upsample = nn.UpsamplingNearest2d(scale_factor=2)  # Matches UpSampling2D
        self.dec1_conv = nn.Conv2d(192, 64, kernel_size=3, padding=1)  # 64 (skip) + 64 (upsampled) = 128 in
        self.dec1_bn = nn.BatchNorm2d(64)

        # Block 2: Upsample + concat with enc1 + 32 filters
        self.dec2_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec2_conv = nn.Conv2d(96, 32, kernel_size=3, padding=1)  # 32 (skip) + 64 (upsampled) = 96 in
        self.dec2_bn = nn.BatchNorm2d(32)

        # Output correction: 1 channel, 3x3 conv, linear activation
        self.correction_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

        # Crop padding: removes (2, 2) from height, (1, 1) from width
        # This is handled in forward by slicing

    def forward(self, x):
        # Apply reflection padding
        x_padded = self.reflection_pad(x)

        # Store VHM0 for residual (from padded input)
        if self.add_vhm0_residual:
            vhm0_raw = x_padded[:, self.vhm0_channel_index:self.vhm0_channel_index+1, :, :]

        # Encoder
        c1 = F.relu(self.enc1_bn(self.enc1_conv(x_padded)))
        p1 = self.enc1_pool(c1)

        c2 = F.relu(self.enc2_bn(self.enc2_conv(p1)))
        p2 = self.enc2_pool(c2)

        # Bottleneck
        c3 = F.relu(self.bottleneck_bn(self.bottleneck_conv(p2)))

        # Decoder
        # Block 1: Upsample c3, concat with c2
        u2 = self.dec1_upsample(c3)
        # Handle dimension mismatch if needed
        if u2.size()[2:] != c2.size()[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u2_concat = torch.cat([u2, c2], dim=1)  # 64 + 64 = 128 channels
        c4 = F.relu(self.dec1_bn(self.dec1_conv(u2_concat)))

        # Block 2: Upsample c4, concat with c1
        u1 = self.dec2_upsample(c4)
        # Handle dimension mismatch if needed
        if u1.size()[2:] != c1.size()[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        u1_concat = torch.cat([u1, c1], dim=1)  # 64 + 32 = 96 channels
        c5 = F.relu(self.dec2_bn(self.dec2_conv(u1_concat)))

        # Output correction
        correction = self.correction_conv(c5)  # Linear activation (no ReLU)

        # Residual connection
        if self.add_vhm0_residual:
            # Ensure vhm0 matches correction dimensions
            if vhm0_raw.shape[2:] != correction.shape[2:]:
                vhm0_raw = F.interpolate(vhm0_raw, size=correction.shape[2:], mode='bilinear', align_corners=False)
            output_padded = correction + vhm0_raw
        else:
            output_padded = correction

        # Crop padding: remove (2, 2) from height, (1, 1) from width
        # If padded input was (B, C, H+4, W+2), output should be (B, C, H, W)
        # Crop top=2, bottom=2, left=1, right=1
        _, _, h, w = output_padded.shape
        output = output_padded[:, :, 2:h-2, 1:w-1]

        return output

class BU_Net_Geo_Nick(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2, add_vhm0_residual=False, vhm0_channel_index=0
    ):
        """
        BU-Net matching notebook architecture exactly.
        Input: (batch, in_channels, H, W)
        Output: (batch, out_channels, H, W)
        """
        super().__init__()
        filters = [32, 64] if filters is None else filters
        self.add_vhm0_residual = add_vhm0_residual
        self.vhm0_channel_index = vhm0_channel_index
        self.filters = filters
        self.dropout = dropout

        # Reflection padding: ((2, 2), (1, 1)) -> (left, right, top, bottom)
        self.reflection_pad = nn.ReflectionPad2d((1, 1, 2, 2))

        # Encoder (automated)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for f in filters:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, f, kernel_size=3, padding=1),
                    nn.BatchNorm2d(f),
                    # nn.GroupNorm(1, f),
                    nn.ReLU(inplace=True),
                )
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            prev_c = f

        # Bottleneck (keep same channels as last encoder to match notebook math: 64 + 64)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(filters[-1], filters[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[-1]),
            # nn.GroupNorm(1, filters[-1]),
            nn.ReLU(inplace=True),
        )

        # Decoder (automated): upsample then Conv2d(current + skip -> skip)
        self.up_samples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = filters[-1]
        for skip_channels in reversed(filters[:-1]):
            self.up_samples.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(current_channels + skip_channels, skip_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_channels),
                    # nn.GroupNorm(1, skip_channels),
                    nn.ReLU(inplace=True),
                )
            )
            current_channels = skip_channels

        # Output correction: 1 channel, 3x3 conv, linear activation
        self.correction_conv = nn.Conv2d(filters[0], out_channels, kernel_size=3, padding=1)

        # Crop padding: removes (2, 2) from height, (1, 1) from width) handled in forward

    def forward(self, x):
        # Apply reflection padding
        x_padded = self.reflection_pad(x)

        # Store VHM0 for residual (from padded input)
        if self.add_vhm0_residual:
            vhm0_raw = x_padded[:, self.vhm0_channel_index:self.vhm0_channel_index+1, :, :]

        # Encoder
        enc_feats = []
        out = x_padded
        for enc, pool in zip(self.encoders, self.pools, strict=False):
            out = enc(out)
            enc_feats.append(out)
            out = pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder (skip the last encoder output as it feeds into bottleneck)
        for up, dec, skip in zip(self.up_samples, self.decoders, reversed(enc_feats[:-1]), strict=False):
            out = up(out)
            if out.size()[2:] != skip.size()[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        # Output correction
        correction = self.correction_conv(out)  # Linear activation (no ReLU)

        # Residual connection
        if self.add_vhm0_residual:
            if vhm0_raw.shape[2:] != correction.shape[2:]:
                vhm0_raw = F.interpolate(vhm0_raw, size=correction.shape[2:], mode='bilinear', align_corners=False)
            output_padded = correction + vhm0_raw
        else:
            output_padded = correction

        # Crop padding: remove (2, 2) from height, (1, 1) from width
        _, _, h, w = output_padded.shape
        output = output_padded[:, :, 2:h-2, 1:w-1]

        return output


class BU_Net_Geo_Nick_Enhanced(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2, add_vhm0_residual=False, vhm0_channel_index=0, upsample_mode="nearest", use_mdn=False
    ):
        """
        Enhanced BU-Net with geophysical padding and deeper architecture.
        Combines Nick's architecture with Geo features:
        - GeoConv layers with circular longitude padding
        - Two conv layers per encoder/decoder block
        - Doubled bottleneck channels
        - Learnable upsampling (ConvTranspose2d)
        - Dropout throughout
        - All skip connections

        Input: (batch, in_channels, H, W)
        Output: (batch, out_channels, H, W)
        """
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512]
        self.dropout = dropout
        self.add_vhm0_residual = add_vhm0_residual
        self.vhm0_channel_index = vhm0_channel_index
        self.filters = filters
        self.use_mdn = use_mdn
        # Encoder with GeoConv
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoder_dropouts = nn.ModuleList()
        prev_c = in_channels
        for f in filters:
            self.encoders.append(
                nn.Sequential(
                    GeoConv(prev_c, f),
                    GeoConv(f, f)
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            self.encoder_dropouts.append(nn.Dropout2d(dropout))
            prev_c = f

        # Bottleneck with doubled channels
        self.bottleneck = nn.Sequential(
            GeoConv(filters[-1], filters[-1] * 2),
            GeoConv(filters[-1] * 2, filters[-1] * 2),
            nn.Dropout2d(dropout),
        )

        # Decoder with learnable upsampling
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_dropouts = nn.ModuleList()
        self.upsample_mode = upsample_mode
        rev_filters = list(reversed(filters))
        prev_c = filters[-1] * 2
        for f in rev_filters:
            if upsample_mode == "nearest":
                self.upconvs.append(nn.UpsamplingNearest2d(scale_factor=2))
                # With nearest upsampling: channels don't change, so after concat: prev_c + f
                decoder_in_channels = prev_c + f
            else:
                self.upconvs.append(nn.ConvTranspose2d(prev_c, f, kernel_size=2, stride=2))
                # With ConvTranspose2d: channels reduced to f, so after concat: f + f = 2*f
                decoder_in_channels = f * 2

            self.decoders.append(
                nn.Sequential(
                    GeoConv(decoder_in_channels, f),
                    GeoConv(f, f)
                )
            )
            self.decoder_dropouts.append(nn.Dropout2d(dropout))
            prev_c = f

        if use_mdn:
            self.mdn_head = MDNHead(filters[-1], K=3)
        else:
            self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Store VHM0 for residual connection
        if self.add_vhm0_residual:
            vhm0_input = x[:, self.vhm0_channel_index:self.vhm0_channel_index+1, :, :]

        # Encoder with dropout
        enc_feats = []
        for enc, pool, dropout in zip(self.encoders, self.pools, self.encoder_dropouts, strict=False):
            x = enc(x)
            x = dropout(x)
            enc_feats.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections and dropout
        for up, dec, dropout, skip in zip(
            self.upconvs, self.decoders, self.decoder_dropouts, reversed(enc_feats), strict=False
        ):
            x = up(x)
            # Align dimensions if needed
            if x.size()[2:] != skip.size()[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
            x = dropout(x)

        # Final output
        x = self.final_conv(x)

        # Add residual connection
        if self.add_vhm0_residual:
            if vhm0_input.shape[2:] != x.shape[2:]:
                vhm0_input = F.interpolate(
                    vhm0_input, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            x = x + vhm0_input

        if self.use_mdn:
            return self.mdn_head(x)
        else:
            return x
