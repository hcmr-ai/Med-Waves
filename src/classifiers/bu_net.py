import sys
from pathlib import Path

import numpy as np
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.commons.losses import (
    masked_mse_loss,
    masked_smooth_l1_loss,
    masked_multi_bin_weighted_mse,
    masked_multi_bin_weighted_smooth_l1
)


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
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2, add_vhm0_residual=False, vhm0_channel_index=0, upsample_mode="nearest"
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
        
        # Final 1×1 conv to output
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
        
        return x


class WaveBiasCorrector(pl.LightningModule):
    def __init__(
        self, in_channels=3, lr=1e-3, loss_type="weighted_mse", lr_scheduler_config=None, predict_bias=False,
        filters=None,
        dropout=0.2,
        add_vhm0_residual=False,
        vhm0_channel_index=0,
        weight_decay=1e-4,
        model_type="nick",  # Options: "nick", "geo", "enhanced"
        upsample_mode="nearest",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Select model architecture
        if model_type == "geo":
            self.model = BU_Net_Geo(
                in_channels=in_channels, filters=filters, dropout=dropout,
                add_vhm0_residual=add_vhm0_residual, vhm0_channel_index=vhm0_channel_index
            )
        elif model_type == "enhanced":
            self.model = BU_Net_Geo_Nick_Enhanced(
                in_channels=in_channels, filters=filters, dropout=dropout,
                add_vhm0_residual=add_vhm0_residual, vhm0_channel_index=vhm0_channel_index,
                upsample_mode=upsample_mode,
            )
        else:  # "nick" or default
            self.model = BU_Net_Geo_Nick(
                in_channels=in_channels, filters=filters, dropout=dropout,
                add_vhm0_residual=add_vhm0_residual, vhm0_channel_index=vhm0_channel_index
            )
        
        self.loss_type = loss_type
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.predict_bias = predict_bias
        if loss_type == "smooth_l1" or loss_type == "multi_bin_weighted_smooth_l1":
            self.criterion = torch.nn.SmoothL1Loss(beta=0.3, reduction="none")

    def forward(self, x):
        # Handle NaN values in input by replacing with zeros
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model(x_clean)

    def compute_loss(self, y_pred, y_true, mask, vhm0_for_reconstruction):
        if self.loss_type == "mse":
            return masked_mse_loss(y_pred, y_true, mask)
        elif self.loss_type == "smooth_l1":
            return masked_smooth_l1_loss(y_pred, y_true, mask, self.criterion)
        elif self.loss_type == "multi_bin_weighted_smooth_l1":
            return masked_multi_bin_weighted_smooth_l1(y_pred, y_true, mask, vhm0_for_reconstruction, self.criterion)
        else:
            return masked_multi_bin_weighted_mse(y_pred, y_true, mask, vhm0_for_reconstruction)

    def training_step(self, batch, batch_idx):
        X, y, mask, vhm0_for_reconstruction = batch

        y_pred = self(X)
        loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction)

        # Enhanced metrics for Comet
        with torch.no_grad():
            # Calculate additional metrics
            min_h = min(y_pred.shape[2], y.shape[2])
            min_w = min(y_pred.shape[3], y.shape[3])
            y_pred = y_pred[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]
            mask = mask[:, :, :min_h, :min_w]

            mae = torch.abs(y_pred - y)[mask].mean()
            mse = ((y_pred - y) ** 2)[mask].mean()
            rmse = torch.sqrt(mse)

            # Log metrics
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_mae", mae, on_step=True, on_epoch=True)
            self.log("train_mse", mse, on_step=True, on_epoch=True)
            self.log("train_rmse", rmse, on_step=True, on_epoch=True)

            # Log data statistics
            self.log("train_y_mean", y[mask].mean(), on_step=True, on_epoch=True)
            self.log("train_y_std", y[mask].std(), on_step=True, on_epoch=True)
            self.log(
                "train_pred_mean", y_pred[mask].mean(), on_step=True, on_epoch=True
            )
            self.log("train_pred_std", y_pred[mask].std(), on_step=True, on_epoch=True)
            self.log(
                "train_valid_pixels", mask.sum().float(), on_step=True, on_epoch=True
            )

            # Log sea-bin metrics for training
            if self.predict_bias and vhm0_for_reconstruction is not None:
                vhm0_for_reconstruction = vhm0_for_reconstruction[mask]
                y_true_wave_heights = vhm0_for_reconstruction + y[mask]  # matches: corrected = vhm0 + bias
                y_pred_wave_heights = vhm0_for_reconstruction + y_pred[mask]
                self._log_sea_bin_metrics(y_true_wave_heights, y_pred_wave_heights, "train")
                self._log_sea_bin_metrics(y_true_wave_heights, vhm0_for_reconstruction, "train_baseline")
            else:
                self._log_sea_bin_metrics(y[mask], y_pred[mask], "train")
                self._log_sea_bin_metrics(y[mask], vhm0_for_reconstruction[mask], "train_baseline")

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, mask, vhm0_for_reconstruction = batch

        y_pred = self(X)
        loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction)

        # Enhanced validation metrics
        with torch.no_grad():
            # Calculate additional metrics
            min_h = min(y_pred.shape[2], y.shape[2])
            min_w = min(y_pred.shape[3], y.shape[3])
            y_pred = y_pred[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]
            mask = mask[:, :, :min_h, :min_w]

            mae = torch.abs(y_pred - y)[mask].mean()
            mse = ((y_pred - y) ** 2)[mask].mean()
            rmse = torch.sqrt(mse)
            log_on_step = False
            # if self.trainer is not None and hasattr(self.trainer, 'val_check_interval'):
            #     # val_check_interval can be None, int (steps), or float (fraction of epoch)
            #     log_on_step = self.trainer.val_check_interval is not None
            # print(f"log_on_step: {log_on_step}")

            # Log validation metrics
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_mae", mae, on_step=log_on_step, on_epoch=True)
            self.log("val_mse", mse, on_step=log_on_step, on_epoch=True)
            self.log("val_rmse", rmse, on_step=log_on_step, on_epoch=True)

            # Log validation data statistics
            self.log("val_y_mean", y[mask].mean(), on_epoch=True)
            self.log("val_y_std", y[mask].std(), on_epoch=True)
            self.log("val_pred_mean", y_pred[mask].mean(), on_epoch=True)
            self.log("val_pred_std", y_pred[mask].std(), on_epoch=True)
            self.log("val_valid_pixels", mask.sum().float(), on_epoch=True)

            # Log sea-bin metrics for validation
            if self.predict_bias and vhm0_for_reconstruction is not None:
                vhm0_for_reconstruction = vhm0_for_reconstruction[mask]
                y_true_wave_heights = vhm0_for_reconstruction + y[mask]
                y_pred_wave_heights = vhm0_for_reconstruction + y_pred[mask]
                self._log_sea_bin_metrics(y_true_wave_heights, y_pred_wave_heights, "val")
                self._log_sea_bin_metrics(y_true_wave_heights, vhm0_for_reconstruction, "val_baseline")
            else:
                self._log_sea_bin_metrics(y[mask], y_pred[mask], "val")
                self._log_sea_bin_metrics(y[mask], vhm0_for_reconstruction[mask], "val_baseline")

        return loss

    def on_train_start(self) -> None:
        """Log scheduler info and other hyperparameters when training starts."""
        # Log optimizer info
        if hasattr(self, 'optimizer_info'):
            for key, value in self.optimizer_info.items():
                self.log(key, value)
        
        # Log scheduler info
        if hasattr(self, 'scheduler_info'):
            for key, value in self.scheduler_info.items():
                self.log(key, value)

    def on_train_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_epoch=True, prog_bar=True)

    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log("grad_norm_clipped", total_norm, on_step=True, on_epoch=True, prog_bar=True)

    def _log_sea_bin_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor, prefix: str):
        """Log sea-bin metrics for different wave height ranges."""
        # Convert to numpy for sea-bin calculation
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # Define sea-bin ranges (same as in config)
        sea_bins = [
            {"name": "calm", "min": 0.0, "max": 1.0},
            {"name": "light", "min": 1.0, "max": 2.0},
            {"name": "moderate", "min": 2.0, "max": 3.0},
            {"name": "rough", "min": 3.0, "max": 4.0},
            {"name": "very_rough", "min": 4.0, "max": 5.0},
            {"name": "extreme_5_6", "min": 5.0, "max": 6.0},
            {"name": "extreme_6_7", "min": 6.0, "max": 7.0},
            {"name": "extreme_7_8", "min": 7.0, "max": 8.0},
            {"name": "extreme_8_9", "min": 8.0, "max": 9.0},
            {"name": "extreme_9_10", "min": 9.0, "max": 10.0},
            {"name": "extreme_10_11", "min": 10.0, "max": 11.0},
            {"name": "extreme_11_12", "min": 11.0, "max": 12.0},
            {"name": "extreme_12_13", "min": 12.0, "max": 13.0},
            {"name": "extreme_13_14", "min": 13.0, "max": 14.0},
            {"name": "extreme_14_15", "min": 14.0, "max": 15.0}
        ]

        for bin_config in sea_bins:
            bin_name = bin_config["name"]
            bin_min = bin_config["min"]
            bin_max = bin_config["max"]

            # Filter data for this sea state bin
            mask = (y_true_np >= bin_min) & (y_true_np < bin_max)
            bin_count = np.sum(mask)

            if bin_count > 0:
                bin_y_true = y_true_np[mask]
                bin_y_pred = y_pred_np[mask]

                # Calculate metrics for this bin
                mae = np.mean(np.abs(bin_y_pred - bin_y_true))
                mse = np.mean((bin_y_pred - bin_y_true) ** 2)
                rmse = np.sqrt(mse)
                bias = np.mean(bin_y_pred - bin_y_true)

                # Log metrics with bin-specific names
                self.log(f"{prefix}_{bin_name}_mae", mae, on_epoch=True)
                self.log(f"{prefix}_{bin_name}_rmse", rmse, on_epoch=True)
                self.log(f"{prefix}_{bin_name}_bias", bias, on_epoch=True)
                self.log(f"{prefix}_{bin_name}_count", bin_count, on_epoch=True)

    def configure_optimizers(self):
        # lr = float(self.hparams.lr) if isinstance(self.hparams.lr, str) else self.hparams.lr
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # Store optimizer info to log later
        self.optimizer_info = {
            "optimizer_weight_decay": self.hparams.weight_decay,
            "optimizer_lr": self.hparams.lr
        }
        
        def get_float(key, default):
            val = scheduler_config.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        # Configure learning rate scheduler
        scheduler_config = self.lr_scheduler_config
        if not scheduler_config or scheduler_config.get("type", "none") == "none":
            return optimizer

        scheduler_type = scheduler_config["type"]

        if scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=get_float("factor", 0.5),
                patience=int(scheduler_config.get("patience", 5)),
                min_lr=get_float("min_lr", 1e-7),
                # verbose=scheduler_config.get("verbose", True),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get("monitor", "val_loss"),
                },
            }

        elif scheduler_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(scheduler_config.get("T_max", 50)), eta_min=get_float("eta_min", 1e-6),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_config.get("step_size", 10)),
                gamma=get_float("gamma", 0.1),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif scheduler_type == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=get_float("gamma", 0.1)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif scheduler_type == "CosineAnnealingWarmupRestarts":
            # Use PyTorch Lightning's estimated_stepping_batches
            total_steps = self.trainer.estimated_stepping_batches
            warmup_ratio = get_float(scheduler_config.get("warmup_steps", 0.1), 0.1)
            warmup_steps = int(warmup_ratio * total_steps)
            
            # Store these values to log them during training
            self.scheduler_info = {
                "total_steps": total_steps,
                "max_epochs": self.trainer.max_epochs,
                "warmup_ratio": warmup_ratio,
                "warmup_steps_calculated": warmup_steps
            }
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            )
            return {"optimizer": optimizer, "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",   # ✅ CRITICAL — warmup MUST be per-step
                    "frequency": 1,
                }}

        else:
            # Unknown scheduler type, return optimizer only
            return optimizer
