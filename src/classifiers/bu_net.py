import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
from commons.losses import masked_mse_loss, masked_weighted_mse


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
        self, in_channels=6, out_channels=1, filters=None, dropout=0.2
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

        # Ensure output matches input dimensions
        # if x.shape[2] != input_h or x.shape[3] != input_w:
        #     x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)

        return x


class BU_Net(nn.Module):
    def __init__(
        self, in_channels=6, out_channels=1, filters=None
    ):
        """
        5-level U-Net with BatchNorm as in Sun et al. (2022).
        Input: (batch, in_channels, 80, 80)
        Output: (batch, 1, 80, 80) corrected SWH
        """
        super().__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 1024]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_c = in_channels
        for f in filters:
            self.encoders.append(conv_block(prev_c, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_c = f

        # Bottleneck
        self.bottleneck = conv_block(filters[-1], filters[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_filters = list(reversed(filters))
        prev_c = filters[-1] * 2
        for f in rev_filters:
            self.upconvs.append(nn.ConvTranspose2d(prev_c, f, kernel_size=2, stride=2))
            self.decoders.append(conv_block(prev_c, f))
            prev_c = f

        # Output layer
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_feats = []
        for enc, pool in zip(self.encoders, self.pools, strict=False):
            x = enc(x)
            enc_feats.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for up, dec, skip in zip(
            self.upconvs, self.decoders, reversed(enc_feats), strict=False
        ):
            x = up(x)
            # Ensure correct size in case of odd dimensions
            if x.size() != skip.size():
                x = nn.functional.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        out = self.final_conv(x)
        return out  # correction field


class WaveBiasCorrector(pl.LightningModule):
    def __init__(
        self, in_channels=3, lr=1e-3, loss_type="weighted_mse", lr_scheduler_config=None, predict_bias=False,
        filters=None,
        dropout=0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = BU_Net_Geo(in_channels=in_channels, filters=filters, dropout=dropout)
        self.loss_type = loss_type
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.predict_bias = predict_bias

    def forward(self, x):
        # Handle NaN values in input by replacing with zeros
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model(x_clean)

    def compute_loss(self, y_pred, y_true, mask):
        if self.loss_type == "mse":
            return masked_mse_loss(y_pred, y_true, mask)
        else:
            return masked_weighted_mse(y_pred, y_true, mask)

    def training_step(self, batch, batch_idx):
        X, y, mask = batch

        y_pred = self(X)
        loss = self.compute_loss(y_pred, y, mask)

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
            self._log_sea_bin_metrics(y[mask], y_pred[mask], "train")
            
            # Log baseline sea-bin metrics (uncorrected VHM0 vs ground truth)
            # Extract uncorrected VHM0 from input features
            vhm0_uncorrected = X[:, 1:2, :min_h, :min_w]
            vhm0_uncorrected_masked = vhm0_uncorrected[mask]
            
            # Calculate ground truth wave heights depending on target type
            # If predicting bias, y = corrected - uncorrected, so y_true = y + uncorrected
            # If predicting corrected directly, y is already the ground truth wave heights
            if hasattr(self.hparams, "predict_bias") and bool(self.hparams.predict_bias):
                y_true_wave_heights = y[mask] + vhm0_uncorrected_masked
            else:
                y_true_wave_heights = y[mask]
            
            # Log baseline sea-bin metrics
            self._log_sea_bin_metrics(y_true_wave_heights, vhm0_uncorrected_masked, "train_baseline")

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, mask = batch

        y_pred = self(X)
        loss = self.compute_loss(y_pred, y, mask)

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
            self._log_sea_bin_metrics(y[mask], y_pred[mask], "val")
            
            # Log baseline sea-bin metrics for validation (uncorrected VHM0 vs ground truth)
            # Extract uncorrected VHM0 from input features
            vhm0_uncorrected = X[:, 1:2, :min_h, :min_w]  # Extract VHM0 channel (index 1)
            vhm0_uncorrected_masked = vhm0_uncorrected[mask]
            
            # Calculate ground truth wave heights depending on target type
            if hasattr(self.hparams, "predict_bias") and bool(self.hparams.predict_bias):
                y_true_wave_heights = y[mask] + vhm0_uncorrected_masked
            else:
                y_true_wave_heights = y[mask]
            
            # Log baseline sea-bin metrics
            self._log_sea_bin_metrics(y_true_wave_heights, vhm0_uncorrected_masked, "val_baseline")

            # Store results for callback (only on first batch of epoch)
            if batch_idx == 0:
                self.last_val_batch = (y, y_pred, mask)

        return loss
    
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
        lr = float(self.hparams.lr) if isinstance(self.hparams.lr, str) else self.hparams.lr
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
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
                optimizer, T_max=int(scheduler_config.get("T_max", 50))
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

        else:
            # Unknown scheduler type, return optimizer only
            return optimizer
