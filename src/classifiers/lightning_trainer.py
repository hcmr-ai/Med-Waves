import sys
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifiers.networks.bunet import (
    BU_Net_Geo,
    BU_Net_Geo_Nick,
    BU_Net_Geo_Nick_Enhanced,
)
from src.classifiers.networks.mdn import mdn_expected_value
from src.classifiers.networks.swin_unet import SwinUNetAgnostic
from src.classifiers.networks.trans_unet import TransUNetGeo
from src.classifiers.networks.trans_unet_gan import WaveTransUNetGAN
from src.commons.loss_functions.perceptual_loss import (
    PerceptualLoss,
    WaveFeatureExtractor,
)
from src.commons.loss_functions.ssim import SSIMLoss
from src.commons.losses import (
    adversarial_loss_D,
    adversarial_loss_G,
    masked_huber_loss,
    masked_mse_loss,
    masked_mse_mdn_loss,
    masked_mse_perceptual_loss,
    masked_mse_ssim_loss,
    masked_multi_bin_weighted_mse,
    masked_multi_bin_weighted_smooth_l1,
    masked_smooth_l1_loss,
    masked_ssim_perceptual_loss,
    masked_weighted_mse,
    mdn_nll_loss,
    pixel_switch_loss_stable,
)


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
        pixel_switch_threshold_m=0.45,
        use_mdn=False,
        optimizer_type="Adam",
        lambda_adv=0.01,
        n_discriminator_updates=3,
        discriminator_lr_multiplier=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_type = loss_type
        self.n_discriminator_updates = n_discriminator_updates
        self.discriminator_lr_multiplier = discriminator_lr_multiplier
        if self.loss_type == "pixel_switch_mse":
            self.pixel_switch_threshold_m = pixel_switch_threshold_m

        if self.loss_type == "mse_perceptual" or self.loss_type == "mse_ssim_perceptual":
            self.perceptual_loss = PerceptualLoss(WaveFeatureExtractor(), layer_weights=[1.0, 1.0, 1.0])

        if self.loss_type == "mse_ssim_perceptual" or self.loss_type == "mse_ssim":
            self.ssim_loss = SSIMLoss()

        self.use_mdn = use_mdn
        self.optimizer_type = optimizer_type
        self.lambda_adv = lambda_adv
        self.model_type = model_type

        if model_type == "transunet_gan":
            self.automatic_optimization = False
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
                use_mdn=use_mdn,
            )
        elif model_type == "transunet":
            self.model = TransUNetGeo(
                in_channels=in_channels, out_channels=1, base_channels=64, bottleneck_dim=1024, patch_size=16, num_layers=8,
                use_mdn=use_mdn,
            )
        elif model_type == "swinunet":
            self.model = SwinUNetAgnostic(
                img_size=(64, 64),
                in_chans=in_channels,
                num_classes=1,
                embed_dim=64,
                depths=(2,2,2,2),
                num_heads=(2,4,8,8),
                window_size=4,
                mlp_ratio=4.,
            )
        elif model_type == "transunet_gan":
            self.model = WaveTransUNetGAN(
                in_channels=in_channels, out_channels=1, base_channels=64, bottleneck_dim=1024, patch_size=16, num_layers=8,
                use_mdn=use_mdn,
            )
        else:  # "nick" or default
            self.model = BU_Net_Geo_Nick(
                in_channels=in_channels, filters=filters, dropout=dropout,
                add_vhm0_residual=add_vhm0_residual, vhm0_channel_index=vhm0_channel_index
            )

        self.lr_scheduler_config = lr_scheduler_config or {}
        self.predict_bias = predict_bias
        if loss_type == "smooth_l1" or loss_type == "multi_bin_weighted_smooth_l1":
            self.criterion = torch.nn.SmoothL1Loss(beta=0.3, reduction="none")

    def forward(self, x):
        # Handle NaN values in input by replacing with zeros
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model(x_clean)

    def compute_loss(self, y_pred, y_true, mask, vhm0_for_reconstruction, pi=None, mu=None, sigma=None):
        if self.loss_type == "mse":
            return masked_mse_loss(y_pred, y_true, mask)
        elif self.loss_type == "smooth_l1":
            return masked_smooth_l1_loss(y_pred, y_true, mask, self.criterion)
        elif self.loss_type == "weighted_mse":
            return masked_weighted_mse(y_pred, y_true, mask, threshold=6.0, high_weight=5.0, epsilon=1e-6)
        elif self.loss_type == "multi_bin_weighted_smooth_l1":
            return masked_multi_bin_weighted_smooth_l1(y_pred, y_true, mask, vhm0_for_reconstruction, self.criterion)
        elif self.loss_type == "pixel_switch_mse":
            return pixel_switch_loss_stable(y_pred, y_true, mask, threshold_m=self.pixel_switch_threshold_m)
        elif self.loss_type == "mse_perceptual":
            return masked_mse_perceptual_loss(y_pred, y_true, mask, self.perceptual_loss)
        elif self.loss_type == "mse_ssim":
            return masked_mse_ssim_loss(y_pred, y_true, mask, ssim_loss=self.ssim_loss)
        elif self.loss_type == "mse_ssim_perceptual":
            return masked_ssim_perceptual_loss(y_pred, y_true, mask, self.ssim_loss, self.perceptual_loss)
        elif self.loss_type == "mse_mdn":
            return masked_mse_mdn_loss(pi, mu, sigma, y_true, mask, eps=1e-9, lambda_mse=0.1, lambda_nll=1.0)
        elif self.loss_type == "mdn":
            return mdn_nll_loss(pi, mu, sigma, y_true, mask, eps=1e-9)
        elif self.loss_type == "mse_gan":
            return masked_mse_loss(y_pred, y_true, mask)
        elif self.loss_type == "huber":
            return masked_huber_loss(y_pred, y_true, mask)
        else:
            return masked_multi_bin_weighted_mse(y_pred, y_true, mask, vhm0_for_reconstruction)

    def _training_step(self, batch, batch_idx):
        X, y, mask, vhm0_for_reconstruction = batch  # _ is the bin_id we don't need
        if self.use_mdn:
            pi, mu, sigma = self(X)
            y_pred = mdn_expected_value(pi, mu)
            loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction, pi, mu, sigma)
        else:
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
            self.log("train_error_min", (y_pred - y)[mask].min(), on_step=True, on_epoch=True)
            self.log("train_error_max", (y_pred - y)[mask].max(), on_step=True, on_epoch=True)
            self.log("train_error_mean", (y_pred - y)[mask].mean(), on_step=True, on_epoch=True)
            self.log("train_error_p95", torch.quantile(torch.abs(y_pred - y)[mask], 0.95), on_step=True, on_epoch=True)

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

    def _training_step_no_gan(self, X, y, mask, vhm0_for_reconstruction):
        if self.use_mdn:
            pi, mu, sigma = self(X)
            y_pred = mdn_expected_value(pi, mu)
            loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction, pi, mu, sigma)
        else:
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
            self.log("train_error_min", (y_pred - y)[mask].min(), on_step=True, on_epoch=True)
            self.log("train_error_max", (y_pred - y)[mask].max(), on_step=True, on_epoch=True)
            self.log("train_error_mean", (y_pred - y)[mask].mean(), on_step=True, on_epoch=True)
            self.log("train_error_p95", torch.quantile(torch.abs(y_pred - y)[mask], 0.95), on_step=True, on_epoch=True)

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

    def _training_step_G(self, X, y, mask, vhm0_for_reconstruction):
        """
        Generator training step (optimizer_idx == 0).
        Includes:
        - MDN or normal forward pass
        - supervised loss via compute_loss()
        - adversarial loss L_G
        - masked discriminator forward
        """

        # ---------------------------------------
        # Forward pass (MDN or normal)
        # ---------------------------------------
        if self.use_mdn:
            pi, mu, sigma = self(X)
            y_pred = mdn_expected_value(pi, mu)
            base_loss = self.compute_loss(
                y_pred, y, mask, vhm0_for_reconstruction,
                pi, mu, sigma
            )
        else:
            y_pred = self(X)
            base_loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction)

        # ---------------------------------------
        # GAN adversarial loss
        # ---------------------------------------
        # Mask land — important!
        y_pred_masked = y_pred * mask

        D_fake = self.model.D(y_pred_masked)
        loss_adv = adversarial_loss_G(D_fake)

        # Combine
        total_loss = base_loss + self.hparams.lambda_adv * loss_adv

        # ---------------------------------------
        # Log (just the GAN component + total)
        # ---------------------------------------
        self.log("train/G_base_loss", base_loss, on_step=True, on_epoch=True)
        self.log("train/G_adv_loss", loss_adv, on_step=True, on_epoch=True)
        self.log("train/G_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def _training_step_D(self, X, y, mask):
        """
        Discriminator training step (optimizer_idx == 1).
        Includes:
        - real masked ocean field
        - fake masked ocean field (stop gradient)
        - LSGAN discriminator loss
        """

        # ---------------------------------------
        # Real field
        # ---------------------------------------
        y_real_masked = y * mask
        D_real = self.model.D(y_real_masked)

        # ---------------------------------------
        # Fake field (detach from generator graph)
        # ---------------------------------------
        with torch.no_grad():
            if self.use_mdn:
                pi, mu, sigma = self(X)
                y_pred = mdn_expected_value(pi, mu)
            else:
                y_pred = self(X)

        y_fake_masked = y_pred * mask
        D_fake = self.model.D(y_fake_masked)

        # ---------------------------------------
        # GAN loss
        # ---------------------------------------
        loss_D = adversarial_loss_D(D_real, D_fake)

        # Log
        self.log("train/D_loss", loss_D, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/D_real_mean", D_real.mean(), on_step=True, on_epoch=True)
        self.log("train/D_fake_mean", D_fake.mean(), on_step=True, on_epoch=True)

        return loss_D

    def training_step(self, batch, batch_idx):
        X, y, mask, vhm0_for_reconstruction = batch

        # Non-GAN models use automatic optimization
        if self.model_type != "transunet_gan":
            return self._training_step_no_gan(X, y, mask, vhm0_for_reconstruction)

        # GAN models use manual optimization
        opt_g, opt_d = self.optimizers()

        # ========================================
        # Train Generator
        # ========================================
        opt_g.zero_grad()

        if self.use_mdn:
            pi, mu, sigma = self(X)
            y_pred = mdn_expected_value(pi, mu)
            base_loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction, pi, mu, sigma)
        else:
            y_pred = self(X)
            base_loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction)

        # Adversarial loss for generator
        y_pred_masked = (torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0) * mask)
        D_fake = self.model.D(y_pred_masked)
        loss_adv = adversarial_loss_G(D_fake)

        total_loss_g = base_loss + self.lambda_adv * loss_adv

        # Manual backward and step
        self.manual_backward(total_loss_g)
        opt_g.step()

        # Log generator metrics
        with torch.no_grad():
            min_h = min(y_pred.shape[2], y.shape[2])
            min_w = min(y_pred.shape[3], y.shape[3])
            y_pred_crop = y_pred[:, :, :min_h, :min_w]
            y_crop = y[:, :, :min_h, :min_w]
            mask_crop = mask[:, :, :min_h, :min_w]

            mae = torch.abs(y_pred_crop - y_crop)[mask_crop].mean()
            mse = ((y_pred_crop - y_crop) ** 2)[mask_crop].mean()
            rmse = torch.sqrt(mse)

            self.log("train/G_base_loss", base_loss, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/G_adv_loss", loss_adv, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/G_total_loss", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_loss", total_loss_g, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_mae", mae, on_step=True, on_epoch=True)
            self.log("train_rmse", rmse, on_step=True, on_epoch=True)

        # ========================================
        # Train Discriminator (multiple times to strengthen it)
        # ========================================
        for _ in range(self.n_discriminator_updates):
            opt_d.zero_grad()

            # Detach generator output for discriminator training
            with torch.no_grad():
                if self.use_mdn:
                    pi, mu, sigma = self(X)
                    y_pred_detached = mdn_expected_value(pi, mu)
                else:
                    y_pred_detached = self(X)

            y_real_masked = (torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) * mask)
            y_pred_masked = (torch.nan_to_num(y_pred_detached, nan=0.0, posinf=0.0, neginf=0.0) * mask)

            D_real = self.model.D(y_real_masked)
            D_fake = self.model.D(y_pred_masked)

            loss_d = adversarial_loss_D(D_real, D_fake)

            # Manual backward and step
            self.manual_backward(loss_d)
            opt_d.step()

        # Log only the last iteration to avoid cluttering logs
        self.log("train/D_loss", loss_d, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/D_real_mean", D_real.mean(), on_step=True, on_epoch=True)
        self.log("train/D_fake_mean", D_fake.mean(), on_step=True, on_epoch=True)

        # Manually step schedulers if they exist
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                # Step both generator and discriminator schedulers
                for sch in schedulers:
                    if sch is not None:
                        sch.step()
            else:
                schedulers.step()

        return total_loss_g

    def on_validation_epoch_start(self):
        print(f"\n>>> ON_VALIDATION_EPOCH_START CALLED - Epoch {self.current_epoch}")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"\n>>> VALIDATION STEP CALLED - Epoch {self.current_epoch}, Batch {batch_idx}")

        try:
            X, y, mask, vhm0_for_reconstruction = batch  # _ is the bin_id
            if batch_idx == 0:
                print(f"Batch unpacked: X={X.shape}, y={y.shape}, mask={mask.shape}")

            if self.use_mdn:
                pi, mu, sigma = self(X)
                y_pred = mdn_expected_value(pi, mu)
                loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction, pi, mu, sigma)
            else:
                y_pred = self(X)
                loss = self.compute_loss(y_pred, y, mask, vhm0_for_reconstruction)

            if self.model_type == "transunet_gan":
                with torch.no_grad():
                    y_pred_masked = y_pred * mask
                    y_real_masked = y * mask

                    # Get discriminator scores
                    D_real = self.model.D(y_real_masked)
                    D_fake = self.model.D(y_pred_masked)

                    # Log discriminator metrics
                    self.log("val_D_real_mean", D_real.mean(), on_step=False, on_epoch=True)
                    self.log("val_D_fake_mean", D_fake.mean(), on_step=False, on_epoch=True)
                    self.log("val_D_diff", (D_real - D_fake).mean(), on_step=False, on_epoch=True)

        except Exception as e:
            print(f"\n!!! ERROR IN VALIDATION_STEP: {e}")
            import traceback
            traceback.print_exc()
            raise

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
            self.log("val_error_min", (y_pred - y)[mask].min(), on_step=log_on_step, on_epoch=True)
            self.log("val_error_max", (y_pred - y)[mask].max(), on_step=log_on_step, on_epoch=True)
            self.log("val_error_mean", (y_pred - y)[mask].mean(), on_step=log_on_step, on_epoch=True)
            self.log("val_error_p95", torch.quantile(torch.abs(y_pred - y)[mask], 0.95), on_step=log_on_step, on_epoch=True)

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

        return {"loss": loss, "pred": y_pred}

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
        if self.model_type == "transunet_gan":
            # Manual optimization - get first optimizer (generator)
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
        else:
            # Automatic optimization
            lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log("learning_rate", lr, on_epoch=True, prog_bar=True)

    def on_after_backward(self):
        # Only clip gradients in automatic optimization mode
        if self.automatic_optimization:
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

    def _build_scheduler(self, optimizer):
        def get_float(key, default):
            val = scheduler_config.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        scheduler_config = self.lr_scheduler_config
        if not scheduler_config or scheduler_config.get("type", "none") == "none":
            return {}

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
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get("monitor", "val_loss"),
                }

        elif scheduler_type == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(scheduler_config.get("T_max", 50)), eta_min=get_float("eta_min", 1e-6),
            )
            return {
                    "scheduler": scheduler,
                    "interval": "step",   # ✅ CRITICAL — warmup MUST be per-step
                    "frequency": 1,
                }

        elif scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_config.get("step_size", 10)),
                gamma=get_float("gamma", 0.1),
            )
            return {
                    "scheduler": scheduler,
                    "interval": "step",   # ✅ CRITICAL — warmup MUST be per-step
                    "frequency": 1,
                }

        elif scheduler_type == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=get_float("gamma", 0.1)
            )
            return {
                    "scheduler": scheduler,
                    "interval": "step",   # ✅ CRITICAL — warmup MUST be per-step
                    "frequency": 1,
                }

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
            return {
                    "scheduler": scheduler,
                    "interval": "step",   # ✅ CRITICAL — warmup MUST be per-step
                    "frequency": 1,
                }
        elif scheduler_type == "LambdaLR":
            import math
            total_steps = self.trainer.estimated_stepping_batches

            warmup_frac = get_float(scheduler_config.get("warmup_steps", 0.1), 0.1)
            warmup_steps = int(warmup_frac * total_steps)
            print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

            def lr_lambda(step):
                # Warmup: linear increase
                if step < warmup_steps:
                    return max(0.01, step / max(1, warmup_steps))

                # Cosine decay: smooth decrease
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }

        else:
            # Unknown scheduler type, return optimizer only
            return {}

    def configure_optimizers(self):
        if self.optimizer_type == "Adam":
            opt_g = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer_type == "AdamW":
            opt_g = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer_type == "SGD":
            opt_g = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

        if self.model_type != "transunet_gan":
            return {
                "optimizer": opt_g,
                "lr_scheduler": self._build_scheduler(opt_g)
            }

        # Discriminator uses separate (potentially higher) learning rate
        d_lr = self.hparams.lr * self.discriminator_lr_multiplier

        if self.optimizer_type == "Adam":
            opt_d = optim.Adam(self.model.D.parameters(), lr=d_lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer_type == "AdamW":
            opt_d = optim.AdamW(self.model.D.parameters(), lr=d_lr, weight_decay=self.hparams.weight_decay)
        else:
            opt_d = optim.SGD(self.model.D.parameters(), lr=d_lr, weight_decay=self.hparams.weight_decay)

        self.optimizer_info = {
            "optimizer_weight_decay": self.hparams.weight_decay,
            "optimizer_lr_G": self.hparams.lr,
            "optimizer_lr_D": d_lr
        }

        return [
            {  # Generator optimizer + scheduler
                "optimizer": opt_g,
                "lr_scheduler": self._build_scheduler(opt_g)
            },
            {  # Discriminator optimizer + scheduler (keeps G/D ratio constant)
                "optimizer": opt_d,
                "lr_scheduler": self._build_scheduler(opt_d)
            }
        ]

    def _configure_optimizers(self):
        # lr = float(self.hparams.lr) if isinstance(self.hparams.lr, str) else self.hparams.lr
        if self.optimizer_type == "Adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.optimizer_type == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_type}")

        if self.model_type == "transunet_gan":
            optimizer_g = optim.Adam(self.model.G.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            optimizer_d = optim.Adam(self.model.D.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            optimizer = [optimizer_g, optimizer_d]

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
        elif scheduler_type == "LambdaLR":
            import math
            total_steps = self.trainer.estimated_stepping_batches

            warmup_frac = get_float(scheduler_config.get("warmup_steps", 0.1), 0.1)
            warmup_steps = int(warmup_frac * total_steps)
            print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")

            def lr_lambda(step):
                # Warmup: linear increase
                if step < warmup_steps:
                    return max(0.01, step / max(1, warmup_steps))

                # Cosine decay: smooth decrease
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                return 0.5 * (1 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

        else:
            # Unknown scheduler type, return optimizer only
            return optimizer
