import sys
from pathlib import Path

import lightning as pl
import numpy as np
import torch
import torch.optim as optim

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifiers.model_factory import create_model
from src.classifiers.networks.mdn import mdn_expected_value
from src.commons.loss_functions.adversarial_loss import (
    adversarial_loss_D,
    adversarial_loss_G,
)
from src.commons.loss_functions.perceptual_loss import (
    PerceptualLoss,
    WaveFeatureExtractor,
)
from src.commons.loss_functions.ssim import SSIMLoss
from src.commons.losses_factory import compute_loss
from src.commons.scheduler_factory import create_scheduler


class WaveBiasCorrector(pl.LightningModule):
    """
    Lightning module for wave bias correction with multi-task learning support.

    Multi-task configuration via tasks_config:
        - Single task (default): tasks_config=None → ['vhm0'] with loss_type from parameter
        - Multi-task: tasks_config=[
            {'name': 'vhm0', 'loss_type': 'weighted_mse', 'weight': 1.0},
            {'name': 'vtm02', 'loss_type': 'mse', 'weight': 0.5}
          ]
        - Task names (auxiliary_tasks) are automatically inferred from tasks_config
    """
    def __init__(
        self,
        tasks_config, # List of tasks: [{'name': 'vhm0', 'loss_type': 'mse', 'weight': 1.0}, ...]
        in_channels=3,
        lr=1e-3,
        loss_type="weighted_mse",
        lr_scheduler_config=None,
        predict_bias=False,
        filters=None,
        dropout=0.2,
        add_vhm0_residual=False,
        vhm0_channel_index=0,
        weight_decay=1e-4,
        model_type="transunet",
        upsample_mode="nearest",
        pixel_switch_threshold_m=0.45,
        use_mdn=False,
        optimizer_type="Adam",
        lambda_adv=0.01,
        n_discriminator_updates=3,
        discriminator_lr_multiplier=1.0,
        normalizer=None,
        normalize_target=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['normalizer'])
        self.loss_type = loss_type
        self.n_discriminator_updates = n_discriminator_updates
        self.discriminator_lr_multiplier = discriminator_lr_multiplier
        if self.loss_type == "pixel_switch_mse":
            self.pixel_switch_threshold_m = pixel_switch_threshold_m

        if self.loss_type == "mse_perceptual" or self.loss_type == "mse_ssim_perceptual":
            self.perceptual_loss = PerceptualLoss(WaveFeatureExtractor(), layer_weights=[1.0, 1.0, 1.0])

        if self.loss_type == "mse_ssim_perceptual" or self.loss_type == "mse_ssim":
            self.ssim_loss = SSIMLoss()

        if self.loss_type == "smooth_l1" or self.loss_type == "multi_bin_weighted_smooth_l1":
            self.criterion = torch.nn.SmoothL1Loss(beta=0.3, reduction="none")

        self.use_mdn = use_mdn
        self.optimizer_type = optimizer_type
        self.lambda_adv = lambda_adv
        self.model_type = model_type

        # Multi-task or single-task configuration: infer auxiliary_tasks from tasks_config
        # Use provided tasks_config and ensure each task has a loss_type
        self.tasks_config = tasks_config
        # Add loss_type to each task if not already present
        for task in self.tasks_config:
            if 'loss_type' not in task:
                task['loss_type'] = self.loss_type
        self.auxiliary_tasks = [task['name'] for task in tasks_config]

        # Store whether we're in multi-task mode
        self.is_multi_task = len(self.auxiliary_tasks) > 1

        if model_type == "transunet_gan":
            self.automatic_optimization = False

        # Create model using factory
        self.model = create_model(
            model_type=model_type,
            in_channels=in_channels,
            filters=filters,
            dropout=dropout,
            add_vhm0_residual=add_vhm0_residual,
            vhm0_channel_index=vhm0_channel_index,
            upsample_mode=upsample_mode,
            use_mdn=use_mdn,
            auxiliary_tasks=self.auxiliary_tasks,
        )

        self.lr_scheduler_config = lr_scheduler_config or {}
        self.predict_bias = predict_bias
        self.normalizer = normalizer
        self.normalize_target = normalize_target

    def forward(self, x):
        # Handle NaN values in input by replacing with zeros
        x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model(x_clean)

    def _denormalize_bias_prediction(self, prediction, task_name):
        """
        Denormalize predictions if normalization was applied during training.
        Works for both bias predictions and direct wave height predictions.
        
        Args:
            prediction: Normalized prediction tensor (any shape)
            task_name: Name of the task (e.g., 'vhm0', 'vtm02')
        
        Returns:
            Denormalized prediction tensor (same shape as input)
        """
        if not self.normalize_target or self.normalizer is None:
            return prediction
        
        # Get the target column name for this task
        # This assumes the normalizer has access to the task column mappings
        # For vhm0, the target is typically 'corrected_VHM0'
        target_column_map = {
            'vhm0': 'corrected_VHM0',
            'vtm02': 'corrected_VTM02',
        }
        target_col = target_column_map.get(task_name, f'corrected_{task_name.upper()}')
        
        # Set up the normalizer to use the correct target stats
        if target_col in self.normalizer.feature_order_:
            target_idx = self.normalizer.feature_order_.index(target_col)
            if target_idx in self.normalizer.stats_:
                self.normalizer.target_stats_ = self.normalizer.stats_[target_idx]
        
        # Denormalize
        original_shape = prediction.shape
        
        # inverse_transform_torch expects (H, W) or (1, H, W) or (H, W, 1)
        # prediction is typically (B, C, H, W) for batches
        if len(original_shape) == 4:  # (B, C, H, W)
            B, C, H, W = original_shape
            # Process each sample in the batch
            denormalized = torch.zeros_like(prediction)
            for b in range(B):
                for c in range(C):
                    slice_2d = prediction[b, c]  # (H, W)
                    denormalized[b, c] = self.normalizer.inverse_transform_torch(slice_2d)
            return denormalized
        elif len(original_shape) == 3:  # (C, H, W) or (B, H, W)
            # Assume (C, H, W) for single sample
            C, H, W = original_shape
            denormalized = torch.zeros_like(prediction)
            for c in range(C):
                slice_2d = prediction[c]  # (H, W)
                denormalized[c] = self.normalizer.inverse_transform_torch(slice_2d)
            return denormalized
        elif len(original_shape) == 2:  # (H, W)
            return self.normalizer.inverse_transform_torch(prediction)
        elif len(original_shape) == 1:  # Flattened (N,)
            # Cannot denormalize flattened without spatial structure
            # Return as-is (should not happen in practice)
            return prediction
        else:
            return prediction

    def compute_loss(self, y_pred, y_true, mask, vhm0_for_reconstruction, pi=None, mu=None, sigma=None):
        """Compute loss using the unified loss wrapper from losses.py"""
        return compute_loss(
            loss_type=self.loss_type,
            y_pred=y_pred,
            y_true=y_true,
            mask=mask,
            vhm0_for_reconstruction=vhm0_for_reconstruction,
            pi=pi,
            mu=mu,
            sigma=sigma,
            criterion=self.criterion if hasattr(self, 'criterion') else None,
            pixel_switch_threshold_m=self.pixel_switch_threshold_m if hasattr(self, 'pixel_switch_threshold_m') else None,
            perceptual_loss=self.perceptual_loss if hasattr(self, 'perceptual_loss') else None,
            ssim_loss=self.ssim_loss if hasattr(self, 'ssim_loss') else None,
        )

    def compute_multi_task_loss(self, predictions, targets, mask, vhm0_for_reconstruction):
        """
        Compute weighted sum of task-specific losses for multi-task learning.

        Args:
            predictions: Dict of predictions {'task_name': tensor} or single tensor for backward compat
            targets: Dict of targets {'task_name': tensor} or single tensor for backward compat
            mask: Valid pixel mask
            vhm0_for_reconstruction: VHM0 values for reconstruction

        Returns:
            total_loss: Weighted sum of all task losses
            task_losses: Dict of individual task losses {'task_name': loss_value}
        """
        # Backward compatibility: single task
        if not isinstance(predictions, dict):
            loss = self.compute_loss(predictions, targets, mask, vhm0_for_reconstruction)
            # Use actual task name instead of hardcoding 'vhm0'
            task_name = self.auxiliary_tasks[0]
            return loss, {task_name: loss}

        # Multi-task: compute weighted sum
        total_loss = 0.0
        task_losses = {}

        for task_config in self.tasks_config:
            task_name = task_config['name']
            weight = task_config['weight']
            loss_type = task_config['loss_type']

            y_pred = predictions[task_name]
            y_true = targets[task_name]

            # Use task-specific loss type
            task_loss = compute_loss(
                loss_type=loss_type,
                y_pred=y_pred,
                y_true=y_true,
                mask=mask,
                vhm0_for_reconstruction=vhm0_for_reconstruction,
                criterion=self.criterion if hasattr(self, 'criterion') else None,
                pixel_switch_threshold_m=self.pixel_switch_threshold_m if hasattr(self, 'pixel_switch_threshold_m') else None,
                perceptual_loss=self.perceptual_loss if hasattr(self, 'perceptual_loss') else None,
                ssim_loss=self.ssim_loss if hasattr(self, 'ssim_loss') else None,
            )

            total_loss += weight * task_loss
            task_losses[task_name] = task_loss.detach()

        return total_loss, task_losses

    def _compute_and_log_task_metrics(self, predictions, targets, mask, prefix="train"):
        """
        Compute and log metrics for each task.

        Args:
            predictions: Dict of predictions or single tensor
            targets: Dict of targets or single tensor
            mask: Valid pixel mask
            prefix: Logging prefix (train/val)
        """
        # Handle backward compatibility: single task
        if not isinstance(predictions, dict):
            # Use actual task name instead of hardcoding 'vhm0'
            task_name = self.auxiliary_tasks[0]
            predictions = {task_name: predictions}
            targets = {task_name: targets}

        # Compute metrics per task
        for task_name in self.auxiliary_tasks:
            y_pred = predictions[task_name]
            y_true = targets[task_name]

            # Align shapes
            min_h = min(y_pred.shape[2], y_true.shape[2])
            min_w = min(y_pred.shape[3], y_true.shape[3])
            y_pred = y_pred[:, :, :min_h, :min_w]
            y_true = y_true[:, :, :min_h, :min_w]
            mask_crop = mask[:, :, :min_h, :min_w]

            # Calculate metrics
            mae = torch.abs(y_pred - y_true)[mask_crop].mean()
            mse = ((y_pred - y_true) ** 2)[mask_crop].mean()
            rmse = torch.sqrt(mse)

            # Task-specific metric names
            task_suffix = f"_{task_name}" if self.is_multi_task else ""

            # Log metrics
            self.log(f"{prefix}_mae{task_suffix}", mae, on_step=True, on_epoch=True)
            self.log(f"{prefix}_mse{task_suffix}", mse, on_step=True, on_epoch=True)
            self.log(f"{prefix}_rmse{task_suffix}", rmse, on_step=True, on_epoch=True)
            self.log(f"{prefix}_error_min{task_suffix}", (y_pred - y_true)[mask_crop].min(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_error_max{task_suffix}", (y_pred - y_true)[mask_crop].max(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_error_mean{task_suffix}", (y_pred - y_true)[mask_crop].mean(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_error_p95{task_suffix}", torch.quantile(torch.abs(y_pred - y_true)[mask_crop], 0.95), on_step=True, on_epoch=True)

            # Log data statistics
            self.log(f"{prefix}_y_mean{task_suffix}", y_true[mask_crop].mean(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_y_std{task_suffix}", y_true[mask_crop].std(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_pred_mean{task_suffix}", y_pred[mask_crop].mean(), on_step=True, on_epoch=True)
            self.log(f"{prefix}_pred_std{task_suffix}", y_pred[mask_crop].std(), on_step=True, on_epoch=True)

    def _training_step_no_gan(self, X, targets, mask, vhm0_for_reconstruction):
        """Training step for non-GAN models with multi-task support."""
        # Forward pass (returns dict for multi-task or tensor for single-task)
        # For MDN: returns (pi, mu, sigma) tuples per task
        model_output = self(X)

        # Handle MDN vs non-MDN
        if self.use_mdn:
            # Single-task MDN (multi-task MDN not fully supported yet)
            if not isinstance(model_output, dict):
                pi, mu, sigma = model_output
                predictions = mdn_expected_value(pi, mu)
                loss = self.compute_loss(predictions, targets, mask, vhm0_for_reconstruction, pi, mu, sigma)
                # Use actual task name instead of hardcoding 'vhm0'
                task_name = self.auxiliary_tasks[0]
                task_losses = {task_name: loss}
            else:
                # Multi-task MDN: extract expected values for each task
                # Note: MDN loss computation for multi-task needs enhancement
                predictions = {}
                for task_name, (pi, mu, _) in model_output.items():
                    predictions[task_name] = mdn_expected_value(pi, mu)
                loss, task_losses = self.compute_multi_task_loss(predictions, targets, mask, vhm0_for_reconstruction)
        else:
            # Non-MDN: standard loss computation
            predictions = model_output
            loss, task_losses = self.compute_multi_task_loss(predictions, targets, mask, vhm0_for_reconstruction)

        # Log total loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log individual task losses
        for task_name, task_loss in task_losses.items():
            if self.is_multi_task:
                self.log(f"train_loss_{task_name}", task_loss, on_step=True, on_epoch=True)

        # Compute and log per-task metrics
        with torch.no_grad():
            self._compute_and_log_task_metrics(predictions, targets, mask, prefix="train")
            self.log("train_valid_pixels", mask.sum().float(), on_step=True, on_epoch=True)

            # Log sea-bin metrics for training
            # For multi-task, log sea-bins for ALL tasks (with task-specific prefixes)
            tasks_to_log = self.auxiliary_tasks if isinstance(predictions, dict) else [self.auxiliary_tasks[0]]
            
            for task_name in tasks_to_log:
                if isinstance(predictions, dict):
                    y_pred_task = predictions[task_name]
                    y_true_task = targets[task_name] if isinstance(targets, dict) else targets
                else:
                    y_pred_task = predictions
                    y_true_task = targets

                # Align shapes for sea-bin computation
                min_h = min(y_pred_task.shape[2], y_true_task.shape[2])
                min_w = min(y_pred_task.shape[3], y_true_task.shape[3])
                y_pred_task = y_pred_task[:, :, :min_h, :min_w]
                y_true_task = y_true_task[:, :, :min_h, :min_w]
                mask_crop = mask[:, :, :min_h, :min_w]

                # Denormalize predictions for sea-bin metrics (if targets were normalized)
                y_pred_task_denorm = self._denormalize_bias_prediction(y_pred_task, task_name)
                y_true_task_denorm = self._denormalize_bias_prediction(y_true_task, task_name)
                
                # Create task-specific prefix for multi-task logging
                prefix = f"train_{task_name}" if isinstance(predictions, dict) else "train"
                baseline_prefix = f"train_baseline_{task_name}" if isinstance(predictions, dict) else "train_baseline"
                
                if self.predict_bias and vhm0_for_reconstruction is not None:
                    # Reconstruct full wave heights from bias
                    vhm0_for_reconstruction_masked = vhm0_for_reconstruction[:, :, :min_h, :min_w][mask_crop]
                    y_true_wave_heights = vhm0_for_reconstruction_masked + y_true_task_denorm[mask_crop]
                    y_pred_wave_heights = vhm0_for_reconstruction_masked + y_pred_task_denorm[mask_crop]
                    self._log_sea_bin_metrics(y_true_wave_heights, y_pred_wave_heights, prefix)
                    self._log_sea_bin_metrics(y_true_wave_heights, vhm0_for_reconstruction_masked, baseline_prefix)
                else:
                    # Direct wave height prediction - use denormalized values
                    self._log_sea_bin_metrics(y_true_task_denorm[mask_crop], y_pred_task_denorm[mask_crop], prefix)
                    vhm0_crop = vhm0_for_reconstruction[:, :, :min_h, :min_w] if vhm0_for_reconstruction is not None else None
                    if vhm0_crop is not None:
                        self._log_sea_bin_metrics(y_true_task_denorm[mask_crop], vhm0_crop[mask_crop], baseline_prefix)

        return loss

    def training_step(self, batch, batch_idx):
        # Unpack batch: targets can be dict (multi-task) or tensor (single-task)
        X, targets, mask, vhm0_for_reconstruction = batch

        # Non-GAN models use automatic optimization
        if self.model_type != "transunet_gan":
            return self._training_step_no_gan(X, targets, mask, vhm0_for_reconstruction)

        # GAN models use manual optimization
        # NOTE: GAN training currently only supports single-task
        if isinstance(targets, dict):
            # Multi-task not supported for GAN yet - use first task
            y = targets[self.auxiliary_tasks[0]]
            print(f"WARNING: GAN training with multi-task not fully implemented. Using task '{self.auxiliary_tasks[0]}' only.")
        else:
            y = targets

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
        """Validation step with multi-task support."""
        if batch_idx == 0:
            print(f"\n>>> VALIDATION STEP CALLED - Epoch {self.current_epoch}, Batch {batch_idx}")

        try:
            # Unpack batch: targets can be dict (multi-task) or tensor (single-task)
            X, targets, mask, vhm0_for_reconstruction = batch
            if batch_idx == 0:
                if isinstance(targets, dict):
                    print(f"Batch unpacked: X={X.shape}, targets={list(targets.keys())}, mask={mask.shape}")
                else:
                    print(f"Batch unpacked: X={X.shape}, targets={targets.shape}, mask={mask.shape}")

            # Forward pass (returns dict for multi-task or tensor for single-task)
            # For MDN: returns (pi, mu, sigma) tuples per task
            model_output = self(X)

            # Handle MDN vs non-MDN
            if self.use_mdn:
                # Single-task MDN (multi-task MDN not fully supported yet)
                if not isinstance(model_output, dict):
                    pi, mu, sigma = model_output
                    predictions = mdn_expected_value(pi, mu)
                    loss = self.compute_loss(predictions, targets, mask, vhm0_for_reconstruction, pi, mu, sigma)
                    # Use actual task name instead of hardcoding 'vhm0'
                    task_name = self.auxiliary_tasks[0]
                    task_losses = {task_name: loss}
                else:
                    # Multi-task MDN: extract expected values for each task
                    # Note: MDN loss computation for multi-task needs enhancement
                    predictions = {}
                    for task_name, (pi, mu, _) in model_output.items():
                        predictions[task_name] = mdn_expected_value(pi, mu)
                    loss, task_losses = self.compute_multi_task_loss(predictions, targets, mask, vhm0_for_reconstruction)
            else:
                # Non-MDN: standard loss computation
                predictions = model_output
                loss, task_losses = self.compute_multi_task_loss(predictions, targets, mask, vhm0_for_reconstruction)

            # GAN-specific validation: log discriminator metrics
            if self.model_type == "transunet_gan":
                with torch.no_grad():
                    # Extract single task prediction for GAN discriminator
                    if isinstance(predictions, dict):
                        y_pred = predictions[self.auxiliary_tasks[0]]
                        y = targets[self.auxiliary_tasks[0]] if isinstance(targets, dict) else targets
                    else:
                        y_pred = predictions
                        y = targets

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

        # Log total loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Log individual task losses
        for task_name, task_loss in task_losses.items():
            if self.is_multi_task:
                self.log(f"val_loss_{task_name}", task_loss, on_step=False, on_epoch=True)

        # Compute and log per-task metrics
        with torch.no_grad():
            self._compute_and_log_task_metrics(predictions, targets, mask, prefix="val")
            self.log("val_valid_pixels", mask.sum().float(), on_epoch=True)

            # Log sea-bin metrics for validation
            # For multi-task, log sea-bins for ALL tasks (with task-specific prefixes)
            tasks_to_log = self.auxiliary_tasks if isinstance(predictions, dict) else [self.auxiliary_tasks[0]]
            
            for task_name in tasks_to_log:
                if isinstance(predictions, dict):
                    y_pred_task = predictions[task_name]
                    y_true_task = targets[task_name] if isinstance(targets, dict) else targets
                else:
                    y_pred_task = predictions
                    y_true_task = targets

                # Align shapes for sea-bin computation
                min_h = min(y_pred_task.shape[2], y_true_task.shape[2])
                min_w = min(y_pred_task.shape[3], y_true_task.shape[3])
                y_pred_task = y_pred_task[:, :, :min_h, :min_w]
                y_true_task = y_true_task[:, :, :min_h, :min_w]
                mask_crop = mask[:, :, :min_h, :min_w]

                # Denormalize predictions for sea-bin metrics (if targets were normalized)
                y_pred_task_denorm = self._denormalize_bias_prediction(y_pred_task, task_name)
                y_true_task_denorm = self._denormalize_bias_prediction(y_true_task, task_name)
                
                # Create task-specific prefix for multi-task logging
                prefix = f"val_{task_name}" if isinstance(predictions, dict) else "val"
                baseline_prefix = f"val_baseline_{task_name}" if isinstance(predictions, dict) else "val_baseline"
                
                if self.predict_bias and vhm0_for_reconstruction is not None:
                    # Reconstruct full wave heights from bias
                    vhm0_for_reconstruction_masked = vhm0_for_reconstruction[:, :, :min_h, :min_w][mask_crop]
                    y_true_wave_heights = vhm0_for_reconstruction_masked + y_true_task_denorm[mask_crop]
                    y_pred_wave_heights = vhm0_for_reconstruction_masked + y_pred_task_denorm[mask_crop]
                    self._log_sea_bin_metrics(y_true_wave_heights, y_pred_wave_heights, prefix)
                    self._log_sea_bin_metrics(y_true_wave_heights, vhm0_for_reconstruction_masked, baseline_prefix)
                else:
                    # Direct wave height prediction - use denormalized values
                    self._log_sea_bin_metrics(y_true_task_denorm[mask_crop], y_pred_task_denorm[mask_crop], prefix)
                    vhm0_crop = vhm0_for_reconstruction[:, :, :min_h, :min_w] if vhm0_for_reconstruction is not None else None
                    if vhm0_crop is not None:
                        self._log_sea_bin_metrics(y_true_task_denorm[mask_crop], vhm0_crop[mask_crop], baseline_prefix)

        return {"loss": loss, "pred": predictions}

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
        # Last bin is unbounded to capture all extreme wave heights
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
            {"name": "extreme_14plus", "min": 14.0, "max": float("inf")}
        ]

        for bin_config in sea_bins:
            bin_name = bin_config["name"]
            bin_min = bin_config["min"]
            bin_max = bin_config["max"]

            # Filter data for this sea state bin
            # Use <= for the last bin to capture all values >= bin_min
            if bin_max == float("inf"):
                mask = y_true_np >= bin_min
            else:
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
        """Build scheduler using scheduler factory"""
        scheduler_cfg, scheduler_metadata = create_scheduler(
            optimizer=optimizer,
            scheduler_config=self.lr_scheduler_config,
            total_steps=self.trainer.estimated_stepping_batches if hasattr(self, 'trainer') else None,
            max_epochs=self.trainer.max_epochs if hasattr(self, 'trainer') else None,
        )

        # Store metadata for logging if provided
        if scheduler_metadata:
            self.scheduler_info = scheduler_metadata

        return scheduler_cfg

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
