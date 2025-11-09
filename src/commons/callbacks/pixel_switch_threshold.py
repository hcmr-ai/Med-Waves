import torch
from lightning.pytorch import Callback

class PixelSwitchThresholdCallback(Callback):
    """
    Computes dynamic threshold each validation epoch based on quantile of abs error.
    Stores the threshold inside `trainer.model.current_threshold` so the loss can use it.
    Logs % hard pixels for monitoring.
    """

    def __init__(self, quantile=0.90):
        super().__init__()
        self.quantile = quantile

    def on_validation_epoch_start(self, trainer, pl_module):
        # Prepare buffer for collecting validation errors
        self.val_errors = []

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collects validation errors over full epoch
        Assumes batch = (X, y_true, mask, vhm0)
        """
        _, y_true, mask, _ = batch
        y_pred = outputs["pred"]  # must be returned by validation_step

        diff = (y_pred - y_true).abs()
        valid_errors = diff[mask]            # 1-D flattened vector of valid errors
        self.val_errors.append(valid_errors)  # Keep on GPU

    def on_validation_epoch_end(self, trainer, pl_module):
        # Concatenate all errors for the epoch (on GPU)
        all_errors = torch.cat(self.val_errors)  # shape (N_valid_pixels,)

        # For very large tensors, subsample to avoid quantile() size limits
        max_samples = 10_000_000  # 10M samples is more than enough for accurate quantile
        if all_errors.numel() > max_samples:
            # Randomly sample
            indices = torch.randperm(all_errors.numel(), device=all_errors.device)[:max_samples]
            sampled_errors = all_errors[indices]
        else:
            sampled_errors = all_errors

        # Compute quantile threshold (on GPU, then move to CPU)
        threshold = torch.quantile(sampled_errors, self.quantile).cpu().item()
        pl_module.current_threshold = threshold

        # Compute % hard pixels on full dataset
        hard_ratio = (all_errors > threshold).float().mean().item() * 100

        # Log to Lightning
        trainer.logger.log_metrics(
            {
                "pixel_switch/threshold": threshold,
                "pixel_switch/hard_pixel_pct": hard_ratio,
            },
            step=trainer.global_step,
        )

        print(f"[PixelSwitch] threshold={threshold:.4f}, hard_pixels={hard_ratio:.2f}%")
