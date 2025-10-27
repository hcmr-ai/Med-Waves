#!/usr/bin/env python3
"""
Enhanced Comet ML callback for wave height prediction
"""
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback


class CometVisualizationCallback(Callback):
    """Custom callback for enhanced Comet ML visualizations"""

    def __init__(self, log_every_n_epochs: int = 5):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        # Store metrics history for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_rmses = []
        self.val_rmses = []
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log visualizations at the end of validation epochs"""
        # Collect metrics for plotting
        self._collect_metrics(trainer, pl_module)

        if trainer.current_epoch % self.log_every_n_epochs == 0:
            # self._log_predictions_vs_actual(trainer, pl_module)
            # self._log_error_distribution(trainer, pl_module)
            # self._log_spatial_error_map(trainer, pl_module)
            self._log_loss_curves(trainer, pl_module)
            self._log_accuracy_curves(trainer, pl_module)

            # Clear stored validation batch to free memory
            if hasattr(pl_module, 'last_val_batch'):
                pl_module.last_val_batch = None

    def _log_predictions_vs_actual(self, trainer, pl_module):
        """Create predictions vs actual scatter plot"""
        # Use stored validation results instead of recomputing
        if not hasattr(pl_module, 'last_val_batch') or pl_module.last_val_batch is None:
            return

        y, y_pred, mask = pl_module.last_val_batch

        # Flatten and filter valid pixels
        y_true_flat = y[mask].cpu().numpy()
        y_pred_flat = y_pred[mask].cpu().numpy()

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=1)
        plt.plot([y_true_flat.min(), y_true_flat.max()],
                [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        plt.xlabel('Actual Wave Height (m)')
        plt.ylabel('Predicted Wave Height (m)')
        plt.title(f'Predictions vs Actual (Epoch {trainer.current_epoch})')

        # Calculate R²
        correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        plt.text(0.05, 0.95, f'R² = {correlation**2:.3f}',
                transform=plt.gca().transAxes, fontsize=12)

        # Log to Comet (find Comet logger in the list)
        comet_logger = None
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_figure'):
                    comet_logger = logger
                    break
        elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_figure'):
            comet_logger = trainer.logger

        if comet_logger:
            comet_logger.experiment.log_figure(
                figure_name=f"predictions_vs_actual_epoch_{trainer.current_epoch}",
                figure=plt.gcf(),
                step=trainer.current_epoch
            )
        plt.close()

    def _log_error_distribution(self, trainer, pl_module):
        """Create error distribution histogram"""
        # Use stored validation results instead of recomputing
        if not hasattr(pl_module, 'last_val_batch') or pl_module.last_val_batch is None:
            return

        y, y_pred, mask = pl_module.last_val_batch

        # Calculate errors
        errors = (y_pred - y)[mask].cpu().numpy()

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (m)')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (Epoch {trainer.current_epoch})')
        plt.axvline(errors.mean(), color='red', linestyle='--',
                   label=f'Mean: {errors.mean():.3f}m')
        plt.legend()

        # Log to Comet (find Comet logger in the list)
        comet_logger = None
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_figure'):
                    comet_logger = logger
                    break
        elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_figure'):
            comet_logger = trainer.logger

        if comet_logger:
            comet_logger.experiment.log_figure(
                figure_name=f"error_distribution_epoch_{trainer.current_epoch}",
                figure=plt.gcf(),
                step=trainer.current_epoch
            )
        plt.close()

    def _log_spatial_error_map(self, trainer, pl_module):
        """Create spatial error map"""
        # Use stored validation results instead of recomputing
        if not hasattr(pl_module, 'last_val_batch') or pl_module.last_val_batch is None:
            return

        y, y_pred, mask = pl_module.last_val_batch

        # Calculate spatial error map
        error_map = (y_pred - y)[0, 0].cpu().numpy()  # First sample
        mask_map = mask[0, 0].cpu().numpy()

        # Mask out invalid areas
        error_map[~mask_map] = np.nan

        # Create spatial plot
        plt.figure(figsize=(12, 8))
        im = plt.imshow(error_map, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im, label='Prediction Error (m)')
        plt.title(f'Spatial Error Map (Epoch {trainer.current_epoch})')
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')

        # Log to Comet (find Comet logger in the list)
        comet_logger = None
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_figure'):
                    comet_logger = logger
                    break
        elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_figure'):
            comet_logger = trainer.logger

        if comet_logger:
            comet_logger.experiment.log_figure(
                figure_name=f"spatial_error_map_epoch_{trainer.current_epoch}",
                figure=plt.gcf(),
                step=trainer.current_epoch
            )
        plt.close()

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training statistics"""
        # Find Comet logger
        comet_logger = None
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_metric'):
                    comet_logger = logger
                    break
        elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_metric'):
            comet_logger = trainer.logger

        if not comet_logger:
            return

        # Log learning rate
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        comet_logger.experiment.log_metric("learning_rate", current_lr,
                                           step=trainer.current_epoch)

        # Log model parameters statistics
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                comet_logger.experiment.log_metric(
                    f"grad_norm_{name.replace('.', '_')}",
                    param.grad.norm().item(),
                    step=trainer.current_epoch
                )

    def _collect_metrics(self, trainer, pl_module):
        """Collect metrics from trainer for plotting"""
        # Get metrics from trainer's logged metrics
        logged_metrics = trainer.logged_metrics

        # Extract epoch-level metrics
        train_loss = logged_metrics.get('train_loss_epoch', None)
        val_loss = logged_metrics.get('val_loss', None)
        train_mae = logged_metrics.get('train_mae_epoch', None)
        val_mae = logged_metrics.get('val_mae', None)
        train_rmse = logged_metrics.get('train_rmse_epoch', None)
        val_rmse = logged_metrics.get('val_rmse', None)

        # Store metrics
        self.epochs.append(trainer.current_epoch)
        self.train_losses.append(train_loss.item() if train_loss is not None else None)
        self.val_losses.append(val_loss.item() if val_loss is not None else None)
        self.train_maes.append(train_mae.item() if train_mae is not None else None)
        self.val_maes.append(val_mae.item() if val_mae is not None else None)
        self.train_rmses.append(train_rmse.item() if train_rmse is not None else None)
        self.val_rmses.append(val_rmse.item() if val_rmse is not None else None)

    def _log_loss_curves(self, trainer, pl_module):
        """Create loss curves plot"""
        if len(self.epochs) < 2:
            return

        # Find Comet logger
        comet_logger = self._get_comet_logger(trainer)
        if not comet_logger:
            return

        plt.figure(figsize=(12, 8))

        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        valid_train_losses = [(e, loss) for e, loss in zip(self.epochs, self.train_losses, strict=False) if loss is not None]
        valid_val_losses = [(e, loss) for e, loss in zip(self.epochs, self.val_losses, strict=False) if loss is not None]

        if valid_train_losses:
            epochs_train, losses_train = zip(*valid_train_losses, strict=False)
            plt.plot(epochs_train, losses_train, 'b-', label='Train Loss', linewidth=2)

        if valid_val_losses:
            epochs_val, losses_val = zip(*valid_val_losses, strict=False)
            plt.plot(epochs_val, losses_val, 'r-', label='Val Loss', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot MAE
        plt.subplot(2, 2, 2)
        valid_train_maes = [(e, m) for e, m in zip(self.epochs, self.train_maes, strict=False) if m is not None]
        valid_val_maes = [(e, m) for e, m in zip(self.epochs, self.val_maes, strict=False) if m is not None]

        if valid_train_maes:
            epochs_train, maes_train = zip(*valid_train_maes, strict=False)
            plt.plot(epochs_train, maes_train, 'b-', label='Train MAE', linewidth=2)

        if valid_val_maes:
            epochs_val, maes_val = zip(*valid_val_maes, strict=False)
            plt.plot(epochs_val, maes_val, 'r-', label='Val MAE', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('MAE (m)')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot RMSE
        plt.subplot(2, 2, 3)
        valid_train_rmses = [(e, r) for e, r in zip(self.epochs, self.train_rmses, strict=False) if r is not None]
        valid_val_rmses = [(e, r) for e, r in zip(self.epochs, self.val_rmses, strict=False) if r is not None]

        if valid_train_rmses:
            epochs_train, rmses_train = zip(*valid_train_rmses, strict=False)
            plt.plot(epochs_train, rmses_train, 'b-', label='Train RMSE', linewidth=2)

        if valid_val_rmses:
            epochs_val, rmses_val = zip(*valid_val_rmses, strict=False)
            plt.plot(epochs_val, rmses_val, 'r-', label='Val RMSE', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel('RMSE (m)')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot loss ratio (overfitting indicator)
        plt.subplot(2, 2, 4)
        if valid_train_losses and valid_val_losses:
            # Find common epochs
            train_epochs = set(epochs_train)
            val_epochs = set(epochs_val)
            common_epochs = sorted(train_epochs.intersection(val_epochs))

            if common_epochs:
                ratios = []
                for epoch in common_epochs:
                    train_idx = epochs_train.index(epoch)
                    val_idx = epochs_val.index(epoch)
                    ratio = losses_val[val_idx] / losses_train[train_idx]
                    ratios.append(ratio)

                plt.plot(common_epochs, ratios, 'g-', label='Val/Train Loss Ratio', linewidth=2)
                plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Fit')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Ratio')
                plt.title('Overfitting Indicator')
                plt.legend()
                plt.grid(True, alpha=0.3)

        plt.tight_layout()

        comet_logger.experiment.log_figure(
            figure_name=f"loss_curves_epoch_{trainer.current_epoch}",
            figure=plt.gcf(),
            step=trainer.current_epoch
        )
        plt.close()

    def _log_accuracy_curves(self, trainer, pl_module):
        """Create accuracy-like curves (R² and correlation)"""
        if len(self.epochs) < 2:
            return

        # Find Comet logger
        comet_logger = self._get_comet_logger(trainer)
        if not comet_logger:
            return

        plt.figure(figsize=(12, 6))

        # Calculate R² and correlation from stored validation batches
        # r2_scores = []
        # correlations = []

        # For now, we'll use a simple approach based on MAE improvement
        # In a real implementation, you'd want to store actual predictions
        plt.subplot(1, 2, 1)

        # Plot MAE improvement (inverse of accuracy)
        valid_train_maes = [(e, m) for e, m in zip(self.epochs, self.train_maes, strict=False) if m is not None]
        valid_val_maes = [(e, m) for e, m in zip(self.epochs, self.val_maes, strict=False) if m is not None]

        if valid_train_maes and valid_val_maes:
            epochs_train, maes_train = zip(*valid_train_maes, strict=False)
            epochs_val, maes_val = zip(*valid_val_maes, strict=False)

            plt.plot(epochs_train, maes_train, 'b-', label='Train MAE', linewidth=2)
            plt.plot(epochs_val, maes_val, 'r-', label='Val MAE', linewidth=2)

            plt.xlabel('Epoch')
            plt.ylabel('MAE (m)')
            plt.title('Accuracy (Lower MAE = Better)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot learning rate schedule
        plt.subplot(1, 2, 2)
        lr_schedule = []
        for _ in self.epochs:
            # Get learning rate from optimizer
            if hasattr(trainer, 'optimizers') and trainer.optimizers:
                lr = trainer.optimizers[0].param_groups[0]['lr']
                lr_schedule.append(lr)
            else:
                lr_schedule.append(None)

        valid_lrs = [(e, lr) for e, lr in zip(self.epochs, lr_schedule, strict=False) if lr is not None]
        if valid_lrs:
            epochs_lr, lrs = zip(*valid_lrs, strict=False)
            plt.plot(epochs_lr, lrs, 'g-', label='Learning Rate', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

        plt.tight_layout()

        comet_logger.experiment.log_figure(
            figure_name=f"accuracy_curves_epoch_{trainer.current_epoch}",
            figure=plt.gcf(),
            step=trainer.current_epoch
        )
        plt.close()

    def on_train_end(self, trainer, pl_module):
        """Log final training summary"""
        comet_logger = self._get_comet_logger(trainer)
        if not comet_logger:
            return

        # Log final metrics
        final_train_loss = self.train_losses[-1] if self.train_losses else None
        final_val_loss = self.val_losses[-1] if self.val_losses else None
        final_train_mae = self.train_maes[-1] if self.train_maes else None
        final_val_mae = self.val_maes[-1] if self.val_maes else None

        if final_train_loss is not None:
            comet_logger.experiment.log_metric("final_train_loss", final_train_loss)
        if final_val_loss is not None:
            comet_logger.experiment.log_metric("final_val_loss", final_val_loss)
        if final_train_mae is not None:
            comet_logger.experiment.log_metric("final_train_mae", final_train_mae)
        if final_val_mae is not None:
            comet_logger.experiment.log_metric("final_val_mae", final_val_mae)

        # Log training summary
        summary_text = f"""
Training Summary:
- Total epochs: {trainer.current_epoch + 1}
- Final train loss: {final_train_loss:.6f if final_train_loss else 'N/A'}
- Final val loss: {final_val_loss:.6f if final_val_loss else 'N/A'}
- Final train MAE: {final_train_mae:.6f if final_train_mae else 'N/A'}
- Final val MAE: {final_val_mae:.6f if final_val_mae else 'N/A'}
- Best val loss: {min(self.val_losses) if self.val_losses else 'N/A'}
- Training completed successfully
        """

        comet_logger.experiment.log_text(summary_text)

        # Log final loss curves
        self._log_loss_curves(trainer, pl_module)
        self._log_accuracy_curves(trainer, pl_module)

    def _get_comet_logger(self, trainer):
        """Helper method to get Comet logger from trainer"""
        comet_logger = None
        if isinstance(trainer.logger, list):
            for logger in trainer.logger:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log_figure'):
                    comet_logger = logger
                    break
        elif hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'log_figure'):
            comet_logger = trainer.logger
        return comet_logger
