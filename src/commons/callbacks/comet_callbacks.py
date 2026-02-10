#!/usr/bin/env python3
"""
Enhanced Comet ML callback for wave height prediction
"""
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback

import logging
logger = logging.getLogger(__name__)

class CometVisualizationCallback(Callback):
    """Custom callback for enhanced Comet ML visualizations"""
    SEA_BINS = ["calm", "light", "moderate", "rough", "very_rough", 
                   "extreme_5_6", "extreme_6_7", "extreme_7_8", "extreme_8_9", 
                   "extreme_9_10", "extreme_10_11", "extreme_11_12", 
                   "extreme_12_13", "extreme_13_14", "extreme_14plus"]

    def __init__(self, log_every_n_epochs: int = 5, tasks: list = None):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        # Track tasks for multi-task learning (None means single-task, will be detected automatically)
        self.tasks = tasks
        self.detected_tasks = []
        
        # Store metrics history for plotting
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_rmses = []
        self.val_rmses = []
        self.epochs = []
        self.val_epochs = []
        
        # Store sea-bin metrics history per task
        # Structure: {task_name: [epoch_dict1, epoch_dict2, ...]}
        self.train_sea_bin_metrics = {}
        self.val_sea_bin_metrics = {}
        
        # Store baseline sea-bin metrics history per task
        self.train_baseline_sea_bin_metrics = {}
        self.val_baseline_sea_bin_metrics = {}
    
    def _detect_tasks_from_metrics(self, logged_metrics, phase='train'):
        """Auto-detect tasks from metric names (single-task vs multi-task)"""
        if self.tasks:
            self.detected_tasks = self.tasks
            return
        
        # Look for sea-bin metrics to detect task names
        # Multi-task pattern: train_vhm0_calm_mae or val_vtm02_calm_mae
        # Single-task pattern: train_calm_mae or val_calm_mae
        import re
        task_pattern = re.compile(rf'{phase}_(\w+?)_({"|".join(self.SEA_BINS)})_mae')
        
        for metric_name in logged_metrics.keys():
            match = task_pattern.match(metric_name)
            if match:
                task_name = match.group(1)
                if task_name not in self.detected_tasks:
                    self.detected_tasks.append(task_name)
        
        # If no tasks detected, assume single-task with default name
        if not self.detected_tasks:
            self.detected_tasks = ['default']
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log training statistics"""
        if not self.epochs or self.epochs[-1] != trainer.current_epoch:
            self.epochs.append(trainer.current_epoch)

        logged_metrics = trainer.logged_metrics
        train_loss = logged_metrics.get('train_loss_epoch', None)
        train_mae = logged_metrics.get('train_mae_epoch', None)
        train_rmse = logged_metrics.get('train_rmse_epoch', None)

        self.train_losses.append(train_loss.item() if train_loss is not None else None)
        self.train_maes.append(train_mae.item() if train_mae is not None else None)
        self.train_rmses.append(train_rmse.item() if train_rmse is not None else None)

        # Auto-detect tasks from metric names if not already detected
        if not self.detected_tasks:
            self._detect_tasks_from_metrics(logged_metrics, 'train')
        
        # Collect sea-bin metrics for each task
        for task_name in self.detected_tasks:
            # Initialize task storage if needed
            if task_name not in self.train_sea_bin_metrics:
                self.train_sea_bin_metrics[task_name] = []
            if task_name not in self.train_baseline_sea_bin_metrics:
                self.train_baseline_sea_bin_metrics[task_name] = []
            
            train_sea_bin_epoch = {}
            train_baseline_sea_bin_epoch = {}
            
            # Determine metric prefix (single-task vs multi-task)
            prefix = f'train_{task_name}_' if len(self.detected_tasks) > 1 else 'train_'
            baseline_prefix = f'train_baseline_{task_name}_' if len(self.detected_tasks) > 1 else 'train_baseline_'
            
            for bin_name in self.SEA_BINS:
                # Training sea-bin metrics
                train_bin_mae = logged_metrics.get(f'{prefix}{bin_name}_mae_epoch', None)
                train_bin_rmse = logged_metrics.get(f'{prefix}{bin_name}_rmse_epoch', None)
                train_bin_bias = logged_metrics.get(f'{prefix}{bin_name}_bias_epoch', None)
                train_bin_count = logged_metrics.get(f'{prefix}{bin_name}_count_epoch', None)
                
                if train_bin_mae is not None:
                    train_sea_bin_epoch[bin_name] = {
                        'mae': train_bin_mae.item() if hasattr(train_bin_mae, 'item') else train_bin_mae,
                        'rmse': train_bin_rmse.item() if hasattr(train_bin_rmse, 'item') else train_bin_rmse,
                        'bias': train_bin_bias.item() if hasattr(train_bin_bias, 'item') else train_bin_bias,
                        'count': train_bin_count.item() if hasattr(train_bin_count, 'item') else train_bin_count
                    }
                
                # Training baseline sea-bin metrics
                train_baseline_bin_mae = logged_metrics.get(f'{baseline_prefix}{bin_name}_mae_epoch', None)
                train_baseline_bin_rmse = logged_metrics.get(f'{baseline_prefix}{bin_name}_rmse_epoch', None)
                train_baseline_bin_bias = logged_metrics.get(f'{baseline_prefix}{bin_name}_bias_epoch', None)
                train_baseline_bin_count = logged_metrics.get(f'{baseline_prefix}{bin_name}_count_epoch', None)
                
                if train_baseline_bin_mae is not None:
                    train_baseline_sea_bin_epoch[bin_name] = {
                        'mae': train_baseline_bin_mae.item() if hasattr(train_baseline_bin_mae, 'item') else train_baseline_bin_mae,
                        'rmse': train_baseline_bin_rmse.item() if hasattr(train_baseline_bin_rmse, 'item') else train_baseline_bin_rmse,
                        'bias': train_baseline_bin_bias.item() if hasattr(train_baseline_bin_bias, 'item') else train_baseline_bin_bias,
                        'count': train_baseline_bin_count.item() if hasattr(train_baseline_bin_count, 'item') else train_baseline_bin_count
                    }
            
            self.train_sea_bin_metrics[task_name].append(train_sea_bin_epoch)
            self.train_baseline_sea_bin_metrics[task_name].append(train_baseline_sea_bin_epoch)

        comet_logger = self._get_comet_logger(trainer)

        if not comet_logger:
            logger.info("No Comet logger found in on_train_epoch_end")
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

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log visualizations at the end of validation epochs"""
        # Collect metrics for plotting
        if trainer.sanity_checking:
            logger.info("Sanity checking, skipping validation epoch end")
            return
            
        self._collect_metrics(trainer)

        if hasattr(pl_module, 'last_val_batch'):
            pl_module.last_val_batch = None
            
        # if trainer.current_epoch % self.log_every_n_epochs == 0:
            # self._log_spatial_error_map(trainer, pl_module)
            # self._log_loss_curves(trainer, pl_module)
            # self._log_accuracy_curves(trainer, pl_module)

            # Clear stored validation batch to free memory
            # if hasattr(pl_module, 'last_val_batch'):
            #     pl_module.last_val_batch = None
    
    def on_train_end(self, trainer, pl_module):
        """Log final training summary"""
        logger.info(f"on_train_end called - trainer.logger type: {type(trainer.logger)}")
        logger.info(f"trainer.logger is list: {isinstance(trainer.logger, list)}")
        
        comet_logger = self._get_comet_logger(trainer)
        if not comet_logger:
            logger.info("No Comet logger found in on_train_end")

            logger.info("Continuing without Comet logging...")
            return

        # Debug: Print metrics arrays
        logger.info(f"Final metrics arrays - train_losses: {len(self.train_losses)}, val_losses: {len(self.val_losses)}")
        logger.info(f"train_losses: {self.train_losses}")
        logger.info(f"val_losses: {self.val_losses}")

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

        # Create final sea-bin analysis plots
        self._log_loss_curves(trainer)
        self._log_accuracy_curves(trainer, pl_module)
        self._log_sea_bin_curves(trainer, pl_module)

        # Log training summary
        summary_text = f"""
Training Summary:
- Total epochs: {trainer.current_epoch + 1}
- Final train loss: {final_train_loss if final_train_loss is not None else 'N/A'}
- Final val loss: {final_val_loss if final_val_loss is not None else 'N/A'}
- Final train MAE: {final_train_mae if final_train_mae is not None else 'N/A'}
- Final val MAE: {final_val_mae if final_val_mae is not None else 'N/A'}
- Best val loss: {min(self.val_losses) if self.val_losses else 'N/A'}
- Training completed successfully
"""

        comet_logger.experiment.log_text(summary_text)

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

    def _collect_metrics(self, trainer):
        """Collect metrics from trainer for plotting"""
        # Get metrics from trainer's logged metrics
        logged_metrics = trainer.logged_metrics
        
        self.val_epochs.append(trainer.current_epoch)
            
        # Debug: Print available metrics
        # logger.info(f"Available logged metrics: {list(logged_metrics.keys())}")

        # Extract epoch-level metrics
        # train_loss = logged_metrics.get('train_loss_step', None)
        val_loss = logged_metrics.get('val_loss', None)
        # train_mae = logged_metrics.get('train_mae_step', None)
        val_mae = logged_metrics.get('val_mae', None)
        # train_rmse = logged_metrics.get('train_rmse_step', None)
        val_rmse = logged_metrics.get('val_rmse', None)
        
        # Debug: Print found metrics
        logger.info(f"Found metrics - val_loss: {val_loss}, val_mae: {val_mae}, val_rmse: {val_rmse}")

        # Auto-detect tasks from metric names if not already detected
        if not self.detected_tasks:
            self._detect_tasks_from_metrics(logged_metrics, 'val')

        # Store overall metrics
        self.val_losses.append(val_loss.item() if val_loss is not None else None)
        self.val_maes.append(val_mae.item() if val_mae is not None else None)
        self.val_rmses.append(val_rmse.item() if val_rmse is not None else None)
        
        # Collect sea-bin metrics for each task
        for task_name in self.detected_tasks:
            # Initialize task storage if needed
            if task_name not in self.val_sea_bin_metrics:
                self.val_sea_bin_metrics[task_name] = []
            if task_name not in self.val_baseline_sea_bin_metrics:
                self.val_baseline_sea_bin_metrics[task_name] = []
            
            val_sea_bin_epoch = {}
            val_baseline_sea_bin_epoch = {}
            
            # Determine metric prefix (single-task vs multi-task)
            prefix = f'val_{task_name}_' if len(self.detected_tasks) > 1 else 'val_'
            baseline_prefix = f'val_baseline_{task_name}_' if len(self.detected_tasks) > 1 else 'val_baseline_'
            
            for bin_name in self.SEA_BINS:
                # Validation sea-bin metrics
                val_bin_mae = logged_metrics.get(f'{prefix}{bin_name}_mae', None)
                val_bin_rmse = logged_metrics.get(f'{prefix}{bin_name}_rmse', None)
                val_bin_bias = logged_metrics.get(f'{prefix}{bin_name}_bias', None)
                val_bin_count = logged_metrics.get(f'{prefix}{bin_name}_count', None)
                
                if val_bin_mae is not None:
                    val_sea_bin_epoch[bin_name] = {
                        'mae': val_bin_mae.item() if hasattr(val_bin_mae, 'item') else val_bin_mae,
                        'rmse': val_bin_rmse.item() if hasattr(val_bin_rmse, 'item') else val_bin_rmse,
                        'bias': val_bin_bias.item() if hasattr(val_bin_bias, 'item') else val_bin_bias,
                        'count': val_bin_count.item() if hasattr(val_bin_count, 'item') else val_bin_count
                    }
                
                # Validation baseline sea-bin metrics
                val_baseline_bin_mae = logged_metrics.get(f'{baseline_prefix}{bin_name}_mae', None)
                val_baseline_bin_rmse = logged_metrics.get(f'{baseline_prefix}{bin_name}_rmse', None)
                val_baseline_bin_bias = logged_metrics.get(f'{baseline_prefix}{bin_name}_bias', None)
                val_baseline_bin_count = logged_metrics.get(f'{baseline_prefix}{bin_name}_count', None)
                
                if val_baseline_bin_mae is not None:
                    val_baseline_sea_bin_epoch[bin_name] = {
                        'mae': val_baseline_bin_mae.item() if hasattr(val_baseline_bin_mae, 'item') else val_baseline_bin_mae,
                        'rmse': val_baseline_bin_rmse.item() if hasattr(val_baseline_bin_rmse, 'item') else val_baseline_bin_rmse,
                        'bias': val_baseline_bin_bias.item() if hasattr(val_baseline_bin_bias, 'item') else val_baseline_bin_bias,
                        'count': val_baseline_bin_count.item() if hasattr(val_baseline_bin_count, 'item') else val_baseline_bin_count
                    }
            
            self.val_sea_bin_metrics[task_name].append(val_sea_bin_epoch)
            self.val_baseline_sea_bin_metrics[task_name].append(val_baseline_sea_bin_epoch)

    def _log_loss_curves(self, trainer):
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
        valid_val_losses = [(e, loss) for e, loss in zip(self.val_epochs, self.val_losses, strict=False) if loss is not None]

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
        valid_val_maes = [(e, m) for e, m in zip(self.val_epochs, self.val_maes, strict=False) if m is not None]

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

    def _log_sea_bin_curves(self, trainer, pl_module):
        """Create sea-bin performance bar charts for all tasks"""
        if len(self.epochs) < 2:
            return

        # Find Comet logger
        comet_logger = self._get_comet_logger(trainer)
        if not comet_logger:
            return

        # Check if we have sea-bin data
        if not self.train_sea_bin_metrics or not self.val_sea_bin_metrics:
            print("No sea-bin metrics available for plotting")
            return

        # Define wave height ranges for each bin
        wave_bins = {
            "calm": (0.0, 1.0),
            "light": (1.0, 2.0),
            "moderate": (2.0, 3.0),
            "rough": (3.0, 4.0),
            "very_rough": (4.0, 5.0),
            "extreme_5_6": (5.0, 6.0),
            "extreme_6_7": (6.0, 7.0),
            "extreme_7_8": (7.0, 8.0),
            "extreme_8_9": (8.0, 9.0),
            "extreme_9_10": (9.0, 10.0),
            "extreme_10_11": (10.0, 11.0),
            "extreme_11_12": (11.0, 12.0),
            "extreme_12_13": (12.0, 13.0),
            "extreme_13_14": (13.0, 14.0),
            "extreme_14plus": (14.0, float('inf'))
        }
        
        # Create separate plots for each task
        for task_name in self.detected_tasks:
            if task_name not in self.train_sea_bin_metrics or task_name not in self.val_sea_bin_metrics:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            task_title = f' - Task: {task_name}' if len(self.detected_tasks) > 1 else ''
            fig.suptitle(f'Sea-Bin Performance Analysis (Final Epoch){task_title}', fontsize=16)

            # Get final epoch data for this task
            final_train_metrics = self.train_sea_bin_metrics[task_name][-1] if self.train_sea_bin_metrics[task_name] else {}
            final_val_metrics = self.val_sea_bin_metrics[task_name][-1] if self.val_sea_bin_metrics[task_name] else {}

            # Get final epoch baseline data for this task
            final_train_baseline_metrics = self.train_baseline_sea_bin_metrics[task_name][-1] if self.train_baseline_sea_bin_metrics[task_name] else {}
            final_val_baseline_metrics = self.val_baseline_sea_bin_metrics[task_name][-1] if self.val_baseline_sea_bin_metrics[task_name] else {}

            # Prepare data for plotting
            bin_labels = []
            train_maes = []
            val_maes = []
            train_rmses = []
            val_rmses = []
            train_biases = []
            val_biases = []
            train_counts = []
            val_counts = []
            
            # Baseline metrics
            train_baseline_maes = []
            val_baseline_maes = []
            train_baseline_rmses = []
            val_baseline_rmses = []
            train_baseline_biases = []
            val_baseline_biases = []

            for bin_name in self.SEA_BINS:
                if bin_name in final_train_metrics and bin_name in final_val_metrics:
                    # Create label with wave height range
                    min_val, max_val = wave_bins[bin_name]
                    if max_val == float('inf'):
                        label = f"{min_val:.1f}+m"
                    else:
                        label = f"{min_val:.1f}-{max_val:.1f}m"
                    bin_labels.append(label)
                    
                    # Collect model metrics
                    train_maes.append(final_train_metrics[bin_name]['mae'])
                    val_maes.append(final_val_metrics[bin_name]['mae'])
                    train_rmses.append(final_train_metrics[bin_name]['rmse'])
                    val_rmses.append(final_val_metrics[bin_name]['rmse'])
                    train_biases.append(final_train_metrics[bin_name]['bias'])
                    val_biases.append(final_val_metrics[bin_name]['bias'])
                    train_counts.append(final_train_metrics[bin_name]['count'])
                    val_counts.append(final_val_metrics[bin_name]['count'])
                    
                    # Collect baseline metrics (if available)
                    # Use NaN for missing data instead of 0 to avoid misleading plots
                    if bin_name in final_train_baseline_metrics:
                        train_baseline_maes.append(final_train_baseline_metrics[bin_name]['mae'])
                        train_baseline_rmses.append(final_train_baseline_metrics[bin_name]['rmse'])
                        train_baseline_biases.append(final_train_baseline_metrics[bin_name]['bias'])
                    else:
                        train_baseline_maes.append(0)  # Will be invisible in plot
                        train_baseline_rmses.append(0)  # Will be invisible in plot
                        train_baseline_biases.append(0)  # Will be invisible in plot
                        
                    if bin_name in final_val_baseline_metrics:
                        val_baseline_maes.append(final_val_baseline_metrics[bin_name]['mae'])
                        val_baseline_rmses.append(final_val_baseline_metrics[bin_name]['rmse'])
                        val_baseline_biases.append(final_val_baseline_metrics[bin_name]['bias'])
                    else:
                        val_baseline_maes.append(0)  # Will be invisible in plot
                        val_baseline_rmses.append(0)  # Will be invisible in plot
                        val_baseline_biases.append(0)  # Will be invisible in plot

            if not bin_labels:
                print(f"No sea-bin data available for final epoch - task: {task_name}")
                continue

            # Set up bar positions for side-by-side comparison
            x = np.arange(len(bin_labels))
            width = 0.2  # Narrower bars for 4 groups

            # Plot 1: MAE comparison (Model vs Baseline)
            ax1 = axes[0, 0]
        bars1_train_model = ax1.bar(x - 1.5*width, train_maes, width, label='Train Model', alpha=0.8, color='skyblue')
        bars1_val_model = ax1.bar(x - 0.5*width, val_maes, width, label='Val Model', alpha=0.8, color='lightcoral')
        bars1_train_baseline = ax1.bar(x + 0.5*width, train_baseline_maes, width, label='Train Baseline', alpha=0.6, color='navy')
        bars1_val_baseline = ax1.bar(x + 1.5*width, val_baseline_maes, width, label='Val Baseline', alpha=0.6, color='darkred')
        
        ax1.set_xlabel('Wave Height Range')
        ax1.set_ylabel('MAE (m)')
        ax1.set_title('Mean Absolute Error: Model vs Baseline')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1_train_model, train_maes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for bar, value in zip(bars1_val_model, val_maes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars1_train_baseline, train_baseline_maes)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_train_baseline_metrics:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars1_val_baseline, val_baseline_maes)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_val_baseline_metrics:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)

        # Plot 2: RMSE comparison (Model vs Baseline)
        ax2 = axes[0, 1]
        bars2_train_model = ax2.bar(x - 1.5*width, train_rmses, width, label='Train Model', alpha=0.8, color='lightgreen')
        bars2_val_model = ax2.bar(x - 0.5*width, val_rmses, width, label='Val Model', alpha=0.8, color='orange')
        bars2_train_baseline = ax2.bar(x + 0.5*width, train_baseline_rmses, width, label='Train Baseline', alpha=0.6, color='darkgreen')
        bars2_val_baseline = ax2.bar(x + 1.5*width, val_baseline_rmses, width, label='Val Baseline', alpha=0.6, color='darkorange')
        
        ax2.set_xlabel('Wave Height Range')
        ax2.set_ylabel('RMSE (m)')
        ax2.set_title('Root Mean Square Error: Model vs Baseline')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2_train_model, train_rmses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for bar, value in zip(bars2_val_model, val_rmses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars2_train_baseline, train_baseline_rmses)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_train_baseline_metrics:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars2_val_baseline, val_baseline_rmses)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_val_baseline_metrics:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)

        # Plot 3: Bias comparison (Model vs Baseline)
        ax3 = axes[1, 0]
        bars3_train_model = ax3.bar(x - 1.5*width, train_biases, width, label='Train Model', alpha=0.8, color='gold')
        bars3_val_model = ax3.bar(x - 0.5*width, val_biases, width, label='Val Model', alpha=0.8, color='purple')
        bars3_train_baseline = ax3.bar(x + 0.5*width, train_baseline_biases, width, label='Train Baseline', alpha=0.6, color='darkgoldenrod')
        bars3_val_baseline = ax3.bar(x + 1.5*width, val_baseline_biases, width, label='Val Baseline', alpha=0.6, color='indigo')
        
        ax3.set_xlabel('Wave Height Range')
        ax3.set_ylabel('Bias (m)')
        ax3.set_title('Bias: Model vs Baseline')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add value labels on bars (for bias, handle negative values correctly)
        for bar, value in zip(bars3_train_model, train_biases):
            y_pos = bar.get_height() + 0.001 if value >= 0 else bar.get_height() - 0.015
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=7)
        for bar, value in zip(bars3_val_model, val_biases):
            y_pos = bar.get_height() + 0.001 if value >= 0 else bar.get_height() - 0.015
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars3_train_baseline, train_baseline_biases)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_train_baseline_metrics:
                y_pos = bar.get_height() + 0.001 if value >= 0 else bar.get_height() - 0.015
                ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=7)
        for i, (bar, value) in enumerate(zip(bars3_val_baseline, val_baseline_biases)):
            # Only show if baseline data exists (check against the bin in the metrics dict)
            if self.SEA_BINS[i] in final_val_baseline_metrics:
                y_pos = bar.get_height() + 0.001 if value >= 0 else bar.get_height() - 0.015
                ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top', fontsize=7)

        # Plot 4: Sample counts (Model only - same for baseline)
        ax4 = axes[1, 1]
        bars4_train = ax4.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='lightblue')
        bars4_val = ax4.bar(x + width/2, val_counts, width, label='Val', alpha=0.8, color='pink')
        
        ax4.set_xlabel('Wave Height Range')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Sample Distribution by Sea State')
        ax4.set_xticks(x)
        ax4.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars4_train, train_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_counts) * 0.01,
                    f'{value:,}', ha='center', va='bottom', fontsize=8)
        for bar, value in zip(bars4_val, val_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_counts) * 0.01,
                    f'{value:,}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()

            figure_name = f"sea_bin_analysis_{task_name}_epoch_{trainer.current_epoch}" if len(self.detected_tasks) > 1 else f"sea_bin_analysis_epoch_{trainer.current_epoch}"
            comet_logger.experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=trainer.current_epoch
            )
            plt.close()

    def _get_comet_logger(self, trainer):
        """Helper method to get Comet logger from trainer"""
        comet_logger = None
        
        # Use trainer.loggers (contains all loggers) instead of trainer.logger (primary only)
        callback_loggers = trainer.loggers if hasattr(trainer, 'loggers') else []
        
        for callback_logger in callback_loggers:
            # Check if this is a Comet logger by looking for the experiment attribute
            # and the log_figure method
            if (hasattr(callback_logger, 'experiment') and 
                hasattr(callback_logger.experiment, 'log_figure')):
                comet_logger = callback_logger
                break
        
        if comet_logger is None:
            logger.info(f"No Comet logger found. Available loggers: {[type(l) for l in callback_loggers]}")
            for i, logger_obj in enumerate(callback_loggers):
                logger.info(f"  Logger {i}: {type(logger_obj)}")
                logger.info(f"    Has experiment: {hasattr(logger_obj, 'experiment')}")
                if hasattr(logger_obj, 'experiment'):
                    logger.info(f"    Has log_figure: {hasattr(logger_obj.experiment, 'log_figure')}")
        
        return comet_logger
