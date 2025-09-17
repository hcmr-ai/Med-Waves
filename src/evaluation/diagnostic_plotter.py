"""
DiagnosticPlotter class for creating model evaluation plots.

This class handles all diagnostic plotting functionality that was previously
embedded in the FullDatasetTrainer class.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DiagnosticPlotter:
    """Handles creation of diagnostic plots for model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DiagnosticPlotter.
        
        Args:
            config: Configuration dictionary containing plotting settings
        """
        self.config = config
        self.diagnostics_config = config.get("diagnostics", {})
        
    def create_diagnostic_plots(self, 
                              trainer: Any, 
                              test_predictions: np.ndarray) -> None:
        """
        Create comprehensive diagnostic plots for model evaluation.
        
        Args:
            trainer: The trainer object containing model, data, and metrics
            test_predictions: Test set predictions
        """
        plots_dir = Path(self.diagnostics_config.get("plots_save_path", "diagnostic_plots"))
        plots_dir.mkdir(exist_ok=True)
        
        # Create individual plot types
        self._create_predictions_vs_actual_plots(trainer, test_predictions, plots_dir)
        self._create_residuals_plots(trainer, test_predictions, plots_dir)
        self._create_learning_curves_plot(trainer, plots_dir)
        self._create_snr_comparison_plot(trainer, plots_dir)
        
        logger.info(f"Diagnostic plots saved to {plots_dir}")
        
        # Log SNR values
        if trainer.current_train_metrics and trainer.current_test_metrics:
            train_snr = trainer.current_train_metrics.get('snr', 0)
            train_snr_db = trainer.current_train_metrics.get('snr_db', 0)
            test_snr = trainer.current_test_metrics.get('snr', 0)
            test_snr_db = trainer.current_test_metrics.get('snr_db', 0)
            logger.info(f"SNR - Train: {train_snr:.1f} ({train_snr_db:.1f} dB), Test: {test_snr:.1f} ({test_snr_db:.1f} dB)")
    
    def _create_predictions_vs_actual_plots(self, 
                                          trainer: Any, 
                                          test_predictions: np.ndarray, 
                                          plots_dir: Path) -> None:
        """Create predictions vs actual plots for both training and test sets."""
        
        # Test Predictions vs Actual
        plt.figure(figsize=(10, 8))
        plt.scatter(trainer.y_test, test_predictions, alpha=0.5, s=1)
        plt.plot([trainer.y_test.min(), trainer.y_test.max()], 
                [trainer.y_test.min(), trainer.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual (Test)')
        plt.ylabel('Predicted (Test)')
        plt.title('Test Set: Predictions vs Actual')
        plt.savefig(plots_dir / 'test_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training Predictions vs Actual (for comparison)
        train_predictions = trainer.model.predict(trainer.X_train)
        plt.figure(figsize=(10, 8))
        plt.scatter(trainer.y_train, train_predictions, alpha=0.5, s=1)
        plt.plot([trainer.y_train.min(), trainer.y_train.max()], 
                [trainer.y_train.min(), trainer.y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual (Train)')
        plt.ylabel('Predicted (Train)')
        plt.title('Training Set: Predictions vs Actual')
        plt.savefig(plots_dir / 'train_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_residuals_plots(self, 
                              trainer: Any, 
                              test_predictions: np.ndarray, 
                              plots_dir: Path) -> None:
        """Create residuals vs predicted plots for both training and test sets."""
        
        # Test Residuals plot
        test_residuals = trainer.y_test - test_predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(test_predictions, test_residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted (Test)')
        plt.ylabel('Residuals (Test)')
        plt.title('Test Set: Residuals vs Predicted')
        plt.savefig(plots_dir / 'test_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training Residuals plot (for comparison)
        train_predictions = trainer.model.predict(trainer.X_train)
        train_residuals = trainer.y_train - train_predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(train_predictions, train_residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted (Train)')
        plt.ylabel('Residuals (Train)')
        plt.title('Training Set: Residuals vs Predicted')
        plt.savefig(plots_dir / 'train_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_learning_curves_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create learning curves plot if training history is available."""
        if trainer.training_history['train_loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(trainer.training_history['train_loss'], label='Training Loss')
            plt.plot(trainer.training_history['val_loss'], label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE')
            plt.title('Learning Curves')
            plt.legend()
            plt.savefig(plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_snr_comparison_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create dual-scale SNR comparison plot."""
        
        # Get SNR values from stored metrics
        if trainer.current_train_metrics is not None and trainer.current_test_metrics is not None:
            # Use already calculated and stored metrics
            train_snr = trainer.current_train_metrics.get('snr', 0)
            test_snr = trainer.current_test_metrics.get('snr', 0)
            train_snr_db = trainer.current_train_metrics.get('snr_db', 0)
            test_snr_db = trainer.current_test_metrics.get('snr_db', 0)
        else:
            # Fallback: calculate from scratch (should not happen in normal flow)
            logger.warning("Metrics not available, calculating SNR from scratch")
            train_predictions = trainer.model.predict(trainer.X_train)
            test_predictions = trainer.model.predict(trainer.X_test)
            train_residuals = trainer.y_train - train_predictions
            test_residuals = trainer.y_test - test_predictions
            
            train_snr = np.var(trainer.y_train) / np.var(train_residuals) if np.var(train_residuals) > 0 else float('inf')
            test_snr = np.var(trainer.y_test) / np.var(test_residuals) if np.var(test_residuals) > 0 else float('inf')
            train_snr_db = 10 * np.log10(train_snr) if train_snr != float('inf') else float('inf')
            test_snr_db = 10 * np.log10(test_snr) if test_snr != float('inf') else float('inf')
        
        # Create dual-scale SNR comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Linear SNR
        datasets = ['Training', 'Test']
        snr_linear_values = [train_snr, test_snr]
        colors = ['blue', 'red']
        
        bars1 = ax1.bar(datasets, snr_linear_values, color=colors, alpha=0.7)
        ax1.set_ylabel('SNR (Linear)')
        ax1.set_title('Signal-to-Noise Ratio (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Add value labels on bars for linear plot
        for bar, value in zip(bars1, snr_linear_values):
            if value != float('inf'):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                        f'{value:.1f}', ha='center', va='bottom')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                        '∞', ha='center', va='bottom')
        
        # Plot 2: dB SNR
        snr_db_values = [train_snr_db, test_snr_db]
        
        bars2 = ax2.bar(datasets, snr_db_values, color=colors, alpha=0.7)
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('Signal-to-Noise Ratio (dB Scale)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars for dB plot
        for bar, value in zip(bars2, snr_db_values):
            if value != float('inf'):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f} dB', ha='center', va='bottom')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        '∞ dB', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'snr_comparison_dual_scale.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_plot_files(self) -> list:
        """Get list of all plot files created."""
        plots_dir = Path(self.diagnostics_config.get("plots_save_path", "diagnostic_plots"))
        return list(plots_dir.glob("*.png"))
