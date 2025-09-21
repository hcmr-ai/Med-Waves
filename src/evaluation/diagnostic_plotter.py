"""
DiagnosticPlotter class for creating model evaluation plots.

This class handles all diagnostic plotting functionality that was previously
embedded in the FullDatasetTrainer class.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List
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
        
        # Create new analysis plots (if enabled in config)
        if self.diagnostics_config.get("create_regional_analysis", True):
            self._create_regional_analysis_plots(trainer, test_predictions, plots_dir)
        
        if self.diagnostics_config.get("create_sea_bin_analysis", True):
            self._create_sea_bin_analysis_plots(trainer, test_predictions, plots_dir)
        
        if self.diagnostics_config.get("create_spatial_maps", True):
            self._create_aggregated_spatial_maps(trainer, test_predictions, plots_dir)
        
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
    
    def _create_regional_analysis_plots(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create regional performance analysis plots."""
        if not hasattr(trainer, 'regions_test') or trainer.regions_test is None:
            logger.warning("No regional information available for regional analysis plots")
            return
        
        logger.info("Creating regional analysis plots...")
        
        # Create regional comparison plot
        self._create_regional_comparison_plot(trainer, plots_dir)
        
        # Create regional predictions vs actual plots
        self._create_regional_predictions_plots(trainer, test_predictions, plots_dir)
        
        # Create regional error analysis plots
        self._create_regional_error_analysis(trainer, test_predictions, plots_dir)
    
    def _create_regional_comparison_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create regional performance comparison plot."""
        if not hasattr(trainer, 'regional_test_metrics') or not trainer.regional_test_metrics:
            logger.warning("No regional test metrics available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Regional Performance Comparison', fontsize=16, fontweight='bold')
        
        regions = list(trainer.regional_test_metrics.keys())
        rmse_values = [trainer.regional_test_metrics[region].get('rmse', 0) for region in regions]
        mae_values = [trainer.regional_test_metrics[region].get('mae', 0) for region in regions]
        pearson_values = [trainer.regional_test_metrics[region].get('pearson', 0) for region in regions]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # RMSE by region
        bars1 = axes[0].bar(regions, rmse_values, color=colors, alpha=0.7)
        axes[0].set_title('RMSE by Region', fontweight='bold')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        for bar, v in zip(bars1, rmse_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        # MAE by region
        bars2 = axes[1].bar(regions, mae_values, color=colors, alpha=0.7)
        axes[1].set_title('MAE by Region', fontweight='bold')
        axes[1].set_ylabel('MAE')
        axes[1].grid(True, alpha=0.3)
        for bar, v in zip(bars2, mae_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Pearson correlation by region
        bars3 = axes[2].bar(regions, pearson_values, color=colors, alpha=0.7)
        axes[2].set_title('Pearson Correlation by Region', fontweight='bold')
        axes[2].set_ylabel('Pearson Correlation')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        for bar, v in zip(bars3, pearson_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'regional_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_regional_predictions_plots(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create regional predictions vs actual plots."""
        if not hasattr(trainer, 'regions_test') or trainer.regions_test is None:
            return
        
        unique_regions = np.unique(trainer.regions_test)
        n_regions = len(unique_regions)
        
        if n_regions == 0:
            return
        
        # Create subplots for each region
        fig, axes = plt.subplots(1, min(n_regions, 3), figsize=(5 * min(n_regions, 3), 5))
        if n_regions == 1:
            axes = [axes]
        
        fig.suptitle('Regional Predictions vs Actual', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, region in enumerate(unique_regions[:3]):  # Limit to 3 regions for readability
            mask = trainer.regions_test == region
            region_y_true = trainer.y_test[mask]
            region_y_pred = test_predictions[mask]
            
            axes[i].scatter(region_y_true, region_y_pred, alpha=0.5, s=1, color=colors[i])
            axes[i].plot([region_y_true.min(), region_y_true.max()], 
                        [region_y_true.min(), region_y_true.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'{region.title()} Region')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate and display R²
            r2 = np.corrcoef(region_y_true, region_y_pred)[0, 1] ** 2
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'regional_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_regional_error_analysis(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create regional error analysis plots."""
        if not hasattr(trainer, 'regions_test') or trainer.regions_test is None:
            return
        
        unique_regions = np.unique(trainer.regions_test)
        n_regions = len(unique_regions)
        
        if n_regions == 0:
            return
        
        # Create subplots for each region
        fig, axes = plt.subplots(1, min(n_regions, 3), figsize=(5 * min(n_regions, 3), 5))
        if n_regions == 1:
            axes = [axes]
        
        fig.suptitle('Regional Error Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, region in enumerate(unique_regions[:3]):  # Limit to 3 regions for readability
            mask = trainer.regions_test == region
            region_y_true = trainer.y_test[mask]
            region_y_pred = test_predictions[mask]
            region_errors = region_y_pred - region_y_true
            
            axes[i].hist(region_errors, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
            axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[i].set_title(f'{region.title()} Region Errors')
            axes[i].set_xlabel('Prediction Error')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(region_errors)
            std_error = np.std(region_errors)
            axes[i].text(0.05, 0.95, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}', 
                        transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'regional_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sea_bin_analysis_plots(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create sea-bin performance analysis plots."""
        if not hasattr(trainer, 'sea_bin_test_metrics') or not trainer.sea_bin_test_metrics:
            logger.warning("No sea-bin test metrics available")
            return
        
        logger.info("Creating sea-bin analysis plots...")
        
        # Create sea-bin performance plot
        self._create_sea_bin_performance_plot(trainer, plots_dir)
        
        # Create sea-bin predictions vs actual plots
        self._create_sea_bin_predictions_plots(trainer, test_predictions, plots_dir)
    
    def _create_sea_bin_performance_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create sea-bin performance metrics plot."""
        sea_bin_metrics = trainer.sea_bin_test_metrics
        
        # Prepare data for plotting
        bin_names = []
        rmse_values = []
        mae_values = []
        pearson_values = []
        counts = []
        percentages = []
        
        for bin_name, metrics in sea_bin_metrics.items():
            bin_names.append(bin_name.replace('_', ' ').title())
            rmse_values.append(metrics.get('rmse', 0))
            mae_values.append(metrics.get('mae', 0))
            pearson_values.append(metrics.get('pearson', 0))
            counts.append(metrics.get('count', 0))
            percentages.append(metrics.get('percentage', 0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sea-Bin Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: RMSE by sea state
        axes[0, 0].bar(bin_names, rmse_values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('RMSE by Sea State', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(rmse_values):
            axes[0, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: MAE by sea state
        axes[0, 1].bar(bin_names, mae_values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('MAE by Sea State', fontweight='bold')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(mae_values):
            axes[0, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Pearson correlation by sea state
        axes[1, 0].bar(bin_names, pearson_values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Pearson Correlation by Sea State', fontweight='bold')
        axes[1, 0].set_ylabel('Pearson Correlation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(pearson_values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Sample distribution by sea state
        axes[1, 1].bar(bin_names, percentages, color='gold', alpha=0.7)
        axes[1, 1].set_title('Sample Distribution by Sea State', fontweight='bold')
        axes[1, 1].set_ylabel('Percentage of Samples (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(percentages):
            axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'sea_bin_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sea_bin_predictions_plots(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create sea-bin predictions vs actual plots."""
        if not hasattr(trainer, 'sea_bin_test_metrics') or not trainer.sea_bin_test_metrics:
            return
        
        # Get sea-bin configuration to determine bin ranges
        sea_bin_config = trainer.config.get('feature_block', {}).get('sea_bin_metrics', {})
        if not sea_bin_config.get('enabled', False):
            return
        
        bins = sea_bin_config.get('bins', [])
        if not bins:
            return
        
        # Create subplots for each sea state bin
        n_bins = len(bins)
        fig, axes = plt.subplots(2, (n_bins + 1) // 2, figsize=(5 * ((n_bins + 1) // 2), 10))
        if n_bins == 1:
            axes = [axes]
        elif n_bins <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle('Sea-Bin Predictions vs Actual', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        for i, bin_config in enumerate(bins):
            if i >= len(axes):
                break
                
            bin_name = bin_config["name"]
            bin_min = bin_config["min"]
            bin_max = bin_config["max"]
            
            # Filter data for this sea state bin
            mask = (trainer.y_test >= bin_min) & (trainer.y_test < bin_max)
            bin_y_true = trainer.y_test[mask]
            bin_y_pred = test_predictions[mask]
            
            if len(bin_y_true) > 0:
                axes[i].scatter(bin_y_true, bin_y_pred, alpha=0.5, s=1, color=colors[i % len(colors)])
                axes[i].plot([bin_y_true.min(), bin_y_true.max()], 
                           [bin_y_true.min(), bin_y_true.max()], 'r--', lw=2)
                axes[i].set_xlabel('Actual')
                axes[i].set_ylabel('Predicted')
                axes[i].set_title(f'{bin_name.replace("_", " ").title()}\n({bin_min}-{bin_max}m)')
                axes[i].grid(True, alpha=0.3)
                
                # Calculate and display R²
                r2 = np.corrcoef(bin_y_true, bin_y_pred)[0, 1] ** 2
                axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nn = {len(bin_y_true):,}', 
                           transform=axes[i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{bin_name.replace("_", " ").title()}\n({bin_min}-{bin_max}m)')
        
        # Hide unused subplots
        for i in range(n_bins, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'sea_bin_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_aggregated_spatial_maps(self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path) -> None:
        """Create aggregated spatial maps for key metrics."""
        if not hasattr(trainer, 'regions_test') or trainer.regions_test is None:
            logger.warning("No regional information available for spatial maps")
            return
        
        logger.info("Creating aggregated spatial maps...")
        
        # Create spatial maps directory
        spatial_dir = plots_dir / "spatial_maps"
        spatial_dir.mkdir(exist_ok=True)
        
        # Calculate spatial metrics
        spatial_metrics = self._calculate_spatial_metrics(trainer, test_predictions)
        
        if spatial_metrics is None or len(spatial_metrics) == 0:
            logger.warning("No spatial metrics calculated")
            return
        
        # Create spatial maps for key metrics
        key_metrics = ['rmse', 'mae', 'bias', 'pearson']
        
        for metric in key_metrics:
            if metric in spatial_metrics.columns:
                self._create_spatial_map(spatial_metrics, metric, spatial_dir)
    
    def _calculate_spatial_metrics(self, trainer: Any, test_predictions: np.ndarray) -> Optional[pd.DataFrame]:
        """Calculate spatial metrics for plotting."""
        try:
            import polars as pl
            
            # Create DataFrame with coordinates, predictions, and actual values
            if hasattr(trainer, 'coords_test') and trainer.coords_test is not None:
                lat_coords = trainer.coords_test[:, 0]
                lon_coords = trainer.coords_test[:, 1]
            else:
                # Fallback: create dummy coordinates
                lat_coords = np.zeros(len(trainer.y_test))
                lon_coords = np.zeros(len(trainer.y_test))
            
            data = {
                'lat': lat_coords,
                'lon': lon_coords,
                'y_true': trainer.y_test,
                'y_pred': test_predictions
            }
            
            df = pl.DataFrame(data)
            
            # Calculate spatial metrics
            spatial_metrics = df.group_by(['lat', 'lon']).agg([
                pl.col('y_true').mean().alias('y_true_mean'),
                pl.col('y_pred').mean().alias('y_pred_mean'),
                ((pl.col('y_pred') - pl.col('y_true')) ** 2).mean().sqrt().alias('rmse'),
                (pl.col('y_pred') - pl.col('y_true')).abs().mean().alias('mae'),
                (pl.col('y_pred') - pl.col('y_true')).mean().alias('bias'),
                pl.corr('y_true', 'y_pred').alias('pearson'),
                pl.len().alias('count')
            ])
            
            # Convert to pandas for plotting
            return spatial_metrics.to_pandas()
            
        except Exception as e:
            logger.warning(f"Could not calculate spatial metrics: {e}")
            return None
    
    def _create_spatial_map(self, spatial_metrics: pd.DataFrame, metric: str, spatial_dir: Path) -> None:
        """Create a spatial map for a specific metric."""
        try:
            from src.analytics.plots.spatial_plots import plot_spatial_feature_map
            
            # Prepare data for plotting
            plot_df = spatial_metrics[['lat', 'lon', metric]].copy()
            plot_df = plot_df.dropna()
            
            if len(plot_df) == 0:
                logger.warning(f"No data available for {metric} spatial map")
                return
            
            # Create spatial map
            plot_spatial_feature_map(
                df_pd=plot_df,
                feature_col=metric,
                save_path=str(spatial_dir / f'spatial_{metric}_map.png'),
                title=f'Spatial {metric.upper()} Map',
                colorbar_label=f'{metric.upper()}',
                s=8,
                alpha=0.85,
                cmap='RdYlBu_r'
            )
            
        except Exception as e:
            logger.warning(f"Could not create spatial map for {metric}: {e}")
    
    def get_plot_files(self) -> list:
        """Get list of all plot files created."""
        plots_dir = Path(self.diagnostics_config.get("plots_save_path", "diagnostic_plots"))
        return list(plots_dir.glob("*.png"))
