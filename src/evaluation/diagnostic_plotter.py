"""
DiagnosticPlotter class for creating model evaluation plots.

This class handles all diagnostic plotting functionality that was previously
embedded in the FullDatasetTrainer class.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classifiers.helpers import reconstruct_vhm0_values
from src.commons.region_mapping import RegionMapper

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

    def create_diagnostic_plots(
        self, trainer: Any, test_predictions: np.ndarray
    ) -> None:
        """
        Create comprehensive diagnostic plots for model evaluation.

        Args:
            trainer: The trainer object containing model, data, and metrics
            test_predictions: Test set predictions
        """
        plots_dir = Path(
            self.diagnostics_config.get("plots_save_path", "diagnostic_plots")
        )
        plots_dir.mkdir(exist_ok=True)

        # Create individual plot types
        if self.diagnostics_config.get("create_prediction_plots", True):
            self._create_predictions_vs_actual_plots(
                trainer, test_predictions, plots_dir
            )
        if self.diagnostics_config.get("create_residual_plots", True):
            self._create_residuals_plots(trainer, test_predictions, plots_dir)
        if self.diagnostics_config.get("create_learning_curves", True):
            self._create_learning_curves_plot(trainer, plots_dir)
        if self.diagnostics_config.get("create_snr_comparison_plot", True):
            self._create_snr_comparison_plot(trainer, plots_dir)

        # Create new analysis plots (if enabled in config)
        if self.diagnostics_config.get("create_regional_analysis", True):
            self._create_regional_analysis_plots(trainer, test_predictions, plots_dir)

        if self.diagnostics_config.get("create_sea_bin_analysis", True):
            self._create_sea_bin_analysis_plots(trainer, test_predictions, plots_dir)

        if self.diagnostics_config.get("create_spatial_maps", True):
            self._create_aggregated_spatial_maps(trainer, test_predictions, plots_dir)

        # Create advanced analysis plots
        if self.diagnostics_config.get("create_regional_error_box_plots", True):
            self._create_regional_error_box_plots(trainer, test_predictions, plots_dir)
        if self.diagnostics_config.get("create_wave_height_performance_plots", True):
            self._create_wave_height_performance_plots(trainer, plots_dir)
        # self._create_residual_geographic_analysis(trainer, test_predictions, plots_dir)
        # self._create_condition_performance_plots(trainer, test_predictions, plots_dir)

        # Create data distribution plots (if enabled in config)
        if self.diagnostics_config.get("create_data_distribution_plots", True):
            self._create_data_distribution_plots(trainer, plots_dir)

        logger.info(f"Diagnostic plots saved to {plots_dir}")

        # Log SNR values
        if trainer.current_train_metrics and trainer.current_test_metrics:
            train_snr = trainer.current_train_metrics.get("snr", 0)
            train_snr_db = trainer.current_train_metrics.get("snr_db", 0)
            test_snr = trainer.current_test_metrics.get("snr", 0)
            test_snr_db = trainer.current_test_metrics.get("snr_db", 0)
            logger.info(
                f"SNR - Train: {train_snr:.1f} ({train_snr_db:.1f} dB), Test: {test_snr:.1f} ({test_snr_db:.1f} dB)"
            )

    def _create_predictions_vs_actual_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create predictions vs actual plots for both training and test sets."""
        logger.info("Creating predictions vs actual plots...")
        # Test Predictions vs Actual
        plt.figure(figsize=(10, 8))
        plt.scatter(trainer.vhm0_y_test, test_predictions, alpha=0.5, s=1)
        plt.plot(
            [trainer.vhm0_y_test.min(), trainer.vhm0_y_test.max()],
            [trainer.vhm0_y_test.min(), trainer.vhm0_y_test.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual (Test)")
        plt.ylabel("Predicted (Test)")
        plt.title("Test Set: Predictions vs Actual")
        plt.savefig(
            plots_dir / "test_predictions_vs_actual.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Training Predictions vs Actual (for comparison)
        train_predictions = trainer.model.predict(trainer.X_train)
        _, train_predictions = reconstruct_vhm0_values(
            predict_bias=self.config.get("feature_block", {}).get("predict_bias", {}),
            predict_bias_log_space=self.config.get("feature_block", {}).get(
                "predict_bias_log_space", {}
            ),
            vhm0_x=trainer.vhm0_x_train,
            y_true=trainer.y_train,
            y_pred=train_predictions,
        )

        plt.figure(figsize=(10, 8))
        plt.scatter(trainer.vhm0_y_train, train_predictions, alpha=0.5, s=1)
        plt.plot(
            [trainer.vhm0_y_train.min(), trainer.vhm0_y_train.max()],
            [trainer.vhm0_y_train.min(), trainer.vhm0_y_train.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual (Train)")
        plt.ylabel("Predicted (Train)")
        plt.title("Training Set: Predictions vs Actual")
        plt.savefig(
            plots_dir / "train_predictions_vs_actual.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_residuals_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create residuals vs predicted plots for both training and test sets."""

        # Test Residuals plot
        logger.info("Creating test residuals vs predicted plot...")
        test_residuals = trainer.y_test - test_predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(test_predictions, test_residuals, alpha=0.5, s=1)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted (Test)")
        plt.ylabel("Residuals (Test)")
        plt.title("Test Set: Residuals vs Predicted")
        plt.savefig(
            plots_dir / "test_residuals_vs_predicted.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Training Residuals plot (for comparison)
        if trainer.X_train is not None and trainer.y_train is not None:
            train_predictions = trainer.model.predict(trainer.X_train)
            train_residuals = trainer.y_train - train_predictions
            plt.figure(figsize=(10, 8))
            plt.scatter(train_predictions, train_residuals, alpha=0.5, s=1)
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel("Predicted (Train)")
            plt.ylabel("Residuals (Train)")
            plt.title("Training Set: Residuals vs Predicted")
            plt.savefig(
                plots_dir / "train_residuals_vs_predicted.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _create_learning_curves_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create learning curves plot if training history is available."""
        if trainer.training_history["train_loss"]:
            logger.info("Creating learning curves plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(trainer.training_history["train_loss"], label="Training Loss")
            plt.plot(trainer.training_history["val_loss"], label="Validation Loss")
            plt.xlabel("Iteration")
            plt.ylabel("RMSE")
            plt.title("Learning Curves")
            plt.legend()
            plt.savefig(plots_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

    def _create_snr_comparison_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create dual-scale SNR comparison plot."""

        # Get SNR values from stored metrics
        if (
            trainer.current_train_metrics is not None
            and trainer.current_test_metrics is not None
        ):
            logger.info("Creating SNR comparison plot...")
            # Use already calculated and stored metrics
            train_snr = trainer.current_train_metrics.get("snr", 0)
            test_snr = trainer.current_test_metrics.get("snr", 0)
            train_snr_db = trainer.current_train_metrics.get("snr_db", 0)
            test_snr_db = trainer.current_test_metrics.get("snr_db", 0)
        else:
            # Fallback: calculate from scratch (should not happen in normal flow)
            logger.warning("Metrics not available, calculating SNR from scratch")
            train_predictions = trainer.model.predict(trainer.X_train)
            test_predictions = trainer.model.predict(trainer.X_test)
            train_residuals = trainer.vhm0_y_train - train_predictions
            test_residuals = trainer.vhm0_y_test - test_predictions

            train_snr = (
                np.var(trainer.vhm0_y_train) / np.var(train_residuals)
                if np.var(train_residuals) > 0
                else float("inf")
            )
            test_snr = (
                np.var(trainer.vhm0_y_test) / np.var(test_residuals)
                if np.var(test_residuals) > 0
                else float("inf")
            )
            train_snr_db = (
                10 * np.log10(train_snr) if train_snr != float("inf") else float("inf")
            )
            test_snr_db = (
                10 * np.log10(test_snr) if test_snr != float("inf") else float("inf")
            )

        # Create dual-scale SNR comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot 1: Linear SNR
        datasets = ["Training", "Test"]
        snr_linear_values = [train_snr, test_snr]
        colors = ["blue", "red"]

        bars1 = ax1.bar(datasets, snr_linear_values, color=colors, alpha=0.7)
        ax1.set_ylabel("SNR (Linear)")
        ax1.set_title("Signal-to-Noise Ratio (Linear Scale)")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")  # Log scale for better visualization

        # Add value labels on bars for linear plot
        for bar, value in zip(bars1, snr_linear_values, strict=False):
            if value != float("inf"):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.1,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )
            else:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.1,
                    "∞",
                    ha="center",
                    va="bottom",
                )

        # Plot 2: dB SNR
        snr_db_values = [train_snr_db, test_snr_db]

        bars2 = ax2.bar(datasets, snr_db_values, color=colors, alpha=0.7)
        ax2.set_ylabel("SNR (dB)")
        ax2.set_title("Signal-to-Noise Ratio (dB Scale)")
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars for dB plot
        for bar, value in zip(bars2, snr_db_values, strict=False):
            if value != float("inf"):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{value:.1f} dB",
                    ha="center",
                    va="bottom",
                )
            else:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    "∞ dB",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "snr_comparison_dual_scale.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_regional_analysis_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create regional performance analysis plots."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
            logger.warning(
                "No regional information available for regional analysis plots"
            )
            return

        logger.info("Creating regional analysis plots...")

        # Create regional comparison plot
        self._create_regional_comparison_plot(trainer, plots_dir)

        # Create regional predictions vs actual plots
        # self._create_regional_predictions_plots(trainer, test_predictions, plots_dir)

        # Create regional error analysis plots
        # self._create_regional_error_analysis(trainer, test_predictions, plots_dir)

    def _create_regional_comparison_plot(self, trainer: Any, plots_dir: Path) -> None:
        """Create regional performance comparison plot with baseline comparison."""
        if (
            not hasattr(trainer, "regional_test_metrics")
            or not trainer.regional_test_metrics
        ):
            logger.warning("No regional test metrics available")
            return
        logger.info("Creating regional performance comparison plot...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Regional Performance Comparison (Model vs Baseline)",
            fontsize=16,
            fontweight="bold",
        )

        region_ids = list(trainer.regional_test_metrics.keys())
        regions = [RegionMapper.get_display_name(rid) for rid in region_ids]

        # Model metrics
        model_rmse_values = [
            trainer.regional_test_metrics[rid].get("rmse", 0) for rid in region_ids
        ]
        model_mae_values = [
            trainer.regional_test_metrics[rid].get("mae", 0) for rid in region_ids
        ]
        model_pearson_values = [
            trainer.regional_test_metrics[rid].get("pearson", 0) for rid in region_ids
        ]

        # Baseline metrics (if available)
        baseline_rmse_values = []
        baseline_mae_values = []
        baseline_pearson_values = []

        if (
            hasattr(trainer, "baseline_regional_test_metrics")
            and trainer.baseline_regional_test_metrics
        ):
            baseline_rmse_values = [
                trainer.baseline_regional_test_metrics[rid].get("rmse", 0)
                for rid in region_ids
            ]
            baseline_mae_values = [
                trainer.baseline_regional_test_metrics[rid].get("mae", 0)
                for rid in region_ids
            ]
            baseline_pearson_values = [
                trainer.baseline_regional_test_metrics[rid].get("pearson", 0)
                for rid in region_ids
            ]

        # Set up bar positions
        x = np.arange(len(regions))
        width = 0.35

        # Colors
        model_color = "#2E86AB"  # Blue for model
        baseline_color = "#A23B72"  # Purple for baseline

        # RMSE by region
        bars1_model = axes[0].bar(
            x - width / 2,
            model_rmse_values,
            width,
            label="Model",
            color=model_color,
            alpha=0.8,
        )
        if baseline_rmse_values:
            bars1_baseline = axes[0].bar(
                x + width / 2,
                baseline_rmse_values,
                width,
                label="Baseline",
                color=baseline_color,
                alpha=0.8,
            )
        axes[0].set_title("RMSE by Region", fontweight="bold")
        axes[0].set_ylabel("RMSE")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(regions)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Add value labels on bars
        for bar, v in zip(bars1_model, model_rmse_values, strict=False):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if baseline_rmse_values:
            for bar, v in zip(bars1_baseline, baseline_rmse_values, strict=False):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # MAE by region
        bars2_model = axes[1].bar(
            x - width / 2,
            model_mae_values,
            width,
            label="Model",
            color=model_color,
            alpha=0.8,
        )
        if baseline_mae_values:
            bars2_baseline = axes[1].bar(
                x + width / 2,
                baseline_mae_values,
                width,
                label="Baseline",
                color=baseline_color,
                alpha=0.8,
            )
        axes[1].set_title("MAE by Region", fontweight="bold")
        axes[1].set_ylabel("MAE")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regions)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Add value labels on bars
        for bar, v in zip(bars2_model, model_mae_values, strict=False):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if baseline_mae_values:
            for bar, v in zip(bars2_baseline, baseline_mae_values, strict=False):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Pearson correlation by region
        bars3_model = axes[2].bar(
            x - width / 2,
            model_pearson_values,
            width,
            label="Model",
            color=model_color,
            alpha=0.8,
        )
        if baseline_pearson_values:
            bars3_baseline = axes[2].bar(
                x + width / 2,
                baseline_pearson_values,
                width,
                label="Baseline",
                color=baseline_color,
                alpha=0.8,
            )
        axes[2].set_title("Pearson Correlation by Region", fontweight="bold")
        axes[2].set_ylabel("Pearson Correlation")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(regions)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)
        axes[2].legend()

        # Add value labels on bars
        for bar, v in zip(bars3_model, model_pearson_values, strict=False):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if baseline_pearson_values:
            for bar, v in zip(bars3_baseline, baseline_pearson_values, strict=False):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(plots_dir / "regional_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _create_regional_predictions_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create regional predictions vs actual plots."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
            return
        logger.info("Creating regional predictions vs actual plots...")
        unique_regions = np.unique(trainer.regions_test)
        n_regions = len(unique_regions)

        if n_regions == 0:
            return

        # Create subplots for each region
        fig, axes = plt.subplots(
            1, min(n_regions, 3), figsize=(5 * min(n_regions, 3), 5)
        )
        if n_regions == 1:
            axes = [axes]

        fig.suptitle("Regional Predictions vs Actual", fontsize=16, fontweight="bold")

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for i, region in enumerate(
            unique_regions[:3]
        ):  # Limit to 3 regions for readability
            mask = trainer.regions_test == region
            region_y_true = trainer.y_test[mask]
            region_y_pred = test_predictions[mask]

            axes[i].scatter(
                region_y_true, region_y_pred, alpha=0.5, s=1, color=colors[i]
            )
            axes[i].plot(
                [region_y_true.min(), region_y_true.max()],
                [region_y_true.min(), region_y_true.max()],
                "r--",
                lw=2,
            )
            axes[i].set_xlabel("Actual")
            axes[i].set_ylabel("Predicted")
            region_name = RegionMapper.get_display_name(region)
            axes[i].set_title(f"{region_name} Region")
            axes[i].grid(True, alpha=0.3)

            # Calculate and display R²
            r2 = np.corrcoef(region_y_true, region_y_pred)[0, 1] ** 2
            axes[i].text(
                0.05,
                0.95,
                f"R² = {r2:.3f}",
                transform=axes[i].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "regional_predictions_vs_actual.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_regional_error_analysis(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create regional error analysis plots."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
            return
        logger.info("Creating regional error analysis plots...")
        unique_regions = np.unique(trainer.regions_test)
        n_regions = len(unique_regions)

        if n_regions == 0:
            return

        # Create subplots for each region
        fig, axes = plt.subplots(
            1, min(n_regions, 3), figsize=(5 * min(n_regions, 3), 5)
        )
        if n_regions == 1:
            axes = [axes]

        fig.suptitle("Regional Error Analysis", fontsize=16, fontweight="bold")

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for i, region in enumerate(
            unique_regions
        ):  # Limit to 3 regions for readability
            mask = trainer.regions_test == region
            region_y_true = trainer.vhm0_y_test[mask]
            region_y_pred = test_predictions[mask]
            region_errors = region_y_pred - region_y_true

            axes[i].hist(
                region_errors, bins=30, alpha=0.7, color=colors[i], edgecolor="black"
            )
            axes[i].axvline(0, color="red", linestyle="--", linewidth=2)
            region_name = RegionMapper.get_display_name(region)
            axes[i].set_title(f"{region_name} Region Errors")
            axes[i].set_xlabel("Prediction Error")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

            # Add statistics
            mean_error = np.mean(region_errors)
            std_error = np.std(region_errors)
            axes[i].text(
                0.05,
                0.95,
                f"Mean: {mean_error:.3f}\nStd: {std_error:.3f}",
                transform=axes[i].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "regional_error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_sea_bin_analysis_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create sea-bin performance analysis plots."""
        if (
            not hasattr(trainer, "sea_bin_test_metrics")
            or not trainer.sea_bin_test_metrics
        ):
            logger.warning("No sea-bin test metrics available")
            return

        logger.info("Creating sea-bin analysis plots...")

        # Create sea-bin performance plot
        self._create_sea_bin_performance_plot(
            trainer.sea_bin_test_metrics,
            getattr(trainer, "baseline_sea_bin_test_metrics", {}),
            plots_dir,
            mode="test",
        )

        # Create sea-bin performance plot
        if hasattr(trainer, "sea_bin_train_metrics"):
            self._create_sea_bin_performance_plot(
                trainer.sea_bin_train_metrics,
                getattr(trainer, "baseline_sea_bin_train_metrics", {}),
                plots_dir,
                mode="train",
            )

        # Create sea-bin predictions vs actual plots
        # self._create_sea_bin_predictions_plots(trainer, test_predictions, plots_dir)

    def _create_sea_bin_performance_plot(
        self,
        sea_bin_metrics: Any,
        baseline_sea_bin_metrics: Any,
        plots_dir: Path,
        mode: str,
    ) -> None:
        """Create sea-bin performance metrics plot with baseline comparison."""
        logger.info("Creating sea-bin performance metrics plot...")

        # Prepare data for plotting
        bin_names = []
        bin_labels = []
        rmse_values = []
        mae_values = []
        pearson_values = []
        counts = []
        percentages = []

        # Baseline data
        baseline_rmse_values = []
        baseline_mae_values = []
        baseline_pearson_values = []

        wave_bins = {}
        for bin_config in (
            self.config.get("feature_block", {})
            .get("sea_bin_metrics", {})
            .get("bins", [])
        ):
            min_val, max_val = bin_config["min"], bin_config["max"]
            bin_name = bin_config["name"]

            # Convert to float in case they come from YAML as strings
            min_val = float(min_val)
            if str(max_val) in [".inf", "float('inf')"] or max_val == float("inf"):
                max_val = float("inf")
            else:
                max_val = float(max_val)

            wave_bins[bin_name] = (min_val, max_val)

        for bin_name, metrics in sea_bin_metrics.items():
            label = f"{wave_bins[bin_name][0]}-{wave_bins[bin_name][1]}m"
            bin_labels.append(label)
            bin_names.append(bin_name)
            # bin_names.append(bin_name.replace('_', ' ').title())
            rmse_values.append(metrics.get("rmse", 0))
            mae_values.append(metrics.get("mae", 0))
            pearson_values.append(metrics.get("pearson", 0))
            counts.append(metrics.get("count", 0))
            percentages.append(metrics.get("percentage", 0))

            # Get baseline metrics for the same bin
            baseline_metrics = baseline_sea_bin_metrics.get(bin_name, {})
            baseline_rmse_values.append(baseline_metrics.get("rmse", 0))
            baseline_mae_values.append(baseline_metrics.get("mae", 0))
            baseline_pearson_values.append(baseline_metrics.get("pearson", 0))

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            "Sea-Bin Performance Analysis (Model vs Baseline)",
            fontsize=16,
            fontweight="bold",
        )

        # Set up bar positions for side-by-side comparison
        x = np.arange(len(bin_names))
        width = 0.35

        # Plot 1: RMSE by sea state
        _ = axes[0, 0].bar(
            x - width / 2, rmse_values, width, label="Model", color="skyblue", alpha=0.8
        )
        _ = axes[0, 0].bar(
            x + width / 2,
            baseline_rmse_values,
            width,
            label="Baseline",
            color="darkblue",
            alpha=0.6,
        )
        axes[0, 0].set_title("RMSE by Sea State", fontweight="bold")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(rmse_values, baseline_rmse_values, strict=False)):
            axes[0, 0].text(
                i - width / 2,
                v1 + 0.001,
                f"{v1:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            axes[0, 0].text(
                i + width / 2,
                v2 + 0.001,
                f"{v2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 2: MAE by sea state
        _ = axes[0, 1].bar(
            x - width / 2,
            mae_values,
            width,
            label="Model",
            color="lightcoral",
            alpha=0.8,
        )
        _ = axes[0, 1].bar(
            x + width / 2,
            baseline_mae_values,
            width,
            label="Baseline",
            color="darkred",
            alpha=0.6,
        )
        axes[0, 1].set_title("MAE by Sea State", fontweight="bold")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(bin_labels, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(mae_values, baseline_mae_values, strict=False)):
            axes[0, 1].text(
                i - width / 2,
                v1 + 0.001,
                f"{v1:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            axes[0, 1].text(
                i + width / 2,
                v2 + 0.001,
                f"{v2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 3: Pearson correlation by sea state
        _ = axes[1, 0].bar(
            x - width / 2,
            pearson_values,
            width,
            label="Model",
            color="lightgreen",
            alpha=0.8,
        )
        _ = axes[1, 0].bar(
            x + width / 2,
            baseline_pearson_values,
            width,
            label="Baseline",
            color="darkgreen",
            alpha=0.6,
        )
        axes[1, 0].set_title("Pearson Correlation by Sea State", fontweight="bold")
        axes[1, 0].set_ylabel("Pearson Correlation")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(bin_labels, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(pearson_values, baseline_pearson_values, strict=False)):
            axes[1, 0].text(
                i - width / 2,
                v1 + 0.01,
                f"{v1:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            axes[1, 0].text(
                i + width / 2,
                v2 + 0.01,
                f"{v2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 4: Sample distribution by sea state
        axes[1, 1].bar(bin_labels, percentages, color="gold", alpha=0.7)
        axes[1, 1].set_title("Sample Distribution by Sea State", fontweight="bold")
        axes[1, 1].set_ylabel("Percentage of Samples (%)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(percentages):
            axes[1, 1].text(
                i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / f"sea_bin_performance_{mode}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_sea_bin_predictions_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create sea-bin predictions vs actual plots."""
        if (
            not hasattr(trainer, "sea_bin_test_metrics")
            or not trainer.sea_bin_test_metrics
        ):
            return

        logger.info("Creating sea-bin predictions vs actual plots...")
        # Get sea-bin configuration to determine bin ranges
        sea_bin_config = trainer.config.get("feature_block", {}).get(
            "sea_bin_metrics", {}
        )
        if not sea_bin_config.get("enabled", False):
            return

        bins = sea_bin_config.get("bins", [])
        if not bins:
            return

        # Create subplots for each sea state bin
        n_bins = len(bins)
        fig, axes = plt.subplots(
            2, (n_bins + 1) // 2, figsize=(5 * ((n_bins + 1) // 2), 10)
        )
        if n_bins == 1:
            axes = [axes]
        elif n_bins <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle("Sea-Bin Predictions vs Actual", fontsize=16, fontweight="bold")

        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
            "#F7DC6F",
        ]

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
                axes[i].scatter(
                    bin_y_true,
                    bin_y_pred,
                    alpha=0.5,
                    s=1,
                    color=colors[i % len(colors)],
                )

                # Set axes to show full bin range with some padding
                padding = (bin_max - bin_min) * 0.1  # 10% padding
                axes[i].set_xlim(bin_min - padding, bin_max + padding)
                axes[i].set_ylim(bin_min - padding, bin_max + padding)

                # Plot perfect prediction line across full range
                axes[i].plot(
                    [bin_min - padding, bin_max + padding],
                    [bin_min - padding, bin_max + padding],
                    "r--",
                    lw=2,
                )

                axes[i].set_xlabel("Actual")
                axes[i].set_ylabel("Predicted")
                axes[i].set_title(
                    f"{bin_name.replace('_', ' ').title()}\n({bin_min}-{bin_max}m)"
                )
                axes[i].grid(True, alpha=0.3)

                # Calculate and display R²
                r2 = np.corrcoef(bin_y_true, bin_y_pred)[0, 1] ** 2
                axes[i].text(
                    0.05,
                    0.95,
                    f"R² = {r2:.3f}\nn = {len(bin_y_true):,}",
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(
                    f"{bin_name.replace('_', ' ').title()}\n({bin_min}-{bin_max}m)"
                )

        # Hide unused subplots
        for i in range(n_bins, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "sea_bin_predictions_vs_actual.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_aggregated_spatial_maps(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create aggregated spatial maps for key metrics."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
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
        key_metrics = ["rmse", "mae", "bias", "pearson"]

        for metric in key_metrics:
            if metric in spatial_metrics.columns:
                self._create_spatial_map(spatial_metrics, metric, spatial_dir)

    def _calculate_spatial_metrics(
        self, trainer: Any, test_predictions: np.ndarray
    ) -> Optional[pd.DataFrame]:
        """Calculate spatial metrics for plotting."""
        logger.info("Calculating spatial metrics...")
        try:
            import polars as pl

            # Create DataFrame with coordinates, predictions, and actual values
            if hasattr(trainer, "coords_test") and trainer.coords_test is not None:
                lat_coords = trainer.coords_test[:, 0]
                lon_coords = trainer.coords_test[:, 1]
            else:
                # Fallback: create dummy coordinates
                lat_coords = np.zeros(len(trainer.y_test))
                lon_coords = np.zeros(len(trainer.y_test))

            data = {
                "lat": lat_coords,
                "lon": lon_coords,
                "y_true": trainer.vhm0_y_test,
                "y_pred": test_predictions,
            }

            df = pl.DataFrame(data)

            # Calculate spatial metrics
            spatial_metrics = df.group_by(["lat", "lon"]).agg(
                [
                    pl.col("y_true").mean().alias("y_true_mean"),
                    pl.col("y_pred").mean().alias("y_pred_mean"),
                    ((pl.col("y_pred") - pl.col("y_true")) ** 2)
                    .mean()
                    .sqrt()
                    .alias("rmse"),
                    (pl.col("y_pred") - pl.col("y_true")).abs().mean().alias("mae"),
                    (pl.col("y_pred") - pl.col("y_true")).mean().alias("bias"),
                    pl.corr("y_true", "y_pred").alias("pearson"),
                    pl.len().alias("count"),
                ]
            )

            # Convert to pandas for plotting
            return spatial_metrics.to_pandas()

        except Exception as e:
            logger.warning(f"Could not calculate spatial metrics: {e}")
            return None

    def _create_spatial_map(
        self, spatial_metrics: pd.DataFrame, metric: str, spatial_dir: Path
    ) -> None:
        """Create a spatial map for a specific metric."""
        logger.info(f"Creating spatial map for {metric}...")
        try:
            from src.analytics.plots.spatial_plots import plot_spatial_feature_map

            # Prepare data for plotting
            plot_df = spatial_metrics[["lat", "lon", metric]].copy()
            plot_df = plot_df.dropna()

            if len(plot_df) == 0:
                logger.warning(f"No data available for {metric} spatial map")
                return

            # Create spatial map
            plot_spatial_feature_map(
                df_pd=plot_df,
                feature_col=metric,
                save_path=str(spatial_dir / f"spatial_{metric}_map.png"),
                title=f"Spatial {metric.upper()} Map",
                colorbar_label=f"{metric.upper()}",
                s=8,
                alpha=0.85,
                cmap="RdYlBu_r",
            )

        except Exception as e:
            logger.warning(f"Could not create spatial map for {metric}: {e}")

    def _create_regional_error_box_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create regional error box plots for detailed error analysis."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
            logger.warning(
                "No regional information available for regional error box plots"
            )
            return

        logger.info("Creating regional error box plots...")

        # Calculate errors
        errors = test_predictions - trainer.vhm0_y_test
        abs_errors = np.abs(errors)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Regional Error Analysis - Box Plots", fontsize=16, fontweight="bold"
        )

        # Get unique regions
        unique_regions = np.unique(trainer.regions_test)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Prepare data for box plots
        error_data = []
        abs_error_data = []
        bias_data = []
        rmse_data = []
        region_labels = []

        for region in unique_regions:
            mask = trainer.regions_test == region
            region_errors = errors[mask]
            region_abs_errors = abs_errors[mask]
            region_bias = region_errors  # Bias is the same as errors
            region_rmse = np.sqrt(
                np.mean(region_errors**2)
            )  # Calculate RMSE for this region

            error_data.append(region_errors)
            abs_error_data.append(region_abs_errors)
            bias_data.append(region_bias)
            rmse_data.append(
                [region_rmse] * len(region_errors)
            )  # Repeat RMSE for box plot
            region_name = RegionMapper.get_display_name(region)
            region_labels.append(region_name)

        # Plot 1: Error distribution by region
        bp1 = axes[0, 0].boxplot(error_data, labels=region_labels, patch_artist=True)
        for patch, color in zip(bp1["boxes"], colors[: len(unique_regions)], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 0].set_title("Error Distribution by Region", fontweight="bold")
        axes[0, 0].set_ylabel("Prediction Error")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Plot 2: Absolute error distribution by region
        bp2 = axes[0, 1].boxplot(
            abs_error_data, labels=region_labels, patch_artist=True
        )
        for patch, color in zip(bp2["boxes"], colors[: len(unique_regions)], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 1].set_title("Absolute Error Distribution by Region", fontweight="bold")
        axes[0, 1].set_ylabel("Absolute Prediction Error")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Error standard deviation by region (as bar chart since it's a single value per region)
        error_std_values = []
        for region in unique_regions:
            mask = trainer.regions_test == region
            region_errors = errors[mask]
            # Calculate standard deviation of errors for this region
            region_std = np.std(region_errors)
            error_std_values.append(region_std)

        bars = axes[1, 0].bar(
            region_labels,
            error_std_values,
            color=colors[: len(unique_regions)],
            alpha=0.7,
        )
        axes[1, 0].set_title("Error Standard Deviation by Region", fontweight="bold")
        axes[1, 0].set_ylabel("Standard Deviation of Errors")
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, error_std_values, strict=False):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 4: RMSE by region (as bar chart since it's a single value per region)
        rmse_values = []
        for region in unique_regions:
            mask = trainer.regions_test == region
            region_errors = errors[mask]
            # Calculate RMSE for this region
            region_rmse = np.sqrt(np.mean(region_errors**2))
            rmse_values.append(region_rmse)

        bars = axes[1, 1].bar(
            region_labels, rmse_values, color=colors[: len(unique_regions)], alpha=0.7
        )
        axes[1, 1].set_title("RMSE by Region", fontweight="bold")
        axes[1, 1].set_ylabel("Root Mean Squared Error")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, rmse_values, strict=False):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "regional_error_box_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_wave_height_performance_plots(
        self, trainer: Any, plots_dir: Path
    ) -> None:
        """Create performance vs wave height by region plots with baseline comparison."""
        logger.info(
            "Creating wave height performance plots with baseline comparison..."
        )

        # Get sea-bin configuration
        sea_bin_config = trainer.config.get("feature_block", {}).get(
            "sea_bin_metrics", {}
        )
        if not sea_bin_config.get("enabled", False):
            logger.warning(
                "Sea-bin metrics not enabled, skipping wave height performance plots"
            )
            return

        bins = sea_bin_config.get("bins", [])
        if not bins:
            logger.warning("No sea-bin configuration found")
            return
        # Get unique regions
        unique_regions = np.unique(trainer.regions_test)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(24, 14))
        fig.suptitle(
            "Performance vs Wave Height by Region (Model vs Baseline)",
            fontsize=16,
            fontweight="bold",
        )

        bins_with_data = trainer.region_sea_bin_metrics.get("bins_with_data")
        model_error_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "model_error_data_by_region_bin"
        )
        model_abs_error_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "model_abs_error_data_by_region_bin"
        )
        model_rmse_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "model_rmse_data_by_region_bin"
        )
        model_count_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "model_count_data_by_region_bin"
        )
        baseline_error_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "baseline_error_data_by_region_bin"
        )
        baseline_abs_error_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "baseline_abs_error_data_by_region_bin"
        )
        baseline_rmse_data_by_region_bin = trainer.region_sea_bin_metrics.get(
            "baseline_rmse_data_by_region_bin"
        )

        if baseline_error_data_by_region_bin is not None:
            baseline_errors = True
        else:
            baseline_errors = None

        # Filter bins to only include those with data
        bins_with_data = sorted(bins_with_data)
        filtered_bins = [bins[i] for i in bins_with_data]
        bin_labels = [
            f"{bin_config['min']:.1f}-{bin_config['max']:.1f}m"
            for bin_config in filtered_bins
        ]

        # Plot 1: Mean Error by region and wave height (Model vs Baseline)
        x_pos = np.arange(len(bin_labels))
        if baseline_errors is not None:
            width = 0.12  # Narrower bars for side-by-side comparison
        else:
            width = 0.25

        for i, region in enumerate(unique_regions):
            # Filter data to only include bins with data
            region_model_errors = [
                model_error_data_by_region_bin[region][j] for j in bins_with_data
            ]
            region_model_mean_errors = [
                np.mean(errors) if len(errors) > 0 else 0
                for errors in region_model_errors
            ]
            region_name = RegionMapper.get_display_name(region)

            if baseline_errors is not None:
                # Model bars
                axes[0, 0].bar(
                    x_pos + i * width * 2,
                    region_model_mean_errors,
                    width,
                    label=f"{region_name} (Model)",
                    color=colors[i],
                    alpha=0.8,
                )

                # Baseline bars
                region_baseline_errors = [
                    baseline_error_data_by_region_bin[region][j] for j in bins_with_data
                ]
                region_baseline_mean_errors = [
                    np.mean(errors) if len(errors) > 0 else 0
                    for errors in region_baseline_errors
                ]
                axes[0, 0].bar(
                    x_pos + i * width * 2 + width,
                    region_baseline_mean_errors,
                    width,
                    label=f"{region_name} (Baseline)",
                    color=colors[i],
                    alpha=0.4,
                    hatch="///",
                )
            else:
                # Only model bars
                axes[0, 0].bar(
                    x_pos + i * width,
                    region_model_mean_errors,
                    width,
                    label=region_name,
                    color=colors[i],
                    alpha=0.7,
                )

        axes[0, 0].set_title("Mean Error by Region and Wave Height", fontweight="bold")
        axes[0, 0].set_ylabel("Mean Prediction Error")
        axes[0, 0].set_xlabel("Wave Height Range")
        if baseline_errors is not None:
            axes[0, 0].set_xticks(x_pos + width * (len(unique_regions) - 1))
        else:
            axes[0, 0].set_xticks(x_pos + width)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Plot 2: Mean Absolute Error by region and wave height (Model vs Baseline)
        for i, region in enumerate(unique_regions):
            region_model_abs_errors = [
                model_abs_error_data_by_region_bin[region][j] for j in bins_with_data
            ]
            region_model_mean_abs_errors = [
                np.mean(errors) if len(errors) > 0 else 0
                for errors in region_model_abs_errors
            ]
            region_name = RegionMapper.get_display_name(region)

            if baseline_errors is not None:
                # Model bars
                axes[0, 1].bar(
                    x_pos + i * width * 2,
                    region_model_mean_abs_errors,
                    width,
                    label=f"{region_name} (Model)",
                    color=colors[i],
                    alpha=0.8,
                )

                # Baseline bars
                region_baseline_abs_errors = [
                    baseline_abs_error_data_by_region_bin[region][j]
                    for j in bins_with_data
                ]
                region_baseline_mean_abs_errors = [
                    np.mean(errors) if len(errors) > 0 else 0
                    for errors in region_baseline_abs_errors
                ]
                axes[0, 1].bar(
                    x_pos + i * width * 2 + width,
                    region_baseline_mean_abs_errors,
                    width,
                    label=f"{region_name} (Baseline)",
                    color=colors[i],
                    alpha=0.4,
                    hatch="///",
                )
            else:
                # Only model bars
                axes[0, 1].bar(
                    x_pos + i * width,
                    region_model_mean_abs_errors,
                    width,
                    label=region_name,
                    color=colors[i],
                    alpha=0.7,
                )

        axes[0, 1].set_title(
            "Mean Absolute Error by Region and Wave Height", fontweight="bold"
        )
        axes[0, 1].set_ylabel("Mean Absolute Error")
        axes[0, 1].set_xlabel("Wave Height Range")
        if baseline_errors is not None:
            axes[0, 1].set_xticks(x_pos + width * (len(unique_regions) - 1))
        else:
            axes[0, 1].set_xticks(x_pos + width)
        axes[0, 1].set_xticklabels(bin_labels, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: RMSE by region and wave height (Model vs Baseline)
        for i, region in enumerate(unique_regions):
            region_model_rmse = [
                model_rmse_data_by_region_bin[region][j] for j in bins_with_data
            ]
            region_name = RegionMapper.get_display_name(region)

            if baseline_errors is not None:
                # Model bars
                axes[1, 0].bar(
                    x_pos + i * width * 2,
                    region_model_rmse,
                    width,
                    label=f"{region_name} (Model)",
                    color=colors[i],
                    alpha=0.8,
                )

                # Baseline bars
                region_baseline_rmse = [
                    baseline_rmse_data_by_region_bin[region][j] for j in bins_with_data
                ]
                axes[1, 0].bar(
                    x_pos + i * width * 2 + width,
                    region_baseline_rmse,
                    width,
                    label=f"{region_name} (Baseline)",
                    color=colors[i],
                    alpha=0.4,
                    hatch="///",
                )
            else:
                # Only model bars
                axes[1, 0].bar(
                    x_pos + i * width,
                    region_model_rmse,
                    width,
                    label=region_name,
                    color=colors[i],
                    alpha=0.7,
                )

        axes[1, 0].set_title("RMSE by Region and Wave Height", fontweight="bold")
        axes[1, 0].set_ylabel("RMSE")
        axes[1, 0].set_xlabel("Wave Height Range")
        if baseline_errors is not None:
            axes[1, 0].set_xticks(x_pos + width * (len(unique_regions) - 1))
        else:
            axes[1, 0].set_xticks(x_pos + width)
        axes[1, 0].set_xticklabels(bin_labels, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Sample count by region and wave height (Model only - same for baseline)
        for i, region in enumerate(unique_regions):
            region_counts = [
                model_count_data_by_region_bin[region][j] for j in bins_with_data
            ]
            region_name = RegionMapper.get_display_name(region)
            axes[1, 1].bar(
                x_pos + i * width,
                region_counts,
                width,
                label=region_name,
                color=colors[i],
                alpha=0.7,
            )

        axes[1, 1].set_title(
            "Sample Count by Region and Wave Height", fontweight="bold"
        )
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].set_xlabel("Wave Height Range")
        axes[1, 1].set_xticks(x_pos + width)
        axes[1, 1].set_xticklabels(bin_labels, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "wave_height_performance_by_region.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_residual_geographic_analysis(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create residuals vs geographic location plots."""
        if not hasattr(trainer, "coords_test") or trainer.coords_test is None:
            logger.warning(
                "No coordinate information available for residual geographic analysis"
            )
            return

        logger.info("Creating residual geographic analysis plots...")

        # Calculate residuals
        residuals = test_predictions - trainer.y_test

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Residuals vs Geographic Location", fontsize=16, fontweight="bold")

        # Plot 1: Residuals vs Latitude
        scatter1 = axes[0, 0].scatter(
            trainer.coords_test[:, 0],
            residuals,
            alpha=0.5,
            s=1,
            c=residuals,
            cmap="RdBu_r",
        )
        axes[0, 0].set_title("Residuals vs Latitude", fontweight="bold")
        axes[0, 0].set_xlabel("Latitude")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.colorbar(scatter1, ax=axes[0, 0], label="Residual Value")

        # Plot 2: Residuals vs Longitude
        scatter2 = axes[0, 1].scatter(
            trainer.coords_test[:, 1],
            residuals,
            alpha=0.5,
            s=1,
            c=residuals,
            cmap="RdBu_r",
        )
        axes[0, 1].set_title("Residuals vs Longitude", fontweight="bold")
        axes[0, 1].set_xlabel("Longitude")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.colorbar(scatter2, ax=axes[0, 1], label="Residual Value")

        # Plot 3: Residuals vs Distance from Center (approximate)
        # center_lat, center_lon = 40.0, 15.0  # Approximate center of Mediterranean
        # distances = np.sqrt((trainer.coords_test[:, 0] - center_lat)**2 + (trainer.coords_test[:, 1] - center_lon)**2)
        # scatter3 = axes[1, 0].scatter(distances, residuals, alpha=0.5, s=1, c=residuals, cmap='RdBu_r')
        # axes[1, 0].set_title('Residuals vs Distance from Center', fontweight='bold')
        # axes[1, 0].set_xlabel('Distance from Center (degrees)')
        # axes[1, 0].set_ylabel('Residuals')
        # axes[1, 0].grid(True, alpha=0.3)
        # axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        # plt.colorbar(scatter3, ax=axes[1, 0], label='Residual Value')

        # Plot 4: Residuals vs Longitude (colored by region if available)
        if hasattr(trainer, "regions_test") and trainer.regions_test is not None:
            unique_regions = np.unique(trainer.regions_test)
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

            for i, region in enumerate(unique_regions):
                mask = trainer.regions_test == region
                region_name = RegionMapper.get_display_name(region)
                axes[1, 1].scatter(
                    trainer.coords_test[mask, 1],
                    residuals[mask],
                    alpha=0.5,
                    s=1,
                    color=colors[i],
                    label=region_name,
                )

            axes[1, 1].set_title(
                "Residuals vs Longitude (by Region)", fontweight="bold"
            )
            axes[1, 1].set_xlabel("Longitude")
            axes[1, 1].set_ylabel("Residuals")
            axes[1, 1].legend()
        else:
            scatter4 = axes[1, 1].scatter(
                trainer.coords_test[:, 1],
                residuals,
                alpha=0.5,
                s=1,
                c=residuals,
                cmap="RdBu_r",
            )
            axes[1, 1].set_title("Residuals vs Longitude", fontweight="bold")
            axes[1, 1].set_xlabel("Longitude")
            axes[1, 1].set_ylabel("Residuals")
            plt.colorbar(scatter4, ax=axes[1, 1], label="Residual Value")

        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "residuals_vs_geographic_location.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_condition_performance_plots(
        self, trainer: Any, test_predictions: np.ndarray, plots_dir: Path
    ) -> None:
        """Create performance under different conditions by region plots."""
        if not hasattr(trainer, "regions_test") or trainer.regions_test is None:
            logger.warning(
                "No regional information available for condition performance plots"
            )
            return

        logger.info("Creating condition performance plots...")

        # Calculate errors
        errors = test_predictions - trainer.y_test
        _ = np.abs(errors)

        # Get unique regions
        unique_regions = np.unique(trainer.regions_test)
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Create figure with subplots
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        fig.suptitle(
            "Performance Under Different Conditions by Region",
            fontsize=16,
            fontweight="bold",
        )

        # Plot 1: Performance vs Wave Height (box plots by region)
        # Create wave height bins
        wave_height_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
        wave_height_labels = [
            f"{wave_height_bins[i]:.1f}-{wave_height_bins[i + 1]:.1f}m"
            for i in range(len(wave_height_bins) - 1)
        ]

        error_data_by_region_wave = {}
        for region in unique_regions:
            region_mask = trainer.regions_test == region
            region_errors = errors[region_mask]
            region_wave_heights = trainer.y_test[region_mask]

            error_data_by_region_wave[region] = []
            for i in range(len(wave_height_bins) - 1):
                bin_mask = (region_wave_heights >= wave_height_bins[i]) & (
                    region_wave_heights < wave_height_bins[i + 1]
                )
                bin_errors = region_errors[bin_mask]
                error_data_by_region_wave[region].append(bin_errors)

        # Create box plots for each region
        x_pos = np.arange(len(wave_height_labels))
        width = 0.25

        for i, region in enumerate(unique_regions):
            region_means = [
                np.mean(errors) if len(errors) > 0 else 0
                for errors in error_data_by_region_wave[region]
            ]
            axes[0, 0].bar(
                x_pos + i * width,
                region_means,
                width,
                label=region.title(),
                color=colors[i],
                alpha=0.7,
            )

        axes[0, 0].set_title("Mean Error vs Wave Height by Region", fontweight="bold")
        axes[0, 0].set_ylabel("Mean Prediction Error")
        axes[0, 0].set_xlabel("Wave Height Range")
        axes[0, 0].set_xticks(x_pos + width)
        axes[0, 0].set_xticklabels(wave_height_labels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5)

        # Plot 2: Performance vs Wind Speed (if available in features)
        # This would require access to input features, which we don't have in the current setup
        # For now, create a placeholder plot
        # axes[0, 1].text(0.5, 0.5, 'Wind Speed Analysis\n(Requires input features)',
        #                ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        # axes[0, 1].set_title('Performance vs Wind Speed by Region', fontweight='bold')

        # Plot 3: Performance vs Distance from Coast (approximate)
        # if hasattr(trainer, 'coords_test') and trainer.coords_test is not None:
        #     # Approximate distance from coast using longitude (rough approximation)
        #     coast_distance = np.abs(trainer.coords_test[:, 1] - 0)  # Distance from 0 longitude
        #     distance_bins = [0, 5, 10, 15, 20, 25, 30, 40]
        #     distance_labels = [f"{distance_bins[i]}-{distance_bins[i+1]}°"
        #                       for i in range(len(distance_bins)-1)]

        #     error_data_by_region_distance = {}
        #     for region in unique_regions:
        #         region_mask = trainer.regions_test == region
        #         region_errors = errors[region_mask]
        #         region_distances = coast_distance[region_mask]

        #         error_data_by_region_distance[region] = []
        #         for i in range(len(distance_bins)-1):
        #             bin_mask = (region_distances >= distance_bins[i]) & (region_distances < distance_bins[i+1])
        #             bin_errors = region_errors[bin_mask]
        #             error_data_by_region_distance[region].append(bin_errors)

        #     x_pos = np.arange(len(distance_labels))
        #     for i, region in enumerate(unique_regions):
        #         region_means = [np.mean(errors) if len(errors) > 0 else 0 for errors in error_data_by_region_distance[region]]
        #         axes[1, 0].bar(x_pos + i * width, region_means, width, label=region.title(),
        #                       color=colors[i], alpha=0.7)

        #     axes[1, 0].set_title('Mean Error vs Distance from Coast by Region', fontweight='bold')
        #     axes[1, 0].set_ylabel('Mean Prediction Error')
        #     axes[1, 0].set_xlabel('Distance from Coast (degrees)')
        #     axes[1, 0].set_xticks(x_pos + width)
        #     axes[1, 0].set_xticklabels(distance_labels, rotation=45)
        #     axes[1, 0].legend()
        #     axes[1, 0].grid(True, alpha=0.3)
        #     axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        # else:
        #     axes[1, 0].text(0.5, 0.5, 'Distance Analysis\n(No coordinates available)',
        #                    ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        #     axes[1, 0].set_title('Performance vs Distance from Coast by Region', fontweight='bold')

        # Plot 4: Performance Consistency (coefficient of variation by region)
        cv_data = []
        region_labels_cv = []
        for region in unique_regions:
            region_mask = trainer.regions_test == region
            region_errors = errors[region_mask]
            if len(region_errors) > 0:
                cv = (
                    np.std(region_errors) / np.mean(np.abs(region_errors))
                    if np.mean(np.abs(region_errors)) > 0
                    else 0
                )
                cv_data.append(cv)
                region_name = RegionMapper.get_display_name(region)
                region_labels_cv.append(region_name)

        bars = axes[1, 1].bar(
            region_labels_cv, cv_data, color=colors[: len(cv_data)], alpha=0.7
        )
        axes[1, 1].set_title("Performance Consistency by Region", fontweight="bold")
        axes[1, 1].set_ylabel("Coefficient of Variation")
        axes[1, 1].set_xlabel("Region")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, cv_data, strict=False):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            plots_dir / "condition_performance_by_region.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_data_distribution_plots(self, trainer: Any, plots_dir: Path) -> None:
        """Create plots showing the data distribution after split."""
        logger.info("Creating data distribution plots...")

        try:
            # Get wave height bins from configuration
            wave_bins = self._get_wave_height_bins_from_config()

            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(
                "Data Distribution After Split", fontsize=16, fontweight="bold"
            )

            # Define colors for wave height bins
            bin_names = list(wave_bins.keys())
            colors = [
                "#2E8B57",
                "#FFD700",
                "#FF8C00",
                "#FF4500",
                "#8B0000",
            ]  # Green to Dark Red

            # Plot 1: Training Data Distribution (Pie Chart)
            ax1 = axes[0, 0]
            if len(trainer.vhm0_y_train) > 0:
                train_bin_counts = self._calculate_bin_counts(
                    trainer.vhm0_y_train, wave_bins
                )
                train_bin_counts_non_zero = [
                    count for count in train_bin_counts if count > 0
                ]
                train_bin_names_non_zero = [
                    name
                    for count, name in zip(train_bin_counts, bin_names, strict=False)
                    if count > 0
                ]
                colors_train = colors[: len(train_bin_counts_non_zero)]

                _, _, _ = ax1.pie(
                    train_bin_counts_non_zero,
                    labels=train_bin_names_non_zero,
                    autopct="%1.1f%%",
                    colors=colors_train,
                    startangle=90,
                )
                ax1.set_title(
                    f"Training Data\n({len(trainer.vhm0_y_train):,} samples)",
                    fontweight="bold",
                )
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No training data",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )
                ax1.set_title("Training Data", fontweight="bold")

            # Plot 2: Validation Data Distribution (Pie Chart)
            ax2 = axes[0, 1]
            if hasattr(trainer, "vhm0_y_val") and len(trainer.vhm0_y_val) > 0:
                val_bin_counts = self._calculate_bin_counts(
                    trainer.vhm0_y_val, wave_bins
                )
                val_bin_counts_non_zero = [
                    count for count in val_bin_counts if count > 0
                ]
                val_bin_names_non_zero = [
                    name for count, name in zip(val_bin_counts, bin_names, strict=False) if count > 0
                ]
                colors_val = colors[: len(val_bin_counts_non_zero)]

                _, _, _ = ax2.pie(
                    val_bin_counts_non_zero,
                    labels=val_bin_names_non_zero,
                    autopct="%1.1f%%",
                    colors=colors_val,
                    startangle=90,
                )
                ax2.set_title(
                    f"Validation Data\n({len(trainer.vhm0_y_val):,} samples)",
                    fontweight="bold",
                )
            else:
                val_bin_counts = []
                val_bin_counts_non_zero = []
                val_bin_names_non_zero = []
                colors_val = []
                ax2.text(
                    0.5,
                    0.5,
                    "No validation data",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Validation Data", fontweight="bold")

            # Plot 3: Test Data Distribution (Pie Chart)
            ax3 = axes[0, 2]
            if len(trainer.vhm0_y_test) > 0:
                test_bin_counts = self._calculate_bin_counts(
                    trainer.vhm0_y_test, wave_bins
                )
                test_bin_counts_non_zero = [
                    count for count in test_bin_counts if count > 0
                ]
                test_bin_names_non_zero = [
                    name for count, name in zip(test_bin_counts, bin_names, strict=False) if count > 0
                ]
                colors_test = colors[: len(test_bin_counts_non_zero)]

                _, _, _ = ax3.pie(
                    test_bin_counts_non_zero,
                    labels=test_bin_names_non_zero,
                    autopct="%1.1f%%",
                    colors=colors_test,
                    startangle=90,
                )
                ax3.set_title(
                    f"Test Data\n({len(trainer.vhm0_y_test):,} samples)",
                    fontweight="bold",
                )
            else:
                test_bin_counts = []
                test_bin_counts_non_zero = []
                test_bin_names_non_zero = []
                colors_test = []
                ax3.text(
                    0.5,
                    0.5,
                    "No test data",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Test Data", fontweight="bold")

            # Plot 4: Comparison Bar Chart
            ax4 = axes[1, 0]
            splits = ["Train", "Val", "Test"]
            split_data = [
                trainer.vhm0_y_train,
                trainer.vhm0_y_val if hasattr(trainer, "vhm0_y_val") else [],
                trainer.vhm0_y_test if hasattr(trainer, "vhm0_y_test") else [],
            ]

            x = np.arange(len(splits))
            width = 0.15
            bin_counts = [
                train_bin_counts_non_zero,
                val_bin_counts_non_zero,
                test_bin_counts_non_zero,
            ]
            bin_names_uniq = list(
                set(
                    train_bin_names_non_zero
                    + val_bin_names_non_zero
                    + test_bin_names_non_zero
                )
            )

            for j, bin_name in enumerate(bin_names_uniq):
                values = [
                    bin_counts[i][j] if bin_counts[i] and len(bin_counts[i]) > j else 0
                    for i in range(len(splits))
                ]
                ax4.bar(
                    x + j * width,
                    values,
                    width,
                    label=bin_name,
                    color=colors[j],
                    alpha=0.8,
                )

            ax4.set_title("Wave Height Distribution Comparison", fontweight="bold")
            ax4.set_ylabel("Number of Samples")
            ax4.set_xticks(x + width * 2)
            ax4.set_xticklabels(splits)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Plot 5: Sample Size Comparison
            ax5 = axes[1, 1]
            split_sizes = [
                len(trainer.vhm0_y_train),
                len(trainer.vhm0_y_val) if hasattr(trainer, "vhm0_y_val") else 0,
                len(trainer.vhm0_y_test) if hasattr(trainer, "vhm0_y_test") else 0,
            ]
            bars = ax5.bar(
                splits, split_sizes, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.7
            )
            ax5.set_title("Data Split Sizes", fontweight="bold")
            ax5.set_ylabel("Number of Samples")

            # Add value labels on bars
            for bar, size in zip(bars, split_sizes, strict=False):
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{size:,}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Plot 6: Extreme Wave Analysis
            ax6 = axes[1, 2]
            extreme_counts = []
            for data in split_data:
                if len(data) > 0:
                    extreme_count = np.sum(data >= 9.0)
                    extreme_counts.append(extreme_count)
                else:
                    extreme_counts.append(0)

            bars = ax6.bar(splits, extreme_counts, color="red", alpha=0.7)
            ax6.set_title("Extreme Waves (>9m) by Split", fontweight="bold")
            ax6.set_ylabel("Number of Extreme Waves")

            # Add value labels on bars
            for bar, count in zip(bars, extreme_counts, strict=False):
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(
                plots_dir / "data_distribution_after_split.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to create data distribution plots: {e}")
            plt.close()  # Ensure figure is closed even on error

    def _get_wave_height_bins_from_config(self) -> Dict[str, tuple]:
        """Get wave height bins from configuration."""
        # Get stratification bins from feature config
        feature_config = self.config.get("feature_block", {})
        stratification_bins = feature_config.get("per_point_stratification_bins", {})

        if not stratification_bins:
            # Fallback to default bins if not configured
            logger.warning(
                "No per_point_stratification_bins found in config, using default bins"
            )
            return {
                "calm": (0.0, 1.0),
                "moderate": (1.0, 3.0),
                "rough": (3.0, 6.0),
                "high": (6.0, 9.0),
                "extreme": (9.0, float("inf")),
            }

        # Convert config bins to the format expected by plotting
        wave_bins = {}
        for bin_name, bin_config in stratification_bins.items():
            min_val, max_val = bin_config["range"]

            # Convert to float in case they come from YAML as strings
            min_val = float(min_val)
            if str(max_val) in [".inf", "float('inf')"] or max_val == float("inf"):
                max_val = float("inf")
            else:
                max_val = float(max_val)

            wave_bins[bin_name] = (min_val, max_val)

        return wave_bins

    def _calculate_bin_counts(
        self, wave_data: np.ndarray, wave_bins: Dict[str, tuple]
    ) -> list:
        """Calculate bin counts for given wave data and bins."""
        bin_counts = []
        for _, (min_val, max_val) in wave_bins.items():
            if max_val == float("inf"):
                count = np.sum(wave_data >= min_val)
            else:
                count = np.sum((wave_data >= min_val) & (wave_data < max_val))
            bin_counts.append(count)
        return bin_counts

    def get_plot_files(self) -> list:
        """Get list of all plot files created."""
        plots_dir = Path(
            self.diagnostics_config.get("plots_save_path", "diagnostic_plots")
        )
        return list(plots_dir.glob("*.png"))
