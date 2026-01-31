"""
Evaluation Plotting Functions for Model Evaluator

Extracted from evaluate_bunet.py to reduce file size and improve organization.
All functions are standalone and take explicit parameters instead of relying on class state.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.visuals import load_coordinates_from_parquet, plot_spatial_rmse_map

logger = logging.getLogger(__name__)


def plot_rmse_maps(
    spatial_errors_model: List[Dict],
    spatial_errors_baseline: List[Dict],
    test_files: List[str],
    subsample_step: int,
    geo_bounds: dict,
    unit: str,
    output_dir: Path,
):
    """Plot spatial RMSE maps for model and baseline."""
    if not test_files or not spatial_errors_model:
        logger.warning("No spatial data available for RMSE maps")
        return

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")

    # Load coordinates from first test file
    try:
        lat_grid, lon_grid = load_coordinates_from_parquet(
            "s3://" + test_files[0], subsample_step=subsample_step
        )
        logger.info(f"Coordinate grid shape: {lat_grid.shape}")
    except Exception as e:
        logger.error(f"Failed to load coordinates: {e}")
        return

    # Aggregate spatial errors across all batches
    total_error_sq_model = np.zeros_like(lat_grid)
    total_count = np.zeros_like(lat_grid)
    total_error_sq_mae_model = np.zeros_like(lat_grid)

    for _i, batch_data in enumerate(spatial_errors_model):
        h, w = batch_data["error_sq"].shape
        total_error_sq_model[:h, :w] += batch_data["error_sq"]
        total_count[:h, :w] += batch_data["count"]
        total_error_sq_mae_model[:h, :w] += batch_data["error_sq_mae"]

    # Calculate RMSE (avoid division by zero)
    rmse_model = np.sqrt(total_error_sq_model / np.maximum(total_count, 1))
    rmse_model[total_count == 0] = np.nan
    mae_model = total_error_sq_mae_model / np.maximum(total_count, 1)
    mae_model[total_count == 0] = np.nan

    # Same for baseline if available
    if spatial_errors_baseline:
        total_error_sq_baseline = np.zeros_like(lat_grid)
        total_error_sq_mae_baseline = np.zeros_like(lat_grid)
        for batch_data in spatial_errors_baseline:
            h, w = batch_data["error_sq"].shape
            total_error_sq_baseline[:h, :w] += batch_data["error_sq"]
            total_error_sq_mae_baseline[:h, :w] += batch_data["error_sq_mae"]
        rmse_baseline = np.sqrt(
            total_error_sq_baseline / np.maximum(total_count, 1)
        )
        rmse_baseline[total_count == 0] = np.nan
        mae_baseline = total_error_sq_mae_baseline / np.maximum(total_count, 1)
        mae_baseline[total_count == 0] = np.nan
    else:
        rmse_baseline = None
        mae_baseline = None

    # Compute appropriate color scale based on actual data
    vmax_model = np.nanpercentile(rmse_model, 98)
    if rmse_baseline is not None:
        vmax_baseline = np.nanpercentile(rmse_baseline, 98)
        vmax_combined = max(vmax_model, vmax_baseline)
    else:
        vmax_combined = vmax_model

    logger.info(
        f"RMSE model - min: {np.nanmin(rmse_model):.3f}, max: {np.nanmax(rmse_model):.3f}, mean: {np.nanmean(rmse_model):.3f}"
    )
    logger.info(f"Using color scale vmax: {vmax_combined:.3f}")
    logger.info(
        f"Valid pixels: {np.sum(~np.isnan(rmse_model))} / {rmse_model.size}"
    )

    # Plot model RMSE
    plot_spatial_rmse_map(
        lat_grid,
        lon_grid,
        rmse_model,
        save_path=output_dir / "rmse_model.png",
        title="Model RMSE",
        vmin=0,
        vmax=vmax_combined,
        cmap=cmap,
        geo_bounds=geo_bounds,
        unit=unit,
    )

    # Plot model MAE
    if mae_baseline is not None:
        vmax_mae = max(np.nanpercentile(mae_model, 98), np.nanpercentile(mae_baseline, 98))
    else:
        vmax_mae = np.nanpercentile(mae_model, 98)

    plot_spatial_rmse_map(
        lat_grid,
        lon_grid,
        mae_model,
        save_path=output_dir / "mae_model.png",
        title="Model MAE",
        vmin=0,
        vmax=vmax_mae,
        cmap=cmap,
        geo_bounds=geo_bounds,
        unit=unit,
    )
    logger.info(f"Saved model RMSE map to {output_dir / 'rmse_model.png'}")

    # Plot baseline RMSE if available
    if rmse_baseline is not None:
        plot_spatial_rmse_map(
            lat_grid,
            lon_grid,
            rmse_baseline,
            save_path=output_dir / "rmse_reference.png",
            title="Reference RMSE",
            vmin=0,
            vmax=vmax_combined,
            cmap=cmap,
            geo_bounds=geo_bounds,
            unit=unit,
        )
        plot_spatial_rmse_map(
            lat_grid,
            lon_grid,
            mae_baseline,
            save_path=output_dir / "mae_reference.png",
            title="Reference MAE",
            vmin=0,
            vmax=vmax_mae,
            cmap=cmap,
            geo_bounds=geo_bounds,
            unit=unit,
        )

        # Plot improvement maps
        improvement = rmse_baseline - rmse_model
        improvement_mae = mae_baseline - mae_model

        # Symmetric scale around zero for diverging colormap
        imp_abs_max = np.nanpercentile(np.abs(improvement), 98)
        imp_mae_abs_max = np.nanpercentile(np.abs(improvement_mae), 95)

        # For improvement plots: Blue for <0 (worse), Red for >=0 (better)
        from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

        colors = ["#0000FF", "#FF0000"]
        cmap_binary = LinearSegmentedColormap.from_list(
            "improvement", colors, N=256
        )
        cmap_binary.set_bad("white")

        norm_binary = BoundaryNorm(
            boundaries=[-imp_abs_max, 0, imp_abs_max], ncolors=256, clip=True
        )
        norm_binary_mae = BoundaryNorm(
            boundaries=[-imp_mae_abs_max, 0, imp_mae_abs_max],
            ncolors=256,
            clip=True,
        )

        plot_spatial_rmse_map(
            lat_grid,
            lon_grid,
            improvement,
            save_path=output_dir / "rmse_improvement_symmetric.png",
            title="RMSE Improvement (Reference - Model)",
            vmin=np.nanpercentile(improvement, 2),
            vmax=imp_abs_max,
            cmap=cmap,
            geo_bounds=geo_bounds,
            unit=unit,
        )

        plot_spatial_rmse_map(
            lat_grid,
            lon_grid,
            improvement,
            save_path=output_dir / "rmse_improvement_binary.png",
            title="RMSE Improvement (Reference - Model)",
            vmin=-imp_abs_max,
            vmax=imp_abs_max,
            cmap=cmap_binary,
            geo_bounds=geo_bounds,
            unit=unit,
            norm=norm_binary,
        )
        plot_spatial_rmse_map(
            lat_grid,
            lon_grid,
            improvement_mae,
            save_path=output_dir / "mae_improvement_binary.png",
            title="MAE Improvement (Reference - Model)",
            vmin=-imp_mae_abs_max,
            vmax=imp_mae_abs_max,
            cmap=cmap_binary,
            geo_bounds=geo_bounds,
            unit=unit,
            norm=norm_binary_mae,
        )


def plot_model_better_percentage(
    sea_bin_metrics: Dict[str, Dict],
    sea_bins: List[Dict],
    var_name_full: str,
    output_dir: Path,
):
    """Plot percentage of samples where model is better than reference for each bin."""
    print("Creating model better percentage plot...")

    # Prepare data
    bin_labels = []
    pct_better = []
    counts_better = []
    pct_worse = []
    counts_worse = []

    # Sort bins by their min value
    sorted_bins = sorted(sea_bins, key=lambda x: x["min"])

    for bin_config in sorted_bins:
        bin_name = bin_config["name"]
        if bin_name not in sea_bin_metrics:
            continue

        metrics = sea_bin_metrics[bin_name]
        if metrics.get("count", 0) == 0:
            continue

        bin_labels.append(metrics.get("label", bin_config["label"]))
        pct_better.append(metrics.get("pct_model_better", 0))
        pct_worse.append(metrics.get("pct_model_worse", 0))
        counts_better.append(metrics.get("count_model_better", 0))
        counts_worse.append(metrics.get("count_model_worse", 0))

    if not bin_labels:
        logger.warning("No data for model better percentage plot")
        return

    # Create figure for "better" percentage
    _, ax = plt.subplots(figsize=(14, 8))
    colors = ["#5cb85c" if pct >= 50 else "#f0ad4e" for pct in pct_better]
    _ = ax.bar(range(len(bin_labels)), pct_better, color=colors, alpha=0.8)

    ax.axhline(
        y=50,
        color="black",
        linestyle="--",
        linewidth=2,
        label="50% threshold",
        alpha=0.7,
    )

    # Add value labels on bars
    for i, (pct, count) in enumerate(zip(pct_better, counts_better, strict=False)):
        label_text = f"{pct:.1f}%\n(n={count:,})"
        y_pos = pct + 2
        ax.text(
            i,
            y_pos,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Formatting
    ax.set_title(
        "Model Better Than Reference (% of Samples)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(f"{var_name_full} Bin", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "% of Samples Where |Model Error| < |Reference Error|",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        output_dir / "model_better_percentage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Saved model better percentage plot to {output_dir / 'model_better_percentage.png'}"
    )

    # Create figure for "worse" percentage
    _, ax = plt.subplots(figsize=(14, 8))
    colors = ["#5cb85c" if pct >= 50 else "#f0ad4e" for pct in pct_worse]
    _ = ax.bar(range(len(bin_labels)), pct_worse, color=colors, alpha=0.8)

    ax.axhline(
        y=50,
        color="black",
        linestyle="--",
        linewidth=2,
        label="50% threshold",
        alpha=0.7,
    )

    # Add value labels on bars
    for i, (pct, count) in enumerate(zip(pct_worse, counts_worse, strict=False)):
        label_text = f"{pct:.1f}%\n(n={count:,})"
        y_pos = pct + 2
        ax.text(
            i,
            y_pos,
            label_text,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Formatting
    ax.set_title(
        "Model Worse Than Reference (% of Samples)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel(f"{var_name_full} Bin", fontsize=13, fontweight="bold")
    ax.set_ylabel(
        "% of Samples Where |Model Error| > |Reference Error|",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        output_dir / "model_worse_percentage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Saved model worse percentage plot to {output_dir / 'model_worse_percentage.png'}"
    )


def plot_sea_bin_metrics(
    sea_bin_metrics: Dict[str, Dict],
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Create sea-bin performance metrics plot with baseline comparison."""
    print("Creating sea-bin performance plot...")

    # Prepare data for plotting
    bin_names = []
    bin_labels = []
    rmse_values = []
    mae_values = []
    counts = []
    percentages = []
    improvement_mae_values = []
    improvement_rmse_values = []
    baseline_rmse_values = []
    baseline_mae_values = []

    # Total count for percentage calculation
    total_count = sum(m.get("count", 0) for m in sea_bin_metrics.values())

    # Sort bins by their min value
    sorted_bins = sorted(sea_bins, key=lambda x: x["min"])

    for bin_config in sorted_bins:
        bin_name = bin_config["name"]
        if bin_name not in sea_bin_metrics:
            continue

        metrics = sea_bin_metrics[bin_name]
        if metrics.get("count", 0) == 0:
            continue

        bin_labels.append(metrics.get("label", bin_config["label"]))
        bin_names.append(bin_name)

        rmse_values.append(metrics.get("rmse", 0))
        mae_values.append(metrics.get("mae", 0))
        counts.append(metrics.get("count", 0))

        # Calculate percentage
        pct = (
            (metrics.get("count", 0) / total_count * 100) if total_count > 0 else 0
        )
        percentages.append(pct)

        # Get baseline metrics
        baseline_rmse = metrics.get("baseline_rmse", 0)
        baseline_mae = metrics.get("baseline_mae", 0)
        baseline_rmse_values.append(
            baseline_rmse if baseline_rmse is not None else 0
        )
        baseline_mae_values.append(baseline_mae if baseline_mae is not None else 0)
        improvement_mae_values.append(metrics.get("mae_improvement_pct", 0))
        improvement_rmse_values.append(metrics.get("rmse_improvement_pct", 0))

    if not bin_names:
        logger.warning("No sea-bin metrics to plot")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Period Range Performance Analysis (Model vs Reference)"
        if target_column == "corrected_VTM02"
        else "Sea-Bin Performance Analysis (Model vs Reference)",
        fontsize=16,
        fontweight="bold",
    )

    # Set up bar positions for side-by-side comparison
    x = np.arange(len(bin_names))
    width = 0.35

    # Plot 1: RMSE by sea state
    axes[0, 0].bar(
        x - width / 2,
        baseline_rmse_values,
        width,
        label="Reference",
        color="darkblue",
        alpha=0.6,
    )
    axes[0, 0].bar(
        x + width / 2, rmse_values, width, label="Model", color="skyblue", alpha=0.8
    )

    axes[0, 0].set_title(
        "RMSE by Period Range"
        if target_column == "corrected_VTM02"
        else "RMSE by Sea State",
        fontweight="bold",
    )
    axes[0, 0].set_ylabel(f"RMSE ({unit})")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(bin_labels, rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(baseline_rmse_values, rmse_values, strict=False)):
        if v1 > 0:
            axes[0, 0].text(
                i - width / 2,
                v1 + max(rmse_values) * 0.01,
                f"{v1:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if v2 > 0:
            axes[0, 0].text(
                i + width / 2,
                v2 + max(baseline_rmse_values) * 0.01
                if baseline_rmse_values
                else 0,
                f"{v2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 2: MAE by sea state
    axes[0, 1].bar(
        x - width / 2,
        baseline_mae_values,
        width,
        label="Reference",
        color="darkred",
        alpha=0.6,
    )
    axes[0, 1].bar(
        x + width / 2,
        mae_values,
        width,
        label="Model",
        color="lightcoral",
        alpha=0.8,
    )

    axes[0, 1].set_title(
        "MAE by Period Range"
        if target_column == "corrected_VTM02"
        else "MAE by Sea State",
        fontweight="bold",
    )
    axes[0, 1].set_ylabel(f"MAE ({unit})")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(bin_labels, rotation=45, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(baseline_mae_values, mae_values, strict=False)):
        if v1 > 0:
            axes[0, 1].text(
                i - width / 2,
                v1 + max(mae_values) * 0.01,
                f"{v1:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if v2 > 0:
            axes[0, 1].text(
                i + width / 2,
                v2 + max(baseline_mae_values) * 0.01 if baseline_mae_values else 0,
                f"{v2:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 3: Improvement percentage (RMSE)
    colors = ["green" if v > 0 else "red" for v in improvement_rmse_values]
    axes[1, 0].bar(bin_labels, improvement_rmse_values, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title(
        "RMSE Improvement by Period Range"
        if target_column == "corrected_VTM02"
        else "RMSE Improvement by Sea State",
        fontweight="bold",
    )
    axes[1, 0].set_ylabel("Improvement (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, v in enumerate(improvement_rmse_values):
        axes[1, 0].text(
            i,
            v + (max(improvement_rmse_values) - min(improvement_rmse_values)) * 0.02
            if improvement_rmse_values
            else 0,
            f"{v:.1f}%",
            ha="center",
            va="bottom" if v > 0 else "top",
            fontsize=9,
        )

    # Plot 4: Sample distribution by sea state
    axes[1, 1].bar(bin_labels, percentages, color="gold", alpha=0.7)
    axes[1, 1].set_title(
        "Sample Distribution by Period Range"
        if target_column == "corrected_VTM02"
        else "Sample Distribution by Sea State",
        fontweight="bold",
    )
    axes[1, 1].set_ylabel("Percentage of Samples (%)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (v, cnt) in enumerate(zip(percentages, counts, strict=False)):
        axes[1, 1].text(
            i,
            v + max(percentages) * 0.02 if percentages else 0,
            f"{v:.1f}%\n({cnt:,})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "sea_bin_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(
        f"Saved sea-bin performance plot to {output_dir / 'sea_bin_performance.png'}"
    )


def plot_error_distribution_histograms(
    sea_bin_error_samples: Dict,
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Plot histogram grid showing error distributions per sea bin."""
    print("Creating error distribution histogram grid...")

    # Filter bins with sufficient samples
    bins_to_plot = []
    for bin_config in sea_bins:
        bin_name = bin_config["name"]
        if len(sea_bin_error_samples[bin_name]["model_errors"]) > 0:
            bins_to_plot.append(bin_config)

    if not bins_to_plot:
        logger.warning("No error samples available for histogram plots")
        return

    # Create grid layout (5 rows x 3 cols for up to 15 bins)
    n_bins = len(bins_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_bins / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle(
        "Error Distribution by Period Range"
        if target_column == "corrected_VTM02"
        else "Error Distribution by Sea State",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, bin_config in enumerate(bins_to_plot):
        ax = axes_flat[idx]
        bin_name = bin_config["name"]
        bin_label = bin_config["label"]

        model_errors = np.array(
            sea_bin_error_samples[bin_name]["model_errors"]
        )
        baseline_errors = np.array(
            sea_bin_error_samples[bin_name]["baseline_errors"]
        )

        # Determine histogram range
        all_errors = (
            np.concatenate([model_errors, baseline_errors])
            if len(baseline_errors) > 0
            else model_errors
        )
        error_range = (np.percentile(all_errors, 1), np.percentile(all_errors, 99))

        # Plot histograms
        bins = np.linspace(error_range[0], error_range[1], 40)

        ax.hist(
            model_errors,
            bins=bins,
            alpha=0.6,
            color="blue",
            label=f"Model (n={len(model_errors):,})",
            density=True,
        )

        if len(baseline_errors) > 0:
            ax.hist(
                baseline_errors,
                bins=bins,
                alpha=0.6,
                color="red",
                label=f"Baseline (n={len(baseline_errors):,})",
                density=True,
            )

        # Add vertical lines for mean
        model_mean = np.mean(model_errors)
        ax.axvline(
            model_mean,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Model μ={model_mean:.3f}m",
        )

        if len(baseline_errors) > 0:
            baseline_mean = np.mean(baseline_errors)
            ax.axvline(
                baseline_mean,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Baseline μ={baseline_mean:.3f}m",
            )

        # Add zero line
        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)

        # Formatting
        ax.set_title(f"{bin_label}", fontweight="bold", fontsize=11)
        ax.set_xlabel(f"Error ({unit})", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Add statistics box
        model_std = np.std(model_errors)
        stats_text = f"Model: μ={model_mean:.3f}, σ={model_std:.3f}"
        if len(baseline_errors) > 0:
            baseline_std = np.std(baseline_errors)
            stats_text += f"\nBaseline: μ={baseline_mean:.3f}, σ={baseline_std:.3f}"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Hide unused subplots
    for idx in range(len(bins_to_plot), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "error_distribution_histograms.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved error distribution histograms to {output_dir / 'error_distribution_histograms.png'}"
    )


def plot_error_boxplots(
    sea_bin_error_samples: Dict,
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Plot box plot comparison of errors across all sea bins."""
    print("Creating error distribution box plots...")

    # Prepare data
    bin_labels = []
    model_error_data = []
    baseline_error_data = []

    for bin_config in sea_bins:
        bin_name = bin_config["name"]
        model_errors = sea_bin_error_samples[bin_name]["model_errors"]
        baseline_errors = sea_bin_error_samples[bin_name]["baseline_errors"]

        if len(model_errors) > 0:
            bin_labels.append(bin_config["label"])
            model_error_data.append(model_errors)
            baseline_error_data.append(
                baseline_errors if len(baseline_errors) > 0 else []
            )

    if not bin_labels:
        logger.warning("No error data available for box plots")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        "Error Distribution Box Plots by Period Range"
        if target_column == "corrected_VTM02"
        else "Error Distribution Box Plots by Sea State",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Model errors
    ax1 = axes[0]
    ax1.boxplot(
        model_error_data,
        labels=bin_labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="blue", linewidth=2),
        meanprops=dict(color="darkblue", linewidth=2, linestyle="--"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        flierprops=dict(
            marker="o", markerfacecolor="blue", markersize=3, alpha=0.3
        ),
    )

    ax1.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax1.set_title("Model Errors", fontweight="bold", fontsize=14)
    ax1.set_ylabel(f"Error ({unit})", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add sample counts
    for i, (_label, data) in enumerate(
        zip(bin_labels, model_error_data, strict=False)
    ):
        ax1.text(
            i + 1,
            ax1.get_ylim()[0],
            f"n={len(data):,}",
            ha="center",
            va="top",
            fontsize=8,
            rotation=0,
        )

    # Plot 2: Model vs Baseline comparison (side-by-side)
    ax2 = axes[1]

    # Prepare positions for side-by-side box plots
    positions_model = np.arange(len(bin_labels)) * 2.5 + 0.6
    positions_baseline = np.arange(len(bin_labels)) * 2.5 + 1.4

    ax2.boxplot(
        model_error_data,
        positions=positions_model,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="blue", linewidth=2),
        meanprops=dict(color="darkblue", linewidth=1.5, linestyle="--"),
        whiskerprops=dict(color="blue"),
        capprops=dict(color="blue"),
        flierprops=dict(
            marker="o", markerfacecolor="blue", markersize=2, alpha=0.3
        ),
    )

    # Only plot baseline if data exists
    has_baseline = any(len(d) > 0 for d in baseline_error_data)
    if has_baseline:
        ax2.boxplot(
            baseline_error_data,
            positions=positions_baseline,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            boxprops=dict(facecolor="lightcoral", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            meanprops=dict(color="darkred", linewidth=1.5, linestyle="--"),
            whiskerprops=dict(color="red"),
            capprops=dict(color="red"),
            flierprops=dict(
                marker="o", markerfacecolor="red", markersize=2, alpha=0.3
            ),
        )

    ax2.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_title("Model vs Baseline Comparison", fontweight="bold", fontsize=14)
    ax2.set_ylabel(f"Error ({unit})", fontsize=12)
    ax2.set_xlabel("Sea State", fontsize=12)

    # Set x-ticks at the center of each pair
    ax2.set_xticks((positions_model + positions_baseline) / 2)
    ax2.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="lightblue", edgecolor="blue", label="Model")
    ]
    if has_baseline:
        legend_elements.append(
            Patch(facecolor="lightcoral", edgecolor="red", label="Baseline")
        )
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(
        output_dir / "error_distribution_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved error distribution box plots to {output_dir / 'error_distribution_boxplots.png'}"
    )


def plot_error_violins(
    sea_bin_error_samples: Dict,
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Plot violin plots showing error distributions per sea bin."""
    print("Creating error distribution violin plots...")

    # Filter bins with sufficient samples
    bins_to_plot = []
    empty_bins = []
    for bin_config in sea_bins:
        bin_name = bin_config["name"]
        n_samples = len(sea_bin_error_samples[bin_name]["model_errors"])
        if n_samples > 5:  # Need more samples for violin plots
            bins_to_plot.append(bin_config)
        else:
            empty_bins.append(f"{bin_config['label']} (n={n_samples})")

    if empty_bins:
        print(f"  Skipping bins with too few samples: {', '.join(empty_bins)}")
    print(f"  Plotting {len(bins_to_plot)} bins with sufficient data")

    if not bins_to_plot:
        logger.warning("No bins with sufficient samples for violin plots")
        return

    # Create grid layout
    n_bins = len(bins_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_bins / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    fig.suptitle(
        "Error Distribution Violin Plots by Period Range"
        if target_column == "corrected_VTM02"
        else "Error Distribution Violin Plots by Sea State",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, bin_config in enumerate(bins_to_plot):
        ax = axes_flat[idx]
        bin_name = bin_config["name"]
        bin_label = bin_config["label"]

        model_errors = np.array(
            sea_bin_error_samples[bin_name]["model_errors"]
        )
        baseline_errors = np.array(
            sea_bin_error_samples[bin_name]["baseline_errors"]
        )

        # Prepare data for violin plot
        plot_data = []
        plot_labels = []
        plot_colors = []

        plot_data.append(model_errors)
        plot_labels.append("Model")
        plot_colors.append("lightblue")

        if len(baseline_errors) > 5:
            plot_data.append(baseline_errors)
            plot_labels.append("Baseline")
            plot_colors.append("lightcoral")

        # Create violin plot
        parts = ax.violinplot(
            plot_data,
            positions=range(len(plot_data)),
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )

        # Color the violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)

        # Style the mean and median lines
        parts["cmeans"].set_color("darkblue")
        parts["cmeans"].set_linewidth(2)
        parts["cmedians"].set_color("red")
        parts["cmedians"].set_linewidth(2)

        # Add zero line
        ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)

        # Formatting
        ax.set_title(f"{bin_label}", fontweight="bold", fontsize=11)
        ax.set_ylabel(f"Error ({unit})", fontsize=9)
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistics
        model_mean = np.mean(model_errors)
        model_median = np.median(model_errors)
        model_std = np.std(model_errors)
        stats_text = f"Model:\nμ={model_mean:.3f}\nmedian={model_median:.3f}\nσ={model_std:.3f}\nn={len(model_errors):,}"

        if len(baseline_errors) > 5:
            baseline_mean = np.mean(baseline_errors)
            baseline_median = np.median(baseline_errors)
            baseline_std = np.std(baseline_errors)
            stats_text += f"\n\nBaseline:\nμ={baseline_mean:.3f}\nmedian={baseline_median:.3f}\nσ={baseline_std:.3f}"

        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Hide unused subplots
    for idx in range(len(bins_to_plot), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        output_dir / "error_distribution_violins.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved error distribution violin plots to {output_dir / 'error_distribution_violins.png'}"
    )


def plot_error_cdfs(
    sea_bin_error_samples: Dict,
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Plot cumulative distribution functions for errors across sea bins."""
    print("Creating error CDF plots...")

    # Prepare data
    bins_with_data = []
    for bin_config in sea_bins:
        bin_name = bin_config["name"]
        if len(sea_bin_error_samples[bin_name]["model_errors"]) > 0:
            bins_with_data.append(bin_config)

    if not bins_with_data:
        logger.warning("No error data available for CDF plots")
        return

    print(f"  Plotting CDFs for {len(bins_with_data)} bins")

    # Create figure with subplots - one for model, one for model vs baseline
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        "Cumulative Distribution Functions of Errors by Period Range"
        if target_column == "corrected_VTM02"
        else "Cumulative Distribution Functions of Errors by Sea State",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Model errors only (all bins)
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(bins_with_data)))

    for idx, bin_config in enumerate(bins_with_data):
        bin_name = bin_config["name"]
        bin_label = bin_config["label"]
        model_errors = np.array(
            sea_bin_error_samples[bin_name]["model_errors"]
        )

        if len(model_errors) > 0:
            # Sort errors for CDF
            sorted_errors = np.sort(model_errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

            ax1.plot(
                sorted_errors,
                cdf,
                label=bin_label,
                color=colors[idx],
                linewidth=2,
                alpha=0.8,
            )

    ax1.axvline(
        0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Zero error"
    )
    ax1.set_xlabel(f"Error ({unit})", fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.set_title("Model Error CDFs", fontweight="bold", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9, ncol=2)

    # Add horizontal lines at key percentiles
    for percentile, alpha_val in [(0.5, 0.3), (0.9, 0.3), (0.95, 0.3)]:
        ax1.axhline(
            percentile, color="gray", linestyle=":", linewidth=1, alpha=alpha_val
        )
        ax1.text(
            ax1.get_xlim()[1],
            percentile,
            f"{int(percentile * 100)}%",
            fontsize=8,
            va="center",
        )

    # Plot 2: Model vs Baseline comparison for selected bins
    ax2 = axes[1]

    # Select a few representative bins to avoid clutter
    representative_bins = [
        bins_with_data[i]
        for i in [
            0,
            len(bins_with_data) // 4,
            len(bins_with_data) // 2,
            min(len(bins_with_data) - 1, 3 * len(bins_with_data) // 4),
        ]
        if i < len(bins_with_data)
    ]

    for bin_config in representative_bins:
        bin_name = bin_config["name"]
        bin_label = bin_config["label"]
        model_errors = np.array(
            sea_bin_error_samples[bin_name]["model_errors"]
        )
        baseline_errors = np.array(
            sea_bin_error_samples[bin_name]["baseline_errors"]
        )

        if len(model_errors) > 0:
            sorted_errors = np.sort(model_errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(
                sorted_errors,
                cdf,
                label=f"{bin_label} (Model)",
                linewidth=2,
                linestyle="-",
            )

        if len(baseline_errors) > 0:
            sorted_errors = np.sort(baseline_errors)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax2.plot(
                sorted_errors,
                cdf,
                label=f"{bin_label} (Baseline)",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
            )

    ax2.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel(f"Error ({unit})", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_title(
        "Model vs Baseline CDF Comparison (Selected Bins)",
        fontweight="bold",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9, ncol=2)

    # Add percentile lines
    for percentile, alpha_val in [(0.5, 0.3), (0.9, 0.3), (0.95, 0.3)]:
        ax2.axhline(
            percentile, color="gray", linestyle=":", linewidth=1, alpha=alpha_val
        )
        ax2.text(
            ax2.get_xlim()[1],
            percentile,
            f"{int(percentile * 100)}%",
            fontsize=8,
            va="center",
        )

    # Add text box with interpretation guide
    guide_text = (
        "CDF Interpretation:\n"
        "• Steeper curve = tighter error distribution\n"
        "• Curve closer to zero = better accuracy\n"
        "• Read percentiles at horizontal lines"
    )
    ax1.text(
        0.02,
        0.98,
        guide_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "error_distribution_cdfs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved error CDF plots to {output_dir / 'error_distribution_cdfs.png'}"
    )


def plot_vhm0_distributions(
    plot_samples: Dict,
    var_name: str,
    var_name_full: str,
    unit: str,
    corrected_label: str,
    model_label: str,
    uncorrected_label: str,
    output_dir: Path,
):
    """Plot distributions of ground truth, predicted, and uncorrected VHM0."""
    print("Creating VHM0 distribution comparison plot...")

    y_true = np.array(plot_samples["y_true"])
    y_pred = np.array(plot_samples["y_pred"])
    y_uncorrected = np.array(plot_samples["y_uncorrected"])

    # Pre-compute KDE for each dataset once (reused across all 3 plots)
    print("Computing KDEs (cached for reuse)...")
    from scipy import stats

    # Create KDE objects
    kde_true = stats.gaussian_kde(
        y_true, bw_method=0.5 * y_true.std() * len(y_true) ** (-1 / 5)
    )
    kde_pred = stats.gaussian_kde(
        y_pred, bw_method=0.5 * y_pred.std() * len(y_pred) ** (-1 / 5)
    )
    kde_uncorrected = stats.gaussian_kde(
        y_uncorrected,
        bw_method=0.5 * y_uncorrected.std() * len(y_uncorrected) ** (-1 / 5),
    )

    # Create evaluation grid
    x_min = 0
    x_max = 15
    x_grid = np.linspace(x_min, x_max, 200)

    # Evaluate KDEs once
    kde_true_values = kde_true(x_grid)
    kde_pred_values = kde_pred(x_grid)
    kde_uncorrected_values = kde_uncorrected(x_grid)
    print("KDE computation complete!")

    # Plot 1: All three distributions
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Use cached KDE results
    ax.plot(
        x_grid,
        kde_true_values,
        label=corrected_label,
        color="green",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.fill_between(x_grid, kde_true_values, alpha=0.2, color="green")

    ax.plot(
        x_grid,
        kde_pred_values,
        label=model_label,
        color="blue",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.fill_between(x_grid, kde_pred_values, alpha=0.2, color="blue")

    ax.plot(
        x_grid,
        kde_uncorrected_values,
        label=uncorrected_label,
        color="red",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.fill_between(x_grid, kde_uncorrected_values, alpha=0.2, color="red")

    ax.set_xlabel(f"{var_name} ({unit})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(f"{var_name_full}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{var_name}_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved VHM0 distribution plot to {output_dir / f'{var_name}_distributions.png'}"
    )

    # Plot 2: Model vs Reference (reusing cached KDEs)
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        x_grid,
        kde_true_values,
        label=corrected_label,
        color="green",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.plot(
        x_grid,
        kde_pred_values,
        label=model_label,
        color="blue",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.set_xlabel(f"{var_name} ({unit})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{var_name_full} Distribution Comparison (Model vs Reference)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{var_name}_distributions_model_vs_reference.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved VHM0 distribution plot to {output_dir / '{var_name}_distributions_model_vs_reference.png'}"
    )

    # Plot 3: Reference vs Uncorrected (reusing cached KDEs)
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(
        x_grid,
        kde_true_values,
        label=corrected_label,
        color="green",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.plot(
        x_grid,
        kde_uncorrected_values,
        label=uncorrected_label,
        color="red",
        linewidth=1.0,
        alpha=0.85,
    )

    ax.set_xlabel(f"{var_name} ({unit})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{var_name_full} Distribution Comparison (Reference vs Uncorrected)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{var_name}_distributions_reference_vs_uncorrected.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved VHM0 distribution plot to {output_dir / '{var_name}_distributions_reference_vs_uncorrected.png'}"
    )
