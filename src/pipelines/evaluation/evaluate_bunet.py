#!/usr/bin/env python3
"""
Comprehensive evaluation script for WaveBiasCorrector model.
Provides detailed metrics, visualizations, and sea-bin analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your model and dataset classes
import logging

from src.classifiers.lightning_trainer import WaveBiasCorrector
from src.classifiers.networks.mdn import mdn_expected_value
from src.commons.dataloaders import CachedWaveDataset, GridPatchWaveDataset
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer
from src.evaluation.evaluation_plots import (
    plot_error_boxplots as plot_error_boxplots_fn,
)
from src.evaluation.evaluation_plots import (
    plot_error_cdfs as plot_error_cdfs_fn,
)
from src.evaluation.evaluation_plots import (
    plot_error_distribution_histograms as plot_error_distribution_histograms_fn,
)
from src.evaluation.evaluation_plots import (
    plot_error_violins as plot_error_violins_fn,
)
from src.evaluation.evaluation_plots import (
    plot_model_better_percentage as plot_model_better_percentage_fn,
)
from src.evaluation.evaluation_plots import (
    plot_rmse_maps as plot_rmse_maps_fn,
)
from src.evaluation.evaluation_plots import (
    plot_sea_bin_metrics as plot_sea_bin_metrics_fn,
)
from src.evaluation.evaluation_plots import (
    plot_vhm0_distributions as plot_vhm0_distributions_fn,
)
from src.evaluation.metrics import (
    compute_overall_metrics_from_accumulators,
    compute_sea_bin_metrics_from_accumulators,
)
from src.evaluation.visuals import load_coordinates_from_parquet
from src.pipelines.training.dnn_trainer import (
    DNNConfig,
    get_file_list,
    split_files_by_year,
)

logger = logging.getLogger(__name__)



def apply_binwise_correction(
    y_pred,
    y_true,
    vhm0,
    bins=None,  # Wave height bins in meters
    normalize=False,
    std=None,
    mean=None,
    mask=None,
):
    """
    Apply post-hoc bin-wise bias correction based on VHM0 (true or predicted).

    Args:
        y_pred: (B, 1, H, W) torch.Tensor, model predictions (normalized or real)
        y_true: (B, 1, H, W) torch.Tensor, ground truth (same units as y_pred)
        vhm0:   (B, 1, H, W) torch.Tensor, original wave height (same units as y_true)
        bins:   List of bin edges in meters for binning vhm0
        normalize: If True, assumes y_pred and y_true are normalized
        std, mean: If normalize=True, std and mean should be provided
        mask: (B, 1, H, W) optional boolean mask of valid sea pixels

    Returns:
        y_corr: (B, 1, H, W) bias-corrected predictions
        bin_biases: dict of {bin_label: bias_value}
        bin_counts: dict of {bin_label: sample_count}
    """

    # 1. Unnormalize if needed
    if bins is None:
        bins = [0, 1, 2, 3, 4, 5, 10]
    if normalize:
        assert std is not None and mean is not None, "Need mean/std to unnormalize"
        y_pred_denorm = y_pred * std + mean
        y_true_denorm = y_true * std + mean
        vhm0_denorm = vhm0 * std + mean
    else:
        y_pred_denorm, y_true_denorm, vhm0_denorm = y_pred, y_true, vhm0

    # 2. Prepare mask
    if mask is None:
        mask = ~torch.isnan(y_true_denorm)

    # 3. Compute residuals (true - pred)
    residuals = y_true_denorm - y_pred_denorm

    # 4. Compute mean bias per bin
    bin_biases = {}
    bin_counts = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_label = f"{low}-{high}m"

        in_bin = (vhm0_denorm >= low) & (vhm0_denorm < high) & mask
        if in_bin.any():
            bin_bias = residuals[in_bin].mean().item()
            bin_count = in_bin.sum().item()
        else:
            bin_bias = 0.0  # no data in this bin → no correction
            bin_count = 0
        bin_biases[bin_label] = bin_bias
        bin_counts[bin_label] = bin_count

    # 5. Apply correction
    y_corr_denorm = y_pred_denorm.clone()
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        bin_label = f"{low}-{high}m"
        in_bin = (vhm0_denorm >= low) & (vhm0_denorm < high) & mask
        y_corr_denorm[in_bin] += bin_biases[bin_label]

    # 6. Normalize back if needed
    if normalize:
        y_corr = (y_corr_denorm - mean) / std
    else:
        y_corr = y_corr_denorm

    return y_corr, bin_biases, bin_counts


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        model: pl.LightningModule,
        test_loader: DataLoader,
        output_dir: Path,
        predict_bias: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalizer: WaveNormalizer = None,
        normalize_target: bool = False,
        test_files: List[str] = None,
        subsample_step: int = None,
        apply_binwise_correction_flag: bool = False,
        bias_loader: DataLoader = None,
        geo_bounds: dict = None,
        use_mdn: bool = False,
        target_column: str = "corrected_VHM0",
        apply_bilateral_filter: bool = False,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.bias_loader = (
            bias_loader  # Separate loader for computing biases (train/val)
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.predict_bias = predict_bias
        self.normalizer = normalizer
        self.normalize_target = normalize_target
        self.apply_binwise_correction_flag = apply_binwise_correction_flag
        self.geo_bounds = geo_bounds  # {'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float}
        self.use_mdn = use_mdn
        self.target_column = target_column
        self.apply_bilateral_filter = apply_bilateral_filter
        if self.apply_bilateral_filter:
            logger.info("Applying bilateral filter to predictions")
        self._configure_sea_bins()
        self._configure_labels()

        self.test_files = test_files
        self.subsample_step = subsample_step

        # Load geographic mask if filtering is requested
        self.geo_mask = None
        if self.geo_bounds and self.test_files:
            self._load_geographic_mask()

        # Add spatial accumulators for RMSE maps
        self.spatial_errors_model = []  # Store (error_map, count_map) for each batch
        self.spatial_errors_baseline = []

        # Initialize accumulators for incremental computation
        self._reset_accumulators()

        # Sample storage for plots (optional, limited size)
        self.plot_samples = {
            "y_true": [],
            "y_pred": [],
            "y_uncorrected": [],
        }

    def _reset_accumulators(self):
        """Reset all metric accumulators."""
        # Overall metrics - using Welford's algorithm for stable variance
        self.total_count = 0
        self.sum_mae = 0.0
        self.sum_mse = 0.0
        self.sum_bias = 0.0
        self.sum_baseline_mae = 0.0
        self.sum_baseline_mse = 0.0
        self.sum_baseline_bias = 0.0

        # For R² and correlation - need sum of squares
        self.sum_y_true = 0.0
        self.sum_y_true_sq = 0.0
        self.sum_y_pred = 0.0
        self.sum_y_pred_sq = 0.0
        self.sum_y_true_y_pred = 0.0

        # Sea-bin accumulators: {bin_name: {'count': 0, 'sum_mae': 0, 'sum_mse': 0, ...}}
        self.sea_bin_accumulators = {
            bin_config["name"]: {
                "count": 0,
                "sum_mae": 0.0,
                "sum_mse": 0.0,
                "sum_bias": 0.0,
                "sum_baseline_mae": 0.0,
                "sum_baseline_mse": 0.0,
                "sum_baseline_bias": 0.0,
                "count_model_better": 0,  # Count of samples where |model_error| < |baseline_error|
                "count_model_worse": 0,  # Count of samples where |model_error| > |baseline_error|
            }
            for bin_config in self.sea_bins
        }

        # Store error samples for distribution plots
        self.sea_bin_error_samples = {
            bin_config["name"]: {
                "model_errors": [],
                "baseline_errors": [],
            }
            for bin_config in self.sea_bins
        }

        self.spatial_rmse_accumulators = {}

    def _configure_labels(self):
        """Configure dynamic labels based on target_column."""
        # Infer variable name and unit from target_column
        target_map = {
            "corrected_VHM0": {
                "var_name": "VHM0",
                "var_name_full": "Significant Wave Height",
                "unit": "m",
                "corrected_label": "Corrected (Reference)",
                "uncorrected_label": "Uncorrected",
                "model_label": "Model Prediction",
            },
            "corrected_VTM02": {
                "var_name": "VTM02",
                "var_name_full": "Wave Period",
                "unit": "s",
                "corrected_label": "Corrected (Reference)",
                "uncorrected_label": "Uncorrected",
                "model_label": "Model Prediction",
            },
        }

        # Get configuration or use defaults
        if self.target_column in target_map:
            config = target_map[self.target_column]
        else:
            # Default fallback for unknown target columns
            config = {
                "var_name": self.target_column.replace("corrected_", "")
                .replace("_", " ")
                .upper(),
                "var_name_full": self.target_column.replace("_", " ").title(),
                "unit": "units",
                "corrected_label": "Corrected (Reference)",
                "uncorrected_label": "Uncorrected",
                "model_label": "Model Prediction",
            }

        # Store as instance variables for easy access
        self.var_name = config["var_name"]
        self.var_name_full = config["var_name_full"]
        self.unit = config["unit"]
        self.corrected_label = config["corrected_label"]
        self.uncorrected_label = config["uncorrected_label"]
        self.model_label = config["model_label"]

        logger.info(
            f"Configured labels for target '{self.target_column}': "
            f"{self.var_name} ({self.unit})"
        )

    def _configure_sea_bins(self):
        """Configure sea bins based on target column."""
        if self.target_column == "corrected_VHM0":
            self.sea_bins = [
                {"name": "calm", "min": 0.0, "max": 1.0, "label": "0.0-1.0m"},
                {"name": "light", "min": 1.0, "max": 2.0, "label": "1.0-2.0m"},
                {"name": "moderate", "min": 2.0, "max": 3.0, "label": "2.0-3.0m"},
                {"name": "rough", "min": 3.0, "max": 4.0, "label": "3.0-4.0m"},
                {"name": "very_rough", "min": 4.0, "max": 5.0, "label": "4.0-5.0m"},
                {"name": "extreme_5_6", "min": 5.0, "max": 6.0, "label": "5.0-6.0m"},
                {"name": "extreme_6_7", "min": 6.0, "max": 7.0, "label": "6.0-7.0m"},
                {"name": "extreme_7_8", "min": 7.0, "max": 8.0, "label": "7.0-8.0m"},
                {"name": "extreme_8_9", "min": 8.0, "max": 9.0, "label": "8.0-9.0m"},
                {"name": "extreme_9_10", "min": 9.0, "max": 10.0, "label": "9.0-10.0m"},
                {
                    "name": "extreme_10_11",
                    "min": 10.0,
                    "max": 11.0,
                    "label": "10.0-11.0m",
                },
                {
                    "name": "extreme_11_12",
                    "min": 11.0,
                    "max": 12.0,
                    "label": "11.0-12.0m",
                },
                {
                    "name": "extreme_12_13",
                    "min": 12.0,
                    "max": 13.0,
                    "label": "12.0-13.0m",
                },
                {
                    "name": "extreme_13_14",
                    "min": 13.0,
                    "max": 14.0,
                    "label": "13.0-14.0m",
                },
                {
                    "name": "extreme_14_15",
                    "min": 14.0,
                    "max": 15.0,
                    "label": "14.0-15.0m",
                },
            ]
        elif self.target_column == "corrected_VTM02":
            self.sea_bins_coarse = [
                {
                    "name": "very_short",
                    "min": 0.0,
                    "max": 3.0,
                    "label": "0.0-3.0s",
                },  # Wind waves/choppy
                {
                    "name": "short",
                    "min": 3.0,
                    "max": 5.0,
                    "label": "3.0-5.0s",
                },  # Young wind seas
                {
                    "name": "moderate_short",
                    "min": 5.0,
                    "max": 7.0,
                    "label": "5.0-7.0s",
                },  # Developed wind seas
                {
                    "name": "moderate",
                    "min": 7.0,
                    "max": 9.0,
                    "label": "7.0-9.0s",
                },  # Mature seas
                {
                    "name": "moderate_long",
                    "min": 9.0,
                    "max": 11.0,
                    "label": "9.0-11.0s",
                },  # Swell influence
                {
                    "name": "long",
                    "min": 11.0,
                    "max": 13.0,
                    "label": "11.0-13.0s",
                },  # Swell dominated
                {
                    "name": "very_long",
                    "min": 13.0,
                    "max": 15.0,
                    "label": "13.0-15.0s",
                },  # Long period swell
                {
                    "name": "extreme_long",
                    "min": 15.0,
                    "max": 20.0,
                    "label": "15.0-20.0s",
                },  # Extreme long swell
            ]
            self.sea_bins = [
                {
                    "name": f"bin_{i:.0f}_{i + 1:.0f}",
                    "min": float(i),
                    "max": float(i + 1),
                    "label": f"{i:.0f}-{i + 1:.0f}s",
                }
                for i in range(0, 20)
            ]

    def _load_geographic_mask(self):
        """Load coordinate grid and create geographic filtering mask."""
        try:
            logger.info("Loading geographic coordinates for filtering...")
            lat_grid, lon_grid = load_coordinates_from_parquet(
                "s3://" + self.test_files[0]
                if not self.test_files[0].startswith("s3://")
                else self.test_files[0],
                subsample_step=self.subsample_step,
            )

            # Create boolean mask based on bounds
            lat_mask = (lat_grid >= self.geo_bounds["lat_min"]) & (
                lat_grid <= self.geo_bounds["lat_max"]
            )
            lon_mask = (lon_grid >= self.geo_bounds["lon_min"]) & (
                lon_grid <= self.geo_bounds["lon_max"]
            )
            geo_mask = lat_mask & lon_mask

            # Convert to torch tensor and store
            self.geo_mask = torch.from_numpy(geo_mask).to(self.device)

            valid_pixels = geo_mask.sum()
            total_pixels = geo_mask.size
            logger.info(
                f"Geographic filter: {valid_pixels}/{total_pixels} pixels "
                f"({100 * valid_pixels / total_pixels:.1f}%) within bounds "
                f"[lat: {self.geo_bounds['lat_min']}-{self.geo_bounds['lat_max']}, "
                f"lon: {self.geo_bounds['lon_min']}-{self.geo_bounds['lon_max']}]"
            )

        except Exception as e:
            logger.warning(
                f"Failed to load geographic mask: {e}. Continuing without geographic filtering."
            )
            self.geo_mask = None

    def _reconstruct_wave_heights(
        self, bias: torch.Tensor, vhm0: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct full wave heights from bias: corrected = vhm0 + bias"""
        return bias + vhm0

    def _process_batch(self, y, mask, vhm0, y_pred):
        """Process a single batch and update accumulators."""
        # Apply geographic mask if available
        if self.geo_mask is not None:
            # Crop geo_mask to match current batch size if needed
            h, w = mask.shape[2], mask.shape[3]
            geo_mask_crop = self.geo_mask[:h, :w]

            # Expand dimensions and apply
            geo_mask_expanded = geo_mask_crop.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask = mask & geo_mask_expanded  # Combine with existing validity mask

        # Align dimensions
        # min_h = min(y_pred.shape[2], y.shape[2])
        # min_w = min(y_pred.shape[3], y.shape[3])
        # y_pred = y_pred[:, :, :min_h, :min_w]
        # y = y[:, :, :min_h, :min_w]
        # mask = mask[:, :, :min_h, :min_w]

        # vhm0 = vhm0[:, :, :min_h, :min_w]

        # if self.normalize_target and self.normalizer is not None:
        #     y_pred = self.normalizer.inverse_transform_torch(y_pred)
        #     y = self.normalizer.inverse_transform_torch(y)

        # ========== COMPUTE SPATIAL ERROR MAPS FIRST (using full 4D tensors) ==========
        if self.predict_bias:
            # Reconstruct full wave heights (4D tensors)
            y_pred_full = self._reconstruct_wave_heights(y_pred, vhm0)
            y_true_full = self._reconstruct_wave_heights(y, vhm0)
            y_baseline_full = vhm0  # Baseline is just vhm0

            error_map = ((y_pred_full - y_true_full) ** 2).cpu().numpy()  # (N, C, H, W)
            error_map_mae = (
                (y_pred_full - y_true_full).abs().cpu().numpy()
            )  # (N, C, H, W)
            error_map_baseline = (
                ((y_baseline_full - y_true_full) ** 2).cpu().numpy()
            )  # (N, C, H, W)
            error_map_baseline_mae = (
                (y_baseline_full - y_true_full).abs().cpu().numpy()
            )  # (N, C, H, W)
        else:
            # Not predicting bias
            error_map = ((y_pred - y) ** 2).cpu().numpy()  # (N, C, H, W)
            error_map_mae = (y_pred - y).abs().cpu().numpy()  # (N, C, H, W)
            if vhm0 is not None:
                error_map_baseline = ((vhm0 - y) ** 2).cpu().numpy()  # (N, C, H, W)
                error_map_baseline_mae = (vhm0 - y).abs().cpu().numpy()  # (N, C, H, W)
            else:
                error_map_baseline = None
                error_map_baseline_mae = None

        count_map = mask.cpu().numpy().astype(np.float32)  # (N, C, H, W)

        # IMPORTANT: Apply mask to errors (zero out invalid pixels)
        error_map = error_map * count_map
        if error_map_baseline is not None:
            error_map_baseline = error_map_baseline * count_map
        if error_map_baseline_mae is not None:
            error_map_baseline_mae = error_map_baseline_mae * count_map
        # Store spatial errors (sum over batch and channel dimensions)
        self.spatial_errors_model.append(
            {
                "error_sq": error_map.sum(axis=(0, 1)),  # (H, W)
                "error_sq_mae": error_map_mae.sum(axis=(0, 1)),  # (H, W)
                "count": count_map.sum(axis=(0, 1)),  # (H, W)
            }
        )

        if error_map_baseline is not None:
            self.spatial_errors_baseline.append(
                {
                    "error_sq": error_map_baseline.sum(axis=(0, 1)),  # (H, W)
                    "error_sq_mae": error_map_baseline_mae.sum(axis=(0, 1)),  # (H, W)
                    "count": count_map.sum(axis=(0, 1)),  # (H, W)
                }
            )

        # Apply mask
        mask_flat = mask.flatten()
        y_true_flat = y.flatten()[mask_flat]
        y_pred_flat = y_pred.flatten()[mask_flat]

        # Reconstruct wave heights if predicting bias
        if self.predict_bias and vhm0 is not None:
            vhm0_flat = vhm0.flatten()[mask_flat]
            y_true_wave_heights = self._reconstruct_wave_heights(y_true_flat, vhm0_flat)
            y_pred_wave_heights = self._reconstruct_wave_heights(y_pred_flat, vhm0_flat)
        else:
            y_true_wave_heights = y_true_flat
            y_pred_wave_heights = y_pred_flat

        # Get uncorrected for baseline
        if vhm0 is not None:
            vhm0_flat = vhm0.flatten()[mask_flat]
            y_uncorrected = vhm0_flat
        else:
            y_uncorrected = None

        # Convert to numpy for binning
        y_true_np = y_true_wave_heights.cpu().numpy()
        y_pred_np = y_pred_wave_heights.cpu().numpy()

        # Update overall metrics
        n = len(y_true_np)
        if n > 0:
            self.total_count += n

            # Model metrics
            errors = y_pred_np - y_true_np
            self.sum_mae += np.sum(np.abs(errors))
            self.sum_mse += np.sum(errors**2)
            self.sum_bias += np.sum(errors)

            # Baseline metrics
            if y_uncorrected is not None:
                y_uncorrected_np = y_uncorrected.cpu().numpy()
                baseline_errors = y_uncorrected_np - y_true_np
                self.sum_baseline_mae += np.sum(np.abs(baseline_errors))
                self.sum_baseline_mse += np.sum(baseline_errors**2)
                self.sum_baseline_bias += np.sum(baseline_errors)

            # For R² and correlation
            self.sum_y_true += np.sum(y_true_np)
            self.sum_y_true_sq += np.sum(y_true_np**2)
            self.sum_y_pred += np.sum(y_pred_np)
            self.sum_y_pred_sq += np.sum(y_pred_np**2)
            self.sum_y_true_y_pred += np.sum(y_true_np * y_pred_np)

            # Update sea-bin metrics
            for bin_config in self.sea_bins:
                bin_name = bin_config["name"]
                bin_min = bin_config["min"]
                bin_max = bin_config["max"]

                # Filter for this bin
                bin_mask = (y_true_np >= bin_min) & (y_true_np < bin_max)
                bin_count = np.sum(bin_mask)

                if bin_count > 0:
                    bin_y_true = y_true_np[bin_mask]
                    bin_y_pred = y_pred_np[bin_mask]
                    bin_errors = bin_y_pred - bin_y_true

                    self.sea_bin_accumulators[bin_name]["count"] += bin_count
                    self.sea_bin_accumulators[bin_name]["sum_mae"] += np.sum(
                        np.abs(bin_errors)
                    )
                    self.sea_bin_accumulators[bin_name]["sum_mse"] += np.sum(
                        bin_errors**2
                    )
                    self.sea_bin_accumulators[bin_name]["sum_bias"] += np.sum(
                        bin_errors
                    )

                    # Store all error samples
                    self.sea_bin_error_samples[bin_name]["model_errors"].extend(
                        bin_errors.tolist()
                    )

                    if y_uncorrected is not None:
                        bin_y_uncorrected = y_uncorrected_np[bin_mask]
                        baseline_bin_errors = bin_y_uncorrected - bin_y_true
                        self.sea_bin_accumulators[bin_name]["sum_baseline_mae"] += (
                            np.sum(np.abs(baseline_bin_errors))
                        )
                        self.sea_bin_accumulators[bin_name]["sum_baseline_mse"] += (
                            np.sum(baseline_bin_errors**2)
                        )
                        self.sea_bin_accumulators[bin_name]["sum_baseline_bias"] += (
                            np.sum(baseline_bin_errors)
                        )

                        # Count samples where model has better (lower) absolute error than baseline
                        model_better = np.abs(bin_errors) <= np.abs(baseline_bin_errors)
                        self.sea_bin_accumulators[bin_name]["count_model_better"] += (
                            np.sum(model_better)
                        )

                        model_worse = np.abs(bin_errors) > np.abs(baseline_bin_errors)
                        self.sea_bin_accumulators[bin_name]["count_model_worse"] += (
                            np.sum(model_worse)
                        )

                        # Store all baseline error samples
                        self.sea_bin_error_samples[bin_name]["baseline_errors"].extend(
                            baseline_bin_errors.tolist()
                        )

            # Store samples for plotting (limited)
            self.plot_samples["y_true"].extend(y_true_np)
            self.plot_samples["y_pred"].extend(y_pred_np)
            if y_uncorrected is not None:
                self.plot_samples["y_uncorrected"].extend(y_uncorrected_np)

    def run_inference(self):
        """Run model inference and compute metrics incrementally."""
        print("Running inference on test set...")
        self.model.eval()
        self._reset_accumulators()

        def pad_to_multiple(x, multiple=16, mode="reflect"):
            import torch.nn.functional as F

            _, _, H, W = x.shape
            pad_h = (multiple - H % multiple) % multiple
            pad_w = (multiple - W % multiple) % multiple
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
            return x, (H, W)

        # If binwise correction is enabled, compute biases from bias_loader first
        if self.apply_binwise_correction_flag:
            if self.bias_loader is None:
                raise ValueError(
                    "bias_loader must be provided when apply_binwise_correction_flag=True"
                )
            self._compute_global_bin_biases()

        with torch.no_grad():
            for _, batch in enumerate(
                tqdm(self.test_loader, desc="Processing batches")
            ):
                # Handle batch format based on predict_bias
                if self.predict_bias:
                    X, y, mask, vhm0 = batch
                    vhm0 = vhm0.to(self.device) if vhm0 is not None else None
                else:
                    X, y, mask, vhm0_batch = batch
                    vhm0 = (
                        vhm0_batch.to(self.device) if vhm0_batch is not None else None
                    )

                X, orig_size = pad_to_multiple(X, multiple=16)

                if y is not None:
                    y, _ = pad_to_multiple(y, multiple=16)
                mask_float = mask.float()
                mask, _ = pad_to_multiple(mask_float, multiple=16)
                mask = mask.bool()

                if vhm0 is not None:
                    vhm0, _ = pad_to_multiple(vhm0, multiple=16)

                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                # Get predictions
                if self.use_mdn:
                    pi, mu, sigma = self.model(X)
                    y_pred = mdn_expected_value(pi, mu)
                else:
                    y_pred = self.model(X)

                # Align dimensions
                min_h = min(y_pred.shape[2], y.shape[2])
                min_w = min(y_pred.shape[3], y.shape[3])
                min_h, min_w = orig_size
                y_pred = y_pred[:, :, :min_h, :min_w]
                y = y[:, :, :min_h, :min_w]
                mask = mask[:, :, :min_h, :min_w]
                if vhm0 is not None:
                    vhm0 = vhm0[:, :, :min_h, :min_w]

                # Unnormalize if needed
                if self.normalize_target and self.normalizer is not None:
                    y_pred = self.normalizer.inverse_transform_torch(y_pred)
                    y = self.normalizer.inverse_transform_torch(y)

                if self.apply_bilateral_filter:  # New flag
                    y_pred = self._apply_bilateral_filter(y_pred, mask)

                # Apply bin-wise correction if enabled
                if self.apply_binwise_correction_flag and vhm0 is not None:
                    y_pred = self._apply_bin_corrections(y_pred, vhm0, mask)

                # Process batch and update accumulators
                self._process_batch(y, mask, vhm0, y_pred)

        print(f"Inference complete. Processed {self.total_count} valid pixels.")

    def _compute_global_bin_biases(self):
        """Compute global bin-wise correction biases from training/validation set."""
        print(
            "Computing global bin-wise correction biases from training/validation set..."
        )

        all_residuals_by_bin = {}
        self.bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # Initialize storage for each bin
        for i in range(len(self.bins) - 1):
            bin_label = f"{self.bins[i]}-{self.bins[i + 1]}m"
            all_residuals_by_bin[bin_label] = []

        with torch.no_grad():
            for batch in tqdm(self.bias_loader, desc="Computing biases from train/val"):
                # Handle batch format
                if self.predict_bias:
                    X, y, mask, vhm0 = batch
                    vhm0 = vhm0.to(self.device) if vhm0 is not None else None
                else:
                    X, y, mask, vhm0_batch = batch
                    vhm0 = (
                        vhm0_batch.to(self.device) if vhm0_batch is not None else None
                    )

                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                # Get predictions
                y_pred = self.model(X)

                # Align dimensions
                min_h = min(y_pred.shape[2], y.shape[2])
                min_w = min(y_pred.shape[3], y.shape[3])
                y_pred = y_pred[:, :, :min_h, :min_w]
                y = y[:, :, :min_h, :min_w]
                mask = mask[:, :, :min_h, :min_w]
                if vhm0 is not None:
                    vhm0 = vhm0[:, :, :min_h, :min_w]

                # Unnormalize
                if self.normalize_target and self.normalizer is not None:
                    y_pred = self.normalizer.inverse_transform_torch(y_pred)
                    y = self.normalizer.inverse_transform_torch(y)

                # Compute residuals and bin them
                residuals = y - y_pred

                for i in range(len(self.bins) - 1):
                    low, high = self.bins[i], self.bins[i + 1]
                    bin_label = f"{low}-{high}m"
                    in_bin = (vhm0 >= low) & (vhm0 < high) & mask
                    if in_bin.any():
                        all_residuals_by_bin[bin_label].append(residuals[in_bin].cpu())

        # Compute global bin biases
        self.global_bin_biases = {}
        print("\nGlobal bin-wise correction biases (from train/val set):")
        print(f"{'Bin':<12} {'Count':<15} {'Bias ({self.unit})':<12}")
        print("-" * 39)

        for bin_label, residual_list in all_residuals_by_bin.items():
            if residual_list:
                all_residuals = torch.cat(residual_list)
                self.global_bin_biases[bin_label] = all_residuals.mean().item()
                bin_count = len(all_residuals)
                print(
                    f"{bin_label:<12} {bin_count:<15,} {self.global_bin_biases[bin_label]:>10.4f}"
                )
            else:
                self.global_bin_biases[bin_label] = 0.0

    def _apply_bin_corrections(self, y_pred, vhm0, mask):
        """Apply pre-computed global bin-wise corrections to predictions."""
        y_pred_corrected = y_pred.clone()
        for i in range(len(self.bins) - 1):
            low, high = self.bins[i], self.bins[i + 1]
            bin_label = f"{low}-{high}m"
            in_bin = (vhm0 >= low) & (vhm0 < high) & mask
            if in_bin.any() and bin_label in self.global_bin_biases:
                y_pred_corrected[in_bin] += self.global_bin_biases[bin_label]
        return y_pred_corrected

    def _apply_bilateral_filter(self, predictions, mask, d=5, sigma_color=0.3, sigma_space=5):
        """
        Apply bilateral filter to smooth extreme predictions while preserving edges.

        Args:
            predictions: [B, 1, H, W] tensor
            mask: [B, 1, H, W] boolean mask (sea pixels)
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in value space (wave height diff tolerance)
            sigma_space: Filter sigma in coordinate space (spatial distance)

        Returns:
            Filtered predictions [B, 1, H, W]
        """
        import cv2

        filtered = torch.zeros_like(predictions)

        for i in range(predictions.shape[0]):
            pred_np = predictions[i, 0].cpu().numpy()
            mask_np = mask[i, 0].cpu().numpy()

            # Only filter sea pixels
            pred_filtered = pred_np.copy()

            # Apply bilateral filter (only on valid data)
            if mask_np.sum() > 0:
                pred_filtered = cv2.bilateralFilter(
                    pred_np.astype(np.float32),
                    d=d,
                    sigmaColor=sigma_color,
                    sigmaSpace=sigma_space
                )

                # Keep land pixels unchanged
                pred_filtered[~mask_np] = pred_np[~mask_np]

            filtered[i, 0] = torch.from_numpy(pred_filtered).to(predictions.device)

        return filtered

    def compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall performance metrics from accumulators."""
        return compute_overall_metrics_from_accumulators(
            total_count=self.total_count,
            sum_mae=self.sum_mae,
            sum_mse=self.sum_mse,
            sum_bias=self.sum_bias,
            sum_baseline_mae=self.sum_baseline_mae,
            sum_baseline_mse=self.sum_baseline_mse,
            sum_baseline_bias=self.sum_baseline_bias,
            sum_y_true=self.sum_y_true,
            sum_y_true_sq=self.sum_y_true_sq,
            sum_y_pred=self.sum_y_pred,
            sum_y_pred_sq=self.sum_y_pred_sq,
            sum_y_true_y_pred=self.sum_y_true_y_pred,
            predict_bias=self.predict_bias,
        )

    def compute_sea_bin_metrics(self) -> Dict[str, Dict]:
        """Compute sea-bin metrics from accumulators."""
        return compute_sea_bin_metrics_from_accumulators(
            sea_bins=self.sea_bins,
            sea_bin_accumulators=self.sea_bin_accumulators,
        )

    def plot_rmse_maps(self):
        """Plot spatial RMSE maps for model and baseline."""
        plot_rmse_maps_fn(
            spatial_errors_model=self.spatial_errors_model,
            spatial_errors_baseline=self.spatial_errors_baseline,
            test_files=self.test_files,
            subsample_step=self.subsample_step,
            geo_bounds=self.geo_bounds,
            unit=self.unit,
            output_dir=self.output_dir,
        )

    def plot_model_better_percentage(self, sea_bin_metrics: Dict[str, Dict]):
        """Plot percentage of samples where model is better than reference for each bin."""
        plot_model_better_percentage_fn(
            sea_bin_metrics=sea_bin_metrics,
            sea_bins=self.sea_bins,
            var_name_full=self.var_name_full,
            output_dir=self.output_dir,
        )

    def plot_sea_bin_metrics(self, sea_bin_metrics: Dict[str, Dict]):
        """Create sea-bin performance metrics plot with baseline comparison."""
        plot_sea_bin_metrics_fn(
            sea_bin_metrics=sea_bin_metrics,
            sea_bins=self.sea_bins,
            target_column=self.target_column,
            unit=self.unit,
            output_dir=self.output_dir,
        )

    def plot_error_distribution_histograms(self):
        """Plot histogram grid showing error distributions per sea bin."""
        plot_error_distribution_histograms_fn(
            sea_bin_error_samples=self.sea_bin_error_samples,
            sea_bins=self.sea_bins,
            target_column=self.target_column,
            unit=self.unit,
            output_dir=self.output_dir,
        )

    def plot_error_boxplots(self):
        """Plot box plot comparison of errors across all sea bins."""
        plot_error_boxplots_fn(
            sea_bin_error_samples=self.sea_bin_error_samples,
            sea_bins=self.sea_bins,
            target_column=self.target_column,
            unit=self.unit,
            output_dir=self.output_dir,
        )

    def plot_error_violins(self):
        """Plot violin plots showing error distributions per sea bin."""
        plot_error_violins_fn(
            sea_bin_error_samples=self.sea_bin_error_samples,
            sea_bins=self.sea_bins,
            target_column=self.target_column,
            unit=self.unit,
            output_dir=self.output_dir,
        )

    def plot_error_cdfs(self):
        """Plot cumulative distribution functions for errors across sea bins."""
        plot_error_cdfs_fn(
            sea_bin_error_samples=self.sea_bin_error_samples,
            sea_bins=self.sea_bins,
            target_column=self.target_column,
            unit=self.unit,
            output_dir=self.output_dir,
        )
        return  # Method extracted to evaluation_plots.py

    def plot_vhm0_distributions(self):
        """Plot distributions of ground truth, predicted, and uncorrected VHM0."""
        plot_vhm0_distributions_fn(
            plot_samples=self.plot_samples,
            var_name=self.var_name,
            var_name_full=self.var_name_full,
            unit=self.unit,
            corrected_label=self.corrected_label,
            model_label=self.model_label,
            uncorrected_label=self.uncorrected_label,
            output_dir=self.output_dir,
        )
        return  # Method extracted to evaluation_plots.py

    def print_summary(self, overall_metrics: Dict, sea_bin_metrics: Dict):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        print("\nOverall Metrics:")
        print(f"  Samples:              {overall_metrics['n_samples']:,}")
        print(f"  MAE:                  {overall_metrics['mae']:.4f} m")
        print(f"  RMSE:                 {overall_metrics['rmse']:.4f} m")
        print(f"  Bias:                 {overall_metrics['bias']:.4f} m")
        print(f"  R²:                   {overall_metrics['r2']:.4f}")
        print(f"  Correlation:          {overall_metrics['correlation']:.4f}")

        if overall_metrics.get("baseline_mae") is not None:
            print("\nBaseline (Uncorrected) Metrics:")
            print(f"  MAE:                  {overall_metrics['baseline_mae']:.4f} m")
            print(f"  RMSE:                 {overall_metrics['baseline_rmse']:.4f} m")
            print(f"  Bias:                 {overall_metrics['baseline_bias']:.4f} m")

            print("\nImprovement:")
            if overall_metrics.get("mae_improvement_pct") is not None:
                print(
                    f"  MAE Improvement:      {overall_metrics['mae_improvement_pct']:.2f}%"
                )
            if overall_metrics.get("rmse_improvement_pct") is not None:
                print(
                    f"  RMSE Improvement:     {overall_metrics['rmse_improvement_pct']:.2f}%"
                )

        print("\nSea-Bin Metrics:")
        print(
            f"{'Bin':<20} {'Count':<10} {'MAE':<10} {'RMSE':<10} {'MAE Improv':<15} {'RMSE Improv':<15} {'% Better':<12}"
        )
        print("-" * 102)

        for _, metrics in sea_bin_metrics.items():
            if metrics["count"] > 0:
                improvement_str = (
                    f"{metrics['mae_improvement_pct']:>7.2f}%"
                    if metrics.get("mae_improvement_pct") is not None
                    else "N/A"
                )
                improvement_rmse_str = (
                    f"{metrics['rmse_improvement_pct']:>7.2f}%"
                    if metrics.get("rmse_improvement_pct") is not None
                    else "N/A"
                )
                pct_better_str = (
                    f"{metrics['pct_model_better']:>7.2f}%"
                    if metrics.get("pct_model_better") is not None
                    else "N/A"
                )
                print(
                    f"{metrics['label']:<20} "
                    f"{metrics['count']:<10,} "
                    f"{metrics['mae']:<10.4f} "
                    f"{metrics['rmse']:<10.4f} "
                    f"{improvement_str:>15} "
                    f"{improvement_rmse_str:>15} "
                    f"{pct_better_str:>12}"
                )

        print("=" * 80 + "\n")

    def evaluate(self):
        """Run full evaluation pipeline."""
        print("Starting evaluation...")

        # Run inference (computes metrics incrementally)
        self.run_inference()

        # Compute final metrics from accumulators
        print("Computing final metrics...")
        overall_metrics = self.compute_overall_metrics()
        sea_bin_metrics = self.compute_sea_bin_metrics()

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "overall": overall_metrics,
                    "sea_bins": sea_bin_metrics,
                    # "spatial": spatial_metrics
                },
                f,
                indent=2,
            )

        # Create plots using samples
        print("Creating plots...")
        self.plot_sea_bin_metrics(sea_bin_metrics)
        self.plot_model_better_percentage(sea_bin_metrics)
        self.plot_rmse_maps()
        # self.plot_vhm0_distributions()
        # self.plot_error_distribution_histograms()
        # self.plot_error_boxplots()
        # self.plot_error_violins()
        # self.plot_error_cdfs()

        # Print summary
        self.print_summary(overall_metrics, sea_bin_metrics)

        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate WaveBiasCorrector model")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Path to model checkpoint file or directory (evaluates all .ckpt files in directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config_dnn.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--apply-binwise-correction",
        action="store_true",
        help="Apply bin-wise bias correction computed from training set",
    )
    parser.add_argument(
        "--apply-geographic-filtering",
        action="store_true",
        help="Apply geographic filtering to the test set",
    )
    args = parser.parse_args()

    config = DNNConfig(args.config)

    training_config = config.config["training"]
    data_config = config.config["data"]
    predict_bias = data_config.get("predict_bias", False)
    target_column = data_config.get("target_column", "corrected_VHM0")

    # Get file list (same as training)
    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )
    _test_files_parq = get_file_list(
        # f"s3://medwav-dev-data/parquet/hourly/year={data_config.get('test_year', [2023])[0]}/",
        "/data/users/aiuser/parquet",
        f"WAVEAN{data_config.get('test_year', [2023])[0]}*.parquet",
    )

    _, _, test_files_parq = split_files_by_year(
        _test_files_parq,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )
    print(test_files_parq[:10])

    logger.info(f"Found {len(files)} files")

    # Split files by year (same as training)
    train_files, _, test_files = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )

    logger.info(f"Test files: {len(test_files)}")
    logger.info(f"Train files: {len(train_files)}")

    # Load normalizer (same as training)
    # normalizer = WaveNormalizer.load_from_s3(
    #     "medwav-dev-data", data_config["normalizer_path"]
    # )
    normalizer = WaveNormalizer.load_from_disk(
        data_config["normalizer_path"]
    )
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")

    # Create test dataset (same parameters as training)
    patch_size = tuple(data_config["patch_size"]) if data_config["patch_size"] else None
    excluded_columns = data_config.get(
        "excluded_columns", ["time", "latitude", "longitude", "timestamp"]
    )
    target_column = data_config.get("target_column", "corrected_VHM0")
    subsample_step = data_config.get("subsample_step", None)

    if None is True:
        test_dataset = GridPatchWaveDataset(
            test_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_column=target_column,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            use_cache=False,
            normalize_target=data_config.get("normalize_target", False),
        )
    else:
        test_dataset = CachedWaveDataset(
            test_files,
            excluded_columns=excluded_columns,
            target_column=target_column,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=False,
            use_cache=False,  # Use cache for evaluation
            normalize_target=data_config.get("normalize_target", False),
        )
    # Create test loader (use training batch size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        persistent_workers=training_config["num_workers"] > 0,
        prefetch_factor=training_config["prefetch_factor"],
    )

    # Create train loader for binwise correction (if needed)
    train_loader = None
    if args.apply_binwise_correction:
        logger.info("Creating train loader for bin-wise correction...")
        if patch_size is not None:
            train_dataset = GridPatchWaveDataset(
                train_files,
                patch_size=patch_size,
                excluded_columns=excluded_columns,
                target_column=target_column,
                predict_bias=predict_bias,
                subsample_step=subsample_step,
                normalizer=normalizer,
                use_cache=False,
                normalize_target=data_config.get("normalize_target", False),
            )
        else:
            train_dataset = CachedWaveDataset(
                train_files,
                excluded_columns=excluded_columns,
                target_column=target_column,
                predict_bias=predict_bias,
                subsample_step=subsample_step,
                normalizer=normalizer,
                enable_profiler=False,
                use_cache=False,
                normalize_target=data_config.get("normalize_target", False),
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
            persistent_workers=training_config["num_workers"] > 0,
            prefetch_factor=training_config["prefetch_factor"],
        )
        logger.info(f"Train loader created with {len(train_dataset)} samples")

    # Get checkpoint path (file or directory)
    # Priority: command line arg > config resume_from_checkpoint > config checkpoint_dir
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif config.config["checkpoint"]["resume_from_checkpoint"]:
        checkpoint_path = config.config["checkpoint"]["resume_from_checkpoint"]
    else:
        # Try to get checkpoint directory
        checkpoint_dir = config.config["checkpoint"].get("checkpoint_dir")
        if checkpoint_dir and Path(checkpoint_dir).exists():
            checkpoint_path = checkpoint_dir
        else:
            raise ValueError("No checkpoint specified. Use --checkpoint or set in config")

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        # Find all .ckpt files in directory
        checkpoint_list = sorted(list(checkpoint_path.glob("*.ckpt")))
        if not checkpoint_list:
            raise ValueError(f"No .ckpt files found in directory: {checkpoint_path}")
        logger.info(f"Found {len(checkpoint_list)} checkpoints to evaluate")
    elif checkpoint_path.is_file():
        checkpoint_list = [checkpoint_path]
    else:
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Loop through all checkpoints and evaluate each one
    for checkpoint in checkpoint_list:
        logger.info("=" * 80)
        logger.info(f"Evaluating checkpoint: {checkpoint}")
        logger.info("=" * 80)

        ckpt = torch.load(checkpoint, map_location="cpu")

        logger.info(f"Loading model from {checkpoint}...")
        model = WaveBiasCorrector.load_from_checkpoint(checkpoint)
        logger.info(f"Model loaded. predict_bias={predict_bias}")

        if "ema_weights" in ckpt and ckpt["ema_weights"] is not None:
            logger.info("Applying EMA weights for evaluation...")
            ema_weights = [w.to(model.device) for w in ckpt["ema_weights"]]

            # Copy into model
            for ema_param, param in zip(ema_weights, model.parameters(), strict=False):
                param.data.copy_(ema_param.data)
        else:
            logger.info("No EMA weights found in checkpoint. Using standard weights.")

        # Create geographic bounds dictionary if filtering is requested
        geo_bounds = None
        if args.apply_geographic_filtering:
            if patch_size is not None:
                logger.warning("=" * 80)
                logger.warning(
                    "Geographic filtering is NOT supported with patch-based datasets!"
                )
                logger.warning("Patches don't maintain spatial coordinate information.")
                logger.warning("Geographic filtering will be DISABLED.")
                logger.warning(
                    "To use geographic filtering, set patch_size to null in config."
                )
                logger.warning("=" * 80)
                geo_bounds = None
            else:
                # Iberian Peninsula bounds
                geo_bounds = {
                    "lat_min": 43.0,
                    "lat_max": 48.0,
                    "lon_min": -8.0,
                    "lon_max": 0.0,
                }
                logger.info(
                    f"Geographic filtering enabled: lat=[{geo_bounds['lat_min']}, {geo_bounds['lat_max']}], "
                    f"lon=[{geo_bounds['lon_min']}, {geo_bounds['lon_max']}]"
                )

        # Create evaluator and run evaluation
        evaluator = ModelEvaluator(
            model=model,
            test_loader=test_loader,
            output_dir=Path(args.output_dir)
            / config.config["logging"]["experiment_name"]
            / checkpoint.stem,  # Use checkpoint filename without extension
            predict_bias=predict_bias,
            device="cuda",
            normalizer=normalizer,
            normalize_target=data_config.get("normalize_target", False),
            test_files=test_files_parq,
            subsample_step=1,
            apply_binwise_correction_flag=args.apply_binwise_correction,
            bias_loader=train_loader,  # Use train set to compute bin biases
            geo_bounds=geo_bounds,
            use_mdn=model.use_mdn,
            target_column=target_column,
            apply_bilateral_filter=False,
        )

        evaluator.evaluate()

        logger.info(f"Completed evaluation for {checkpoint.name}")
        logger.info("=" * 80)

    logger.info(f"\nAll evaluations complete! Evaluated {len(checkpoint_list)} checkpoint(s)")


if __name__ == "__main__":
    main()
