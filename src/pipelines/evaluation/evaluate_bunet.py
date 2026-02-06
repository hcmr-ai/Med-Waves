#!/usr/bin/env python3
"""
Comprehensive evaluation script for WaveBiasCorrector model.
Provides detailed metrics, visualizations, and sea-bin analysis.
"""

import argparse
import csv
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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import your model and dataset classes
import logging

from src.classifiers.lightning_trainer import WaveBiasCorrector
from src.classifiers.networks.mdn import mdn_expected_value
from src.commons.dataloaders import CachedWaveDataset, GridPatchWaveDataset
from src.commons.helpers import SeasonHelper
from src.commons.postprocessing.post_processing import (
    apply_bilateral_filter,
    apply_bin_corrections,
    compute_global_bin_biases,
)
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
    compute_snr,
)
from src.evaluation.visuals import load_coordinates_from_parquet
from src.pipelines.training.dnn_trainer import (
    DNNConfig,
    get_file_list,
    split_files_by_year,
)

logger = logging.getLogger(__name__)


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
        target_columns: dict = {"vhm0": "corrected_VHM0"},
        apply_bilateral_filter: bool = False,
    ):
        self.model = model.to(device)
        self.model.eval()
        
        # Load bin-specific model for 0-2m waves (HARDCODED FOR TESTING)
        self.low_wave_model = None
        try:
            low_wave_ckpt = "s3://medwav-dev-data/checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_64_lambda_lr/epoch=36-val_loss=0.02.ckpt"
            low_wave_ckpt = ""
            # low_wave_ckpt = "s3://medwav-dev-data/checkpoints/dnn_training_subsample_step_5_100_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_patch_bin_balanced/epoch=19-val_loss=0.01.ckpt"
            logger.info(f"Loading specialized model for 0-2m waves from {low_wave_ckpt}")
            
            # Load checkpoint manually to extract hyperparameters
            import s3fs
            if low_wave_ckpt.startswith("s3://"):
                fs = s3fs.S3FileSystem()
                with fs.open(low_wave_ckpt, "rb") as f:
                    ckpt = torch.load(f, map_location="cpu")
            else:
                ckpt = torch.load(low_wave_ckpt, map_location="cpu")
            
            # Extract hyperparameters from checkpoint
            hparams = ckpt.get("hyper_parameters", {})
            logger.info(f"Checkpoint hyperparameters: {list(hparams.keys())}")
            
            # Create model instance with checkpoint hyperparameters
            from src.classifiers.lightning_trainer import WaveBiasCorrector
            
            # Reconstruct model with saved hyperparameters (using correct parameter names)
            self.low_wave_model = WaveBiasCorrector(
                tasks_config=hparams.get("tasks_config", [{"name": "vhm0", "loss_type": "mse", "weight": 1.0}]),
                in_channels=hparams.get("in_channels", 15),
                lr=hparams.get("lr", 1e-4),
                loss_type=hparams.get("loss_type", "mse"),
                predict_bias=hparams.get("predict_bias", False),
                model_type=hparams.get("model_type", "transunet"),
                filters=hparams.get("filters", [64, 128, 256]),
                dropout=hparams.get("dropout", 0.2),
                use_mdn=hparams.get("use_mdn", False),
            )
            
            # Load state dict with key mapping (old single-task → new multi-task format)
            state_dict = ckpt["state_dict"]
            
            # Check if we need to remap keys from single-task to multi-task format
            if "model.final.weight" in state_dict and "model.task_heads.vhm0.weight" not in state_dict:
                logger.info("Remapping single-task checkpoint to multi-task format")
                # Rename final layer keys: model.final.* → model.task_heads.vhm0.*
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model.final."):
                        new_key = key.replace("model.final.", "model.task_heads.vhm0.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            self.low_wave_model.load_state_dict(state_dict, strict=False)
            self.low_wave_model.to(device)
            self.low_wave_model.eval()
            logger.info("✓ Successfully loaded 0-2m specialized model from state_dict")
        except Exception as e:
            logger.warning(f"Failed to load specialized 0-2m model: {e}. Using default model for all predictions.")
            self.low_wave_model = None
            import traceback
            logger.debug(traceback.format_exc())
        
        # Load bin-specific model for >=9m waves (HARDCODED FOR TESTING)
        self.high_wave_model = None
        try:
            high_wave_ckpt = "s3://medwav-dev-data/checkpoints/checkpoints_full_20-21_huber_64_lambda_lr_256/last-v1.ckpt"  # TODO: Replace with actual checkpoint path
            high_wave_ckpt = ""
            logger.info(f"Loading specialized model for >=9m waves from {high_wave_ckpt}")
            
            # Load checkpoint manually
            import s3fs
            if high_wave_ckpt.startswith("s3://"):
                fs = s3fs.S3FileSystem()
                with fs.open(high_wave_ckpt, "rb") as f:
                    ckpt = torch.load(f, map_location="cpu")
            else:
                ckpt = torch.load(high_wave_ckpt, map_location="cpu")
            
            hparams = ckpt.get("hyper_parameters", {})
            logger.info(f"High-wave checkpoint hyperparameters: {list(hparams.keys())}")
            
            from src.classifiers.lightning_trainer import WaveBiasCorrector
            self.high_wave_model = WaveBiasCorrector(
                tasks_config=hparams.get("tasks_config", [{"name": "vhm0", "loss_type": "mse", "weight": 1.0}]),
                in_channels=hparams.get("in_channels", 15),
                lr=hparams.get("lr", 1e-4),
                loss_type=hparams.get("loss_type", "mse"),
                predict_bias=hparams.get("predict_bias", False),
                model_type=hparams.get("model_type", "transunet"),
                filters=hparams.get("filters", [64, 128, 256]),
                dropout=hparams.get("dropout", 0.2),
                use_mdn=hparams.get("use_mdn", False),
            )
            
            # Load state dict with key remapping
            state_dict = ckpt["state_dict"]
            if "model.final.weight" in state_dict and "model.task_heads.vhm0.weight" not in state_dict:
                logger.info("Remapping single-task checkpoint to multi-task format (high-wave)")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model.final."):
                        new_key = key.replace("model.final.", "model.task_heads.vhm0.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            self.high_wave_model.load_state_dict(state_dict, strict=False)
            self.high_wave_model.to(device)
            self.high_wave_model.eval()
            logger.info("✓ Successfully loaded >=9m specialized model from state_dict")
        except Exception as e:
            logger.warning(f"Failed to load specialized >=9m model: {e}. Using default model for high waves.")
            self.high_wave_model = None
            import traceback
            logger.debug(traceback.format_exc())
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
        self.target_columns = target_columns
        
        # For backward compatibility and single-task evaluation, use first target
        self.target_column = list(self.target_columns.values())[0]
        self.task_name = list(self.target_columns.keys())[0]
        print(self.target_column, self.task_name)
        print(self.target_columns)
        
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

        # Timestamp cache for seasonal analysis
        self._timestamps_cache = {}

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

        # Category breakdown accumulators: corrected vs not_corrected
        self.category_breakdown = {}
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            self.category_breakdown[bin_name] = {
                'corrected': {
                    'count': 0,
                    'feature_sums': {},  # Dict: {feature_idx: sum}
                    'feature_sq_sums': {},  # For std computation
                    'snr_sum': 0.0,
                    'confidence_sum': 0.0,
                    'seasons': {'winter': 0, 'spring': 0, 'summer': 0, 'autumn': 0}
                },
                'not_corrected': {
                    'count': 0,
                    'feature_sums': {},
                    'feature_sq_sums': {},
                    'snr_sum': 0.0,
                    'confidence_sum': 0.0,
                    'seasons': {'winter': 0, 'spring': 0, 'summer': 0, 'autumn': 0}
                }
            }

        # Overall breakdown (across all bins)
        self.overall_breakdown = {
            'corrected': {
                'count': 0,
                'feature_sums': {},
                'feature_sq_sums': {},
                'snr_sum': 0.0,
                'confidence_sum': 0.0,
                'seasons': {'winter': 0, 'spring': 0, 'summer': 0, 'autumn': 0}
            },
            'not_corrected': {
                'count': 0,
                'feature_sums': {},
                'feature_sq_sums': {},
                'snr_sum': 0.0,
                'confidence_sum': 0.0,
                'seasons': {'winter': 0, 'spring': 0, 'summer': 0, 'autumn': 0}
            }
        }

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

    def _process_batch(self, X, y, mask, vhm0, y_pred, timestamps=None, confidence=None):
        """Process a single batch and update accumulators.

        Args:
            X: Input features (B, C, H, W)
            y: Ground truth targets (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W)
            vhm0: Uncorrected wave heights (B, 1, H, W)
            y_pred: Model predictions (B, 1, H, W)
            timestamps: Batch timestamps for season extraction (optional)
            confidence: Model confidence values (B, H, W) (optional)
        """
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

                        # NEW: Track category breakdown (corrected vs not_corrected)
                        # Categorize samples: corrected = model improved or maintained quality
                        corrected_mask = model_better
                        not_corrected_mask = model_worse

                        # Extract input features for this bin (if X is provided)
                        if X is not None:
                            try:
                                # X shape: (B, C, H, W)
                                X_np = X.cpu().numpy()
                                # Flatten spatial dimensions
                                X_flat = X_np.reshape(X_np.shape[0], X_np.shape[1], -1)  # (B, C, H*W)
                                X_flat = X_flat.transpose(0, 2, 1)  # (B, H*W, C)
                                X_flat = X_flat.reshape(-1, X_np.shape[1])  # (B*H*W, C)
                                # Apply mask to get valid pixels only
                                X_masked = X_flat[mask_flat.cpu().numpy()]  # (N_valid, C)
                                # Get features for this specific bin
                                bin_X = X_masked[bin_mask]  # (N_bin, C)

                                # Prepare confidence for this bin
                                bin_confidence = None
                                if confidence is not None:
                                    confidence_flat = confidence.flatten()[mask_flat]
                                    bin_confidence = confidence_flat.cpu().numpy()[bin_mask]

                                # Prepare timestamps for this bin
                                bin_timestamps = None
                                if timestamps is not None:
                                    # timestamps should already be aligned with valid samples
                                    if len(timestamps) == len(mask_flat):
                                        bin_timestamps = timestamps[bin_mask]

                                # Update corrected category
                                if corrected_mask.sum() > 0:
                                    self._update_category_stats(
                                        bin_name=bin_name,
                                        category='corrected',
                                        features=bin_X[corrected_mask],
                                        y_true=bin_y_true[corrected_mask],
                                        y_pred=bin_y_pred[corrected_mask],
                                        timestamps=bin_timestamps[corrected_mask] if bin_timestamps is not None else None,
                                        confidence=bin_confidence[corrected_mask] if bin_confidence is not None else None
                                    )

                                # Update not_corrected category
                                if not_corrected_mask.sum() > 0:
                                    self._update_category_stats(
                                        bin_name=bin_name,
                                        category='not_corrected',
                                        features=bin_X[not_corrected_mask],
                                        y_true=bin_y_true[not_corrected_mask],
                                        y_pred=bin_y_pred[not_corrected_mask],
                                        timestamps=bin_timestamps[not_corrected_mask] if bin_timestamps is not None else None,
                                        confidence=bin_confidence[not_corrected_mask] if bin_confidence is not None else None
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to update category stats for bin {bin_name}: {e}")

            # Store samples for plotting (limited)
            self.plot_samples["y_true"].extend(y_true_np)
            self.plot_samples["y_pred"].extend(y_pred_np)
            if y_uncorrected is not None:
                self.plot_samples["y_uncorrected"].extend(y_uncorrected_np)

    def _update_category_stats(self, bin_name, category, features, y_true, y_pred, timestamps, confidence):
        """Update statistics for a category (corrected/not_corrected).

        Args:
            bin_name: Name of the sea bin
            category: 'corrected' or 'not_corrected'
            features: numpy array of shape (N, C) where N=samples, C=channels
            y_true: numpy array of ground truth values for this category
            y_pred: numpy array of predicted values for this category
            timestamps: numpy array of timestamps (optional)
            confidence: numpy array of confidence values (optional)
        """
        if len(features) == 0:
            return

        n = len(features)
        stats = self.category_breakdown[bin_name][category]
        overall_stats = self.overall_breakdown[category]

        # Update feature statistics (both bin-specific and overall)
        for i in range(features.shape[1]):  # For each feature channel
            feature_vals = features[:, i]

            # Bin-specific
            if i not in stats['feature_sums']:
                stats['feature_sums'][i] = 0.0
                stats['feature_sq_sums'][i] = 0.0
            stats['feature_sums'][i] += np.sum(feature_vals)
            stats['feature_sq_sums'][i] += np.sum(feature_vals**2)

            # Overall
            if i not in overall_stats['feature_sums']:
                overall_stats['feature_sums'][i] = 0.0
                overall_stats['feature_sq_sums'][i] = 0.0
            overall_stats['feature_sums'][i] += np.sum(feature_vals)
            overall_stats['feature_sq_sums'][i] += np.sum(feature_vals**2)

        # Compute and accumulate SNR from prediction quality
        # SNR = 10 * log10(signal_power / noise_power)
        # signal_power = var(y_true), noise_power = var(y_true - y_pred)
        try:
            signal_power = np.var(y_true)
            residuals = y_true - y_pred
            noise_power = np.var(residuals)
            
            if noise_power > 0 and signal_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # Accumulate SNR (multiply by sample count for proper averaging)
                stats['snr_sum'] += snr_db * n
                overall_stats['snr_sum'] += snr_db * n
        except Exception as e:
            logger.debug(f"Failed to compute SNR for {bin_name}/{category}: {e}")

        # Update seasons
        if timestamps is not None and len(timestamps) > 0:
            try:
                seasons = SeasonHelper.get_seasons_from_timestamps(timestamps)
                for season in seasons:
                    if season in stats['seasons']:
                        stats['seasons'][season] += 1
                        overall_stats['seasons'][season] += 1
            except Exception as e:
                logger.debug(f"Failed to extract seasons for {bin_name}/{category}: {e}")

        # Update confidence
        if confidence is not None and len(confidence) > 0:
            stats['confidence_sum'] += np.sum(confidence)
            overall_stats['confidence_sum'] += np.sum(confidence)

        # Update counts
        stats['count'] += n
        overall_stats['count'] += n

    def _get_timestamps_for_file(self, file_path):
        """Get timestamps from a parquet file with caching.

        Args:
            file_path: Path to parquet file

        Returns:
            Numpy array of timestamps or None if not available
        """
        if file_path not in self._timestamps_cache:
            try:
                _, _, timestamps = load_coordinates_from_parquet(
                    file_path,
                    subsample_step=self.subsample_step,
                    return_timestamps=True
                )
                self._timestamps_cache[file_path] = timestamps
                # logger.info(f"Loaded {len(timestamps) if timestamps is not None else 0} timestamps from {file_path}")
            except Exception as e:
                logger.debug(f"Failed to load timestamps from {file_path}: {e}")
                self._timestamps_cache[file_path] = None
        return self._timestamps_cache[file_path]

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
            for batch_idx, batch in enumerate(
                tqdm(self.test_loader, desc="Processing batches")
            ):
                # Unpack batch
                X, y, mask, vhm0_batch = batch
                vhm0 = vhm0_batch.to(self.device) if vhm0_batch is not None else None
                
                # Handle multi-task vs single-task format
                # If y is a dict (multi-task), extract the target for the task we're evaluating
                if isinstance(y, dict):
                    # Multi-task: extract the specific target we're evaluating
                    y = y[self.task_name]
                
                X, orig_size = pad_to_multiple(X, multiple=16)

                if y is not None:
                    y, _ = pad_to_multiple(y, multiple=16)
                mask_float = mask.float()
                mask, _ = pad_to_multiple(mask_float, multiple=16)
                mask = mask.bool()

                if vhm0 is not None:
                    vhm0, _ = pad_to_multiple(vhm0, multiple=16)

                # Load timestamps from test files for seasonal analysis
                timestamps = None
                if self.test_files and len(self.test_files) > 0:
                    try:
                        # Get file index for this batch (cycle through files)
                        file_idx = batch_idx % len(self.test_files)
                        file_path = self.test_files[file_idx]

                        # Ensure s3:// prefix for S3 files
                        if not file_path.startswith("s3://") and not file_path.startswith("/"):
                            file_path = f"s3://{file_path}"

                        # Get timestamps for this file
                        timestamps_raw = self._get_timestamps_for_file(file_path)

                        if timestamps_raw is not None:
                            # Match timestamps to valid pixels in this batch
                            n_valid = mask.sum().item()

                            if timestamps_raw.ndim == 1:
                                # Single timestamp or 1D array
                                if len(timestamps_raw) == 1:
                                    # Single timestamp per file - replicate for all valid pixels
                                    timestamps = np.full(n_valid, timestamps_raw[0], dtype='datetime64[ns]')
                                else:
                                    # Multiple timestamps - assume they correspond to flattened grid
                                    # Apply mask to get only valid pixels
                                    mask_np = mask.cpu().numpy().flatten()
                                    if len(timestamps_raw) == len(mask_np):
                                        timestamps = timestamps_raw[mask_np]
                                    else:
                                        # Fallback: use first timestamp
                                        timestamps = np.full(n_valid, timestamps_raw[0], dtype='datetime64[ns]')
                            elif timestamps_raw.ndim == 2:
                                # 2D timestamps (H, W) - flatten and mask
                                timestamps_flat = timestamps_raw.flatten()
                                mask_np = mask.cpu().numpy().flatten()
                                timestamps = timestamps_flat[mask_np]
                    except Exception as e:
                        logger.debug(f"Could not load timestamps for batch {batch_idx}: {e}")
                        timestamps = None

                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)

                # Get predictions and compute confidence
                confidence = None
                if self.use_mdn:
                    pi, mu, sigma = self.model(X)
                    y_pred = mdn_expected_value(pi, mu)

                    # Compute confidence from MDN uncertainty
                    # Lower sigma = higher confidence
                    # sigma shape: (B, num_components, 1, H, W)
                    # Average across mixture components, then take inverse as confidence
                    sigma_mean = sigma.mean(dim=1).squeeze(1)  # (B, H, W)
                    confidence = 1.0 / (sigma_mean + 1e-8)  # Higher value = more confident
                else:
                    # Use bin-specific model routing if available
                    if (self.low_wave_model is not None or self.high_wave_model is not None) and vhm0 is not None:
                        # Create masks for different wave height ranges
                        low_wave_mask = (vhm0 >= 0.0) & (vhm0 <= 1.0)
                        high_wave_mask = (vhm0 >= 12.0) & (vhm0 < 14.0)
                        mid_wave_mask = ~(low_wave_mask | high_wave_mask)
                        
                        # Get predictions from all models
                        y_pred_default = self.model(X)
                        
                        # Handle multi-task: extract task before combining
                        if isinstance(y_pred_default, dict):
                            y_pred_default = y_pred_default[self.task_name]
                        
                        # Start with default predictions
                        y_pred = y_pred_default.clone()
                        
                        # Helper function to align spatial dimensions
                        def align_predictions(pred_source, pred_target, is_mask=False):
                            """Align pred_source spatial dims to match pred_target.
                            
                            Args:
                                is_mask: If True, use nearest-neighbor to avoid boundary bleeding
                            """
                            if pred_source.shape != pred_target.shape:
                                # Resize to match target dimensions
                                import torch.nn.functional as F
                                mode = 'nearest' if is_mask else 'bilinear'
                                return F.interpolate(
                                    pred_source,
                                    size=(pred_target.shape[2], pred_target.shape[3]),
                                    mode=mode,
                                    align_corners=False if mode == 'bilinear' else None
                                )
                            return pred_source
                        
                        # Apply low-wave specialized model if available
                        if self.low_wave_model is not None and low_wave_mask.any():
                            y_pred_low = self.low_wave_model(X)
                            if isinstance(y_pred_low, dict):
                                y_pred_low = y_pred_low[self.task_name]
                            
                            # Debug: Check if shapes match
                            if batch_idx == 0:
                                logger.info(f"Shape check - Default: {y_pred.shape}, Low-wave: {y_pred_low.shape}, VHM0: {vhm0.shape}")
                            
                            # Only align if shapes differ
                            if y_pred_low.shape != y_pred.shape:
                                logger.warning(f"Shape mismatch! Aligning low-wave model output from {y_pred_low.shape} to {y_pred.shape}")
                                y_pred_low = align_predictions(y_pred_low, y_pred, is_mask=False)
                                low_wave_mask_aligned = align_predictions(low_wave_mask.float(), y_pred, is_mask=True).bool()
                            else:
                                low_wave_mask_aligned = low_wave_mask
                            
                            y_pred = torch.where(low_wave_mask_aligned, y_pred_low, y_pred)
                        
                        # Apply high-wave specialized model if available
                        if self.high_wave_model is not None and high_wave_mask.any():
                            y_pred_high = self.high_wave_model(X)
                            if isinstance(y_pred_high, dict):
                                y_pred_high = y_pred_high[self.task_name]
                            
                            # Debug: Check if shapes match
                            if batch_idx == 0:
                                logger.info(f"Shape check - Default: {y_pred.shape}, High-wave: {y_pred_high.shape}, VHM0: {vhm0.shape}")
                            
                            # Only align if shapes differ
                            if y_pred_high.shape != y_pred.shape:
                                logger.warning(f"Shape mismatch! Aligning high-wave model output from {y_pred_high.shape} to {y_pred.shape}")
                                y_pred_high = align_predictions(y_pred_high, y_pred, is_mask=False)
                                high_wave_mask_aligned = align_predictions(high_wave_mask.float(), y_pred, is_mask=True).bool()
                            else:
                                high_wave_mask_aligned = high_wave_mask
                            
                            y_pred = torch.where(high_wave_mask_aligned, y_pred_high, y_pred)
                        
                        if batch_idx == 0:
                            low_pixels = low_wave_mask.sum().item()
                            mid_pixels = mid_wave_mask.sum().item()
                            high_pixels = high_wave_mask.sum().item()
                            total_pixels = low_wave_mask.numel()
                            logger.info(f"Bin-specific routing:")
                            logger.info(f"  0-2m: {low_pixels}/{total_pixels} pixels ({100*low_pixels/total_pixels:.1f}%)" + 
                                       (" → specialized model" if self.low_wave_model is not None else " → default model"))
                            logger.info(f"  2-9m: {mid_pixels}/{total_pixels} pixels ({100*mid_pixels/total_pixels:.1f}%) → default model")
                            logger.info(f"  ≥9m: {high_pixels}/{total_pixels} pixels ({100*high_pixels/total_pixels:.1f}%)" + 
                                       (" → specialized model" if self.high_wave_model is not None else " → default model"))
                    else:
                        y_pred = self.model(X)
                
                # Handle multi-task predictions (for non-bin-routed case)
                # If y_pred is a dict (multi-task model), extract the prediction for the task we're evaluating
                if isinstance(y_pred, dict):
                    y_pred = y_pred[self.task_name]

                # Align dimensions
                min_h = min(y_pred.shape[2], y.shape[2])
                min_w = min(y_pred.shape[3], y.shape[3])
                min_h, min_w = orig_size

                # Crop to original size
                y_pred = y_pred[:, :, :min_h, :min_w]
                y = y[:, :, :min_h, :min_w]
                mask = mask[:, :, :min_h, :min_w]
                X_cropped = X[:, :, :min_h, :min_w]  # Crop X to match

                if vhm0 is not None:
                    vhm0 = vhm0[:, :, :min_h, :min_w]

                if confidence is not None:
                    confidence = confidence[:, :min_h, :min_w]

                # Unnormalize if needed
                if self.normalize_target and self.normalizer is not None:
                    # CRITICAL: Set target_stats_ for the correct target column
                    # The dataset may have left it set to a different task during normalization
                    if self.target_column in self.normalizer.feature_order_:
                        target_idx = self.normalizer.feature_order_.index(self.target_column)
                        if target_idx in self.normalizer.stats_:
                            self.normalizer.target_stats_ = self.normalizer.stats_[target_idx]
                    
                    y_pred = self.normalizer.inverse_transform_torch(y_pred)
                    y = self.normalizer.inverse_transform_torch(y)

                if self.apply_bilateral_filter:  # New flag
                    y_pred = self._apply_bilateral_filter(y_pred, mask)
                # Apply bin-wise correction if enabled
                if self.apply_binwise_correction_flag and vhm0 is not None:
                    y_pred = self._apply_bin_corrections(y_pred, vhm0, mask)
                # Process batch and update accumulators
                self._process_batch(X_cropped, y, mask, vhm0, y_pred, timestamps, confidence)

        print(f"Inference complete. Processed {self.total_count} valid pixels.")

        # Report timestamp availability for seasonal analysis
        if self._timestamps_cache:
            loaded_files = [k for k, v in self._timestamps_cache.items() if v is not None]
            if loaded_files:
                logger.info(f"Timestamps loaded from {len(loaded_files)} file(s) - seasonal analysis enabled")
            else:
                logger.info("No timestamps found in data files - seasonal analysis disabled")
        else:
            logger.info("No timestamp loading attempted - seasonal analysis disabled")

    def _compute_global_bin_biases(self):
        """Compute global bin-wise correction biases from training/validation set."""
        self.bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        self.global_bin_biases = compute_global_bin_biases(
            model=self.model,
            data_loader=self.bias_loader,
            device=self.device,
            bins=self.bins,
            predict_bias=self.predict_bias,
            normalize_target=self.normalize_target,
            normalizer=self.normalizer,
            unit=self.unit,
            task_name=self.task_name,
        )

    def _apply_bin_corrections(self, y_pred, vhm0, mask):
        """Apply pre-computed global bin-wise corrections to predictions."""
        return apply_bin_corrections(
            y_pred=y_pred,
            vhm0=vhm0,
            mask=mask,
            bins=self.bins,
            global_bin_biases=self.global_bin_biases,
        )

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
        return apply_bilateral_filter(
            predictions=predictions,
            mask=mask,
            d=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )

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

    def compute_category_breakdown(self) -> dict:
        """Compute breakdown metrics for corrected vs not_corrected categories.

        Returns:
            Dictionary containing breakdown statistics for each bin and overall
        """
        results = {'bins': {}, 'overall': {}}

        # Process each bin
        for bin_name, bin_data in self.category_breakdown.items():
            results['bins'][bin_name] = {}
            total_count = bin_data['corrected']['count'] + bin_data['not_corrected']['count']

            for category in ['corrected', 'not_corrected']:
                stats = bin_data[category]
                count = stats['count']

                if count > 0:
                    # Compute feature means and stds
                    feature_stats = {}
                    for i in sorted(stats['feature_sums'].keys()):
                        mean = stats['feature_sums'][i] / count
                        variance = (stats['feature_sq_sums'][i] / count) - (mean ** 2)
                        std = np.sqrt(max(0, variance))
                        feature_stats[f'feature_{i}'] = {
                            'mean': float(mean),
                            'std': float(std)
                        }

                    # Compute mean SNR
                    mean_snr = stats['snr_sum'] / count if stats['snr_sum'] != 0 else None

                    # Compute season percentages
                    total_seasons = sum(stats['seasons'].values())
                    season_pcts = {
                        season: (cnt / total_seasons * 100) if total_seasons > 0 else 0
                        for season, cnt in stats['seasons'].items()
                    }

                    # Compute mean confidence
                    mean_confidence = stats['confidence_sum'] / count if stats['confidence_sum'] != 0 else None

                    results['bins'][bin_name][category] = {
                        'count': int(count),
                        'percentage': float(count / total_count * 100) if total_count > 0 else 0,
                        'features': feature_stats,
                        'snr_mean': float(mean_snr) if mean_snr is not None else None,
                        'seasons': season_pcts,
                        'confidence_mean': float(mean_confidence) if mean_confidence is not None else None
                    }
                else:
                    results['bins'][bin_name][category] = None

        # Process overall
        total_overall = self.overall_breakdown['corrected']['count'] + self.overall_breakdown['not_corrected']['count']
        for category in ['corrected', 'not_corrected']:
            stats = self.overall_breakdown[category]
            count = stats['count']

            if count > 0:
                feature_stats = {}
                for i in sorted(stats['feature_sums'].keys()):
                    mean = stats['feature_sums'][i] / count
                    variance = (stats['feature_sq_sums'][i] / count) - (mean ** 2)
                    std = np.sqrt(max(0, variance))
                    feature_stats[f'feature_{i}'] = {
                        'mean': float(mean),
                        'std': float(std)
                    }

                mean_snr = stats['snr_sum'] / count if stats['snr_sum'] != 0 else None
                total_seasons = sum(stats['seasons'].values())
                season_pcts = {
                    season: (cnt / total_seasons * 100) if total_seasons > 0 else 0
                    for season, cnt in stats['seasons'].items()
                }
                mean_confidence = stats['confidence_sum'] / count if stats['confidence_sum'] != 0 else None

                results['overall'][category] = {
                    'count': int(count),
                    'percentage': float(count / total_overall * 100) if total_overall > 0 else 0,
                    'features': feature_stats,
                    'snr_mean': float(mean_snr) if mean_snr is not None else None,
                    'seasons': season_pcts,
                    'confidence_mean': float(mean_confidence) if mean_confidence is not None else None
                }

        return results

    def save_category_breakdown_csv(self, breakdown: dict, output_path: Path):
        """Save category breakdown table to CSV files.

        Args:
            breakdown: The breakdown dictionary from compute_category_breakdown
            output_path: Base path for CSV files (will create multiple files)
        """
        # Save overall breakdown
        overall_csv = output_path / "category_breakdown_overall.csv"
        with open(overall_csv, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Category', 'Count', 'Percentage', 'SNR Mean', 'Confidence Mean',
                           'Winter %', 'Spring %', 'Summer %', 'Autumn %', 'Features'])

            for category in ['corrected', 'not_corrected']:
                if category in breakdown['overall'] and breakdown['overall'][category]:
                    data = breakdown['overall'][category]

                    # Format features as string
                    features_str = '; '.join([
                        f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                        for k, v in data['features'].items()
                    ])

                    writer.writerow([
                        category.replace('_', ' ').title(),
                        data['count'],
                        f"{data['percentage']:.2f}",
                        f"{data['snr_mean']:.4f}" if data['snr_mean'] is not None else 'N/A',
                        f"{data['confidence_mean']:.4f}" if data['confidence_mean'] is not None else 'N/A',
                        f"{data['seasons']['winter']:.2f}",
                        f"{data['seasons']['spring']:.2f}",
                        f"{data['seasons']['summer']:.2f}",
                        f"{data['seasons']['autumn']:.2f}",
                        features_str
                    ])

        logger.info(f"Saved overall category breakdown to {overall_csv}")

        # Save per-bin breakdown
        bins_csv = output_path / "category_breakdown_per_bin.csv"
        with open(bins_csv, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Bin', 'Category', 'Count', 'Percentage', 'SNR Mean', 'Confidence Mean',
                           'Winter %', 'Spring %', 'Summer %', 'Autumn %', 'Features'])

            for bin_name, bin_data in breakdown['bins'].items():
                # Get bin label for readability
                bin_label = next((b['label'] for b in self.sea_bins if b['name'] == bin_name), bin_name)

                for category in ['corrected', 'not_corrected']:
                    if category in bin_data and bin_data[category]:
                        data = bin_data[category]

                        # Format features as string
                        features_str = '; '.join([
                            f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                            for k, v in data['features'].items()
                        ])

                        writer.writerow([
                            bin_label,
                            category.replace('_', ' ').title(),
                            data['count'],
                            f"{data['percentage']:.2f}",
                            f"{data['snr_mean']:.4f}" if data['snr_mean'] is not None else 'N/A',
                            f"{data['confidence_mean']:.4f}" if data['confidence_mean'] is not None else 'N/A',
                            f"{data['seasons']['winter']:.2f}",
                            f"{data['seasons']['spring']:.2f}",
                            f"{data['seasons']['summer']:.2f}",
                            f"{data['seasons']['autumn']:.2f}",
                            features_str
                        ])

        logger.info(f"Saved per-bin category breakdown to {bins_csv}")

    def save_category_breakdown_wide_format(self, breakdown: dict, output_path: Path):
        """Save category breakdown in wide pivot table format (bins as columns, metrics as rows).

        Args:
            breakdown: The breakdown dictionary from compute_category_breakdown
            output_path: Base path for CSV files
        """
        import pandas as pd

        # Prepare data structure for wide format

        # Get all bin names in order
        bin_names = [bin_config['name'] for bin_config in self.sea_bins]
        bin_labels = {bin_config['name']: bin_config['label'] for bin_config in self.sea_bins}

        # Filter to only bins with data
        bins_with_data = [bn for bn in bin_names if bn in breakdown['bins'] and
                          (breakdown['bins'][bn].get('corrected') or breakdown['bins'][bn].get('not_corrected'))]

        # Collect all feature indices that exist
        all_feature_indices = set()
        for bin_name in bins_with_data:
            bin_data = breakdown['bins'][bin_name]
            for category in ['corrected', 'not_corrected']:
                if bin_data.get(category) and bin_data[category].get('features'):
                    all_feature_indices.update([int(f.split('_')[1]) for f in bin_data[category]['features'].keys()])

        # Build rows for each metric
        metric_rows = []

        # Count row
        count_row = {'Metric': 'Count'}
        count_pct_row = {'Metric': 'Percentage'}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown['bins'][bin_name]

            for category in ['Corrected', 'Not corrected']:
                cat_key = category.lower().replace(' ', '_')
                col_name = f"{bin_label}_{category}"

                if bin_data.get(cat_key):
                    count_row[col_name] = bin_data[cat_key]['count']
                    count_pct_row[col_name] = f"{bin_data[cat_key]['percentage']:.1f}%"
                else:
                    count_row[col_name] = 0
                    count_pct_row[col_name] = '0.0%'

        metric_rows.append(count_row)
        metric_rows.append(count_pct_row)

        # Feature rows (mean only for simplicity in wide format)
        for feat_idx in sorted(all_feature_indices):
            feat_row = {'Metric': f'Mean feature {feat_idx}'}
            for bin_name in bins_with_data:
                bin_label = bin_labels[bin_name]
                bin_data = breakdown['bins'][bin_name]

                for category in ['Corrected', 'Not corrected']:
                    cat_key = category.lower().replace(' ', '_')
                    col_name = f"{bin_label}_{category}"

                    if bin_data.get(cat_key) and bin_data[cat_key].get('features'):
                        feat_key = f'feature_{feat_idx}'
                        if feat_key in bin_data[cat_key]['features']:
                            feat_row[col_name] = f"{bin_data[cat_key]['features'][feat_key]['mean']:.4f}"
                        else:
                            feat_row[col_name] = 'N/A'
                    else:
                        feat_row[col_name] = 'N/A'

            metric_rows.append(feat_row)

        # SNR row
        snr_row = {'Metric': 'SNR (mean dB)'}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown['bins'][bin_name]

            for category in ['Corrected', 'Not corrected']:
                cat_key = category.lower().replace(' ', '_')
                col_name = f"{bin_label}_{category}"

                if bin_data.get(cat_key) and bin_data[cat_key].get('snr_mean') is not None:
                    snr_row[col_name] = f"{bin_data[cat_key]['snr_mean']:.4f}"
                else:
                    snr_row[col_name] = 'N/A'

        metric_rows.append(snr_row)

        # Season rows
        for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
            season_row = {'Metric': season}
            season_key = season.lower()

            for bin_name in bins_with_data:
                bin_label = bin_labels[bin_name]
                bin_data = breakdown['bins'][bin_name]

                for category in ['Corrected', 'Not corrected']:
                    cat_key = category.lower().replace(' ', '_')
                    col_name = f"{bin_label}_{category}"

                    if bin_data.get(cat_key) and bin_data[cat_key].get('seasons'):
                        season_row[col_name] = f"{bin_data[cat_key]['seasons'][season_key]:.1f}%"
                    else:
                        season_row[col_name] = 'N/A'

            metric_rows.append(season_row)

        # Model confidence row
        conf_row = {'Metric': 'Model confidence'}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown['bins'][bin_name]

            for category in ['Corrected', 'Not corrected']:
                cat_key = category.lower().replace(' ', '_')
                col_name = f"{bin_label}_{category}"

                if bin_data.get(cat_key) and bin_data[cat_key].get('confidence_mean') is not None:
                    conf_row[col_name] = f"{bin_data[cat_key]['confidence_mean']:.4f}"
                else:
                    conf_row[col_name] = 'N/A'

        metric_rows.append(conf_row)

        # Create DataFrame
        df = pd.DataFrame(metric_rows)

        # Reorder columns: Metric first, then bins in order
        columns = ['Metric']
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            columns.append(f"{bin_label}_Corrected")
            columns.append(f"{bin_label}_Not corrected")

        df = df[columns]

        # Save to CSV
        wide_csv = output_path / "category_breakdown_wide_format.csv"
        df.to_csv(wide_csv, index=False)
        logger.info(f"Saved wide-format category breakdown to {wide_csv}")

    def print_category_breakdown(self, breakdown: dict):
        """Print the category breakdown table to console.

        Args:
            breakdown: The breakdown dictionary from compute_category_breakdown
        """
        print("\n" + "=" * 140)
        print("ERROR ANALYSIS - CORRECTED vs NOT CORRECTED BREAKDOWN")
        print("=" * 140)

        # Print overall first
        print("\nOVERALL (All Bins Combined):")
        print("-" * 140)
        for category in ['corrected', 'not_corrected']:
            if category in breakdown['overall'] and breakdown['overall'][category]:
                data = breakdown['overall'][category]
                print(f"\n  {category.upper().replace('_', ' ')}:")
                print(f"    Count: {data['count']:,} ({data['percentage']:.1f}%)")

                if data['features']:
                    print("    Features (mean ± std):")
                    for feat_name, feat_data in sorted(data['features'].items()):
                        print(f"      {feat_name}: {feat_data['mean']:.4f} ± {feat_data['std']:.4f}")

                if data['snr_mean'] is not None:
                    print(f"    SNR (mean): {data['snr_mean']:.4f} dB")

                print("    Seasons:")
                for season in ['winter', 'spring', 'summer', 'autumn']:
                    pct = data['seasons'][season]
                    print(f"      {season.capitalize()}: {pct:.1f}%")

                if data['confidence_mean'] is not None:
                    print(f"    Model Confidence (mean): {data['confidence_mean']:.4f}")

        # Print per-bin breakdown
        print("\n\nPER-BIN BREAKDOWN:")
        print("=" * 140)

        for bin_name, bin_data in breakdown['bins'].items():
            # Get readable bin label
            bin_label = next((b['label'] for b in self.sea_bins if b['name'] == bin_name), bin_name)

            print(f"\n{bin_label}:")
            print("-" * 140)

            for category in ['corrected', 'not_corrected']:
                if category in bin_data and bin_data[category]:
                    data = bin_data[category]
                    print(f"\n  {category.upper().replace('_', ' ')}:")
                    print(f"    Count: {data['count']:,} ({data['percentage']:.1f}%)")

                    if data['features']:
                        print("    Features (mean ± std):")
                        for feat_name, feat_data in sorted(data['features'].items()):
                            print(f"      {feat_name}: {feat_data['mean']:.4f} ± {feat_data['std']:.4f}")

                    if data['snr_mean'] is not None:
                        print(f"    SNR (mean): {data['snr_mean']:.4f} dB")

                    print("    Seasons:")
                    for season in ['winter', 'spring', 'summer', 'autumn']:
                        pct = data['seasons'][season]
                        print(f"      {season.capitalize()}: {pct:.1f}%")

                    if data['confidence_mean'] is not None:
                        print(f"    Model Confidence (mean): {data['confidence_mean']:.4f}")

        print("=" * 140 + "\n")

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

        # NEW: Compute category breakdown
        print("Computing category breakdown (corrected vs not_corrected)...")
        category_breakdown = self.compute_category_breakdown()

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "overall": overall_metrics,
                    "sea_bins": sea_bin_metrics,
                    "category_breakdown": category_breakdown,  # NEW: Add breakdown
                },
                f,
                indent=2,
            )

        # NEW: Save category breakdown to CSV
        print("Saving category breakdown to CSV...")
        self.save_category_breakdown_csv(category_breakdown, self.output_dir)
        self.save_category_breakdown_wide_format(category_breakdown, self.output_dir)

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

        # Print summaries
        self.print_summary(overall_metrics, sea_bin_metrics)

        # NEW: Print category breakdown
        self.print_category_breakdown(category_breakdown)

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
    
    # Support both old target_column (str) and new target_columns (dict)
    target_columns = data_config.get("target_columns", None)
    if target_columns is None:
        # Fall back to old single-task format
        target_column = data_config.get("target_column", "corrected_VHM0")
        target_columns = {"vhm0": target_column}

    # Get file list (same as training)
    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )
    _test_files_parq = get_file_list(
        f"s3://medwav-dev-data/parquet/hourly/year={data_config.get('test_year', [2023])[0]}/",
        # "/data/users/aiuser/parquet",
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
    normalizer = WaveNormalizer.load_from_s3(
        "medwav-dev-data", data_config["normalizer_path"]
    )
    # normalizer = WaveNormalizer.load_from_disk(
    #     data_config["normalizer_path"]
    # )
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")
    
    # CRITICAL: Set target_stats_ for the target column we're evaluating
    # Without this, inverse_transform_torch falls back to the last channel!
    first_target_col = list(target_columns.values())[0]
    if first_target_col in normalizer.feature_order_:
        target_idx = normalizer.feature_order_.index(first_target_col)
        if target_idx in normalizer.stats_:
            normalizer.target_stats_ = normalizer.stats_[target_idx]
            logger.info(f"Set normalizer target_stats_ for '{first_target_col}' (index {target_idx})")
        else:
            logger.warning(f"Target index {target_idx} not found in normalizer stats!")
    else:
        logger.warning(f"Target column '{first_target_col}' not found in normalizer feature_order!")

    # Create test dataset (same parameters as training)
    patch_size = tuple(data_config["patch_size"]) if data_config["patch_size"] else None
    excluded_columns = data_config.get(
        "excluded_columns", ["time", "latitude", "longitude", "timestamp"]
    )
    subsample_step = data_config.get("subsample_step", None)

    if None is True:
        test_dataset = GridPatchWaveDataset(
            test_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_columns=target_columns,
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
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=False,
            use_cache=False,  # Use cache for evaluation
            normalize_target=data_config.get("normalize_target", False),
        )
    # Create test loader (use training batch size)
    # Note: num_workers=0 for reproducible evaluation
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=training_config["batch_size"],
    #     shuffle=False,
    #     num_workers=0,  # Single-threaded for deterministic batch order
    #     pin_memory=training_config["pin_memory"],
    # )

    def seed_worker(worker_id):
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(42),  # Crucial!
        persistent_workers=False,  # Avoid state carryover
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
                target_columns=target_columns,
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
                target_columns=target_columns,
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
            num_workers=0,  # Single-threaded for deterministic batch order
            pin_memory=training_config["pin_memory"],
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
            subsample_step=subsample_step if subsample_step is not None else 5,  # Match preprocessed data subsampling
            apply_binwise_correction_flag=args.apply_binwise_correction,
            bias_loader=train_loader,  # Use train set to compute bin biases
            geo_bounds=geo_bounds,
            use_mdn=model.use_mdn,
            target_columns=target_columns,
            apply_bilateral_filter=False,
        )

        evaluator.evaluate()

        logger.info(f"Completed evaluation for {checkpoint.name}")
        logger.info("=" * 80)

    logger.info(f"\nAll evaluations complete! Evaluated {len(checkpoint_list)} checkpoint(s)")


if __name__ == "__main__":
    main()
