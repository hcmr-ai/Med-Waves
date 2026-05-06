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
from typing import Dict, List, Optional

import joblib
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
from src.commons.datasets.cache_wave_dataset import CachedWaveDataset
from src.commons.datasets.grid_patched_dataset import GridPatchWaveDataset
from src.commons.helpers import (
    DNNConfig,
    SeasonHelper,
    get_file_list,
    split_files_by_year,
)
from src.commons.postprocessing.post_processing import (
    apply_bilateral_filter,
    apply_bin_corrections,
    compute_global_bin_biases,
)
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer
from src.evaluation.evaluation_plots import (
    plot_coastal_distance_improvement as plot_coastal_distance_improvement_fn,
)
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
    plot_low_bin_advanced_diagnostics as plot_low_bin_advanced_diagnostics_fn,
)
from src.evaluation.evaluation_plots import (
    plot_low_bin_spatial_maps as plot_low_bin_spatial_maps_fn,
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

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        model: pl.LightningModule,
        test_loader: DataLoader,
        output_dir: Path,
        predict_bias: bool = False,
        predict_residual_to_prior: bool = False,
        residual_prior_task: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalizer: WaveNormalizer = None,
        normalize_target: bool = False,
        test_files: List[str] = None,
        subsample_step: int = None,
        apply_binwise_correction_flag: bool = False,
        bias_loader: DataLoader = None,
        geo_bounds: dict = None,
        use_mdn: bool = False,
        target_columns: dict = None,
        apply_bilateral_filter: bool = False,
        apply_delta_corrector_flag: bool = False,
        region_filter: str = None,
        low_wave_ckpt: str = None,
        high_wave_ckpt: str = None,
        static_bias_map_path: str = None,
        blend_sigma: float = None,
        uncertainty_blend_sigma: float = None,
        domain_mean_recalibration: bool = False,
        edcdf_model_path: str = None,
        edcdf_blend_sigma: float = None,
        edcdf_hard_fallback_bins: List[List[float]] = None,
        edcdf_fallback_bin_source: str = "raw",
        prior_hard_fallback_bins: List[List[float]] = None,
        prior_fallback_bin_source: str = "raw",
        prior_fallback_target: str = "prior",
        low_bin_affine_params: List[dict] = None,
        low_bin_affine_source: str = "raw",
        sampled_points_csv: Optional[str] = None,
        timestamps_csv: Optional[str] = None,
        eval_task: Optional[str] = None,
        save_predictions: bool = False,
        denoise_abs_threshold: Optional[float] = None,
    ):
        if target_columns is None:
            target_columns = {"vhm0": "corrected_VHM0"}
        self.region_filter = region_filter
        self.model = model.to(device)
        self.model.eval()

        # Load bin-specific model for 0-2m waves (HARDCODED FOR TESTING)
        self.low_wave_model = None
        try:
            # low_wave_ckpt = "s3://medwav-dev-data/checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_64_lambda_lr/epoch=36-val_loss=0.02.ckpt"
            # low_wave_ckpt = "/opt/dlami/nvme/checkpoints/dnn_training_extended_subsampled_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_cos_delta/epoch=16-val_loss=0.02.ckpt"
            # low_wave_ckpt = ""
            low_wave_ckpt = low_wave_ckpt
            # low_wave_ckpt = "s3://medwav-dev-data/checkpoints/dnn_training_subsample_step_5_100_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_patch_bin_balanced/epoch=19-val_loss=0.01.ckpt"
            logger.info(
                f"[LOW-WAVE] Loading specialized model for 0-2m waves from {low_wave_ckpt}"
            )

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
            logger.info(f"[LOW-WAVE] Checkpoint hyperparameters: {list(hparams.keys())}")

            # Create model instance with checkpoint hyperparameters
            from src.classifiers.lightning_trainer import WaveBiasCorrector

            # Reconstruct model with saved hyperparameters (using correct parameter names)
            self.low_wave_model = WaveBiasCorrector(
                tasks_config=hparams.get(
                    "tasks_config",
                    [{"name": "vhm0", "loss_type": "mse", "weight": 1.0}],
                ),
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
            if (
                "model.final.weight" in state_dict
                and "model.task_heads.vhm0.weight" not in state_dict
            ):
                logger.info("[LOW-WAVE] Remapping single-task checkpoint to multi-task format")
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
            logger.info("[LOW-WAVE] ✓ Successfully loaded 0-2m specialized model from state_dict")
        except Exception as e:
            logger.warning(
                f"[LOW-WAVE] Failed to load specialized 0-2m model: {e}. Using default model for all predictions."
            )
            self.low_wave_model = None
            import traceback

            logger.debug(traceback.format_exc())

        # Load bin-specific model for >=9m waves (HARDCODED FOR TESTING)
        self.high_wave_model = None
        try:
            # high_wave_ckpt = "s3://medwav-dev-data/checkpoints/checkpoints_full_20-21_huber_64_lambda_lr_256/last-v1.ckpt"  # TODO: Replace with actual checkpoint path
            # high_wave_ckpt = "/opt/dlami/nvme/checkpoints_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_huber_tail_64_lambda_lr_bias_correction_mediterranean_filtered/epoch=14-val_loss=0.03.ckpt"
            # high_wave_ckpt = "/opt/dlami/nvme/checkpoints/dnn_training_extended_subsampled_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_cos_delta_mediterranean/epoch=10-val_loss=0.01.ckpt"
            high_wave_ckpt = high_wave_ckpt
            print(
                f"[HIGH-WAVE] Loading specialized model for 8-9m waves from {high_wave_ckpt}"
            )

            # Load checkpoint manually
            import s3fs

            if high_wave_ckpt.startswith("s3://"):
                fs = s3fs.S3FileSystem()
                with fs.open(high_wave_ckpt, "rb") as f:
                    ckpt = torch.load(f, map_location="cpu")
            else:
                ckpt = torch.load(high_wave_ckpt, map_location="cpu")

            hparams = ckpt.get("hyper_parameters", {})
            print(f"[HIGH-WAVE] Checkpoint hyperparameters: {list(hparams.keys())}")

            from src.classifiers.lightning_trainer import WaveBiasCorrector

            self.high_wave_model = WaveBiasCorrector(
                tasks_config=hparams.get(
                    "tasks_config",
                    [{"name": "vhm0", "loss_type": "mse", "weight": 1.0}],
                ),
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
            if (
                "model.final.weight" in state_dict
                and "model.task_heads.vhm0.weight" not in state_dict
            ):
                print(
                    "[HIGH-WAVE] Remapping single-task checkpoint to multi-task format"
                )
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
            print("✓ [HIGH-WAVE] Successfully loaded 8-9m specialized model from state_dict")
        except Exception as e:
            print(
                f"✗ [HIGH-WAVE] Failed to load specialized 8-9m model: {e}. Using default model for high waves."
            )
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
        self.predict_residual_to_prior = predict_residual_to_prior
        self.normalizer = normalizer
        self.normalize_target = normalize_target
        self.apply_binwise_correction_flag = apply_binwise_correction_flag
        self.apply_delta_corrector_flag = apply_delta_corrector_flag
        self.geo_bounds = geo_bounds  # {'lat_min': float, 'lat_max': float, 'lon_min': float, 'lon_max': float}
        self.use_mdn = use_mdn
        self.target_columns = target_columns

        # Which task to score (default: first key in dict / YAML order)
        if eval_task is not None:
            if eval_task not in self.target_columns:
                raise ValueError(
                    f"eval_task {eval_task!r} not in target_columns "
                    f"{list(self.target_columns.keys())}"
                )
            self.task_name = eval_task
            self.target_column = self.target_columns[eval_task]
        else:
            self.target_column = list(self.target_columns.values())[0]
            self.task_name = list(self.target_columns.keys())[0]
        if residual_prior_task is None:
            if "vhm0" in self.target_columns:
                residual_prior_task = "vhm0"
            else:
                residual_prior_task = self.task_name
        self.residual_prior_task = residual_prior_task
        self.eval_in_bias_mode = self.predict_bias or self.predict_residual_to_prior
        print(self.target_column, self.task_name)
        print(self.target_columns)

        self.apply_bilateral_filter = apply_bilateral_filter
        if self.apply_bilateral_filter:
            logger.info("Applying bilateral filter to predictions")
        if self.apply_delta_corrector_flag:
            logger.info("Applying DeltaCorrector to predictions for bins >= 11m")
        self._configure_sea_bins()
        self._configure_labels()
        self.low_bin_spatial_subbins = [(0.0, 0.1), (0.1, 0.2)]
        self.coastal_distance_bins_km = [
            (0.0, 10.0),
            (10.0, 25.0),
            (25.0, 50.0),
            (50.0, float("inf")),
        ]
        self.coastal_distance_km_map = None
        self.coastal_distance_bin_idx_map = None

        self.test_files = test_files
        self.subsample_step = subsample_step

        # Load geographic mask if filtering is requested
        self.geo_mask = None
        if self.geo_bounds and self.test_files:
            self._load_geographic_mask()

        # Build static spatial exclusion mask (Bay of Biscay: Atlantic water above Gibraltar)
        self.atlantic_exclusion_mask = None
        # self._build_atlantic_exclusion_mask()

        # Add spatial accumulators for RMSE maps
        self.spatial_errors_model = []  # Store (error_map, count_map) for each batch
        self.spatial_errors_baseline = []

        # Sampled grid-point time-series recording
        self.sampled_points_csv = sampled_points_csv
        self.timestamps_csv = timestamps_csv
        self.save_predictions = save_predictions
        self.denoise_abs_threshold = denoise_abs_threshold
        self._denoise_warned_no_baseline = False
        self._grid_point_indices: Optional[List[dict]] = None  # set by _setup_grid_point_sampling
        self._grid_point_records: List[dict] = []
        self._gp_ts_map: dict = self._load_ts_map(timestamps_csv)

        # Initialize accumulators for incremental computation
        self._reset_accumulators()
        self._init_coastal_distance_diagnostics()

        # Sample storage for plots (optional, limited size)
        self.plot_samples = {
            "y_true": [],
            "y_pred": [],
            "y_uncorrected": [],
            "vhm0": [],
            "lat": [],
            "lon": [],
        }

        # Load coordinate grids once for plot_samples coordinate accumulation
        self._coord_lat_grid: Optional[np.ndarray] = None
        self._coord_lon_grid: Optional[np.ndarray] = None
        try:
            dataset = self.test_loader.dataset
            if hasattr(dataset, "get_coordinates"):
                self._coord_lat_grid, self._coord_lon_grid = dataset.get_coordinates()
        except Exception as e:
            logger.warning(f"Could not load coordinate grids for plot_samples: {e}")

        # Timestamp cache for seasonal analysis
        self._timestamps_cache = {}

        # Static bias map for soft blend and domain-mean recalibration
        self.static_bias_map = None
        self.static_bias_valid = None
        self.static_domain_mean = None
        self.blend_sigma = blend_sigma
        self.uncertainty_blend_sigma = uncertainty_blend_sigma
        self.domain_mean_recalibration = domain_mean_recalibration
        if static_bias_map_path and (
            blend_sigma is not None
            or uncertainty_blend_sigma is not None
            or domain_mean_recalibration
            or (prior_hard_fallback_bins and prior_fallback_target == "static")
        ):
            self._load_static_bias_map(static_bias_map_path)

        # Optional EDCDF prior for Gaussian trust blending of predicted bias
        self.edcdf_corrector = None
        self.edcdf_blend_sigma = edcdf_blend_sigma
        self.edcdf_hard_fallback_bins = self._parse_wave_bin_ranges(
            edcdf_hard_fallback_bins
        )
        self.edcdf_fallback_bin_source = (
            str(edcdf_fallback_bin_source).strip().lower()
            if edcdf_fallback_bin_source is not None
            else "raw"
        )
        if self.edcdf_fallback_bin_source not in {"raw", "edcdf", "true"}:
            logger.warning(
                f"Unknown edcdf_fallback_bin_source='{edcdf_fallback_bin_source}', using 'raw'"
            )
            self.edcdf_fallback_bin_source = "raw"
        if edcdf_model_path and (
            edcdf_blend_sigma is not None or len(self.edcdf_hard_fallback_bins) > 0
        ):
            self._load_edcdf_corrector(edcdf_model_path)
        self._edcdf_fallback_total_valid = 0
        self._edcdf_fallback_total_applied = 0
        self._edcdf_fallback_applied_per_bin = {
            f"[{lo},{hi})": 0 for lo, hi in self.edcdf_hard_fallback_bins
        }
        if self.edcdf_hard_fallback_bins and self.edcdf_fallback_bin_source != "true":
            logger.info(
                "EDCDF hard fallback gating is not using true bins "
                f"(source='{self.edcdf_fallback_bin_source}'). "
                "Sea-bin plots use true bins, so fallback coverage may differ from plotted bins."
            )
        self.prior_hard_fallback_bins = self._parse_wave_bin_ranges(
            prior_hard_fallback_bins
        )
        self.prior_fallback_bin_source = (
            str(prior_fallback_bin_source).strip().lower()
            if prior_fallback_bin_source is not None
            else "raw"
        )
        if self.prior_fallback_bin_source not in {"raw", "true"}:
            logger.warning(
                f"Unknown prior_fallback_bin_source='{prior_fallback_bin_source}', using 'raw'"
            )
            self.prior_fallback_bin_source = "raw"
        self.prior_fallback_target = (
            str(prior_fallback_target).strip().lower()
            if prior_fallback_target is not None
            else "prior"
        )
        if self.prior_fallback_target not in {"prior", "raw", "static"}:
            logger.warning(
                f"Unknown prior_fallback_target='{prior_fallback_target}', using 'prior'"
            )
            self.prior_fallback_target = "prior"
        self._prior_fallback_total_valid = 0
        self._prior_fallback_total_applied = 0
        self._prior_fallback_applied_per_bin = {
            f"[{lo},{hi})": 0 for lo, hi in self.prior_hard_fallback_bins
        }
        if self.prior_hard_fallback_bins and self.prior_fallback_bin_source != "true":
            logger.info(
                "Prior hard fallback gating is not using true bins "
                f"(source='{self.prior_fallback_bin_source}'). "
                "Sea-bin plots use true bins, so fallback coverage may differ from plotted bins."
            )

        self.low_bin_affine_params = self._parse_low_bin_affine_params(
            low_bin_affine_params
        )
        self.low_bin_affine_source = (
            str(low_bin_affine_source).strip().lower()
            if low_bin_affine_source is not None
            else "raw"
        )
        if self.low_bin_affine_source not in {"raw", "true"}:
            logger.warning(
                f"Unknown low_bin_affine_source='{low_bin_affine_source}', using 'raw'"
            )
            self.low_bin_affine_source = "raw"

    def _parse_wave_bin_ranges(self, bins) -> List[tuple]:
        """Parse [[lo, hi], ...] wave bins into validated [(lo, hi), ...]."""
        if bins is None:
            return []

        parsed = []
        for idx, item in enumerate(bins):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                logger.warning(
                    f"Skipping invalid EDCDF fallback bin at index {idx}: {item}"
                )
                continue
            try:
                lo = float(item[0])
                hi = float(item[1])
            except (TypeError, ValueError):
                logger.warning(
                    f"Skipping non-numeric EDCDF fallback bin at index {idx}: {item}"
                )
                continue
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                logger.warning(
                    f"Skipping invalid EDCDF fallback bin bounds at index {idx}: {item}"
                )
                continue
            parsed.append((lo, hi))

        if parsed:
            logger.info(f"EDCDF hard fallback bins enabled: {parsed}")
        return parsed

    def _parse_low_bin_affine_params(self, params) -> List[dict]:
        """Parse low-bin affine calibration config into validated entries."""
        if params is None:
            return []
        parsed = []
        for i, p in enumerate(params):
            if not isinstance(p, dict):
                logger.warning(f"Skipping invalid low_bin_affine_params[{i}]: {p}")
                continue
            try:
                lo = float(p["min"])
                hi = float(p["max"])
                a = float(p["a"])
                c = float(p["c"])
            except (KeyError, TypeError, ValueError):
                logger.warning(f"Skipping malformed low_bin_affine_params[{i}]: {p}")
                continue
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                logger.warning(f"Skipping invalid low-bin range low_bin_affine_params[{i}]: {p}")
                continue
            parsed.append({"min": lo, "max": hi, "a": a, "c": c})
        if parsed:
            logger.info(f"Low-bin affine calibration enabled with {len(parsed)} range(s): {parsed}")
        return parsed

    def _load_static_bias_map(self, path: str):
        """Load static bias map and crop to match dataset region."""
        try:
            raw = np.load(path)
            valid = ~np.isnan(raw)
            raw = np.nan_to_num(raw, nan=0.0)

            dataset = self.test_loader.dataset
            crop_h = getattr(dataset, "crop_h_indices", None)
            crop_w = getattr(dataset, "crop_w_indices", None)
            if crop_h is not None and crop_w is not None:
                raw = raw[crop_h.numpy(), :][:, crop_w.numpy()]
                valid = valid[crop_h.numpy(), :][:, crop_w.numpy()]

            self.static_bias_map = torch.from_numpy(raw).float().to(self.device)
            self.static_bias_valid = torch.from_numpy(valid).to(self.device)
            self.static_domain_mean = self.static_bias_map[self.static_bias_valid].mean().item()
            logger.info(
                f"Loaded static bias map ({self.static_bias_map.shape}) "
                f"with blend_sigma={self.blend_sigma}, "
                f"domain_mean_recal={self.domain_mean_recalibration}, "
                f"static_domain_mean={self.static_domain_mean:.6f}"
            )
        except Exception as e:
            logger.warning(f"Could not load static bias map: {e}")
            self.static_bias_map = None
            self.static_bias_valid = None

    def _load_edcdf_corrector(self, path: str):
        """Load fitted EDCDFCorrector used as a dynamic prior."""
        try:
            self.edcdf_corrector = joblib.load(path)
            n_models = len(getattr(self.edcdf_corrector, "cdf_models", {}))
            logger.info(
                f"Loaded EDCDF model from {path} with {n_models} variable model(s), "
                f"edcdf_blend_sigma={self.edcdf_blend_sigma}"
            )
        except Exception as e:
            logger.warning(f"Could not load EDCDF model from {path}: {e}")
            self.edcdf_corrector = None

    def _format_distance_bin_label(self, lo: float, hi: float) -> str:
        if np.isinf(hi):
            return f">={int(lo)}km"
        return f"{int(lo)}-{int(hi)}km"

    def _init_coastal_distance_diagnostics(self):
        """Build distance-to-coast map (km) and per-pixel distance-bin map."""
        if distance_transform_edt is None:
            logger.warning(
                "scipy.ndimage.distance_transform_edt unavailable; coastal diagnostics disabled."
            )
            return
        try:
            dataset = self.test_loader.dataset
            if not hasattr(dataset, "get_coordinates"):
                logger.warning("Dataset has no get_coordinates; coastal diagnostics disabled.")
                return

            lat_grid, lon_grid = dataset.get_coordinates()
            lat_grid = np.asarray(lat_grid, dtype=np.float64)
            lon_grid = np.asarray(lon_grid, dtype=np.float64)
            if lat_grid.ndim != 2 or lon_grid.ndim != 2:
                logger.warning("Invalid coordinate grids for coastal diagnostics.")
                return

            # Use first sample mask as static sea/land proxy.
            sample = dataset[0]
            if not isinstance(sample, (tuple, list)) or len(sample) < 3:
                logger.warning("Could not read sample mask for coastal diagnostics.")
                return
            sample_mask = sample[2]
            if isinstance(sample_mask, torch.Tensor):
                sample_mask_np = sample_mask.detach().cpu().numpy()
            else:
                sample_mask_np = np.asarray(sample_mask)
            if sample_mask_np.ndim == 3:
                sea_mask = sample_mask_np[0].astype(bool)
            elif sample_mask_np.ndim == 2:
                sea_mask = sample_mask_np.astype(bool)
            else:
                logger.warning("Unexpected sample mask shape for coastal diagnostics.")
                return

            # Estimate km spacing per pixel from coordinate grid.
            lat_diffs = np.abs(np.diff(lat_grid, axis=0))
            lon_diffs = np.abs(np.diff(lon_grid, axis=1))
            lat_step_deg = np.nanmedian(lat_diffs[np.isfinite(lat_diffs)])
            lon_step_deg = np.nanmedian(lon_diffs[np.isfinite(lon_diffs)])
            mean_lat = np.nanmean(lat_grid[np.isfinite(lat_grid)])
            lat_km = max(1e-6, 111.32 * lat_step_deg)
            lon_km = max(1e-6, 111.32 * np.cos(np.deg2rad(mean_lat)) * lon_step_deg)

            if not np.isfinite(lat_km) or not np.isfinite(lon_km):
                logger.warning("Invalid coordinate spacing; coastal diagnostics disabled.")
                return

            # Distance to nearest land for sea pixels.
            dist_km = distance_transform_edt(sea_mask, sampling=(lat_km, lon_km)).astype(
                np.float64
            )
            dist_km[~sea_mask] = np.nan

            bin_idx = np.full(dist_km.shape, -1, dtype=np.int16)
            for i, (lo, hi) in enumerate(self.coastal_distance_bins_km):
                if np.isinf(hi):
                    m = sea_mask & (dist_km >= lo)
                else:
                    m = sea_mask & (dist_km >= lo) & (dist_km < hi)
                bin_idx[m] = i

            self.coastal_distance_km_map = dist_km
            self.coastal_distance_bin_idx_map = bin_idx
            logger.info(
                "Initialized coastal diagnostics with bins: %s",
                [self._format_distance_bin_label(lo, hi) for lo, hi in self.coastal_distance_bins_km],
            )
        except Exception as e:
            logger.warning(f"Failed to initialize coastal diagnostics: {e}")
            self.coastal_distance_km_map = None
            self.coastal_distance_bin_idx_map = None

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
        self._temporal_all_model_sq = np.zeros((24, 12), dtype=np.float64)
        self._temporal_all_base_sq = np.zeros((24, 12), dtype=np.float64)
        self._temporal_all_count = np.zeros((24, 12), dtype=np.float64)
        self.denoise_total_candidate = 0
        self.denoise_total_kept = 0

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

        # Bin-conditional spatial error accumulators keyed by (lo, hi) true-wave bin
        self.bin_spatial_accumulators = {
            (b["min"], b["max"]): {"error_sq": [], "count": []}
            for b in self.sea_bins
        }
        self.spatial_rmse_accumulators = {}
        # Per-pixel relative improvement accumulator: (|base_err| - |model_err|) / |base_err| * 100
        self._rel_improvement_samples = []
        self.low_bin_spatial_accumulators = {
            f"{lo:.1f}_{hi:.1f}": {
                "sum_delta_abs_err": None,
                "sum_model_err": None,
                "sum_base_err": None,
                "count": None,
                "count_worse": None,
            }
            for lo, hi in self.low_bin_spatial_subbins
        }
        self.low_bin_plot_sample_limit_per_subbin = 250000
        self.low_bin_plot_samples = {
            f"{lo:.1f}_{hi:.1f}": {
                "true_wave": [],
                "pred_wave": [],
                "raw_wave": [],
                "true_bias": [],
                "pred_bias": [],
                "prior_bias": [],
            }
            for lo, hi in self.low_bin_spatial_subbins
        }
        self.coastal_distance_accumulators = {
            i: {
                "label": self._format_distance_bin_label(lo, hi),
                "count": 0,
                "sum_mae": 0.0,
                "sum_mse": 0.0,
                "sum_bias": 0.0,
                "sum_baseline_mae": 0.0,
                "sum_baseline_mse": 0.0,
                "sum_baseline_bias": 0.0,
            }
            for i, (lo, hi) in enumerate(self.coastal_distance_bins_km)
        }

        # Category breakdown accumulators: corrected vs not_corrected
        self.category_breakdown = {}
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            self.category_breakdown[bin_name] = {
                "corrected": {
                    "count": 0,
                    "feature_sums": {},  # Dict: {feature_idx: sum}
                    "feature_sq_sums": {},  # For std computation
                    "snr_sum": 0.0,
                    "confidence_sum": 0.0,
                    "seasons": {"winter": 0, "spring": 0, "summer": 0, "autumn": 0},
                },
                "not_corrected": {
                    "count": 0,
                    "feature_sums": {},
                    "feature_sq_sums": {},
                    "snr_sum": 0.0,
                    "confidence_sum": 0.0,
                    "seasons": {"winter": 0, "spring": 0, "summer": 0, "autumn": 0},
                },
            }

        # Overall breakdown (across all bins)
        self.overall_breakdown = {
            "corrected": {
                "count": 0,
                "feature_sums": {},
                "feature_sq_sums": {},
                "snr_sum": 0.0,
                "confidence_sum": 0.0,
                "seasons": {"winter": 0, "spring": 0, "summer": 0, "autumn": 0},
            },
            "not_corrected": {
                "count": 0,
                "feature_sums": {},
                "feature_sq_sums": {},
                "snr_sum": 0.0,
                "confidence_sum": 0.0,
                "seasons": {"winter": 0, "spring": 0, "summer": 0, "autumn": 0},
            },
        }

        # Detailed diagnostics for first two bins (calm/light)
        self.low_bin_sample_limit = 200000
        self.low_bin_diagnostics = {}
        for bin_config in self.sea_bins:
            if bin_config["name"] not in {"calm", "light"}:
                continue
            bmin = float(bin_config["min"])
            bmax = float(bin_config["max"])
            step = 0.1
            n_true = max(1, int(round((bmax - bmin) / step)))
            self.low_bin_diagnostics[bin_config["name"]] = {
                "label": bin_config["label"],
                "bin_min": bmin,
                "bin_max": bmax,
                "step": step,
                "count": 0,
                "raw_in_same_bin_count": 0,
                "sum_true": 0.0,
                "sum_pred": 0.0,
                "sum_raw": 0.0,
                "sum_model_err": 0.0,
                "sum_model_abs_err": 0.0,
                "sum_model_sq_err": 0.0,
                "sum_base_err": 0.0,
                "sum_base_abs_err": 0.0,
                "sum_base_sq_err": 0.0,
                "true_subbin": {
                    "count": np.zeros(n_true, dtype=np.int64),
                    "sum_model_err": np.zeros(n_true, dtype=np.float64),
                    "sum_model_abs_err": np.zeros(n_true, dtype=np.float64),
                    "sum_model_sq_err": np.zeros(n_true, dtype=np.float64),
                    "sum_base_err": np.zeros(n_true, dtype=np.float64),
                    "sum_base_abs_err": np.zeros(n_true, dtype=np.float64),
                    "sum_base_sq_err": np.zeros(n_true, dtype=np.float64),
                    "sum_true": np.zeros(n_true, dtype=np.float64),
                    "sum_pred": np.zeros(n_true, dtype=np.float64),
                    "sum_raw": np.zeros(n_true, dtype=np.float64),
                },
                "samples": {
                    "true": [],
                    "pred": [],
                    "raw": [],
                    "model_err": [],
                    "base_err": [],
                },
            }

    def _update_low_bin_diagnostics(
        self,
        bin_name: str,
        bin_y_true: np.ndarray,
        bin_y_pred: np.ndarray,
        bin_y_uncorrected: np.ndarray,
    ) -> None:
        """Update detailed diagnostics for calm/light bins."""
        if bin_name not in self.low_bin_diagnostics:
            return
        if len(bin_y_true) == 0:
            return

        stats = self.low_bin_diagnostics[bin_name]
        n = len(bin_y_true)
        model_err = bin_y_pred - bin_y_true
        base_err = bin_y_uncorrected - bin_y_true

        stats["count"] += n
        stats["raw_in_same_bin_count"] += int(
            np.sum(
                (bin_y_uncorrected >= stats["bin_min"])
                & (bin_y_uncorrected < stats["bin_max"])
            )
        )
        stats["sum_true"] += float(np.sum(bin_y_true))
        stats["sum_pred"] += float(np.sum(bin_y_pred))
        stats["sum_raw"] += float(np.sum(bin_y_uncorrected))
        stats["sum_model_err"] += float(np.sum(model_err))
        stats["sum_model_abs_err"] += float(np.sum(np.abs(model_err)))
        stats["sum_model_sq_err"] += float(np.sum(model_err**2))
        stats["sum_base_err"] += float(np.sum(base_err))
        stats["sum_base_abs_err"] += float(np.sum(np.abs(base_err)))
        stats["sum_base_sq_err"] += float(np.sum(base_err**2))

        # True sub-bin diagnostics (0.1m granularity)
        sub = stats["true_subbin"]
        n_sub = len(sub["count"])
        idx = np.floor((bin_y_true - stats["bin_min"]) / stats["step"]).astype(int)
        idx = np.clip(idx, 0, n_sub - 1)
        np.add.at(sub["count"], idx, 1)
        np.add.at(sub["sum_model_err"], idx, model_err)
        np.add.at(sub["sum_model_abs_err"], idx, np.abs(model_err))
        np.add.at(sub["sum_model_sq_err"], idx, model_err**2)
        np.add.at(sub["sum_base_err"], idx, base_err)
        np.add.at(sub["sum_base_abs_err"], idx, np.abs(base_err))
        np.add.at(sub["sum_base_sq_err"], idx, base_err**2)
        np.add.at(sub["sum_true"], idx, bin_y_true)
        np.add.at(sub["sum_pred"], idx, bin_y_pred)
        np.add.at(sub["sum_raw"], idx, bin_y_uncorrected)

        # Save a bounded sample for quantile diagnostics
        samples = stats["samples"]
        remaining = self.low_bin_sample_limit - len(samples["true"])
        if remaining > 0:
            take = min(remaining, n)
            samples["true"].extend(bin_y_true[:take].tolist())
            samples["pred"].extend(bin_y_pred[:take].tolist())
            samples["raw"].extend(bin_y_uncorrected[:take].tolist())
            samples["model_err"].extend(model_err[:take].tolist())
            samples["base_err"].extend(base_err[:take].tolist())

    def _update_low_bin_spatial_accumulators(
        self,
        y_true_4d: torch.Tensor,
        y_pred_4d: torch.Tensor,
        y_base_4d: torch.Tensor,
        valid_mask_4d: torch.Tensor,
    ) -> None:
        """Accumulate spatial diagnostics for ultra-calm true-wave sub-bins."""
        if y_base_4d is None:
            return

        y_true_np = y_true_4d.detach().cpu().numpy()
        y_pred_np = y_pred_4d.detach().cpu().numpy()
        y_base_np = y_base_4d.detach().cpu().numpy()
        valid_np = valid_mask_4d.detach().cpu().numpy().astype(bool)

        abs_err_model = np.abs(y_pred_np - y_true_np)
        abs_err_base = np.abs(y_base_np - y_true_np)
        delta_abs_err = abs_err_model - abs_err_base
        model_err = y_pred_np - y_true_np
        base_err = y_base_np - y_true_np

        spatial_shape = y_true_np.shape[-2:]
        for lo, hi in self.low_bin_spatial_subbins:
            key = f"{lo:.1f}_{hi:.1f}"
            stats = self.low_bin_spatial_accumulators[key]
            if stats["count"] is None:
                stats["sum_delta_abs_err"] = np.zeros(spatial_shape, dtype=np.float64)
                stats["sum_model_err"] = np.zeros(spatial_shape, dtype=np.float64)
                stats["sum_base_err"] = np.zeros(spatial_shape, dtype=np.float64)
                stats["count"] = np.zeros(spatial_shape, dtype=np.float64)
                stats["count_worse"] = np.zeros(spatial_shape, dtype=np.float64)

            submask = valid_np & (y_true_np >= lo) & (y_true_np < hi)
            if not np.any(submask):
                continue

            stats["sum_delta_abs_err"] += (delta_abs_err * submask).sum(axis=(0, 1))
            stats["sum_model_err"] += (model_err * submask).sum(axis=(0, 1))
            stats["sum_base_err"] += (base_err * submask).sum(axis=(0, 1))
            stats["count"] += submask.sum(axis=(0, 1))
            stats["count_worse"] += (
                ((abs_err_model > abs_err_base) & submask).sum(axis=(0, 1))
            )

    def _save_low_bin_diagnostics(self, sea_bin_metrics: Dict[str, dict]) -> None:
        """Write detailed low-bin diagnostics (JSON + CSV)."""
        if not self.low_bin_diagnostics:
            return
        low_bin_dir = self.output_dir / "low_bin_spatial_maps"
        low_bin_dir.mkdir(parents=True, exist_ok=True)

        report = {}
        rows = []
        quantiles = [0.01, 0.05, 0.5, 0.95, 0.99]

        for bin_name, stats in self.low_bin_diagnostics.items():
            count = int(stats["count"])
            if count == 0:
                continue

            def _q(arr):
                if len(arr) == 0:
                    return None
                vals = np.quantile(np.array(arr, dtype=np.float64), quantiles)
                return {
                    "q01": float(vals[0]),
                    "q05": float(vals[1]),
                    "q50": float(vals[2]),
                    "q95": float(vals[3]),
                    "q99": float(vals[4]),
                }

            overall = {
                "label": stats["label"],
                "count": count,
                "raw_in_same_bin_pct": 100.0
                * float(stats["raw_in_same_bin_count"])
                / max(1, count),
                "model_mae": float(stats["sum_model_abs_err"]) / count,
                "model_rmse": float(np.sqrt(stats["sum_model_sq_err"] / count)),
                "model_bias": float(stats["sum_model_err"]) / count,
                "baseline_mae": float(stats["sum_base_abs_err"]) / count,
                "baseline_rmse": float(np.sqrt(stats["sum_base_sq_err"] / count)),
                "baseline_bias": float(stats["sum_base_err"]) / count,
                "mean_true": float(stats["sum_true"]) / count,
                "mean_pred": float(stats["sum_pred"]) / count,
                "mean_raw": float(stats["sum_raw"]) / count,
                "sea_bin_metrics": sea_bin_metrics.get(bin_name, {}),
                "sample_quantiles": {
                    "true": _q(stats["samples"]["true"]),
                    "pred": _q(stats["samples"]["pred"]),
                    "raw": _q(stats["samples"]["raw"]),
                    "model_err": _q(stats["samples"]["model_err"]),
                    "baseline_err": _q(stats["samples"]["base_err"]),
                },
            }

            # Per-0.1m sub-bin metrics inside each target bin
            sub = stats["true_subbin"]
            sub_rows = []
            for i in range(len(sub["count"])):
                c = int(sub["count"][i])
                lo = stats["bin_min"] + i * stats["step"]
                hi = lo + stats["step"]
                if c == 0:
                    continue
                rmse_model = float(np.sqrt(sub["sum_model_sq_err"][i] / c))
                rmse_base = float(np.sqrt(sub["sum_base_sq_err"][i] / c))
                mae_model = float(sub["sum_model_abs_err"][i] / c)
                mae_base = float(sub["sum_base_abs_err"][i] / c)
                row = {
                    "bin_name": bin_name,
                    "true_subbin_label": f"{lo:.1f}-{hi:.1f}",
                    "count": c,
                    "model_rmse": rmse_model,
                    "baseline_rmse": rmse_base,
                    "rmse_improvement_pct": (
                        (rmse_base - rmse_model) / rmse_base * 100.0
                        if rmse_base > 0
                        else None
                    ),
                    "model_mae": mae_model,
                    "baseline_mae": mae_base,
                    "mae_improvement_pct": (
                        (mae_base - mae_model) / mae_base * 100.0 if mae_base > 0 else None
                    ),
                    "model_bias": float(sub["sum_model_err"][i] / c),
                    "baseline_bias": float(sub["sum_base_err"][i] / c),
                    "mean_true": float(sub["sum_true"][i] / c),
                    "mean_pred": float(sub["sum_pred"][i] / c),
                    "mean_raw": float(sub["sum_raw"][i] / c),
                }
                sub_rows.append(row)
                rows.append(row)

            overall["true_subbin_metrics"] = sub_rows
            report[bin_name] = overall

        if report:
            with open(low_bin_dir / "low_bin_diagnostics.json", "w") as f:
                json.dump(report, f, indent=2)

        if rows:
            csv_path = low_bin_dir / "low_bin_subbin_metrics.csv"
            fieldnames = [
                "bin_name",
                "true_subbin_label",
                "count",
                "model_rmse",
                "baseline_rmse",
                "rmse_improvement_pct",
                "model_mae",
                "baseline_mae",
                "mae_improvement_pct",
                "model_bias",
                "baseline_bias",
                "mean_true",
                "mean_pred",
                "mean_raw",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _update_low_bin_plot_samples(
        self,
        y_true_wave: np.ndarray,
        y_pred_wave: np.ndarray,
        raw_wave: np.ndarray,
        y_true_bias: np.ndarray,
        y_pred_bias: np.ndarray,
        prior_bias: np.ndarray = None,
    ) -> None:
        """Collect bounded sample sets for low-bin advanced diagnostics."""
        if raw_wave is None:
            return
        if y_true_wave is None or y_pred_wave is None:
            return

        for lo, hi in self.low_bin_spatial_subbins:
            key = f"{lo:.1f}_{hi:.1f}"
            buf = self.low_bin_plot_samples.get(key)
            if buf is None:
                continue

            mask = (
                np.isfinite(y_true_wave)
                & np.isfinite(y_pred_wave)
                & np.isfinite(raw_wave)
                & (y_true_wave >= lo)
                & (y_true_wave < hi)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            remaining = self.low_bin_plot_sample_limit_per_subbin - len(buf["true_wave"])
            if remaining <= 0:
                continue

            if len(idx) > remaining:
                idx = np.random.choice(idx, size=remaining, replace=False)

            buf["true_wave"].extend(y_true_wave[idx].tolist())
            buf["pred_wave"].extend(y_pred_wave[idx].tolist())
            buf["raw_wave"].extend(raw_wave[idx].tolist())
            if y_true_bias is not None:
                buf["true_bias"].extend(y_true_bias[idx].tolist())
            if y_pred_bias is not None:
                buf["pred_bias"].extend(y_pred_bias[idx].tolist())
            if prior_bias is not None:
                pb = prior_bias[idx]
                pb = np.where(np.isfinite(pb), pb, np.nan)
                buf["prior_bias"].extend(pb.tolist())

    def _update_coastal_distance_accumulators(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_base: np.ndarray,
        coastal_bin_ids: np.ndarray,
    ) -> None:
        """Update per-distance-to-coast error accumulators."""
        if y_base is None or self.coastal_distance_bin_idx_map is None:
            return
        if len(y_true) == 0:
            return

        valid = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
            & np.isfinite(y_base)
            & np.isfinite(coastal_bin_ids)
            & (coastal_bin_ids >= 0)
        )
        if not np.any(valid):
            return

        yt = y_true[valid]
        yp = y_pred[valid]
        yb = y_base[valid]
        ids = coastal_bin_ids[valid].astype(np.int64)

        err = yp - yt
        berr = yb - yt

        for i in np.unique(ids):
            m = ids == i
            if not np.any(m):
                continue
            stats = self.coastal_distance_accumulators.get(int(i))
            if stats is None:
                continue
            stats["count"] += int(np.sum(m))
            stats["sum_mae"] += float(np.sum(np.abs(err[m])))
            stats["sum_mse"] += float(np.sum(err[m] ** 2))
            stats["sum_bias"] += float(np.sum(err[m]))
            stats["sum_baseline_mae"] += float(np.sum(np.abs(berr[m])))
            stats["sum_baseline_mse"] += float(np.sum(berr[m] ** 2))
            stats["sum_baseline_bias"] += float(np.sum(berr[m]))

    def _save_coastal_distance_diagnostics(self) -> None:
        """Save coastal-distance metrics to CSV and plot."""
        if not self.coastal_distance_accumulators:
            return

        rows = []
        for i in sorted(self.coastal_distance_accumulators.keys()):
            s = self.coastal_distance_accumulators[i]
            c = int(s["count"])
            if c > 0:
                rmse = float(np.sqrt(s["sum_mse"] / c))
                mae = float(s["sum_mae"] / c)
                bias = float(s["sum_bias"] / c)
                brmse = float(np.sqrt(s["sum_baseline_mse"] / c))
                bmae = float(s["sum_baseline_mae"] / c)
                bbias = float(s["sum_baseline_bias"] / c)
                rmse_imp = (brmse - rmse) / brmse * 100.0 if brmse > 0 else None
                mae_imp = (bmae - mae) / bmae * 100.0 if bmae > 0 else None
            else:
                rmse = None
                mae = None
                bias = None
                brmse = None
                bmae = None
                bbias = None
                rmse_imp = None
                mae_imp = None
            rows.append(
                {
                    "distance_bin_km": s["label"],
                    "count": c,
                    "model_rmse": rmse,
                    "baseline_rmse": brmse,
                    "rmse_improvement_pct": rmse_imp,
                    "model_mae": mae,
                    "baseline_mae": bmae,
                    "mae_improvement_pct": mae_imp,
                    "model_bias": bias,
                    "baseline_bias": bbias,
                }
            )

        csv_path = self.output_dir / "coastal_distance_metrics.csv"
        fieldnames = [
            "distance_bin_km",
            "count",
            "model_rmse",
            "baseline_rmse",
            "rmse_improvement_pct",
            "model_mae",
            "baseline_mae",
            "mae_improvement_pct",
            "model_bias",
            "baseline_bias",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        plot_coastal_distance_improvement_fn(rows, self.output_dir)

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
        """Load coordinate grid and create geographic filtering mask.

        Note: If region_filter is used in dataset, coordinates are already cropped
        and geo_bounds filtering is redundant. This method is for backward compatibility
        with manual geo_bounds filtering.
        """
        try:
            logger.info("Loading geographic coordinates for filtering...")

            # Try to get coordinates from dataset (handles region filtering automatically)
            dataset = self.test_loader.dataset
            if hasattr(dataset, "get_coordinates"):
                lat_grid, lon_grid = dataset.get_coordinates()
                logger.info(
                    "Loaded coordinates from dataset (respects region filtering if enabled)"
                )
            else:
                # Fallback to loading from file (legacy)
                coord_file = self.test_files[0]
                lat_grid, lon_grid = load_coordinates_from_parquet(
                    coord_file,
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

            GIBRALTAR_LON = -5.5
            BISCAY_LAT = 43.0
            BISCAY_LON = 0.0
            biscay = (lat_grid > BISCAY_LAT) & (lon_grid < BISCAY_LON)
            AEGEAN_LON_MIN, AEGEAN_LON_MAX = 23.0, 28.0
            AEGEAN_LAT_MIN, AEGEAN_LAT_MAX = 35.0, 42.0

            if self.region_filter == "mediterranean":
                geo_mask = geo_mask & (lon_grid >= GIBRALTAR_LON) & ~biscay
            elif self.region_filter == "atlantic":
                geo_mask = geo_mask & ((lon_grid < GIBRALTAR_LON) | biscay)
            elif self.region_filter == "aegean":
                geo_mask = geo_mask & (
                    (lat_grid >= AEGEAN_LAT_MIN)
                    & (lat_grid <= AEGEAN_LAT_MAX)
                    & (lon_grid >= AEGEAN_LON_MIN)
                    & (lon_grid <= AEGEAN_LON_MAX)
                )

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

    def _build_atlantic_exclusion_mask(self):
        """Build a static 2D mask excluding Bay of Biscay pixels (lat > 43.5° AND lon < 0°).

        These are Atlantic water pixels that leak through the simple lon >= -5.5° Mediterranean cut.
        The mask is True for pixels to KEEP (Mediterranean) and False for excluded pixels.
        """
        try:
            dataset = self.test_loader.dataset
            if hasattr(dataset, "get_coordinates"):
                lat_grid, lon_grid = dataset.get_coordinates()
                exclude = (lat_grid > 43.0) & (lon_grid < 0.0)
                keep_mask = ~exclude
                n_excluded = exclude.sum()
                if n_excluded > 0:
                    self.atlantic_exclusion_mask = torch.from_numpy(keep_mask).to(
                        self.device
                    )
                    logger.info(
                        f"Atlantic exclusion mask: excluding {n_excluded} Bay of Biscay pixels "
                        f"(lat > 43.0° AND lon < 0°)"
                    )
        except Exception as e:
            logger.warning(f"Could not build Atlantic exclusion mask: {e}")

    def _reconstruct_wave_heights(
        self, bias: torch.Tensor, vhm0: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct full wave heights from bias: corrected = vhm0 + bias"""
        return bias + vhm0

    def _apply_static_blend(self, dnn_bias: torch.Tensor) -> torch.Tensor:
        """Blend DNN bias toward static map when the DNN deviates too far.

        trust = exp(-deviation^2 / (2*sigma^2))
        blended = trust * dnn_bias + (1-trust) * static_bias
        """
        if self.static_bias_map is None or self.blend_sigma is None:
            return dnn_bias
        h, w = dnn_bias.shape[2], dnn_bias.shape[3]
        static = self.static_bias_map[:h, :w].unsqueeze(0).unsqueeze(0)
        valid = self.static_bias_valid[:h, :w].unsqueeze(0).unsqueeze(0)
        deviation = (dnn_bias - static).abs()
        trust = torch.exp(-deviation ** 2 / (2 * self.blend_sigma ** 2))
        blended = trust * dnn_bias + (1 - trust) * static
        return torch.where(valid, blended, dnn_bias)

    def _apply_uncertainty_blend(
        self, dnn_bias: torch.Tensor, uncertainty: torch.Tensor | None
    ) -> torch.Tensor:
        """Blend DNN bias toward static map using MoE gate uncertainty as trust signal.

        trust = exp(-uncertainty^2 / (2*sigma^2))
        blended = trust * dnn_bias + (1-trust) * static_bias

        For K=3 experts, uncertainty ∈ [0, 0.67]; calibrate sigma in that range.
        """
        if (
            self.static_bias_map is None
            or self.uncertainty_blend_sigma is None
            or uncertainty is None
        ):
            return dnn_bias
        h, w = dnn_bias.shape[2], dnn_bias.shape[3]
        static = self.static_bias_map[:h, :w].unsqueeze(0).unsqueeze(0)
        valid = self.static_bias_valid[:h, :w].unsqueeze(0).unsqueeze(0)
        u = uncertainty[:, :h, :w].unsqueeze(1)  # [B, 1, H, W]
        trust = torch.exp(-u ** 2 / (2 * self.uncertainty_blend_sigma ** 2))
        blended = trust * dnn_bias + (1 - trust) * static
        return torch.where(valid, blended, dnn_bias)

    def _apply_edcdf_blend(
        self,
        dnn_bias: torch.Tensor,
        raw_uncorrected: torch.Tensor,
        y_true_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Blend DNN bias toward EDCDF-implied bias using Gaussian trust:
            trust = exp(-|b_dnn - b_edcdf|^2 / (2*sigma^2))
            b_blend = trust*b_dnn + (1-trust)*b_edcdf
        """
        if self.edcdf_corrector is None or raw_uncorrected is None:
            return dnn_bias

        use_soft_blend = self.edcdf_blend_sigma is not None
        use_hard_fallback = len(self.edcdf_hard_fallback_bins) > 0
        if not use_soft_blend and not use_hard_fallback:
            return dnn_bias

        var_name = self.target_column.replace("corrected_", "")
        cdf_models = getattr(self.edcdf_corrector, "cdf_models", {})
        if var_name not in cdf_models:
            return dnn_bias

        f_model_cdf, f_obs_quantile, p_min, p_max = cdf_models[var_name]

        # Evaluate EDCDF map in numpy and convert back to tensor
        raw_np = raw_uncorrected.detach().cpu().numpy().astype(float)
        prob = np.clip(f_model_cdf(raw_np), p_min, p_max)
        edcdf_corrected_np = f_obs_quantile(prob)
        edcdf_corrected = torch.from_numpy(edcdf_corrected_np).to(
            device=dnn_bias.device, dtype=dnn_bias.dtype
        )

        edcdf_bias = edcdf_corrected - raw_uncorrected
        valid = torch.isfinite(edcdf_bias)
        out_bias = dnn_bias

        if use_soft_blend:
            deviation = (out_bias - edcdf_bias).abs()
            trust = torch.exp(-deviation ** 2 / (2 * self.edcdf_blend_sigma ** 2))
            blended = trust * out_bias + (1 - trust) * edcdf_bias
            out_bias = torch.where(valid, blended, out_bias)

        if use_hard_fallback:
            if self.edcdf_fallback_bin_source == "raw":
                gate_values = raw_uncorrected
            elif self.edcdf_fallback_bin_source == "edcdf":
                gate_values = edcdf_corrected
            elif self.edcdf_fallback_bin_source == "true":
                if y_true_bias is None:
                    logger.warning(
                        "edcdf_fallback_bin_source='true' requires y_true_bias; falling back to raw."
                    )
                    gate_values = raw_uncorrected
                else:
                    gate_values = y_true_bias + raw_uncorrected
            else:
                gate_values = raw_uncorrected

            fallback_mask = torch.zeros_like(raw_uncorrected, dtype=torch.bool)
            for lo, hi in self.edcdf_hard_fallback_bins:
                fallback_mask = fallback_mask | (
                    (gate_values >= lo) & (gate_values < hi)
                )
            fallback_mask = fallback_mask & valid
            self._edcdf_fallback_total_valid += int(valid.sum().item())
            self._edcdf_fallback_total_applied += int(fallback_mask.sum().item())
            for lo, hi in self.edcdf_hard_fallback_bins:
                key = f"[{lo},{hi})"
                bin_mask = ((gate_values >= lo) & (gate_values < hi) & valid)
                self._edcdf_fallback_applied_per_bin[key] = (
                    self._edcdf_fallback_applied_per_bin.get(key, 0)
                    + int(bin_mask.sum().item())
                )
            out_bias = torch.where(fallback_mask, edcdf_bias, out_bias)

        return out_bias

    def _apply_low_bin_affine_calibration(
        self,
        dnn_bias: torch.Tensor,
        raw_uncorrected: torch.Tensor,
        y_true_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Apply piecewise affine calibration in selected low-wave bins:
            b' = a * b + c
        """
        if not self.low_bin_affine_params or raw_uncorrected is None:
            return dnn_bias

        if self.low_bin_affine_source == "true":
            if y_true_bias is None:
                logger.warning(
                    "low_bin_affine_source='true' requires y_true_bias; using 'raw' source."
                )
                gate_values = raw_uncorrected
            else:
                gate_values = y_true_bias + raw_uncorrected
        else:
            gate_values = raw_uncorrected

        out = dnn_bias
        for p in self.low_bin_affine_params:
            mask = (gate_values >= p["min"]) & (gate_values < p["max"])
            adjusted = p["a"] * out + p["c"]
            out = torch.where(mask, adjusted, out)
        return out

    def _apply_prior_hard_fallback(
        self,
        dnn_bias: torch.Tensor,
        prior_bias: torch.Tensor,
        raw_uncorrected: torch.Tensor,
        y_true_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Replace DNN bias with configured fallback target in selected bins.

        prior_fallback_target:
          - "raw":    zero correction (bias=0, corrected = raw vhm0)
          - "static": static bias map (requires static_bias_map_path to be set)
          - "prior":  residual prior bias (only in predict_residual_to_prior mode)
        """
        if not self.prior_hard_fallback_bins or raw_uncorrected is None:
            return dnn_bias

        if self.prior_fallback_bin_source == "true":
            if y_true_bias is None:
                logger.warning(
                    "prior_fallback_bin_source='true' requires y_true_bias; using 'raw' source."
                )
                gate_values = raw_uncorrected
            else:
                gate_values = y_true_bias + raw_uncorrected
        else:
            gate_values = raw_uncorrected

        if self.prior_fallback_target == "raw":
            replacement_bias = torch.zeros_like(dnn_bias)
            valid = torch.isfinite(raw_uncorrected)
        elif self.prior_fallback_target == "static":
            if self.static_bias_map is None:
                logger.warning(
                    "prior_fallback_target='static' requires static_bias_map_path; skipping fallback."
                )
                return dnn_bias
            h, w = dnn_bias.shape[2], dnn_bias.shape[3]
            replacement_bias = self.static_bias_map[:h, :w].unsqueeze(0).unsqueeze(0).expand_as(dnn_bias)
            valid = self.static_bias_valid[:h, :w].unsqueeze(0).unsqueeze(0).expand_as(dnn_bias)
        else:
            if prior_bias is None:
                return dnn_bias
            replacement_bias = prior_bias
            valid = torch.isfinite(prior_bias)

        fallback_mask = torch.zeros_like(raw_uncorrected, dtype=torch.bool)
        for lo, hi in self.prior_hard_fallback_bins:
            fallback_mask = fallback_mask | ((gate_values >= lo) & (gate_values < hi))
        fallback_mask = fallback_mask & valid

        self._prior_fallback_total_valid += int(valid.sum().item())
        self._prior_fallback_total_applied += int(fallback_mask.sum().item())
        for lo, hi in self.prior_hard_fallback_bins:
            key = f"[{lo},{hi})"
            bin_mask = ((gate_values >= lo) & (gate_values < hi) & valid)
            self._prior_fallback_applied_per_bin[key] = (
                self._prior_fallback_applied_per_bin.get(key, 0)
                + int(bin_mask.sum().item())
            )

        return torch.where(fallback_mask, replacement_bias, dnn_bias)

    def _recalibrate_domain_mean(
        self, dnn_bias: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Re-anchor the DNN's domain-average level to the static map's level.

        Preserves the DNN's spatial patterns but shifts the overall mean
        to match the static map, correcting systematic year-level drift.
        """
        if not self.domain_mean_recalibration or self.static_domain_mean is None:
            return dnn_bias
        valid = mask.bool() if mask is not None else torch.ones_like(dnn_bias, dtype=torch.bool)
        dnn_mean = dnn_bias[valid.expand_as(dnn_bias)].mean()
        offset = dnn_mean - self.static_domain_mean
        return dnn_bias - offset

    # ------------------------------------------------------------------
    # Sampled grid-point time-series
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ts_map(timestamps_csv: Optional[str]) -> dict:
        """Load the pt_stem × hour_idx → datetime map produced by build_pt_timestamp_map.py.

        Returns a dict keyed by (pt_stem, hour_idx) → datetime, or an empty dict
        if no CSV is provided.
        """
        if timestamps_csv is None:
            return {}

        import csv as _csv
        from datetime import datetime, timezone

        ts_map: dict = {}
        with open(timestamps_csv, newline="") as fh:
            for row in _csv.DictReader(fh):
                key = (row["pt_stem"], int(row["hour_idx"]))
                ts_map[key] = datetime.fromisoformat(row["timestamp"]).replace(
                    tzinfo=timezone.utc
                )
        logger.info(f"Loaded {len(ts_map):,} timestamp entries from '{timestamps_csv}'")
        return ts_map

    def _setup_grid_point_sampling(self) -> None:
        """Read the sampled-points CSV and map each lat/lon to a grid (row, col).

        Also pre-builds a timestamp array (one entry per dataset item) from the
        .pt filename pattern WAVEAN{year}{month}{day} + hour_idx so that every
        recorded row has a proper datetime even when no Parquet timestamps exist.
        """
        dataset = self.test_loader.dataset
        # ------------------------------------------------------------------
        # Timestamp map setup is useful beyond sampled points (e.g. all-pixels
        # temporal heatmap), so initialize it regardless of sampled_points_csv.
        # ------------------------------------------------------------------
        self._gp_item_ts: Optional[List] = None
        self._gp_sample_counter: int = 0
        from torch.utils.data import RandomSampler
        if isinstance(self.test_loader.sampler, RandomSampler):
            raise ValueError(
                "test_loader uses shuffle=True, which breaks timestamp mapping. "
                "Set shuffle=False for evaluation."
            )
        if hasattr(dataset, "index_map") and hasattr(dataset, "file_paths"):
            if self._gp_ts_map:
                item_ts: List = []
                for file_idx, hour_idx in dataset.index_map:
                    pt_stem = Path(dataset.file_paths[file_idx]).stem
                    item_ts.append(self._gp_ts_map.get((pt_stem, hour_idx)))
                self._gp_item_ts = item_ts
                n_resolved = sum(t is not None for t in item_ts)
                logger.info(
                    f"Grid-point timestamps resolved: {n_resolved}/{len(item_ts)} "
                    f"from timestamp map"
                )
            else:
                logger.warning(
                    "No timestamp map loaded; timestamps will be None in grid_point_timeseries.csv "
                    "and all-pixels temporal heatmap. "
                    "Pass --timestamps-csv to enable."
                )

        if self.sampled_points_csv is None:
            return

        import csv as _csv
        if not hasattr(dataset, "get_coordinates"):
            logger.warning("Dataset has no get_coordinates(); grid-point sampling disabled.")
            return

        lat_grid, lon_grid = dataset.get_coordinates()
        lat_grid = np.asarray(lat_grid, dtype=np.float64)
        lon_grid = np.asarray(lon_grid, dtype=np.float64)

        sampled = []
        with open(self.sampled_points_csv, newline="") as fh:
            for row in _csv.DictReader(fh):
                sampled.append({"region": row["region"],
                                 "lat": float(row["latitude"]),
                                 "lon": float(row["longitude"])})

        # Tolerance for exact coordinate matching (1e-5 deg ≈ 1 m — survives CSV
        # round-trip at 4–6 decimal places while staying well within one grid cell).
        COORD_TOL = 1e-5

        indices = []
        unmatched = []
        for pt in sampled:
            match = np.where(
                (np.abs(lat_grid - pt["lat"]) < COORD_TOL)
                & (np.abs(lon_grid - pt["lon"]) < COORD_TOL)
            )
            if len(match[0]) == 0:
                unmatched.append(pt)
                logger.warning(
                    f"Sampled point ({pt['lat']:.6f}, {pt['lon']:.6f}) has no exact match "
                    f"in the coordinate grid (tol={COORD_TOL}); skipping."
                )
                continue
            # If multiple cells match (shouldn't happen on a regular grid), take the first.
            r, c = int(match[0][0]), int(match[1][0])
            indices.append({
                "region":        pt["region"],
                "requested_lat": pt["lat"],
                "requested_lon": pt["lon"],
                "grid_lat":      float(lat_grid[r, c]),
                "grid_lon":      float(lon_grid[r, c]),
                "row":           r,
                "col":           c,
            })
            logger.debug(
                f"  ({pt['lat']:.6f}, {pt['lon']:.6f}) → grid cell [row={r}, col={c}]"
            )

        if unmatched:
            logger.error(
                f"{len(unmatched)} sampled point(s) could not be matched to the grid. "
                "Make sure the CSV was produced from the same dataset used for evaluation."
            )

        self._grid_point_indices = indices
        self._grid_point_records = []
        logger.info(
            f"Grid-point sampling ready: {len(indices)}/{len(sampled)} points matched "
            f"from '{self.sampled_points_csv}'"
        )

    def _update_grid_point_records(
        self,
        y_true_4d: torch.Tensor,
        y_pred_4d: torch.Tensor,
        y_base_4d: Optional[torch.Tensor],
        batch_idx: int,
        valid_mask_4d: Optional[torch.Tensor] = None,
    ) -> None:
        """Extract reference / uncorrected / corrected values at sampled grid points.

        Parameters
        ----------
        y_true_4d  : (B, 1, H, W) ground-truth wave heights
        y_pred_4d  : (B, 1, H, W) model-corrected wave heights
        y_base_4d  : (B, 1, H, W) uncorrected (raw) wave heights, or None
        batch_idx  : int, index of this batch in the DataLoader loop
        """
        if self._grid_point_indices is None:
            return

        y_true_np = y_true_4d.detach().cpu().numpy()   # (B, 1, H, W)
        y_pred_np = y_pred_4d.detach().cpu().numpy()
        y_base_np = y_base_4d.detach().cpu().numpy() if y_base_4d is not None else None
        valid_mask_np = (
            valid_mask_4d.detach().cpu().numpy().astype(bool)
            if valid_mask_4d is not None
            else None
        )

        B = y_true_np.shape[0]

        # Resolve one timestamp per batch item via the sequential counter.
        def _ts(b):
            if self._gp_item_ts is not None:
                idx = self._gp_sample_counter + b
                t = self._gp_item_ts[idx] if idx < len(self._gp_item_ts) else None
                return t.isoformat() if t is not None else None
            return None

        for b in range(B):
            ts = _ts(b)
            for pt in self._grid_point_indices:
                r, c = pt["row"], pt["col"]
                H, W = y_true_np.shape[2], y_true_np.shape[3]
                if r >= H or c >= W:
                    continue  # point outside padded region
                if valid_mask_np is not None and not valid_mask_np[b, :, r, c].any():
                    continue
                self._grid_point_records.append({
                    "timestamp":     ts,
                    "batch_idx":     batch_idx,
                    "sample_in_batch": b,
                    "region":        pt["region"],
                    "requested_lat": pt["requested_lat"],
                    "requested_lon": pt["requested_lon"],
                    "grid_lat":      pt["grid_lat"],
                    "grid_lon":      pt["grid_lon"],
                    "reference":     float(y_true_np[b, 0, r, c]),
                    "uncorrected":   float(y_base_np[b, 0, r, c]) if y_base_np is not None else None,
                    "corrected":     float(y_pred_np[b, 0, r, c]),
                })

    def _save_grid_point_csv(self) -> None:
        """Write the accumulated grid-point time-series records to CSV."""
        if not self._grid_point_records:
            return

        import csv as _csv

        out_path = self.output_dir / "grid_point_timeseries.csv"
        fieldnames = [
            "timestamp", "batch_idx", "sample_in_batch",
            "region", "requested_lat", "requested_lon", "grid_lat", "grid_lon",
            "reference", "uncorrected", "corrected",
        ]
        with open(out_path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._grid_point_records)

        logger.info(
            f"Saved grid-point time-series ({len(self._grid_point_records)} rows) → {out_path}"
        )
        print(f"  Grid-point CSV saved → {out_path}")

    def _update_all_points_temporal_accumulators(
        self,
        error_map: np.ndarray,
        error_map_baseline: Optional[np.ndarray],
        count_map: np.ndarray,
        batch_size: int,
    ) -> None:
        """Accumulate hour×month RMSE stats over all valid pixels."""
        if error_map_baseline is None or self._gp_item_ts is None:
            return

        for b in range(batch_size):
            idx = self._gp_sample_counter + b
            if idx >= len(self._gp_item_ts):
                continue
            ts = self._gp_item_ts[idx]
            if ts is None:
                continue
            h = int(ts.hour)
            m = int(ts.month) - 1

            cnt = float(count_map[b].sum())
            if cnt <= 0:
                continue

            self._temporal_all_count[h, m] += cnt
            self._temporal_all_model_sq[h, m] += float(error_map[b].sum())
            self._temporal_all_base_sq[h, m] += float(error_map_baseline[b].sum())

    def _save_all_points_temporal_heatmap(self) -> None:
        """Save hour×month RMSE improvement heatmap over all valid pixels."""
        if float(self._temporal_all_count.sum()) <= 0:
            logger.info("Skipping all-points temporal heatmap: no timestamp-aligned counts.")
            return

        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        with np.errstate(divide="ignore", invalid="ignore"):
            rmse_unc = np.where(
                self._temporal_all_count > 0,
                np.sqrt(self._temporal_all_base_sq / self._temporal_all_count),
                np.nan,
            )
            rmse_cor = np.where(
                self._temporal_all_count > 0,
                np.sqrt(self._temporal_all_model_sq / self._temporal_all_count),
                np.nan,
            )
            rmse_impr = 100.0 * (rmse_unc - rmse_cor) / rmse_unc

        finite = rmse_impr[np.isfinite(rmse_impr)]
        if finite.size == 0:
            # Fallback: compute improvement directly in MSE space when RMSE
            # normalization yields all-NaN (e.g., zero-denominator edge cases).
            with np.errstate(divide="ignore", invalid="ignore"):
                rmse_impr = np.where(
                    (self._temporal_all_count > 0) & (self._temporal_all_base_sq > 0),
                    100.0
                    * (self._temporal_all_base_sq - self._temporal_all_model_sq)
                    / self._temporal_all_base_sq,
                    np.nan,
                )
            finite = rmse_impr[np.isfinite(rmse_impr)]
            logger.warning(
                "All-points RMSE-improvement matrix had no finite values; used MSE-improvement fallback."
            )

        if finite.size == 0:
            logger.warning(
                "All-points temporal heatmap still has no finite values after fallback; skipping save."
            )
            return

        fig, ax = plt.subplots(figsize=(10, 5.4))
        vmax = float(np.nanpercentile(np.abs(finite), 98))
        if not np.isfinite(vmax) or vmax < 1e-6:
            vmax = max(float(np.nanmax(np.abs(finite))), 1e-3)
        sns.heatmap(
            rmse_impr,
            ax=ax,
            vmin=-vmax,
            vmax=vmax,
            cmap="RdBu",
            center=0,
            linewidths=0.2,
            linecolor="#f0f0f0",
            cbar_kws={"label": "RMSE improvement (%)", "shrink": 1, "pad": 0.02},
        )
        ax.set_xlabel("Month (1-12)")
        ax.set_ylabel("Hour of day")
        ax.set_title("Temporal RMSE improvement (all valid pixels)")
        fig.tight_layout()

        png_path = self.output_dir / "heatmap_rmse_improvement_all_points.png"
        pdf_path = self.output_dir / "heatmap_rmse_improvement_all_points.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Save numeric diagnostics for transparency and reproducibility.
        pd.DataFrame(rmse_impr).to_csv(
            self.output_dir / "heatmap_rmse_improvement_all_points_values.csv",
            index=True,
        )
        pd.DataFrame(self._temporal_all_count).to_csv(
            self.output_dir / "heatmap_rmse_improvement_all_points_counts.csv",
            index=True,
        )
        logger.info(f"Saved all-points temporal heatmap → {png_path}")

    # ------------------------------------------------------------------

    def _process_batch(
        self, X, y, mask, vhm0, y_pred, timestamps=None, confidence=None, prior_bias=None,
        batch_idx: int = 0, timestamps_raw=None, moe_uncertainty=None,
    ):
        """Process a single batch and update accumulators.

        Args:
            X: Input features (B, C, H, W)
            y: Ground truth targets (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W)
            vhm0: Uncorrected wave heights (B, 1, H, W)
            y_pred: Model predictions (B, 1, H, W)
            timestamps: Batch timestamps for season extraction (optional)
            confidence: Model confidence values (B, H, W) (optional)
            prior_bias: Static prior bias used in residual-to-prior mode (optional)
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

        if (
            self.predict_residual_to_prior
            and prior_bias is not None
            and self.task_name == self.residual_prior_task
        ):
            # Dataset yields residuals (bias - prior); reconstruct bias first.
            y = y + prior_bias
            y_pred = y_pred + prior_bias

        if self.eval_in_bias_mode:
            y_pred = self._recalibrate_domain_mean(y_pred, mask)
            y_pred = self._apply_static_blend(y_pred)
            y_pred = self._apply_uncertainty_blend(y_pred, moe_uncertainty)
            y_pred = self._apply_low_bin_affine_calibration(
                y_pred, vhm0, y_true_bias=y
            )
            y_pred = self._apply_prior_hard_fallback(
                y_pred, prior_bias, vhm0, y_true_bias=y
            )
            # Keep EDCDF blend/fallback last so configured hard-fallback bins
            # are guaranteed to end up with EDCDF bias.
            y_pred = self._apply_edcdf_blend(y_pred, vhm0, y_true_bias=y)

        # ========== COMPUTE SPATIAL ERROR MAPS FIRST (using full 4D tensors) ==========
        if self.eval_in_bias_mode:
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
            y_true_wave_4d = y_true_full
            y_pred_wave_4d = y_pred_full
            y_base_wave_4d = y_baseline_full
        else:
            # Not predicting bias
            error_map = ((y_pred - y) ** 2).cpu().numpy()  # (N, C, H, W)
            error_map_mae = (y_pred - y).abs().cpu().numpy()  # (N, C, H, W)
            if vhm0 is not None:
                error_map_baseline = ((vhm0 - y) ** 2).cpu().numpy()  # (N, C, H, W)
                error_map_baseline_mae = (vhm0 - y).abs().cpu().numpy()  # (N, C, H, W)
                y_base_wave_4d = vhm0
            else:
                error_map_baseline = None
                error_map_baseline_mae = None
                y_base_wave_4d = None
            y_true_wave_4d = y
            y_pred_wave_4d = y_pred

        denoise_mask = None
        if (
            self.denoise_abs_threshold is not None
            and self.denoise_abs_threshold > 0
        ):
            if y_base_wave_4d is not None:
                unc_abs_err = (y_base_wave_4d - y_true_wave_4d).abs()
                denoise_mask = torch.isfinite(unc_abs_err) & (
                    unc_abs_err > self.denoise_abs_threshold
                )
            elif not self._denoise_warned_no_baseline:
                logger.warning(
                    "Denoising requested but baseline (vhm0) is unavailable; "
                    "denoise filter is not applied."
                )
                self._denoise_warned_no_baseline = True

        combined_mask = mask.bool()
        if denoise_mask is not None:
            self.denoise_total_candidate += int(combined_mask.sum().item())
            combined_mask = combined_mask & denoise_mask
            self.denoise_total_kept += int(combined_mask.sum().item())

        if self.atlantic_exclusion_mask is not None:
            # Broadcast 2D (H, W) mask to match 4D (N, C, H, W)
            h, w = combined_mask.shape[2], combined_mask.shape[3]
            exc = self.atlantic_exclusion_mask[:h, :w].bool()
            combined_mask = combined_mask & exc.unsqueeze(0).unsqueeze(0)

        self._update_low_bin_spatial_accumulators(
            y_true_4d=y_true_wave_4d,
            y_pred_4d=y_pred_wave_4d,
            y_base_4d=y_base_wave_4d,
            valid_mask_4d=combined_mask,
        )

        # Extract values at sampled grid points (no-op when sampled_points_csv is None)
        self._update_grid_point_records(
            y_true_4d=y_true_wave_4d,
            y_pred_4d=y_pred_wave_4d,
            y_base_4d=y_base_wave_4d,
            batch_idx=batch_idx,
            valid_mask_4d=combined_mask,
        )
        count_map = combined_mask.float().cpu().numpy().astype(np.float32)  # (N, C, H, W)

        # IMPORTANT: Apply mask to errors (zero out invalid pixels)
        # Use nan_to_num first: NaN * 0 stays NaN in numpy, which can poison
        # downstream sums/heatmaps.
        error_map = np.nan_to_num(error_map, nan=0.0, posinf=0.0, neginf=0.0)
        error_map_mae = np.nan_to_num(error_map_mae, nan=0.0, posinf=0.0, neginf=0.0)
        error_map = error_map * count_map
        if error_map_baseline is not None:
            error_map_baseline = np.nan_to_num(
                error_map_baseline, nan=0.0, posinf=0.0, neginf=0.0
            )
            error_map_baseline = error_map_baseline * count_map
        if error_map_baseline_mae is not None:
            error_map_baseline_mae = np.nan_to_num(
                error_map_baseline_mae, nan=0.0, posinf=0.0, neginf=0.0
            )
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
        self._update_all_points_temporal_accumulators(
            error_map=error_map,
            error_map_baseline=error_map_baseline,
            count_map=count_map,
            batch_size=y_true_wave_4d.shape[0],
        )
        self._gp_sample_counter += y_true_wave_4d.shape[0]

        # Bin-conditional spatial accumulators (binned by true wave height)
        if self.eval_in_bias_mode and vhm0 is not None:
            true_wave_np = (y + vhm0).cpu().numpy()  # (N, C, H, W)
            for (lo, hi), acc in self.bin_spatial_accumulators.items():
                bin_mask = ((true_wave_np >= lo) & (true_wave_np < hi)).astype(np.float32)
                bin_mask = bin_mask * count_map
                if bin_mask.sum() > 0:
                    acc["error_sq"].append(
                        (error_map * bin_mask).sum(axis=(0, 1))  # (H, W)
                    )
                    acc["count"].append(bin_mask.sum(axis=(0, 1)))  # (H, W)

        # Per-pixel relative improvement accumulation
        if error_map_baseline is not None:
            eps = 1e-6
            model_err_flat = np.sqrt(error_map[count_map > 0])
            base_err_flat  = np.sqrt(error_map_baseline[count_map > 0])
            rel_imp = (base_err_flat - model_err_flat) / (base_err_flat + eps) * 100.0
            self._rel_improvement_samples.append(rel_imp.astype(np.float32))

        # Apply mask (including Atlantic exclusion)
        mask_combined = combined_mask
        mask_flat = mask_combined.flatten()
        y_true_flat = y.flatten()[mask_flat]
        y_pred_flat = y_pred.flatten()[mask_flat]

        # Reconstruct wave heights if predicting bias
        if self.eval_in_bias_mode and vhm0 is not None:
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
        if prior_bias is not None:
            prior_bias_flat = prior_bias.flatten()[mask_flat]
        else:
            prior_bias_flat = None

        # Filter out extreme wave heights (true VHM0 >= 11m)
        # valid_range_mask = y_true_wave_heights < 11.0
        # valid_range_mask_np = valid_range_mask.cpu().numpy()
        # y_true_wave_heights = y_true_wave_heights[valid_range_mask]
        # y_pred_wave_heights = y_pred_wave_heights[valid_range_mask]
        # if y_uncorrected is not None:
        #     y_uncorrected = y_uncorrected[valid_range_mask]

        # Convert to numpy for binning
        y_true_np = y_true_wave_heights.cpu().numpy()
        y_pred_np = y_pred_wave_heights.cpu().numpy()
        y_true_bias_np = y_true_flat.cpu().numpy() if self.eval_in_bias_mode else None
        y_pred_bias_np = y_pred_flat.cpu().numpy() if self.eval_in_bias_mode else None
        y_uncorrected_np = y_uncorrected.cpu().numpy() if y_uncorrected is not None else None
        prior_bias_np = (
            prior_bias_flat.cpu().numpy() if prior_bias_flat is not None else None
        )
        coastal_bin_ids = None
        if self.coastal_distance_bin_idx_map is not None:
            h, w = mask_combined.shape[2], mask_combined.shape[3]
            idx2d = self.coastal_distance_bin_idx_map[:h, :w]
            idx_t = torch.from_numpy(idx2d).to(mask_combined.device)
            idx4d = idx_t.unsqueeze(0).unsqueeze(0).expand(
                mask_combined.shape[0], mask_combined.shape[1], h, w
            )
            coastal_bin_ids = idx4d.flatten()[mask_flat].cpu().numpy()

        if self.eval_in_bias_mode and y_uncorrected_np is not None:
            self._update_low_bin_plot_samples(
                y_true_wave=y_true_np,
                y_pred_wave=y_pred_np,
                raw_wave=y_uncorrected_np,
                y_true_bias=y_true_bias_np,
                y_pred_bias=y_pred_bias_np,
                prior_bias=prior_bias_np,
            )
        if y_uncorrected_np is not None and coastal_bin_ids is not None:
            self._update_coastal_distance_accumulators(
                y_true=y_true_np,
                y_pred=y_pred_np,
                y_base=y_uncorrected_np,
                coastal_bin_ids=coastal_bin_ids,
            )

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

                        # Extra diagnostics for first two bins (0-1m, 1-2m)
                        self._update_low_bin_diagnostics(
                            bin_name=bin_name,
                            bin_y_true=bin_y_true,
                            bin_y_pred=bin_y_pred,
                            bin_y_uncorrected=bin_y_uncorrected,
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
                                X_flat = X_np.reshape(
                                    X_np.shape[0], X_np.shape[1], -1
                                )  # (B, C, H*W)
                                X_flat = X_flat.transpose(0, 2, 1)  # (B, H*W, C)
                                X_flat = X_flat.reshape(-1, X_np.shape[1])  # (B*H*W, C)
                                # Apply mask to get valid pixels only
                                X_masked = X_flat[
                                    mask_flat.cpu().numpy()
                                ]  # (N_valid, C)
                                # Get features for this specific bin
                                bin_X = X_masked[bin_mask]  # (N_bin, C)

                                # Prepare confidence for this bin
                                bin_confidence = None
                                if confidence is not None:
                                    confidence_flat = confidence.flatten()[mask_flat]
                                    bin_confidence = confidence_flat.cpu().numpy()[
                                        bin_mask
                                    ]

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
                                        category="corrected",
                                        features=bin_X[corrected_mask],
                                        y_true=bin_y_true[corrected_mask],
                                        y_pred=bin_y_pred[corrected_mask],
                                        timestamps=bin_timestamps[corrected_mask]
                                        if bin_timestamps is not None
                                        else None,
                                        confidence=bin_confidence[corrected_mask]
                                        if bin_confidence is not None
                                        else None,
                                    )

                                # Update not_corrected category
                                if not_corrected_mask.sum() > 0:
                                    self._update_category_stats(
                                        bin_name=bin_name,
                                        category="not_corrected",
                                        features=bin_X[not_corrected_mask],
                                        y_true=bin_y_true[not_corrected_mask],
                                        y_pred=bin_y_pred[not_corrected_mask],
                                        timestamps=bin_timestamps[not_corrected_mask]
                                        if bin_timestamps is not None
                                        else None,
                                        confidence=bin_confidence[not_corrected_mask]
                                        if bin_confidence is not None
                                        else None,
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to update category stats for bin {bin_name}: {e}"
                                )

            # Store samples for plotting (limited)
            self.plot_samples["y_true"].extend(y_true_np)
            self.plot_samples["y_pred"].extend(y_pred_np)
            if y_uncorrected is not None:
                self.plot_samples["y_uncorrected"].extend(y_uncorrected_np)
                self.plot_samples["vhm0"].extend(y_uncorrected_np)
            else:
                self.plot_samples["vhm0"].extend(y_true_np)

            # Accumulate coordinates aligned to the same mask
            if self._coord_lat_grid is not None and self._coord_lon_grid is not None:
                h = mask_combined.shape[2]
                w = mask_combined.shape[3]
                B = mask_combined.shape[0]
                lat_crop = self._coord_lat_grid[:h, :w]  # (H, W)
                lon_crop = self._coord_lon_grid[:h, :w]
                # tile B times to match (B, 1, H, W) flatten layout
                lat_tiled = np.tile(lat_crop[np.newaxis, np.newaxis], (B, 1, 1, 1)).flatten()
                lon_tiled = np.tile(lon_crop[np.newaxis, np.newaxis], (B, 1, 1, 1)).flatten()
                mask_np = mask_combined.flatten().cpu().numpy().astype(bool)
                self.plot_samples["lat"].extend(lat_tiled[mask_np])
                self.plot_samples["lon"].extend(lon_tiled[mask_np])

    def _get_denoising_summary(self) -> Dict:
        enabled = (
            self.denoise_abs_threshold is not None
            and self.denoise_abs_threshold > 0
        )
        summary = {
            "enabled": bool(enabled),
            "abs_threshold_m": float(self.denoise_abs_threshold) if enabled else None,
            "candidate_pixels": int(self.denoise_total_candidate) if enabled else 0,
            "kept_pixels": int(self.denoise_total_kept) if enabled else 0,
            "kept_pct": None,
        }
        if enabled and self.denoise_total_candidate > 0:
            summary["kept_pct"] = (
                100.0 * self.denoise_total_kept / self.denoise_total_candidate
            )
        return summary

    def _update_category_stats(
        self, bin_name, category, features, y_true, y_pred, timestamps, confidence
    ):
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
            if i not in stats["feature_sums"]:
                stats["feature_sums"][i] = 0.0
                stats["feature_sq_sums"][i] = 0.0
            stats["feature_sums"][i] += np.sum(feature_vals)
            stats["feature_sq_sums"][i] += np.sum(feature_vals**2)

            # Overall
            if i not in overall_stats["feature_sums"]:
                overall_stats["feature_sums"][i] = 0.0
                overall_stats["feature_sq_sums"][i] = 0.0
            overall_stats["feature_sums"][i] += np.sum(feature_vals)
            overall_stats["feature_sq_sums"][i] += np.sum(feature_vals**2)

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
                stats["snr_sum"] += snr_db * n
                overall_stats["snr_sum"] += snr_db * n
        except Exception as e:
            logger.debug(f"Failed to compute SNR for {bin_name}/{category}: {e}")

        # Update seasons
        if timestamps is not None and len(timestamps) > 0:
            try:
                seasons = SeasonHelper.get_seasons_from_timestamps(timestamps)
                for season in seasons:
                    if season in stats["seasons"]:
                        stats["seasons"][season] += 1
                        overall_stats["seasons"][season] += 1
            except Exception as e:
                logger.debug(
                    f"Failed to extract seasons for {bin_name}/{category}: {e}"
                )

        # Update confidence
        if confidence is not None and len(confidence) > 0:
            stats["confidence_sum"] += np.sum(confidence)
            overall_stats["confidence_sum"] += np.sum(confidence)

        # Update counts
        stats["count"] += n
        overall_stats["count"] += n

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
                    return_timestamps=True,
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
        self._gp_sample_counter = 0
        self._setup_grid_point_sampling()

        def pad_to_multiple(x, multiple=16, mode="reflect"):
            import torch.nn.functional as F

            _, _, H, W = x.shape
            pad_h = (multiple - H % multiple) % multiple
            pad_w = (multiple - W % multiple) % multiple
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
            return x, (H, W)

        def _extract_task_tensor(value, value_name):
            """Normalize nested multi-task payloads to a tensor for current eval task."""
            unwrap_order = (self.task_name, "target", "targets", "prediction", "y")
            max_unwrap_depth = 8

            for _ in range(max_unwrap_depth):
                if isinstance(value, dict):
                    selected_key = next((k for k in unwrap_order if k in value), None)
                    if selected_key is None:
                        raise KeyError(
                            f"{value_name} is dict but has no recognized tensor key. "
                            f"Expected one of {list(unwrap_order)}; available keys: {list(value.keys())}"
                        )
                    value = value[selected_key]
                    continue

                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                    continue

                if hasattr(value, "shape"):
                    return value

                raise TypeError(
                    f"{value_name} could not be converted to tensor. "
                    f"Got type: {type(value)}"
                )

            raise RuntimeError(
                f"{value_name} exceeded max nested unwrap depth ({max_unwrap_depth}). "
                "Check batch/model output structure."
            )

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
                prior_bias_batch = None
                if len(batch) == 5:
                    X, y, mask, vhm0_batch, prior_bias_batch = batch
                else:
                    X, y, mask, vhm0_batch = batch
                vhm0 = vhm0_batch.to(self.device) if vhm0_batch is not None else None

                # Handle multi-task vs single-task format
                # If y is a dict (multi-task), extract the target for the task we're evaluating
                if isinstance(y, dict):
                    # Multi-task: extract the specific target we're evaluating
                    y = _extract_task_tensor(y, "target batch")

                X, orig_size = pad_to_multiple(X, multiple=16)

                if y is not None:
                    y, _ = pad_to_multiple(y, multiple=16)
                mask_float = mask.float()
                mask, _ = pad_to_multiple(mask_float, multiple=16)
                mask = mask.bool()

                if vhm0 is not None:
                    vhm0, _ = pad_to_multiple(vhm0, multiple=16)
                if prior_bias_batch is not None:
                    prior_bias_batch = prior_bias_batch.to(self.device)
                    prior_bias_batch, _ = pad_to_multiple(prior_bias_batch, multiple=16)

                # Load timestamps from test files for seasonal analysis
                timestamps = None
                timestamps_raw = None
                if self.test_files and len(self.test_files) > 0:
                    try:
                        # Get file index for this batch (cycle through files)
                        file_idx = batch_idx % len(self.test_files)
                        file_path = self.test_files[file_idx]

                        # Ensure s3:// prefix for S3 files
                        if not file_path.startswith(
                            "s3://"
                        ) and not file_path.startswith("/"):
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
                                    timestamps = np.full(
                                        n_valid,
                                        timestamps_raw[0],
                                        dtype="datetime64[ns]",
                                    )
                                else:
                                    # Multiple timestamps - assume they correspond to flattened grid
                                    # Apply mask to get only valid pixels
                                    mask_np = mask.cpu().numpy().flatten()
                                    if len(timestamps_raw) == len(mask_np):
                                        timestamps = timestamps_raw[mask_np]
                                    else:
                                        # Fallback: use first timestamp
                                        timestamps = np.full(
                                            n_valid,
                                            timestamps_raw[0],
                                            dtype="datetime64[ns]",
                                        )
                            elif timestamps_raw.ndim == 2:
                                # 2D timestamps (H, W) - flatten and mask
                                timestamps_flat = timestamps_raw.flatten()
                                mask_np = mask.cpu().numpy().flatten()
                                timestamps = timestamps_flat[mask_np]
                    except Exception as e:
                        logger.debug(
                            f"Could not load timestamps for batch {batch_idx}: {e}"
                        )
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
                    confidence = 1.0 / (
                        sigma_mean + 1e-8
                    )  # Higher value = more confident
                else:
                    # Use bin-specific model routing if available
                    if (
                        self.low_wave_model is not None
                        or self.high_wave_model is not None
                    ) and vhm0 is not None:
                        y_for_wave_mask = y
                        if (
                            self.predict_residual_to_prior
                            and prior_bias_batch is not None
                            and self.task_name == self.residual_prior_task
                        ):
                            y_for_wave_mask = y + prior_bias_batch
                        # Create masks for different wave height ranges
                        low_wave_mask = ((vhm0 + y_for_wave_mask) >= 0.0) & (
                            (vhm0 + y_for_wave_mask) <= 3.0
                        )
                        # high_wave_mask = (vhm0 >= 8.0) & (vhm0 <= 9.0)
                        high_wave_mask = ((vhm0 + y_for_wave_mask) >= 5.0) & (
                            (vhm0 + y_for_wave_mask) <= 11.0
                        )
                        mid_wave_mask = ~(low_wave_mask | high_wave_mask)

                        # Get predictions from all models
                        y_pred_default = self.model(X)

                        # Handle MoE diagnostic dicts and multi-task outputs.
                        if isinstance(y_pred_default, dict):
                            y_pred_default = (
                                y_pred_default["prediction"]
                                if "prediction" in y_pred_default
                                else y_pred_default[self.task_name]
                            )

                        # Start with default predictions
                        y_pred = y_pred_default.clone()
                        # y_pred = y_pred.clamp(-0.0325, 0.0325)

                        # Helper function to align spatial dimensions
                        def align_predictions(pred_source, pred_target, is_mask=False):
                            """Align pred_source spatial dims to match pred_target.

                            Args:
                                is_mask: If True, use nearest-neighbor to avoid boundary bleeding
                            """
                            if pred_source.shape != pred_target.shape:
                                # Resize to match target dimensions
                                import torch.nn.functional as F

                                mode = "nearest" if is_mask else "bilinear"
                                return F.interpolate(
                                    pred_source,
                                    size=(pred_target.shape[2], pred_target.shape[3]),
                                    mode=mode,
                                    align_corners=False if mode == "bilinear" else None,
                                )
                            return pred_source

                        # Apply low-wave specialized model if available
                        if self.low_wave_model is not None and low_wave_mask.any():
                            y_pred_low = self.low_wave_model(X)
                            if isinstance(y_pred_low, dict):
                                y_pred_low = (
                                    y_pred_low["prediction"]
                                    if "prediction" in y_pred_low
                                    else y_pred_low[self.task_name]
                                )

                            # Debug: Check if shapes match
                            if batch_idx == 0:
                                logger.info(
                                    f"Shape check - Default: {y_pred.shape}, Low-wave: {y_pred_low.shape}, VHM0: {vhm0.shape}"
                                )

                            # Only align if shapes differ
                            if y_pred_low.shape != y_pred.shape:
                                logger.warning(
                                    f"Shape mismatch! Aligning low-wave model output from {y_pred_low.shape} to {y_pred.shape}"
                                )
                                y_pred_low = align_predictions(
                                    y_pred_low, y_pred, is_mask=False
                                )
                                low_wave_mask_aligned = align_predictions(
                                    low_wave_mask.float(), y_pred, is_mask=True
                                ).bool()
                            else:
                                low_wave_mask_aligned = low_wave_mask

                            y_pred = torch.where(
                                low_wave_mask_aligned, y_pred_low, y_pred
                            )
                        if self.apply_delta_corrector_flag:
                            y_pred = y_pred.clone()

                            # --- Alternative A: outlier ratio clamping ---
                            max_bias_ratio = 1.5
                            delta = 0.035
                            if self.eval_in_bias_mode:
                                corrected = vhm0 + y_pred
                                true_wave = vhm0 + y_for_wave_mask  # ground truth
                                outlier_mask = (corrected.abs() > (max_bias_ratio * vhm0.abs().clamp(min=0.1))) & (true_wave >= 11.0)
                                outlier_mask = (true_wave >= 11.0)
                                if outlier_mask.any():
                                    y_pred[outlier_mask] = delta
                            else:
                                true_wave = y  # ground truth is already absolute
                                outlier_mask = (y_pred.abs() > (max_bias_ratio * vhm0.abs().clamp(min=0.1))) & (true_wave >= 11.0)
                                if outlier_mask.any():
                                    y_pred[outlier_mask] = vhm0[outlier_mask] + delta

                            # --- Alternative B: tail fade-out with power scaling ---
                            # start, end = 10.8, 13.0
                            # if self.predict_bias:
                            #     tail_mask = (vhm0 + y) >= start
                            #     if tail_mask.any():
                            #         scale_lin = 1.0 - (((vhm0 + y) - start) / (end - start)).clamp(0.0, 1.0)
                            #         p = 3.0
                            #         scale = scale_lin ** p
                            #         y_pred[tail_mask] = y_pred[tail_mask] * scale[tail_mask]
                            #         delta = 0.035
                            #         y_pred[tail_mask] = y_pred[tail_mask] + (1.0 - scale[tail_mask]) * delta
                            #         max_bias_tail = 0.3
                            #         y_pred[tail_mask] = y_pred[tail_mask].clamp(-max_bias_tail, max_bias_tail)
                            # else:
                            #     tail_mask = y_pred >= start
                            #     if tail_mask.any():
                            #         scale_lin = 1.0 - ((y_pred - start) / (end - start)).clamp(0.0, 1.0)
                            #         p = 3.0
                            #         scale = scale_lin ** p
                            #         y_pred[tail_mask] = vhm0[tail_mask] + (y_pred[tail_mask] - vhm0[tail_mask]) * scale[tail_mask]
                            #         delta = 0.035
                            #         y_pred[tail_mask] = y_pred[tail_mask] + (1.0 - scale[tail_mask]) * delta
                            #         max_bias_tail = 0.3
                            #         bias_tail = y_pred[tail_mask] - vhm0[tail_mask]
                            #         y_pred[tail_mask] = vhm0[tail_mask] + bias_tail.clamp(-max_bias_tail, max_bias_tail)

                        # Apply high-wave specialized model if available
                        if self.high_wave_model is not None and high_wave_mask.any():
                            y_pred_high = self.high_wave_model(X)
                            if isinstance(y_pred_high, dict):
                                y_pred_high = (
                                    y_pred_high["prediction"]
                                    if "prediction" in y_pred_high
                                    else y_pred_high[self.task_name]
                                )

                            if batch_idx == 0:
                                print(
                                    f"[HIGH-WAVE] Shape check - Default: {y_pred.shape}, High-wave: {y_pred_high.shape}, VHM0: {vhm0.shape}"
                                )
                                print(
                                    f"[HIGH-WAVE] Routing {high_wave_mask.sum().item()} pixels to specialized model"
                                )

                            # Only align if shapes differ
                            if y_pred_high.shape != y_pred.shape:
                                logger.warning(
                                    f"Shape mismatch! Aligning high-wave model output from {y_pred_high.shape} to {y_pred.shape}"
                                )
                                y_pred_high = align_predictions(
                                    y_pred_high, y_pred, is_mask=False
                                )
                                high_wave_mask_aligned = align_predictions(
                                    high_wave_mask.float(), y_pred, is_mask=True
                                ).bool()
                            else:
                                high_wave_mask_aligned = high_wave_mask

                            y_pred = torch.where(
                                high_wave_mask_aligned, y_pred_high, y_pred
                            )

                        if batch_idx == 0:
                            low_pixels = low_wave_mask.sum().item()
                            mid_pixels = mid_wave_mask.sum().item()
                            high_pixels = high_wave_mask.sum().item()
                            total_pixels = low_wave_mask.numel()
                            print("Bin-specific routing:")
                            print(
                                f"  0-1m: {low_pixels}/{total_pixels} pixels ({100 * low_pixels / total_pixels:.1f}%)"
                                + (
                                    " → specialized model"
                                    if self.low_wave_model is not None
                                    else " → default model"
                                )
                            )
                            print(
                                f"  1-8m: {mid_pixels}/{total_pixels} pixels ({100 * mid_pixels / total_pixels:.1f}%) → default model"
                            )
                            print(
                                f"  8-9m: {high_pixels}/{total_pixels} pixels ({100 * high_pixels / total_pixels:.1f}%)"
                                + (
                                    " → specialized model"
                                    if self.high_wave_model is not None
                                    else " → default model"
                                )
                            )
                            if self.apply_delta_corrector_flag:
                                print(
                                    f"  DeltaCorrector: applying correction {0.0325} for bins >= 11m"
                                )
                    else:
                        y_pred = self.model(X)

                # Extract MoE uncertainty before the dict is consumed.
                moe_uncertainty = None
                if isinstance(y_pred, dict) and "uncertainty" in y_pred:
                    moe_uncertainty = y_pred["uncertainty"].detach()  # [B, H, W]

                # Normalize multi-task payloads to tensors before alignment.
                y_pred = _extract_task_tensor(y_pred, "model prediction")
                y = _extract_task_tensor(y, "target batch")

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
                if prior_bias_batch is not None:
                    prior_bias_batch = prior_bias_batch[:, :, :min_h, :min_w]

                if confidence is not None:
                    confidence = confidence[:, :min_h, :min_w]
                if moe_uncertainty is not None:
                    moe_uncertainty = moe_uncertainty[:, :min_h, :min_w]

                # Unnormalize if needed
                if self.normalize_target and self.normalizer is not None:
                    # CRITICAL: Set target_stats_ for the correct target column
                    # The dataset may have left it set to a different task during normalization
                    if self.target_column in self.normalizer.feature_order_:
                        target_idx = self.normalizer.feature_order_.index(
                            self.target_column
                        )
                        if target_idx in self.normalizer.stats_:
                            self.normalizer.target_stats_ = self.normalizer.stats_[
                                target_idx
                            ]

                    y_pred = self.normalizer.inverse_transform_torch(y_pred)
                    y = self.normalizer.inverse_transform_torch(y)

                if self.apply_bilateral_filter:  # New flag
                    y_pred = self._apply_bilateral_filter(y_pred, mask)
                # Apply bin-wise correction if enabled
                if self.apply_binwise_correction_flag and vhm0 is not None:
                    y_pred = self._apply_bin_corrections(y_pred, vhm0, mask)
                # Process batch and update accumulators
                self._process_batch(
                    X_cropped,
                    y,
                    mask,
                    vhm0,
                    y_pred,
                    timestamps,
                    confidence,
                    prior_bias_batch,
                    batch_idx=batch_idx,
                    timestamps_raw=timestamps_raw,
                    moe_uncertainty=moe_uncertainty,
                )

        print(f"Inference complete. Processed {self.total_count} valid pixels.")

        # Report timestamp availability for seasonal analysis
        if self._timestamps_cache:
            loaded_files = [
                k for k, v in self._timestamps_cache.items() if v is not None
            ]
            if loaded_files:
                logger.info(
                    f"Timestamps loaded from {len(loaded_files)} file(s) - seasonal analysis enabled"
                )
            else:
                logger.info(
                    "No timestamps found in data files - seasonal analysis disabled"
                )
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
            predict_bias=self.eval_in_bias_mode,
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

    def _apply_bilateral_filter(
        self, predictions, mask, d=5, sigma_color=0.3, sigma_space=5
    ):
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
            predict_bias=self.eval_in_bias_mode,
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
        results = {"bins": {}, "overall": {}}

        # Process each bin
        for bin_name, bin_data in self.category_breakdown.items():
            results["bins"][bin_name] = {}
            total_count = (
                bin_data["corrected"]["count"] + bin_data["not_corrected"]["count"]
            )

            for category in ["corrected", "not_corrected"]:
                stats = bin_data[category]
                count = stats["count"]

                if count > 0:
                    # Compute feature means and stds
                    feature_stats = {}
                    for i in sorted(stats["feature_sums"].keys()):
                        mean = stats["feature_sums"][i] / count
                        variance = (stats["feature_sq_sums"][i] / count) - (mean**2)
                        std = np.sqrt(max(0, variance))
                        feature_stats[f"feature_{i}"] = {
                            "mean": float(mean),
                            "std": float(std),
                        }

                    # Compute mean SNR
                    mean_snr = (
                        stats["snr_sum"] / count if stats["snr_sum"] != 0 else None
                    )

                    # Compute season percentages
                    total_seasons = sum(stats["seasons"].values())
                    season_pcts = {
                        season: (cnt / total_seasons * 100) if total_seasons > 0 else 0
                        for season, cnt in stats["seasons"].items()
                    }

                    # Compute mean confidence
                    mean_confidence = (
                        stats["confidence_sum"] / count
                        if stats["confidence_sum"] != 0
                        else None
                    )

                    results["bins"][bin_name][category] = {
                        "count": int(count),
                        "percentage": float(count / total_count * 100)
                        if total_count > 0
                        else 0,
                        "features": feature_stats,
                        "snr_mean": float(mean_snr) if mean_snr is not None else None,
                        "seasons": season_pcts,
                        "confidence_mean": float(mean_confidence)
                        if mean_confidence is not None
                        else None,
                    }
                else:
                    results["bins"][bin_name][category] = None

        # Process overall
        total_overall = (
            self.overall_breakdown["corrected"]["count"]
            + self.overall_breakdown["not_corrected"]["count"]
        )
        for category in ["corrected", "not_corrected"]:
            stats = self.overall_breakdown[category]
            count = stats["count"]

            if count > 0:
                feature_stats = {}
                for i in sorted(stats["feature_sums"].keys()):
                    mean = stats["feature_sums"][i] / count
                    variance = (stats["feature_sq_sums"][i] / count) - (mean**2)
                    std = np.sqrt(max(0, variance))
                    feature_stats[f"feature_{i}"] = {
                        "mean": float(mean),
                        "std": float(std),
                    }

                mean_snr = stats["snr_sum"] / count if stats["snr_sum"] != 0 else None
                total_seasons = sum(stats["seasons"].values())
                season_pcts = {
                    season: (cnt / total_seasons * 100) if total_seasons > 0 else 0
                    for season, cnt in stats["seasons"].items()
                }
                mean_confidence = (
                    stats["confidence_sum"] / count
                    if stats["confidence_sum"] != 0
                    else None
                )

                results["overall"][category] = {
                    "count": int(count),
                    "percentage": float(count / total_overall * 100)
                    if total_overall > 0
                    else 0,
                    "features": feature_stats,
                    "snr_mean": float(mean_snr) if mean_snr is not None else None,
                    "seasons": season_pcts,
                    "confidence_mean": float(mean_confidence)
                    if mean_confidence is not None
                    else None,
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
        with open(overall_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Category",
                    "Count",
                    "Percentage",
                    "SNR Mean",
                    "Confidence Mean",
                    "Winter %",
                    "Spring %",
                    "Summer %",
                    "Autumn %",
                    "Features",
                ]
            )

            for category in ["corrected", "not_corrected"]:
                if category in breakdown["overall"] and breakdown["overall"][category]:
                    data = breakdown["overall"][category]

                    # Format features as string
                    features_str = "; ".join(
                        [
                            f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                            for k, v in data["features"].items()
                        ]
                    )

                    writer.writerow(
                        [
                            category.replace("_", " ").title(),
                            data["count"],
                            f"{data['percentage']:.2f}",
                            f"{data['snr_mean']:.4f}"
                            if data["snr_mean"] is not None
                            else "N/A",
                            f"{data['confidence_mean']:.4f}"
                            if data["confidence_mean"] is not None
                            else "N/A",
                            f"{data['seasons']['winter']:.2f}",
                            f"{data['seasons']['spring']:.2f}",
                            f"{data['seasons']['summer']:.2f}",
                            f"{data['seasons']['autumn']:.2f}",
                            features_str,
                        ]
                    )

        logger.info(f"Saved overall category breakdown to {overall_csv}")

        # Save per-bin breakdown
        bins_csv = output_path / "category_breakdown_per_bin.csv"
        with open(bins_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Bin",
                    "Category",
                    "Count",
                    "Percentage",
                    "SNR Mean",
                    "Confidence Mean",
                    "Winter %",
                    "Spring %",
                    "Summer %",
                    "Autumn %",
                    "Features",
                ]
            )

            for bin_name, bin_data in breakdown["bins"].items():
                # Get bin label for readability
                bin_label = next(
                    (b["label"] for b in self.sea_bins if b["name"] == bin_name),
                    bin_name,
                )

                for category in ["corrected", "not_corrected"]:
                    if category in bin_data and bin_data[category]:
                        data = bin_data[category]

                        # Format features as string
                        features_str = "; ".join(
                            [
                                f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                                for k, v in data["features"].items()
                            ]
                        )

                        writer.writerow(
                            [
                                bin_label,
                                category.replace("_", " ").title(),
                                data["count"],
                                f"{data['percentage']:.2f}",
                                f"{data['snr_mean']:.4f}"
                                if data["snr_mean"] is not None
                                else "N/A",
                                f"{data['confidence_mean']:.4f}"
                                if data["confidence_mean"] is not None
                                else "N/A",
                                f"{data['seasons']['winter']:.2f}",
                                f"{data['seasons']['spring']:.2f}",
                                f"{data['seasons']['summer']:.2f}",
                                f"{data['seasons']['autumn']:.2f}",
                                features_str,
                            ]
                        )

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
        bin_names = [bin_config["name"] for bin_config in self.sea_bins]
        bin_labels = {
            bin_config["name"]: bin_config["label"] for bin_config in self.sea_bins
        }

        # Filter to only bins with data
        bins_with_data = [
            bn
            for bn in bin_names
            if bn in breakdown["bins"]
            and (
                breakdown["bins"][bn].get("corrected")
                or breakdown["bins"][bn].get("not_corrected")
            )
        ]

        # Collect all feature indices that exist
        all_feature_indices = set()
        for bin_name in bins_with_data:
            bin_data = breakdown["bins"][bin_name]
            for category in ["corrected", "not_corrected"]:
                if bin_data.get(category) and bin_data[category].get("features"):
                    all_feature_indices.update(
                        [
                            int(f.split("_")[1])
                            for f in bin_data[category]["features"].keys()
                        ]
                    )

        # Build rows for each metric
        metric_rows = []

        # Count row
        count_row = {"Metric": "Count"}
        count_pct_row = {"Metric": "Percentage"}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown["bins"][bin_name]

            for category in ["Corrected", "Not corrected"]:
                cat_key = category.lower().replace(" ", "_")
                col_name = f"{bin_label}_{category}"

                if bin_data.get(cat_key):
                    count_row[col_name] = bin_data[cat_key]["count"]
                    count_pct_row[col_name] = f"{bin_data[cat_key]['percentage']:.1f}%"
                else:
                    count_row[col_name] = 0
                    count_pct_row[col_name] = "0.0%"

        metric_rows.append(count_row)
        metric_rows.append(count_pct_row)

        # Feature rows (mean only for simplicity in wide format)
        for feat_idx in sorted(all_feature_indices):
            feat_row = {"Metric": f"Mean feature {feat_idx}"}
            for bin_name in bins_with_data:
                bin_label = bin_labels[bin_name]
                bin_data = breakdown["bins"][bin_name]

                for category in ["Corrected", "Not corrected"]:
                    cat_key = category.lower().replace(" ", "_")
                    col_name = f"{bin_label}_{category}"

                    if bin_data.get(cat_key) and bin_data[cat_key].get("features"):
                        feat_key = f"feature_{feat_idx}"
                        if feat_key in bin_data[cat_key]["features"]:
                            feat_row[col_name] = (
                                f"{bin_data[cat_key]['features'][feat_key]['mean']:.4f}"
                            )
                        else:
                            feat_row[col_name] = "N/A"
                    else:
                        feat_row[col_name] = "N/A"

            metric_rows.append(feat_row)

        # SNR row
        snr_row = {"Metric": "SNR (mean dB)"}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown["bins"][bin_name]

            for category in ["Corrected", "Not corrected"]:
                cat_key = category.lower().replace(" ", "_")
                col_name = f"{bin_label}_{category}"

                if (
                    bin_data.get(cat_key)
                    and bin_data[cat_key].get("snr_mean") is not None
                ):
                    snr_row[col_name] = f"{bin_data[cat_key]['snr_mean']:.4f}"
                else:
                    snr_row[col_name] = "N/A"

        metric_rows.append(snr_row)

        # Season rows
        for season in ["Summer", "Autumn", "Winter", "Spring"]:
            season_row = {"Metric": season}
            season_key = season.lower()

            for bin_name in bins_with_data:
                bin_label = bin_labels[bin_name]
                bin_data = breakdown["bins"][bin_name]

                for category in ["Corrected", "Not corrected"]:
                    cat_key = category.lower().replace(" ", "_")
                    col_name = f"{bin_label}_{category}"

                    if bin_data.get(cat_key) and bin_data[cat_key].get("seasons"):
                        season_row[col_name] = (
                            f"{bin_data[cat_key]['seasons'][season_key]:.1f}%"
                        )
                    else:
                        season_row[col_name] = "N/A"

            metric_rows.append(season_row)

        # Model confidence row
        conf_row = {"Metric": "Model confidence"}
        for bin_name in bins_with_data:
            bin_label = bin_labels[bin_name]
            bin_data = breakdown["bins"][bin_name]

            for category in ["Corrected", "Not corrected"]:
                cat_key = category.lower().replace(" ", "_")
                col_name = f"{bin_label}_{category}"

                if (
                    bin_data.get(cat_key)
                    and bin_data[cat_key].get("confidence_mean") is not None
                ):
                    conf_row[col_name] = f"{bin_data[cat_key]['confidence_mean']:.4f}"
                else:
                    conf_row[col_name] = "N/A"

        metric_rows.append(conf_row)

        # Create DataFrame
        df = pd.DataFrame(metric_rows)

        # Reorder columns: Metric first, then bins in order
        columns = ["Metric"]
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
        for category in ["corrected", "not_corrected"]:
            if category in breakdown["overall"] and breakdown["overall"][category]:
                data = breakdown["overall"][category]
                print(f"\n  {category.upper().replace('_', ' ')}:")
                print(f"    Count: {data['count']:,} ({data['percentage']:.1f}%)")

                if data["features"]:
                    print("    Features (mean ± std):")
                    for feat_name, feat_data in sorted(data["features"].items()):
                        print(
                            f"      {feat_name}: {feat_data['mean']:.4f} ± {feat_data['std']:.4f}"
                        )

                if data["snr_mean"] is not None:
                    print(f"    SNR (mean): {data['snr_mean']:.4f} dB")

                print("    Seasons:")
                for season in ["winter", "spring", "summer", "autumn"]:
                    pct = data["seasons"][season]
                    print(f"      {season.capitalize()}: {pct:.1f}%")

                if data["confidence_mean"] is not None:
                    print(f"    Model Confidence (mean): {data['confidence_mean']:.4f}")

        # Print per-bin breakdown
        print("\n\nPER-BIN BREAKDOWN:")
        print("=" * 140)

        for bin_name, bin_data in breakdown["bins"].items():
            # Get readable bin label
            bin_label = next(
                (b["label"] for b in self.sea_bins if b["name"] == bin_name), bin_name
            )

            print(f"\n{bin_label}:")
            print("-" * 140)

            for category in ["corrected", "not_corrected"]:
                if category in bin_data and bin_data[category]:
                    data = bin_data[category]
                    print(f"\n  {category.upper().replace('_', ' ')}:")
                    print(f"    Count: {data['count']:,} ({data['percentage']:.1f}%)")

                    if data["features"]:
                        print("    Features (mean ± std):")
                        for feat_name, feat_data in sorted(data["features"].items()):
                            print(
                                f"      {feat_name}: {feat_data['mean']:.4f} ± {feat_data['std']:.4f}"
                            )

                    if data["snr_mean"] is not None:
                        print(f"    SNR (mean): {data['snr_mean']:.4f} dB")

                    print("    Seasons:")
                    for season in ["winter", "spring", "summer", "autumn"]:
                        pct = data["seasons"][season]
                        print(f"      {season.capitalize()}: {pct:.1f}%")

                    if data["confidence_mean"] is not None:
                        print(
                            f"    Model Confidence (mean): {data['confidence_mean']:.4f}"
                        )

        print("=" * 140 + "\n")

    def plot_rmse_maps(self):
        """Plot spatial RMSE maps for model and baseline."""
        # Get coordinates from dataset (respects region cropping)
        dataset_coords = None
        try:
            dataset = self.test_loader.dataset
            if hasattr(dataset, "get_coordinates"):
                lat_grid, lon_grid = dataset.get_coordinates()
                dataset_coords = (lat_grid, lon_grid)
                logger.info(
                    f"Using dataset coordinates for RMSE maps: {lat_grid.shape}"
                )
        except Exception as e:
            logger.warning(f"Could not get dataset coordinates: {e}")

        plot_rmse_maps_fn(
            spatial_errors_model=self.spatial_errors_model,
            spatial_errors_baseline=self.spatial_errors_baseline,
            test_files=self.test_files,
            subsample_step=self.subsample_step,
            geo_bounds=self.geo_bounds,
            unit=self.unit,
            output_dir=self.output_dir,
            dataset_coords=dataset_coords,
        )

    def plot_bin_spatial_rmse_maps(self):
        """Plot spatial RMSE maps conditioned on true wave height bin."""
        import matplotlib.pyplot as plt

        output_dir = self.output_dir / "bin_spatial_rmse"
        output_dir.mkdir(parents=True, exist_ok=True)

        for (lo, hi), acc in self.bin_spatial_accumulators.items():
            if not acc["error_sq"]:
                continue
            sum_sq = np.sum(acc["error_sq"], axis=0)   # (H, W)
            count  = np.sum(acc["count"],    axis=0)   # (H, W)
            with np.errstate(invalid="ignore", divide="ignore"):
                rmse = np.where(count > 0, np.sqrt(sum_sq / count), np.nan)

            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(rmse, origin="upper", cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax, label="RMSE (m)")
            hi_label = f"{hi:.0f}" if hi < float("inf") else "∞"
            ax.set_title(f"Spatial RMSE | true wave {lo:.0f}–{hi_label}m  (n={int(count.sum()):,})")
            ax.axis("off")
            fname = f"spatial_rmse_true_{lo:.0f}_{hi_label}m.png"
            fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved bin spatial RMSE map → {output_dir / fname}")

    def plot_low_bin_spatial_maps(self):
        """Plot spatial diagnostics for ultra-calm true-wave sub-bins."""
        dataset_coords = None
        try:
            dataset = self.test_loader.dataset
            if hasattr(dataset, "get_coordinates"):
                lat_grid, lon_grid = dataset.get_coordinates()
                dataset_coords = (lat_grid, lon_grid)
        except Exception as e:
            logger.warning(f"Could not get dataset coordinates for low-bin maps: {e}")

        plot_low_bin_spatial_maps_fn(
            low_bin_spatial_accumulators=self.low_bin_spatial_accumulators,
            low_bin_spatial_subbins=self.low_bin_spatial_subbins,
            test_files=self.test_files,
            subsample_step=self.subsample_step,
            geo_bounds=self.geo_bounds,
            output_dir=self.output_dir,
            dataset_coords=dataset_coords,
        )

    def plot_low_bin_advanced_diagnostics(self):
        """Create low-bin CDF/hist/hexbin diagnostics from sampled points."""
        plot_low_bin_advanced_diagnostics_fn(
            low_bin_plot_samples=self.low_bin_plot_samples,
            low_bin_spatial_subbins=self.low_bin_spatial_subbins,
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

    def plot_vhm0_distributions(self, vhm0_range=None):
        """Plot distributions of ground truth, predicted, and uncorrected VHM0.

        Args:
            vhm0_range: Optional (lo, hi) tuple to filter by raw VHM0 range (metres).
        """
        plot_vhm0_distributions_fn(
            plot_samples=self.plot_samples,
            var_name=self.var_name,
            var_name_full=self.var_name_full,
            unit=self.unit,
            corrected_label=self.corrected_label,
            model_label=self.model_label,
            uncorrected_label=self.uncorrected_label,
            output_dir=self.output_dir,
            vhm0_range=vhm0_range,
        )

    def compute_per_point_improvement_stats(self, epsilon_m: float = 0.01) -> Dict:
        """Compute per-pixel relative improvement statistics.

        epsilon_m: minimum absolute improvement (m) to count as genuinely improved.
        """
        if not self._rel_improvement_samples:
            return {}
        all_vals = np.concatenate(self._rel_improvement_samples)
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) == 0:
            return {}
        improved = np.sum(finite > (epsilon_m * 100 / 1.0))  # epsilon as relative %
        return {
            "mean_pct": float(np.mean(finite)),
            "median_pct": float(np.median(finite)),
            "std_pct": float(np.std(finite)),
            "pct_improved": float(100.0 * improved / len(finite)),
            "n_pixels": int(len(finite)),
            "epsilon_m": epsilon_m,
        }

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
        denoise_summary = self._get_denoising_summary()
        if denoise_summary["enabled"]:
            kept_pct = denoise_summary["kept_pct"]
            kept_pct_str = "n/a" if kept_pct is None else f"{kept_pct:.2f}%"
            print("\nDenoising:")
            print(f"  Abs threshold:        {denoise_summary['abs_threshold_m']:.4f} m")
            print(
                "  Coverage:             "
                f"{denoise_summary['kept_pixels']:,}/{denoise_summary['candidate_pixels']:,} "
                f"({kept_pct_str})"
            )

        per_point = self.compute_per_point_improvement_stats()
        if per_point:
            print("\n=== Per-point Relative Improvement (%) ===")
            print(f"  Mean:              {per_point['mean_pct']:.2f}%")
            print(f"  Median:            {per_point['median_pct']:.2f}%")
            print(f"  Std:               {per_point['std_pct']:.2f}%")
            print(f"  Improved samples:  {per_point['pct_improved']:.2f}%  (ε={per_point['epsilon_m']}m)")

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
        if self.edcdf_hard_fallback_bins:
            valid = max(1, self._edcdf_fallback_total_valid)
            pct = 100.0 * self._edcdf_fallback_total_applied / valid
            logger.info(
                "EDCDF hard fallback coverage: "
                f"{self._edcdf_fallback_total_applied:,}/{self._edcdf_fallback_total_valid:,} "
                f"valid pixels ({pct:.2f}%) using source='{self.edcdf_fallback_bin_source}'"
            )
            logger.info(
                f"EDCDF hard fallback per-bin applied counts: {self._edcdf_fallback_applied_per_bin}"
            )
        if self.prior_hard_fallback_bins:
            valid = max(1, self._prior_fallback_total_valid)
            pct = 100.0 * self._prior_fallback_total_applied / valid
            logger.info(
                "Prior hard fallback coverage: "
                f"{self._prior_fallback_total_applied:,}/{self._prior_fallback_total_valid:,} "
                f"valid pixels ({pct:.2f}%) using source='{self.prior_fallback_bin_source}'"
            )
            logger.info(
                f"Prior hard fallback per-bin applied counts: {self._prior_fallback_applied_per_bin}"
            )

        # NEW: Compute category breakdown
        print("Computing category breakdown (corrected vs not_corrected)...")
        category_breakdown = self.compute_category_breakdown()

        # Save metrics
        per_point_stats = self.compute_per_point_improvement_stats()
        denoise_summary = self._get_denoising_summary()
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "overall": overall_metrics,
                    "sea_bins": sea_bin_metrics,
                    "category_breakdown": category_breakdown,
                    "per_point_improvement": per_point_stats,
                    "denoising": denoise_summary,
                },
                f,
                indent=2,
            )

        # NEW: Save category breakdown to CSV
        print("Saving category breakdown to CSV...")
        # self.save_category_breakdown_csv(category_breakdown, self.output_dir)
        # self.save_category_breakdown_wide_format(category_breakdown, self.output_dir)
        print("Saving detailed low-bin diagnostics...")
        # self._save_low_bin_diagnostics(sea_bin_metrics)
        print("Saving coastal-distance diagnostics...")
        # self._save_coastal_distance_diagnostics()
        print("Saving grid-point time-series CSV...")
        self._save_grid_point_csv()
        print("Saving all-points temporal heatmap...")
        self._save_all_points_temporal_heatmap()

        # Save raw prediction samples for offline plot experimentation
        if self.save_predictions:
            predictions_path = self.output_dir / "plot_samples.npz"
            np.savez_compressed(
                predictions_path,
                y_true=np.array(self.plot_samples["y_true"]),
                y_pred=np.array(self.plot_samples["y_pred"]),
                vhm0=np.array(self.plot_samples["vhm0"]),
                lat=np.array(self.plot_samples["lat"]),
                lon=np.array(self.plot_samples["lon"]),
            )
            print(f"  Prediction samples saved → {predictions_path}")

        # Create plots using samples
        print("Creating plots...")
        self.plot_sea_bin_metrics(sea_bin_metrics)
        self.plot_model_better_percentage(sea_bin_metrics)
        self.plot_rmse_maps()
        # self.plot_bin_spatial_rmse_maps()
        # self.plot_low_bin_spatial_maps()
        # self.plot_low_bin_advanced_diagnostics()
        # self.plot_vhm0_distributions()
        # self.plot_vhm0_distributions(vhm0_range=(0, 1))
        # self.plot_vhm0_distributions(vhm0_range=(1, 2))
        # self.plot_vhm0_distributions(vhm0_range=(11, 12))
        # self.plot_vhm0_distributions(vhm0_range=(12, 13))
        # self.plot_error_distribution_histograms()
        # self.plot_error_boxplots()
        # self.plot_error_violins()
        # self.plot_error_cdfs()

        # Print summaries
        self.print_summary(overall_metrics, sea_bin_metrics)

        # NEW: Print category breakdown
        # self.print_category_breakdown(category_breakdown)

        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main(evaluator_class=None):
    if evaluator_class is None:
        evaluator_class = ModelEvaluator
    parser = argparse.ArgumentParser(description="Evaluate WaveBiasCorrector model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint file or directory (evaluates all .ckpt files in directory)",
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
        "--eval-task",
        type=str,
        default=None,
        help=(
            "Task name to evaluate (must be a key in data.target_columns). "
            "Use when multiple targets are configured; overrides the default first key."
        ),
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
    parser.add_argument(
        "--apply-delta-corrector-flag",
        action="store_true",
        default=False,
        help="Apply delta corrector to predictions for bins >= 11m",
    )
    parser.add_argument(
        "--region-filter",
        type=str,
        default=None,
        choices=["atlantic", "mediterranean", "aegean"],
        help="Region to filter metrics (applied via geo_mask, not dataset cropping)",
    )
    parser.add_argument(
        "--sampled-points-csv",
        type=str,
        default=None,
        help=(
            "Path to CSV produced by sample_grid_points.py "
            "(columns: region, latitude, longitude). "
            "When provided, records reference / uncorrected / corrected values at each "
            "sampled grid point for every time step and writes grid_point_timeseries.csv "
            "to the output directory."
        ),
    )
    parser.add_argument(
        "--timestamps-csv",
        type=str,
        default=None,
        help=(
            "Path to CSV produced by build_pt_timestamp_map.py "
            "(columns: pt_stem, hour_idx, timestamp). "
            "Required for correct timestamps in grid_point_timeseries.csv."
        ),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save plot_samples (y_true, y_pred, vhm0) to plot_samples.npz in the output dir",
    )
    parser.add_argument(
        "--denoise-abs-threshold",
        type=float,
        default=None,
        help=(
            "Optional denoising filter applied during evaluation: keep only pixels "
            "with |uncorrected-reference| > threshold (meters)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


    config = DNNConfig(args.config)

    training_config = config.config["training"]
    data_config = config.config["data"]
    predict_bias = data_config.get("predict_bias", False)
    predict_residual_to_prior = data_config.get("predict_residual_to_prior", False)
    residual_prior_task = data_config.get("residual_prior_task", None)
    prior_source = data_config.get("prior_source", "none")
    if predict_bias and predict_residual_to_prior:
        raise ValueError(
            "Only one of data.predict_bias or data.predict_residual_to_prior can be enabled for evaluation."
        )

    # Support both old target_column (str) and new target_columns (dict)
    target_columns = data_config.get("target_columns", None)
    if target_columns is None:
        # Fall back to old single-task format
        target_column = data_config.get("target_column", "corrected_VHM0")
        target_columns = {"vhm0": target_column}

    if args.eval_task is not None:
        if args.eval_task not in target_columns:
            raise ValueError(
                f"--eval-task {args.eval_task!r} is not in data.target_columns "
                f"{list(target_columns.keys())}"
            )

    # Get file list (same as training)
    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )
    test_year_cfg = data_config.get("test_year", [2023])
    test_year = test_year_cfg[0] if isinstance(test_year_cfg, list) else test_year_cfg
    parquet_data_path = data_config.get(
        "diagnostics_parquet_data_path",
        "/mnt/blobstorage/parquet/hourly/",
    )
    parquet_file_pattern = data_config.get(
        "diagnostics_parquet_file_pattern", f"WAVEAN{test_year}*.parquet"
    )
    test_files_parq = []
    try:
        _test_files_parq = get_file_list(parquet_data_path, parquet_file_pattern)
        # Enforce test-year filtering even when a broad file pattern/path is used.
        year_prefix = f"WAVEAN{int(test_year)}"
        _test_files_parq = [
            fp for fp in _test_files_parq if fp.rsplit("/", 1)[-1].startswith(year_prefix)
        ]
        _, _, test_files_parq = split_files_by_year(
            _test_files_parq,
            train_year=data_config.get("train_year", 2021),
            val_year=data_config.get("val_year", 2022),
            test_year=data_config.get("test_year", 2023),
            val_months=data_config.get("val_months", []),
            test_months=data_config.get("test_months", []),
        )
        print(test_files_parq[:10])
    except Exception as e:
        logger.warning(
            "Could not list diagnostics parquet files (%s). "
            "Continuing without timestamp-based diagnostics.",
            e,
        )

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

    # Load normalizer (supports both local and s3:// paths)
    normalizer_path = str(data_config["normalizer_path"])
    if normalizer_path.startswith("s3://"):
        s3_uri = normalizer_path.replace("s3://", "", 1)
        bucket, key = s3_uri.split("/", 1)
        normalizer = WaveNormalizer.load_from_s3(bucket, key)
    else:
        normalizer = WaveNormalizer().load(normalizer_path)
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Loaded normalizer from {normalizer_path}")

    # CRITICAL: Set target_stats_ for the target column we're evaluating
    # Without this, inverse_transform_torch falls back to the last channel!
    eval_target_col = (
        target_columns[args.eval_task]
        if args.eval_task
        else list(target_columns.values())[0]
    )
    if eval_target_col in normalizer.feature_order_:
        target_idx = normalizer.feature_order_.index(eval_target_col)
        if target_idx in normalizer.stats_:
            normalizer.target_stats_ = normalizer.stats_[target_idx]
            logger.info(
                f"Set normalizer target_stats_ for '{eval_target_col}' (index {target_idx})"
            )
        else:
            logger.warning(f"Target index {target_idx} not found in normalizer stats!")
    else:
        logger.warning(
            f"Target column '{eval_target_col}' not found in normalizer feature_order!"
        )

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
            predict_residual_to_prior=predict_residual_to_prior,
            prior_source=prior_source,
            static_bias_map_path=data_config.get("static_bias_map_path", None),
            residual_prior_task=residual_prior_task,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=False,
            use_cache=False,  # Use cache for evaluation
            normalize_target=data_config.get("normalize_target", False),
            region_filter=data_config.get("region_filter", None),
            add_sea_mask_channel=data_config.get("add_sea_mask_channel", False),
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
                predict_residual_to_prior=predict_residual_to_prior,
                prior_source=prior_source,
                static_bias_map_path=data_config.get("static_bias_map_path", None),
                residual_prior_task=residual_prior_task,
                subsample_step=subsample_step,
                normalizer=normalizer,
                enable_profiler=False,
                use_cache=False,
                normalize_target=data_config.get("normalize_target", False),
                region_filter=data_config.get("region_filter", None),
                add_sea_mask_channel=data_config.get("add_sea_mask_channel", False),
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
            raise ValueError(
                "No checkpoint specified. Use --checkpoint or set in config"
            )

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_dir():
        # Find all .ckpt files in directory
        checkpoint_list = sorted(list(checkpoint_path.glob("epoch=1*-val_loss=*.ckpt")))
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
        logger.info(
            "Model loaded. predict_bias=%s, predict_residual_to_prior=%s",
            predict_bias,
            predict_residual_to_prior,
        )

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
        region_filter = args.region_filter
        if args.apply_geographic_filtering:
            if patch_size is not None:
                logger.warning("=" * 80)
                logger.warning(
                    "Geographic filtering is NOT supported with patch-based datasets!"
                )
                logger.warning("Patches don't maintain spatial coordinate information.")
                logger.warning("Geographic filtering will be DISABLED.")
                logger.warning("=" * 80)
                geo_bounds = None
            else:
                geo_bounds = {
                    "lat_min": 30.0,
                    "lat_max": 46.0,
                    "lon_min": -18.5,
                    "lon_max": 36.5,
                }
                logger.info(
                    f"Geographic filtering enabled: region_filter={region_filter}, "
                    f"geo_bounds={geo_bounds}"
                )

        # Create evaluator and run evaluation
        evaluator = evaluator_class(
            model=model,
            test_loader=test_loader,
            output_dir=Path(args.output_dir)
            / config.config["logging"]["experiment_name"]
            / str(test_year) /checkpoint.stem,  # Use checkpoint filename without extension
            predict_bias=predict_bias,
            predict_residual_to_prior=predict_residual_to_prior,
            residual_prior_task=residual_prior_task,
            device="cuda",
            normalizer=normalizer,
            normalize_target=data_config.get("normalize_target", False),
            test_files=test_files_parq,
            subsample_step=subsample_step
            if subsample_step is not None
            else 5,  # Match preprocessed data subsampling
            apply_binwise_correction_flag=args.apply_binwise_correction,
            bias_loader=train_loader,  # Use train set to compute bin biases
            geo_bounds=geo_bounds,
            use_mdn=model.use_mdn,
            target_columns=target_columns,
            apply_bilateral_filter=False,
            apply_delta_corrector_flag=args.apply_delta_corrector_flag,
            region_filter=region_filter,
            eval_task=args.eval_task,
            low_wave_ckpt=config.config["checkpoint"]["low_wave_ckpt"],
            high_wave_ckpt=config.config["checkpoint"]["high_wave_ckpt"],
            static_bias_map_path=data_config.get("static_bias_map_path", None),
            blend_sigma=data_config.get("blend_sigma", None),
            uncertainty_blend_sigma=data_config.get("uncertainty_blend_sigma", None),
            domain_mean_recalibration=data_config.get("domain_mean_recalibration", False),
            edcdf_model_path=data_config.get("edcdf_model_path", None),
            edcdf_blend_sigma=data_config.get("edcdf_blend_sigma", None),
            edcdf_hard_fallback_bins=data_config.get("edcdf_hard_fallback_bins", None),
            edcdf_fallback_bin_source=data_config.get("edcdf_fallback_bin_source", "raw"),
            prior_hard_fallback_bins=data_config.get("prior_hard_fallback_bins", None),
            prior_fallback_bin_source=data_config.get("prior_fallback_bin_source", "raw"),
            prior_fallback_target=data_config.get("prior_fallback_target", "prior"),
            low_bin_affine_params=data_config.get("low_bin_affine_params", None),
            low_bin_affine_source=data_config.get("low_bin_affine_source", "raw"),
            sampled_points_csv=args.sampled_points_csv,
            timestamps_csv=args.timestamps_csv,
            save_predictions=args.save_predictions,
            denoise_abs_threshold=args.denoise_abs_threshold,
        )

        evaluator.evaluate()

        logger.info(f"Completed evaluation for {checkpoint.name}")
        logger.info("=" * 80)

    logger.info(
        f"\nAll evaluations complete! Evaluated {len(checkpoint_list)} checkpoint(s)"
    )


if __name__ == "__main__":
    main()
