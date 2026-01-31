import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Setup logging
def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# import pytorch_lightning as lightning
import lightning
import s3fs
import torch
from classifiers.lightning_trainer import WaveBiasCorrector
from commons.aws.utils import download_s3_checkpoint
from commons.callbacks.comet_callbacks import CometVisualizationCallback
from commons.callbacks.exponential_moving_average import EMAWeightAveraging
from commons.callbacks.freeze_layers import FreezeEncoderCallback
from commons.callbacks.pixel_switch_threshold import PixelSwitchThresholdCallback
from commons.callbacks.s3_callback import S3CheckpointSyncCallback
from commons.dataloaders import CachedWaveDataset
from commons.datasets.grid_patched_dataset import GridPatchWaveDataset
from commons.datasets.samplers import WaveBinBalancedSampler
from commons.preprocessing.bu_net_preprocessing import WaveNormalizer
from lightning import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader


def _log_training_artifacts(comet_logger, config_file):
    """Log all training scripts and configuration to Comet ML"""
    import glob
    import os

    # Log the main configuration file
    comet_logger.experiment.log_asset(config_file, file_name="config.yaml")

    # Log all Python scripts involved in training
    script_patterns = [
        "src/pipelines/training/*.py",
        "src/classifiers/*.py",
        "src/commons/*.py",
        "src/commons/preprocessing/*.py",
        "src/commons/callbacks/*.py",
    ]

    for pattern in script_patterns:
        scripts = glob.glob(pattern)
        for script in scripts:
            if os.path.exists(script):
                comet_logger.experiment.log_asset(
                    script, file_name=os.path.basename(script)
                )

    # Log the main training script
    comet_logger.experiment.log_asset(__file__, file_name="dnn_trainer.py")

    # Log project metadata
    comet_logger.experiment.log_asset("pyproject.toml", file_name="pyproject.toml")
    comet_logger.experiment.log_asset("README.md", file_name="README.md")

    # Log git information if available
    try:
        import subprocess

        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        comet_logger.experiment.log_other("git_hash", git_hash)
        comet_logger.experiment.log_other("git_branch", git_branch)
    except Exception as e:
        logger.error(f"Failed to log git information: {e}")
        pass  # Git not available or not a git repo


class DNNConfig:
    """Configuration class for DNN training"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()
        if config_path:
            self._load_config(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "data": {
                "data_path": "s3://medwav-dev-data/parquet/hourly/",
                "file_pattern": "*.parquet",
                "train_year": 2021,
                "val_year": 2022,
                "test_year": 2023,
                "patch_size": [256, 256],
                "max_files": None,
                "random_seed": 42,
                "excluded_columns": ["time", "latitude", "longitude", "timestamp"],
                "target_columns": {"vhm0": "corrected_VHM0"},
                "predict_bias": False,
            },
            "model": {
                "in_channels": 14,
                "learning_rate": 1e-4,
                "loss_type": "weighted_mse",
                "filters": [64, 128, 256, 512, 1024],
                "weight_decay": 0,
            },
            "training": {
                "batch_size": 8,
                "max_epochs": 20,
                "num_workers": 4,
                "pin_memory": True,
                "accelerator": "gpu",
                "devices": 1,
                "precision": 16,
                "log_every_n_steps": 10,
                "early_stopping_patience": 5,
                "save_top_k": 3,
                "monitor": "val_loss",
                "mode": "min",
            },
            "checkpoint": {
                "resume_from_checkpoint": None,
                "checkpoint_dir": "checkpoints",
                "save_last": True,
            },
            "logging": {
                "log_dir": "logs",
                "experiment_name": "dnn_wave_correction",
                "use_comet": True,
            },
        }

    def _load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Deep merge user config with defaults
        self._deep_update(self.config, user_config)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


def get_file_list(
    data_path: str, file_pattern: str, max_files: Optional[int] = None
) -> list:
    """Get list of files from S3 or local path"""
    if data_path.startswith("s3://"):
        fs = s3fs.S3FileSystem()
        # Search in both the directory itself and subdirectories
        data_path_clean = data_path.rstrip("/")
        pattern = f"{data_path_clean}/{file_pattern}"
        print(f"Searching S3 with pattern: {pattern}")
        files = fs.glob(pattern)
        # Also add files from subdirectories
        pattern_recursive = f"{data_path_clean}/**/{file_pattern}"
        files_recursive = fs.glob(pattern_recursive)
        # Combine and deduplicate
        files = list(set(files + files_recursive))
        # Ensure s3:// prefix
        files = [f if f.startswith("s3://") else f"s3://{f}" for f in files]
    else:
        files = list(Path(data_path).glob(f"**/{file_pattern}"))
        files = [str(f) for f in files]

    if max_files:
        files = files[:max_files]

    return files


def split_files_by_year(
    files: list,
    train_year: int | list = 2021,
    val_year: int | list = 2022,
    test_year: int | list = 2023,
    val_months: list = None,
    test_months: list = None,
) -> tuple:
    """Split files into train/val/test based on year lists and validation months.

    Assumptions:
    - `train_year`, `val_year`, `test_year` are lists of years.
    - A file goes to validation only if (year in `val_year`) AND (month in `val_months`).
    - Otherwise it goes to train/test based on year membership.
    """
    train_files = []
    val_files = []
    test_files = []

    # Normalize inputs to sets of years for easy membership checks
    def _to_year_set(y):
        if isinstance(y, (list, tuple, set)):
            return set(int(v) for v in y)
        return {int(y)}

    train_years = _to_year_set(train_year)
    val_years = _to_year_set(val_year)
    test_years = _to_year_set(test_year)
    test_months_set = set(int(m) for m in test_months) if test_months else set()
    val_months_set = set(int(m) for m in val_months) if val_months else set()

    def _parse_year_month(name: str) -> tuple[int | None, int | None]:
        """Parse year and month from filename assuming pattern like WAVEANYYYYMM...
        Returns (year, month) where either can be None if not parsed.
        """
        try:
            marker = "WAVEAN"
            idx = name.find(marker)
            if idx != -1 and len(name) >= idx + 6 + 6:  # WAVEAN + YYYYMM
                year_str = name[idx + 6 : idx + 10]
                month_str = name[idx + 10 : idx + 12]
                year_val = int(year_str)
                month_val = int(month_str)
                if 1 <= month_val <= 12:
                    return year_val, month_val
                return year_val, None
        except Exception:
            pass

        # Fallback: find first 4-digit year and optional following month
        import re

        match = re.search(r"(20\d{2})(?:[^\d]?([01]?\d))?", name)
        if match:
            try:
                y = int(match.group(1))
                m = match.group(2)
                m_val = int(m) if m is not None else None
                if m_val is not None and not (1 <= m_val <= 12):
                    m_val = None
                return y, m_val
            except Exception:
                return None, None
        return None, None

    for file_path in files:
        filename = Path(file_path).name
        year, month = _parse_year_month(filename)

        if year is None:
            logger.warning(f"Skipping file {filename} - could not parse year")
            continue

        # Validation: require both year and month match (simple rule)
        if year in val_years and month in val_months_set:
            val_files.append(file_path)
            continue

        # Test split
        if year in test_years and month in test_months_set:
            test_files.append(file_path)
            continue

        # Train split
        if year in train_years:
            train_files.append(file_path)
            continue

        # Not in any target years
        logger.warning(f"Skipping file {filename} - year {year} not in target years")

    return train_files, val_files, test_files


def create_data_loaders(config: DNNConfig, fs: s3fs.S3FileSystem) -> tuple:
    """Create train and validation data loaders"""
    data_config = config.config["data"]
    training_config = config.config["training"]

    # Get file list
    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )

    logger.info(f"Found {len(files)} files")

    # Split files by year (2021=train, 2022=val, 2023=test)
    train_files, val_files, test_files = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Val files: {len(val_files)}")
    logger.info(f"Test files: {len(test_files)}")

    # Create datasets
    patch_size = tuple(data_config["patch_size"]) if data_config["patch_size"] else None
    excluded_columns = data_config.get(
        "excluded_columns", ["time", "latitude", "longitude", "timestamp"]
    )
    target_columns = data_config.get("target_columns", {"vhm0": "corrected_VHM0"})
    predict_bias = data_config.get("predict_bias", False)
    subsample_step = data_config.get("subsample_step", None)

    # normalizer = WaveNormalizer.load_from_s3("medwav-dev-data",data_config["normalizer_path"])
    normalizer = WaveNormalizer.load_from_disk(data_config["normalizer_path"])
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Normalizer stats: {normalizer.stats_}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")
    if data_config.get("patch_size_deactivate", None) is not None:
        train_dataset = GridPatchWaveDataset(
            train_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            use_cache=data_config.get("use_cache", False),
            normalize_target=data_config.get("normalize_target", False),
            min_valid_pixels=data_config.get("min_valid_pixels", 0.3),  # Only keep patches with >30% sea pixels
            max_cache_size=data_config.get("max_cache_size", 20),
            fs=fs
        )
    else:
        train_dataset = CachedWaveDataset(
            train_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=True,
            use_cache=data_config.get("use_cache", False),
            normalize_target=data_config.get("normalize_target", False),
            fs=fs,
            max_cache_size=data_config.get("max_cache_size", 20)
        )

    if data_config.get("patch_size_deactivate", None) is not None:
        val_dataset = GridPatchWaveDataset(
            val_files,
            patch_size=patch_size,
            stride=data_config.get("stride", None),
            excluded_columns=excluded_columns,
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            use_cache=data_config.get("use_cache", False),
            normalize_target=data_config.get("normalize_target", False),
            min_valid_pixels=data_config.get("min_valid_pixels", 0.3),  # Only keep patches with >30% sea pixels
            fs=fs,
            max_cache_size=data_config.get("max_cache_size", 20)
        )
    else:
        val_dataset = CachedWaveDataset(
            val_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=True,
            use_cache=data_config.get("use_cache", False),
            normalize_target=data_config.get("normalize_target", False),
            fs=fs,
            max_cache_size=data_config.get("max_cache_size", 20)
        )

    # Pre-compute wave bins and filter patches (if using patched dataset)
    use_balanced_sampling = data_config.get("use_balanced_sampling", False)  # Set to False for uniform random sampling

    if patch_size is not None:
        # ALWAYS compute bins to filter out invalid patches (regardless of balanced sampling)
        # logger.info("Computing wave bins and filtering invalid patches...")
        # train_dataset.compute_all_bins()
        # logger.info(f"Training dataset after filtering: {len(train_dataset)} patches")

        # Also filter validation dataset
        # val_dataset.compute_all_bins()
        logger.info(f"Validation dataset after filtering: {len(val_dataset)} patches")

        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty after filtering! Check val_year/val_months or lower min_valid_pixels threshold.")

        if use_balanced_sampling:
            logger.info("Using balanced sampling (equal samples per wave height bin)")
        else:
            logger.info("Using uniform random sampling (shuffle=True, no sampler)")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True if patch_size is None or not use_balanced_sampling else False,  # Don't shuffle when using sampler
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        persistent_workers=training_config.get("persistent_workers", training_config["num_workers"] > 0),
        prefetch_factor=training_config["prefetch_factor"],
        sampler=WaveBinBalancedSampler(train_dataset, training_config["batch_size"]) if (patch_size is not None and use_balanced_sampling) else None,
        # timeout=300  # 5 minute timeout for S3 loading
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        persistent_workers=training_config.get("persistent_workers", training_config["num_workers"] > 0),
        prefetch_factor=None,
        sampler=None,
        # timeout=300  # 5 minute timeout for S3 loading
    )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader

def create_callbacks(config: DNNConfig) -> list:
    """Create training callbacks"""
    callbacks = []

    checkpoint_config = config.config["checkpoint"]
    training_config = config.config["training"]

    # Model checkpoint callback
    logger.info("Adding checkpoint callback")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config["checkpoint_dir"],
        filename="{epoch:02d}-{val_loss:.2f}",
        monitor=training_config["monitor"],
        mode=training_config["mode"],
        save_top_k=training_config["save_top_k"],
        save_last=checkpoint_config["save_last"],
        save_on_train_epoch_end=False,  # Only save when validation runs
    )
    callbacks.append(checkpoint_callback)

    # Add S3 sync callback for spot instances
    if checkpoint_config.get("s3_sync_dir"):
        logger.info("Adding S3 sync callback")
        s3_sync_callback = S3CheckpointSyncCallback(
            s3_dir=checkpoint_config["s3_sync_dir"],
            local_dir=checkpoint_config["checkpoint_dir"],
            sync_frequency=checkpoint_config.get("sync_frequency", 5),
        )
        callbacks.append(s3_sync_callback)

    # Early stopping callback (only if patience is set)
    if training_config["early_stopping_patience"] is not None:
        logger.info("Adding Early Stopping callback")
        early_stopping = EarlyStopping(
            monitor=training_config["monitor"],
            patience=training_config["early_stopping_patience"],
            mode=training_config["mode"],
            check_on_train_epoch_end=False,  # Only check when validation runs
        )
        callbacks.append(early_stopping)
    else:
        logger.info("Early stopping disabled - will train for full max_epochs")

    logger.info("Adding Learning Rate Monitor callback")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Add Comet visualization callback if using Comet
    if config.config["logging"]["use_comet"]:
        logger.info("Adding Comet visualization callback")
        comet_callback = CometVisualizationCallback(log_every_n_epochs=1)
        callbacks.append(comet_callback)

    if config.config["training"]["use_swa"]:
        logger.info("Adding SWA callback")
        swa_callback = StochasticWeightAveraging(
            swa_lrs=1e-3,       # Same LR or lower than initial LR
            annealing_epochs=10,
            swa_epoch_start=5        # When to start SWA averaging
        )
        callbacks.append(swa_callback)

    if config.config["training"]["use_ema"]:
        logger.info("Adding EMA callback")
        ema_callback = EMAWeightAveraging(
            decay=0.999,          # Higher decay => slower updates, smoother EMA
            start_step=100,
        )
        callbacks.append(ema_callback)

    if config.config["model"]["loss_type"] == "pixel_switch_mse":
        logger.info("Adding Pixel Switch Threshold callback")
        pixel_switch_threshold_callback = PixelSwitchThresholdCallback(quantile=0.90)
        callbacks.append(pixel_switch_threshold_callback)

    if config.config["training"]["freeze_encoder_layers"]:
        logger.info("Adding Freeze Encoder Layers callback")
        freeze_encoder_callback = FreezeEncoderCallback(aggressive_freeze=config.config["training"]["aggressive_freeze"])
        callbacks.append(freeze_encoder_callback)

    return callbacks


def main():
    # Optimize for Tensor Cores on modern GPUs (fixes the warning)
    torch.set_float32_matmul_precision('medium')

    # Use new API to avoid deprecation warnings
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'

    parser = argparse.ArgumentParser(description="Train DNN for wave height correction")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--data_path", type=str, help="Override data path")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--max_epochs", type=int, help="Override max epochs")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")

    args = parser.parse_args()

    # Load configuration
    config = DNNConfig(args.config)

    # Override with command line arguments
    if args.data_path:
        config.config["data"]["data_path"] = args.data_path
    if args.batch_size:
        config.config["training"]["batch_size"] = args.batch_size
    if args.max_epochs:
        config.config["training"]["max_epochs"] = args.max_epochs
    if args.learning_rate:
        config.config["model"]["learning_rate"] = args.learning_rate
    if args.resume:
        config.config["checkpoint"]["resume_from_checkpoint"] = args.resume

    # Create directories
    os.makedirs(config.config["checkpoint"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config.config["logging"]["log_dir"], exist_ok=True)

    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem()

    # Create data loaders
    logger.info("Creating data loaders...")

    # Check if we should pre-download data for multiprocessing
    if config.config["training"]["num_workers"] > 0 and config.config["data"]["data_path"].startswith("s3://"):
        logger.warning("S3FS detected with num_workers > 0. This may cause issues.")
        logger.warning("Consider setting num_workers=0 or pre-downloading data locally.")

    train_loader, val_loader = create_data_loaders(config, fs)

    # Create model
    model_config = config.config["model"]
    logger.info(f"Creating model with {model_config['in_channels']} input channels")
    logger.info(f"Learning rate: {model_config['learning_rate']}")
    logger.info(f"Loss type: {model_config['loss_type']}")

    # Log LR scheduler configuration
    lr_scheduler_config = model_config.get("lr_scheduler", {})
    if lr_scheduler_config and lr_scheduler_config.get("type", "none") != "none":
        logger.info(f"LR Scheduler: {lr_scheduler_config['type']}")
        if lr_scheduler_config["type"] == "ReduceLROnPlateau":
            logger.info(
                f"  - Monitor: {lr_scheduler_config.get('monitor', 'val_loss')}"
            )
            logger.info(f"  - Patience: {lr_scheduler_config.get('patience', 5)}")
            logger.info(f"  - Factor: {lr_scheduler_config.get('factor', 0.5)}")
    else:
        logger.info("LR Scheduler: None")

    local_predict_bias = config.config.get("data", {}).get("predict_bias", False)

    # Handle checkpoint resuming (local or S3)
    resume_path = config.config["checkpoint"]["resume_from_checkpoint"]
    if resume_path and resume_path.startswith("s3://"):
        # Download S3 checkpoint to local first
        resume_path = download_s3_checkpoint(
            resume_path, config.config["checkpoint"]["checkpoint_dir"]
        )
    logger.info(f"Resume path: {resume_path}")

    finetune_model = config.config["training"]["finetune_model"]
    if finetune_model:
        logger.info("Finetuning model")
        model = WaveBiasCorrector.load_from_checkpoint(
            resume_path,
            loss_type=model_config["loss_type"],
            lr=float(model_config["learning_rate"]),
            lr_scheduler_config=model_config.get("lr_scheduler", {}),
            dropout=model_config.get("dropout", 0.2),
            add_vhm0_residual=model_config.get("add_vhm0_residual", False),
            vhm0_channel_index=model_config.get("vhm0_channel_index", 0),
            weight_decay=float(model_config.get("weight_decay", 0)),
            pixel_switch_threshold_m=model_config.get("pixel_switch_threshold_m", 0.45),
            use_mdn=model_config.get("use_mdn", False),
            optimizer_type=model_config.get("optimizer_type", "Adam"),
            lambda_adv=model_config.get("lambda_adv", 0.01),
            n_discriminator_updates=model_config.get("n_discriminator_updates", 3),
            discriminator_lr_multiplier=model_config.get("discriminator_lr_multiplier", 1.0),
            tasks_config=model_config.get("tasks_config", None),
        )
    else:
        logger.info("Training new model")
        model = WaveBiasCorrector(
            in_channels=model_config["in_channels"],
            lr=float(model_config["learning_rate"]),
            loss_type=model_config["loss_type"],
            lr_scheduler_config=model_config.get("lr_scheduler", {}),
            predict_bias=local_predict_bias,
            filters=model_config["filters"],
            dropout=model_config.get("dropout", 0.2),
            add_vhm0_residual=model_config.get("add_vhm0_residual", False),
            vhm0_channel_index=model_config.get("vhm0_channel_index", 0),
            weight_decay=float(model_config.get("weight_decay", 0)),
            model_type=model_config.get("model_type", "nick"),  # Options: "nick", "geo", "enhanced"
            upsample_mode=model_config.get("upsample_mode", "nearest"),
            pixel_switch_threshold_m=model_config.get("pixel_switch_threshold_m", 0.45),
            use_mdn=model_config.get("use_mdn", False),
            optimizer_type=model_config.get("optimizer_type", "Adam"),
            lambda_adv=model_config.get("lambda_adv", 0.01),
            n_discriminator_updates=model_config.get("n_discriminator_updates", 3),
            discriminator_lr_multiplier=model_config.get("discriminator_lr_multiplier", 1.0),
            tasks_config=model_config.get("tasks_config", None),
        )

    # Create callbacks
    callbacks = create_callbacks(config)

    # Create loggers
    tensorboard_logger = TensorBoardLogger(
        save_dir=config.config["logging"]["log_dir"],
        name=config.config["logging"]["experiment_name"],
    )
    comet_logger = CometLogger(
        api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
        workspace="ioannisgkinis",
        project="hcmr-ai",
        name=config.config["logging"]["experiment_name"],
        tags=config.config["logging"].get("comet_tags", []),
        # Simplified logging options to avoid parameter issues
        log_graph=True,
        auto_metric_logging=True,
        auto_param_logging=True,
    )

    # Log training artifacts (scripts, config, git info)
    if config.config["logging"]["use_comet"]:
        logger.info(f"Comet experiment URL: {comet_logger.experiment.url}")
        _log_training_artifacts(comet_logger, args.config)

        # Log additional experiment metadata
        comet_logger.experiment.log_other("python_version", sys.version)
        comet_logger.experiment.log_other("pytorch_version", torch.__version__)
        comet_logger.experiment.log_other("lightning_version", lightning.__version__)
        comet_logger.experiment.log_other("experiment_type", "wave_height_correction")
        comet_logger.experiment.log_other("model_architecture", "U-Net")
        comet_logger.experiment.log_other(
            "training_data_year", config.config["data"]["train_year"]
        )
        comet_logger.experiment.log_other(
            "validation_data_year", config.config["data"]["val_year"]
        )
        comet_logger.experiment.log_other(
            "target_columns", config.config["data"]["target_columns"]
        )
        comet_logger.experiment.log_other(
            "predict_bias", config.config["data"]["predict_bias"]
        )

    # Use both loggers
    loggers = (
        [tensorboard_logger]
        if not config.config["logging"]["use_comet"]
        else [tensorboard_logger, comet_logger]
    )

    # Create trainer
    training_config = config.config["training"]
    logger.info("Training configuration:")
    logger.info(f"  - Accelerator: {training_config['accelerator']}")
    logger.info(f"  - Devices: {training_config['devices']}")
    logger.info(f"  - Max epochs: {training_config['max_epochs']}")
    logger.info(f"  - Precision: {training_config['precision']}")
    logger.info(f"  - Fast dev run: {training_config['fast_dev_run']}")

    trainer = Trainer(
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
        max_epochs=training_config["max_epochs"],
        precision=training_config["precision"],
        log_every_n_steps=training_config["log_every_n_steps"],
        callbacks=callbacks,
        logger=loggers,
        fast_dev_run=training_config["fast_dev_run"],
        check_val_every_n_epoch=training_config["check_val_every_n_epoch"],
        gradient_clip_val=training_config["gradient_clip_val"],  # Add gradient clipping
        gradient_clip_algorithm=training_config["gradient_clip_algorithm"],
        num_sanity_val_steps=training_config["num_sanity_val_steps"],
        benchmark=training_config["benchmark"],
        val_check_interval=training_config["val_check_interval"],
    )

    # Train model
    logger.info("Starting training...")
    logger.info(f"Training with {len(train_loader)} train batches and {len(val_loader)} val batches")

    # Only pass ckpt_path if we actually have a checkpoint to resume from
    if config.config["training"]["finetune_model"]:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    elif resume_path is not None:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_path)
    else:
        logger.info("Training from scratch (no checkpoint)")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
