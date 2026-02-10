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
from commons.helpers import DNNConfig
from commons.dataloaders import create_data_loaders
# from commons.datasets.grid_patched_dataset import GridPatchWaveDataset
# from commons.datasets.time_step_patch_dataset import TimestepPatchWaveDataset, PatchSamplingConfig
# from commons.datasets.samplers import WaveBinBalancedSampler
# from commons.preprocessing.bu_net_preprocessing import WaveNormalizer
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
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode (slower but fully reproducible)")

    args = parser.parse_args()

    # Load configuration
    config = DNNConfig(args.config)

    # Set random seed for reproducibility
    random_seed = config.config["data"].get("random_seed", 42)
    lightning.seed_everything(random_seed, workers=True)
    logger.info(f"Set random seed to {random_seed} for reproducible training")

    # Enable deterministic mode if requested (slower but fully reproducible)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("Deterministic mode enabled (training will be slower but fully reproducible)")
    else:
        logger.info("Using non-deterministic optimizations for faster training")

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

    train_loader, val_loader, normalizer = create_data_loaders(config, fs)

    # Create model
    model_config = config.config["model"]
    data_config = config.config["data"]
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
            normalizer=normalizer,
            normalize_target=data_config.get("normalize_target", False),
            use_patch_sampling=data_config.get("use_patch_sampling", False),
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
            normalizer=normalizer,
            normalize_target=data_config.get("normalize_target", False),
            use_patch_sampling=data_config.get("use_patch_sampling", False),
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
