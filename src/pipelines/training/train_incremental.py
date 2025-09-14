"""
Script to run different feature engineering experiments with the IncrementalTrainer.

Usage:
    # Normal runs (local data)
    python src/pipelines/training/train_incremental.py --config config_baseline_sgd
    python src/pipelines/training/train_incremental.py --config config_elasticnet_sgd
    python src/pipelines/training/train_incremental.py --config config_poly_lite
    python src/pipelines/training/train_incremental.py --config config_poly_selection
    python src/pipelines/training/train_incremental.py --config config_poly_dimred

    # Debug runs (quick testing with minimal data)
    python src/pipelines/training/train_incremental.py --config config_debug --debug
    python src/pipelines/training/train_incremental.py --config config_debug --debug --debug-train-days 2 --debug-test-days 1

    # S3 runs
    python src/pipelines/training/train_incremental.py --config config_baseline_sgd --s3-bucket medwav-dev-data --s3-prefix parquet/hourly
    python src/pipelines/training/train_incremental.py --config config_baseline_sgd --s3-bucket medwav-dev-data --s3-prefix parquet/hourly --aws-profile myprofile
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from src.classifiers.inceremental_trainer import IncrementalTrainer
from src.commons.aws.utils import list_s3_parquet_files
from src.data_engineering.split import time_based_split


def main():
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Run feature engineering experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file name (without .yaml extension)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="src/configs",
        help="Directory containing configuration files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with only a few days of data"
    )
    parser.add_argument(
        "--debug-train-days",
        type=int,
        default=1,
        help="Number of days to use for training in debug mode"
    )
    parser.add_argument(
        "--debug-test-days",
        type=int,
        default=1,
        help="Number of days to use for testing in debug mode"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="S3 bucket name for data loading (overrides local data_dir)"
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="",
        help="S3 prefix for data files (e.g., 'parquet/hourly/')"
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile to use for S3 access"
    )

    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("STARTING INCREMENTAL TRAINING EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Debug mode: {args.debug}")
    if args.debug:
        logger.info(f"Debug train days: {args.debug_train_days}")
        logger.info(f"Debug test days: {args.debug_test_days}")
    if args.s3_bucket:
        logger.info(f"S3 bucket: {args.s3_bucket}")
        logger.info(f"S3 prefix: {args.s3_prefix}")
        if args.aws_profile:
            logger.info(f"AWS profile: {args.aws_profile}")

    # Load configuration
    config_path = Path(args.config_dir) / f"{args.config}.yaml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Running experiment with config: {args.config}")
    logger.info("Feature configuration:")
    for key, value in config.get("feature_block", {}).items():
        logger.info(f"  {key}: {value}")

    # Initialize trainer
    logger.info("Initializing IncrementalTrainer...")
    trainer = IncrementalTrainer(config)

    # Get data files (parquet files contain both input and target data)
    if args.s3_bucket:
        # Load from S3
        logger.info(f"Loading data from S3 bucket: {args.s3_bucket}")
        if args.s3_prefix:
            logger.info(f"Using S3 prefix: {args.s3_prefix}")
        data_files = list_s3_parquet_files(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            aws_profile=args.aws_profile
        )
        if not data_files:
            logger.error(f"No parquet files found in s3://{args.s3_bucket}/{args.s3_prefix}")
            sys.exit(1)
        logger.info(f"Found {len(data_files)} parquet files in S3")
    else:
        # Load from local directory
        logger.info(f"Loading data from local directory: {trainer.data_dir}")
        data_files = sorted(glob.glob(os.path.join(trainer.data_dir, "*.parquet")))
        if not data_files:
            logger.error(f"No parquet files found in {trainer.data_dir}")
            sys.exit(1)
        logger.info(f"Found {len(data_files)} parquet files locally")

    # Use time-based split for proper temporal evaluation
    x_train, y_train, x_test, y_test = time_based_split(
        data_files, data_files,
        train_end_year=2021,
        test_start_year=2023,
        debug_mode=args.debug,
        debug_train_days=args.debug_train_days,
        debug_test_days=args.debug_test_days
    )

    logger.info(f"Training files: {len(x_train)}")
    logger.info(f"Test files: {len(x_test)}")
    logger.info(f"Training files: {x_train}")
    logger.info(f"Test files: {x_test}")

    # Run experiment
    logger.info("Starting training and evaluation...")
    try:
        trainer.train(x_train, y_train)
        logger.info("Training completed successfully!")
        
        trainer.evaluate(x_test, y_test)
        logger.info("Evaluation completed successfully!")
        
        logger.info("="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
