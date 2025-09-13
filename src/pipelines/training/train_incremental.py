"""
Script to run different feature engineering experiments with the IncrementalTrainer.

Usage:
    # Normal runs
    python src/pipelines/training/train_incremental.py --config config_baseline_sgd
    python src/pipelines/training/train_incremental.py --config config_elasticnet_sgd
    python src/pipelines/training/train_incremental.py --config config_poly_lite
    python src/pipelines/training/train_incremental.py --config config_poly_selection
    python src/pipelines/training/train_incremental.py --config config_poly_dimred

    # Debug runs (quick testing with minimal data)
    python src/pipelines/training/train_incremental.py --config config_baseline_sgd --debug
    python src/pipelines/training/train_incremental.py --config config_poly_lite --debug --debug-train-days 2 --debug-test-days 1
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.insert(0, str(project_root))
from src.classifiers.inceremental_trainer import IncrementalTrainer
from src.data_engineering.split import time_based_split


def main():
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

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config_dir) / f"{args.config}.yaml"
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Running experiment with config: {args.config}")
    print("Feature configuration:")
    for key, value in config.get("feature_block", {}).items():
        print(f"  {key}: {value}")

    # Initialize trainer
    trainer = IncrementalTrainer(config)

    # Get data files (parquet files contain both input and target data)
    data_files = sorted(glob.glob(os.path.join(trainer.data_dir, "*.parquet")))

    if not data_files:
        print(f"No parquet files found in {trainer.data_dir}")
        sys.exit(1)

    print(f"Found {len(data_files)} parquet files")

    # Use time-based split for proper temporal evaluation
    x_train, y_train, x_test, y_test = time_based_split(
        data_files, data_files,
        train_end_year=2022,
        test_start_year=2023,
        debug_mode=args.debug,
        debug_train_days=args.debug_train_days,
        debug_test_days=args.debug_test_days
    )

    print(f"Training files: {len(x_train)}")
    print(f"Test files: {len(x_test)}")

    # Run experiment
    try:
        trainer.train(x_train, y_train)
        trainer.evaluate(x_test, y_test)
        print("Experiment completed successfully!")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
