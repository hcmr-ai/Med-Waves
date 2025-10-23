"""
Training script for full dataset trainer.

This script loads all data into memory and trains models on the complete dataset,
providing better convergence and more robust results for research purposes.
"""

import argparse
import glob
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from comet_ml import Experiment

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.classifiers.model_per_point import ModelPerPointTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_files(
    data_path: str, file_pattern: str = "*.parquet", config: Dict[str, Any] = None
) -> List[str]:
    """
    Get list of data files to process from local or S3 paths.

    Args:
        data_path: Path to data (can be local path or S3 URI)
        file_pattern: File pattern for local directories
        config: Configuration dictionary for month filtering

    Returns:
        List of file paths
    """
    if data_path.startswith("s3://"):
        # Handle S3 URI
        return get_s3_data_files(data_path, config)
    else:
        # Handle local path
        return get_local_data_files(data_path, file_pattern)


def get_s3_data_files(s3_uri: str, config: Dict[str, Any] = None) -> List[str]:
    """Get list of parquet files from S3."""
    from src.commons.aws.utils import list_s3_parquet_files

    # Parse S3 URI
    s3_path = s3_uri[5:]  # Remove 's3://'
    if "/" in s3_path:
        bucket, prefix = s3_path.split("/", 1)
    else:
        bucket = s3_path
        prefix = ""

    # Get year-based filtering configuration
    filter_months = None
    train_end_year = None
    test_start_year = None
    if config:
        split_config = config.get("data", {}).get("split", {})
        filter_months = split_config.get("eval_months", None)
        train_end_year = split_config.get("train_end_year", None)
        test_start_year = split_config.get("test_start_year", None)

    logger.info(f"Listing S3 files in bucket: {bucket}, prefix: {prefix}")
    logger.info(
        f"Year-based filtering: train_end_year={train_end_year}, test_start_year={test_start_year}"
    )
    if filter_months:
        logger.info(f"Month filtering for test years: {filter_months}")

    # List parquet files with year-aware filtering
    parquet_files = list_s3_parquet_files(
        bucket,
        prefix,
        filter_months=filter_months,
        train_end_year=train_end_year,
        test_start_year=test_start_year,
    )

    logger.info(f"Found {len(parquet_files)} parquet files in S3")
    return parquet_files


def get_local_data_files(data_path: str, file_pattern: str = "*.parquet") -> List[str]:
    """Get list of data files from local path."""
    data_path = Path(data_path)

    if data_path.is_file():
        # Single file
        if data_path.suffix == ".parquet":
            return [str(data_path)]
        else:
            raise ValueError(f"File {data_path} is not a parquet file")
    elif data_path.is_dir():
        # Directory - find all matching files
        pattern = str(data_path / file_pattern)
        files = glob.glob(pattern)
        files.sort()  # Sort for consistent ordering
        return files
    else:
        raise ValueError(f"Data path {data_path} does not exist")


def run_experiment(
    config: Dict[str, Any], data_files: List[str], save_path: str
) -> Dict[str, Any]:
    """Run the Model Per point training experiment."""
    logger.info(
        f"Starting Model Per point training experiment with {len(data_files)} files"
    )

    # Initialize trainer
    trainer = ModelPerPointTrainer(config)

    # Load data
    logger.info("Loading data...")
    (
        X,
        y,
        regions,
        coords,
        successful_files,
        actual_wave_heights,
        years,
        months,
        cluster_ids,
    ) = trainer.load_data(data_files, config["data"]["target_column"])
    x_shape = X.shape
    y_shape = y.shape

    # Split data
    logger.info("Splitting data...")
    trainer.split_data(
        X,
        y,
        regions,
        coords,
        successful_files,
        trainer.vhm0_x_raw,
        actual_wave_heights,
        years,
        months,
        cluster_ids,
    )

    # 🚀 MEMORY OPTIMIZATION: Delete original data after splitting
    del X, y, regions, coords, actual_wave_heights, years, months, cluster_ids
    import gc

    gc.collect()
    logger.info("Freed original data after splitting")

    # Preprocess data
    logger.info("Preprocessing data...")
    trainer.preprocess_data()

    # Train model
    logger.info("Training+Evaluating model...")
    training_results, evaluation_results = trainer.train()

    # # Evaluate model
    # logger.info("Evaluating model...")
    # evaluation_results = trainer.evaluate()

    # Save model
    model_save_path = config["output"]["model_save_path"]
    # trainer.save_model(model_save_path)

    # Compile results
    results = {
        "experiment_name": config["output"]["experiment_name"],
        "timestamp": datetime.now().isoformat(),
        "data_files_count": len(successful_files),
        "data_shape": {"X": x_shape, "y": y_shape},
        "training_results": training_results,
        "evaluation_results": evaluation_results,
        "model_config": config["model"],
        "data_config": config["data"],
    }

    # Save results locally
    save_results(
        convert_numpy_types(results), save_path, trainer.experiment_logger.experiment
    )

    # Save results to S3 if enabled
    s3_config = config["output"].get("s3", {})
    if s3_config.get("enabled", False):
        logger.info("Saving results to S3...")
        plots_dir = config.get("diagnostics", {}).get(
            "plots_save_path", "diagnostic_plots"
        )
        s3_upload_results = trainer.save_results_to_s3(
            convert_numpy_types(results), model_save_path, plots_dir
        )

        # Log S3 URLs
        if s3_upload_results:
            s3_files = trainer.s3_results_saver.list_experiment_files()
            logger.info(f"S3 experiment files: {len(s3_files)} files uploaded")
            for s3_key in s3_files[:5]:  # Show first 5 files
                s3_url = trainer.s3_results_saver.get_s3_url(s3_key)
                logger.info(f"  - {s3_url}")
            if len(s3_files) > 5:
                logger.info(f"  ... and {len(s3_files) - 5} more files")

    # End Comet experiment
    trainer.end_experiment()

    return results


def save_results(
    results: Dict[str, Any], save_path: str, comet_experiment: Experiment = None
) -> None:
    """Save experiment results."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    import json

    results_file = save_path / f"{results['experiment_name']}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_file = save_path / f"{results['experiment_name']}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment: {results['experiment_name']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Data files: {results['data_files_count']}\n")
        f.write(f"Data shape: {results['data_shape']}\n")
        f.write("\nTraining Results:\n")
        f.write(
            f"  Train RMSE: {results['training_results']['train_metrics']['rmse']:.4f}\n"
        )
        f.write(
            f"  Train MAE: {results['training_results']['train_metrics']['mae']:.4f}\n"
        )
        f.write(
            f"  Train Pearson: {results['training_results']['train_metrics']['pearson']:.4f}\n"
        )
        f.write(
            f"  Val RMSE: {results['training_results']['val_metrics']['rmse']:.4f}\n"
        )
        f.write(f"  Val MAE: {results['training_results']['val_metrics']['mae']:.4f}\n")
        f.write(
            f"  Val Pearson: {results['training_results']['val_metrics']['pearson']:.4f}\n"
        )
        f.write("\nTest Results:\n")
        f.write(
            f"  Test RMSE: {results['evaluation_results']['test_metrics']['rmse']:.4f}\n"
        )
        f.write(
            f"  Test MAE: {results['evaluation_results']['test_metrics']['mae']:.4f}\n"
        )
        f.write(
            f"  Test Pearson: {results['evaluation_results']['test_metrics']['pearson']:.4f}\n"
        )

    logger.info(f"Results saved to {save_path}")
    if comet_experiment is not None:
        comet_experiment.log_asset_folder(str(save_path))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model on full dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    # parser.add_argument("--data-path", type=str, required=True,
    #                    help="Path to data files (local path or S3 URI like s3://bucket/prefix)")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.parquet",
        help="File pattern for local data files (ignored for S3)",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help="AWS profile to use for S3 access (optional)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override S3 settings if provided
    if args.aws_profile:
        config["data"]["s3"]["aws_profile"] = args.aws_profile

    # Get data files
    data_files = get_data_files(
        config["data"]["data_path"], config["data"]["file_pattern"], config
    )

    if not data_files:
        raise ValueError(
            f"No data files found in {args.data_path} with pattern {args.file_pattern}"
        )

    logger.info(f"Found {len(data_files)} data files")

    # Run experiment
    results = run_experiment(config, data_files, args.output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Experiment: {results['experiment_name']}")
    print(f"Data files: {results['data_files_count']}")
    print(f"Data shape: {results['data_shape']}")
    print("\nTraining Results:")
    print(f"  Train RMSE: {results['training_results']['train_metrics']['rmse']:.4f}")
    print(f"  Train MAE: {results['training_results']['train_metrics']['mae']:.4f}")
    print(
        f"  Train Pearson: {results['training_results']['train_metrics']['pearson']:.4f}"
    )
    print(f"  Val RMSE: {results['training_results']['val_metrics']['rmse']:.4f}")
    print(f"  Val MAE: {results['training_results']['val_metrics']['mae']:.4f}")
    print(f"  Val Pearson: {results['training_results']['val_metrics']['pearson']:.4f}")

    # Print regional training metrics
    if (
        "regional_train_metrics" in results["training_results"]
        and results["training_results"]["regional_train_metrics"]
    ):
        print("\nRegional Training Metrics:")
        for region, metrics in results["training_results"][
            "regional_train_metrics"
        ].items():
            # Handle both integer region IDs and string region names
            if isinstance(region, (int, np.integer)):
                from src.commons.region_mapping import RegionMapper

                region_name = RegionMapper.get_display_name(region)
            else:
                region_name = region.title()
            print(
                f"  {region_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Pearson: {metrics['pearson']:.4f}"
            )

    # Print regional validation metrics
    if (
        "regional_val_metrics" in results["training_results"]
        and results["training_results"]["regional_val_metrics"]
    ):
        print("\nRegional Validation Metrics:")
        for region, metrics in results["training_results"][
            "regional_val_metrics"
        ].items():
            # Handle both integer region IDs and string region names
            if isinstance(region, (int, np.integer)):
                from src.commons.region_mapping import RegionMapper

                region_name = RegionMapper.get_display_name(region)
            else:
                region_name = region.title()
            print(
                f"  {region_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Pearson: {metrics['pearson']:.4f}"
            )
    print("\nTest Results:")
    print(f"  Test RMSE: {results['evaluation_results']['test_metrics']['rmse']:.4f}")
    print(f"  Test MAE: {results['evaluation_results']['test_metrics']['mae']:.4f}")
    print(
        f"  Test Pearson: {results['evaluation_results']['test_metrics']['pearson']:.4f}"
    )

    # Print regional test metrics
    if (
        "regional_test_metrics" in results["evaluation_results"]
        and results["evaluation_results"]["regional_test_metrics"]
    ):
        print("\nRegional Test Metrics:")
        for region_id, metrics in results["evaluation_results"][
            "regional_test_metrics"
        ].items():
            # Convert region ID to region name
            from src.data_engineering.feature_engineer import RegionMapper

            region_name = RegionMapper.get_display_name(region_id)
            print(
                f"  {region_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Pearson: {metrics['pearson']:.4f}"
            )

    # Print sea-bin test metrics
    if (
        "sea_bin_test_metrics" in results["evaluation_results"]
        and results["evaluation_results"]["sea_bin_test_metrics"]
    ):
        print("\nSea-Bin Test Metrics:")
        for bin_name, metrics in results["evaluation_results"][
            "sea_bin_test_metrics"
        ].items():
            count = metrics.get("count", 0)
            percentage = metrics.get("percentage", 0)
            print(
                f"  {bin_name.title()} ({count:,} samples, {percentage:.1f}%) - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Pearson: {metrics['pearson']:.4f}"
            )

    print("=" * 50)


if __name__ == "__main__":
    main()
