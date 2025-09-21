"""
ExperimentLogger class for handling Comet ML experiment logging.

This class handles all Comet ML logging functionality that was previously
embedded in the FullDatasetTrainer class.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Handles Comet ML experiment logging and tracking."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ExperimentLogger.
        
        Args:
            config: Configuration dictionary containing logging settings
        """
        self.config = config
        self.logging_config = config.get("logging", {})
        self.experiment: Optional[Experiment] = None
        self._init_comet_experiment()
    
    def _init_comet_experiment(self):
        """Initialize Comet ML experiment."""
        try:
            if self.logging_config.get("use_comet", False) and Experiment is not None:
                # Get experiment name from config or use default
                experiment_name = self.config.get("output", {}).get("experiment_name", "full_dataset_training")
                experiment_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{experiment_name}"
                
                # Initialize Comet experiment
                self.experiment = Experiment(
                    api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                    project_name=self.logging_config.get("project_name", "hcmr-ai"),
                    workspace=self.logging_config.get("workspace", "ioannisgkinis"),
                    experiment_name=experiment_name,
                    auto_param_logging=False,
                    auto_metric_logging=False,
                )
                self.experiment.set_name(experiment_name)
            
                logger.info(f"Comet ML experiment initialized: {experiment_name}")
                
                # Log experiment parameters
                self.log_experiment_params()
            else:
                logger.info("Comet ML disabled or not available")
                
        except Exception as e:
            logger.warning(f"Failed to initialize Comet ML: {e}")
            self.experiment = None
    
    def log_experiment_params(self):
        """Log experiment parameters to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log model parameters
            self.experiment.log_parameters(self.config.get("model", {}))
            
            # Log data configuration
            self.experiment.log_parameters(self.config.get("data", {}), prefix="data")
            
            # Log feature configuration
            self.experiment.log_parameters(self.config.get("features", {}), prefix="features")
            
            # Log evaluation configuration
            self.experiment.log_parameters(self.config.get("evaluation", {}), prefix="evaluation")
            
            # Log diagnostics configuration
            self.experiment.log_parameters(self.config.get("diagnostics", {}), prefix="diagnostics")
            
            # Log system info
            self.experiment.log_parameter("python_version", sys.version)
            
            logger.info("Experiment parameters logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log parameters to Comet ML: {e}")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """Log dataset information to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log dataset metrics
            self.experiment.log_metric("dataset_total_samples", dataset_info.get("total_samples", 0))
            self.experiment.log_metric("dataset_total_features", dataset_info.get("total_features", 0))
            self.experiment.log_metric("dataset_files_processed", dataset_info.get("files_processed", 0))
            
            # Log sampling info
            sampling_info = dataset_info.get("sampling_info", {})
            if sampling_info:
                self.experiment.log_metric("sampling_max_per_file", sampling_info.get("max_per_file", 0))
                self.experiment.log_metric("sampling_reduction_percent", sampling_info.get("reduction_percent", 0))
                self.experiment.log_parameter("sampling_strategy", sampling_info.get("strategy", "none"))
            
            # Log target statistics
            target_stats = dataset_info.get("target_stats", {})
            if target_stats:
                self.experiment.log_metric("target_mean", target_stats.get("mean", 0))
                self.experiment.log_metric("target_std", target_stats.get("std", 0))
                self.experiment.log_metric("target_min", target_stats.get("min", 0))
                self.experiment.log_metric("target_max", target_stats.get("max", 0))
            
            logger.info("Dataset information logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log dataset info to Comet ML: {e}")
    
    def log_split_info(self, split_info: Dict[str, Any]):
        """Log data split information to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log split sizes
            self.experiment.log_metric("split_train_samples", split_info.get("train_samples", 0))
            self.experiment.log_metric("split_val_samples", split_info.get("val_samples", 0))
            self.experiment.log_metric("split_test_samples", split_info.get("test_samples", 0))
            
            # Log split ratios
            self.experiment.log_metric("split_train_ratio", split_info.get("train_ratio", 0))
            self.experiment.log_metric("split_val_ratio", split_info.get("val_ratio", 0))
            self.experiment.log_metric("split_test_ratio", split_info.get("test_ratio", 0))
            
            logger.info("Split information logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log split info to Comet ML: {e}")
    
    def log_training_results(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training results to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log training metrics
            for metric, value in train_metrics.items():
                self.experiment.log_metric(f"train_{metric}", value)
            
            # Log validation metrics
            for metric, value in val_metrics.items():
                self.experiment.log_metric(f"val_{metric}", value)
            
            logger.info("Training results logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log training results to Comet ML: {e}")
    
    def log_evaluation_results(self, test_metrics: Dict[str, float]):
        """Log evaluation results to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log test metrics with eval_ prefix
            for metric, value in test_metrics.items():
                self.experiment.log_metric(f"eval_{metric}", value)
            
            logger.info("Evaluation results logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation results to Comet ML: {e}")
    
    def log_diagnostic_plots(self, plots_dir: Path):
        """Log diagnostic plots to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log all plot files
            for plot_file in plots_dir.glob("*.png"):
                self.experiment.log_image(str(plot_file), name=plot_file.stem)
            
            logger.info("Diagnostic plots logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log diagnostic plots to Comet ML: {e}")
    
    def log_model_artifacts(self, save_path: Path):
        """Log model artifacts to Comet ML."""
        if self.experiment is None or self.experiment.disabled:
            return
        
        try:
            # Log the entire model directory
            self.experiment.log_asset_folder(str(save_path), log_file_name="model_artifacts")
            
            logger.info("Model artifacts logged to Comet ML")
            
        except Exception as e:
            logger.warning(f"Failed to log model artifacts to Comet ML: {e}")
    
    def end_experiment(self):
        """End the Comet ML experiment."""
        if self.experiment is not None and not self.experiment.disabled:
            try:
                self.experiment.end()
                logger.info("Comet ML experiment ended")
            except Exception as e:
                logger.warning(f"Failed to end Comet ML experiment: {e}")
    
    @property
    def is_enabled(self) -> bool:
        """Check if Comet ML logging is enabled."""
        return self.experiment is not None and not self.experiment.disabled
