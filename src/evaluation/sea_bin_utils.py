"""
Utility functions for sea-bin metrics calculation.
Shared between training and evaluation pipelines.
"""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def calculate_sea_bin_metrics(y_true: np.ndarray, y_pred: np.ndarray, sea_bin_config: Dict[str, Any], enable_logging: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different sea state bins based on wave height.
    
    Args:
        y_true: Actual wave heights
        y_pred: Predicted wave heights
        sea_bin_config: Sea-bin configuration dictionary
        enable_logging: Whether to enable logging (default: False for evaluation)
        
    Returns:
        Dictionary with sea-bin metrics
    """
    from src.evaluation.metrics import evaluate_model
    
    sea_bin_metrics = {}
    
    if not sea_bin_config.get("enabled", False):
        return sea_bin_metrics
    
    bins = sea_bin_config.get("bins", [])
    if not bins:
        return sea_bin_metrics
    
    if enable_logging:
        logger.info("Calculating sea-bin performance metrics...")
    
    for bin_config in bins:
        bin_name = bin_config["name"]
        bin_min = bin_config["min"]
        bin_max = bin_config["max"]
        bin_description = bin_config.get("description", "")
        
        # Filter data for this sea state bin
        mask = (y_true >= bin_min) & (y_true < bin_max)
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_y_true = y_true[mask]
            bin_y_pred = y_pred[mask]
            
            # Calculate metrics for this sea state bin using evaluate_model
            bin_metrics = evaluate_model(bin_y_pred, bin_y_true)  # Note: evaluate_model expects (y_pred, y_true)
            bin_metrics["count"] = bin_count
            bin_metrics["percentage"] = (bin_count / len(y_true)) * 100
            sea_bin_metrics[bin_name] = bin_metrics
            
            if enable_logging:
                logger.info(f"{bin_name.title()} ({bin_description}) - Count: {bin_count:,} ({bin_metrics['percentage']:.1f}%), RMSE: {bin_metrics['rmse']:.4f}, MAE: {bin_metrics['mae']:.4f}")
        else:
            if enable_logging:
                logger.info(f"{bin_name.title()} ({bin_description}) - No samples in this range")
    
    return sea_bin_metrics
