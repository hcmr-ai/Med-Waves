"""
MetricsCalculator for Wave Height Bias Correction Research

This module provides the MetricsCalculator class for handling all metrics calculations
including regional, sea-bin, and spatial metrics in a modular and reusable way.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.evaluation.metrics import evaluate_model
from src.evaluation.sea_bin_utils import calculate_sea_bin_metrics
from src.data_engineering.feature_engineer import RegionMapper

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Handles all metrics calculations for model evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MetricsCalculator with configuration."""
        self.config = config
        self.feature_config = config.get("feature_block", {})
        self.evaluation_config = config.get("evaluation", {})
        self.logger = logging.getLogger(__name__)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic metrics (RMSE, MAE, bias, Pearson, SNR).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with basic metrics
        """
        return evaluate_model(y_true, y_pred)
    
    def calculate_regional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, regions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics per region for monitoring regional performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            regions: Region information for each sample
            
        Returns:
            Dictionary with metrics per region
        """
        regional_metrics = {}
        
        for region_id in np.unique(regions):
            mask = regions == region_id
            if np.sum(mask) > 0:
                region_y_true = y_true[mask]
                region_y_pred = y_pred[mask]
                
                # Calculate metrics for this region
                region_metrics = evaluate_model(region_y_true, region_y_pred)
                regional_metrics[region_id] = region_metrics
                
                region_name = RegionMapper.get_display_name(region_id)
                self.logger.info(f"{region_name} metrics - RMSE: {region_metrics['rmse']:.4f}, "
                              f"MAE: {region_metrics['mae']:.4f}, Pearson: {region_metrics['pearson']:.4f}")
        
        return regional_metrics
    
    def calculate_sea_bin_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, enable_logging: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different sea state bins based on wave height.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            enable_logging: Whether to enable detailed logging
            
        Returns:
            Dictionary with metrics per sea bin
        """
        # Get sea-bin configuration
        sea_bin_config = self.feature_config.get("sea_bin_metrics", {})
        
        # Use the shared utility function
        return calculate_sea_bin_metrics(y_true, y_pred, sea_bin_config, enable_logging=enable_logging)
    
    def calculate_spatial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, coords: np.ndarray, 
                                 grid_resolution: float = 1.0) -> Dict[str, Any]:
        """
        Calculate spatial metrics by aggregating errors over geographic grid.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            coords: Coordinate information (lat, lon) for each sample
            grid_resolution: Grid resolution in degrees
            
        Returns:
            Dictionary with spatial metrics
        """
        if coords is None or len(coords) == 0:
            self.logger.warning("No coordinate information available for spatial metrics")
            return {}
        
        # Calculate errors
        errors = y_pred - y_true
        rmse_errors = (y_pred - y_true) ** 2
        
        # Create spatial grid
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        # Define grid bounds
        lat_min, lat_max = np.floor(lats.min()), np.ceil(lats.max())
        lon_min, lon_max = np.floor(lons.min()), np.ceil(lons.max())
        
        # Create grid
        lat_grid = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        spatial_metrics = {
            'grid_resolution': grid_resolution,
            'lat_bounds': [float(lat_min), float(lat_max)],
            'lon_bounds': [float(lon_min), float(lon_max)],
            'grid_shape': [len(lat_grid), len(lon_grid)],
            'rmse_grid': np.zeros((len(lat_grid), len(lon_grid))),
            'mae_grid': np.zeros((len(lat_grid), len(lon_grid))),
            'bias_grid': np.zeros((len(lat_grid), len(lon_grid))),
            'pearson_grid': np.zeros((len(lat_grid), len(lon_grid))),
            'sample_count_grid': np.zeros((len(lat_grid), len(lon_grid)), dtype=int)
        }
        
        # Calculate metrics for each grid cell
        for i, lat_center in enumerate(lat_grid):
            for j, lon_center in enumerate(lon_grid):
                # Find samples in this grid cell
                lat_mask = (lats >= lat_center - grid_resolution/2) & (lats < lat_center + grid_resolution/2)
                lon_mask = (lons >= lon_center - grid_resolution/2) & (lons < lon_center + grid_resolution/2)
                cell_mask = lat_mask & lon_mask
                
                if np.sum(cell_mask) > 0:
                    cell_y_true = y_true[cell_mask]
                    cell_y_pred = y_pred[cell_mask]
                    
                    # Calculate metrics for this cell
                    cell_metrics = evaluate_model(cell_y_true, cell_y_pred)
                    
                    spatial_metrics['rmse_grid'][i, j] = cell_metrics['rmse']
                    spatial_metrics['mae_grid'][i, j] = cell_metrics['mae']
                    spatial_metrics['bias_grid'][i, j] = cell_metrics['bias']
                    spatial_metrics['pearson_grid'][i, j] = cell_metrics['pearson']
                    spatial_metrics['sample_count_grid'][i, j] = len(cell_y_true)
        
        return spatial_metrics
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      regions: Optional[np.ndarray] = None,
                                      coords: Optional[np.ndarray] = None,
                                      enable_logging: bool = False) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics including basic, regional, sea-bin, and spatial metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            regions: Region information for each sample (optional)
            coords: Coordinate information for each sample (optional)
            enable_logging: Whether to enable detailed logging
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['basic'] = self.calculate_basic_metrics(y_true, y_pred)
        
        # Regional metrics (if regions available)
        if regions is not None:
            metrics['regional'] = self.calculate_regional_metrics(y_true, y_pred, regions)
        
        # Sea-bin metrics
        metrics['sea_bin'] = self.calculate_sea_bin_metrics(y_true, y_pred, enable_logging)
        
        # Spatial metrics (if coordinates available)
        if coords is not None:
            grid_resolution = self.evaluation_config.get("spatial_grid_resolution", 1.0)
            metrics['spatial'] = self.calculate_spatial_metrics(y_true, y_pred, coords, grid_resolution)
        
        return metrics
    
    def log_metrics_summary(self, metrics: Dict[str, Any], stage: str = "Evaluation"):
        """
        Log a summary of calculated metrics.
        
        Args:
            metrics: Dictionary with calculated metrics
            stage: Stage name for logging (e.g., "Training", "Validation", "Test")
        """
        basic = metrics.get('basic', {})
        
        self.logger.info(f"{stage} Metrics Summary:")
        self.logger.info(f"  RMSE: {basic.get('rmse', 0):.4f}")
        self.logger.info(f"  MAE:  {basic.get('mae', 0):.4f}")
        self.logger.info(f"  Bias: {basic.get('bias', 0):.4f}")
        self.logger.info(f"  Pearson: {basic.get('pearson', 0):.4f}")
        self.logger.info(f"  SNR: {basic.get('snr', 0):.1f} ({basic.get('snr_db', 0):.1f} dB)")
        
        # Log regional metrics if available
        if 'regional' in metrics:
            self.logger.info(f"  Regional analysis: {len(metrics['regional'])} regions")
        
        # Log sea-bin metrics if available
        if 'sea_bin' in metrics:
            self.logger.info(f"  Sea-bin analysis: {len(metrics['sea_bin'])} bins")
        
        # Log spatial metrics if available
        if 'spatial' in metrics:
            spatial = metrics['spatial']
            self.logger.info(f"  Spatial analysis: {spatial['grid_shape']} grid, {spatial['grid_resolution']}° resolution")
    
    def get_metrics_for_logging(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics suitable for experiment logging (flattened structure).
        
        Args:
            metrics: Dictionary with calculated metrics
            
        Returns:
            Flattened dictionary suitable for logging
        """
        logging_metrics = {}
        
        # Basic metrics
        basic = metrics.get('basic', {})
        for key, value in basic.items():
            logging_metrics[f"basic_{key}"] = value
        
        # Regional metrics (average across regions)
        if 'regional' in metrics:
            regional = metrics['regional']
            if regional:
                # Calculate average regional metrics
                regional_rmse = np.mean([r['rmse'] for r in regional.values()])
                regional_mae = np.mean([r['mae'] for r in regional.values()])
                regional_bias = np.mean([r['bias'] for r in regional.values()])
                regional_pearson = np.mean([r['pearson'] for r in regional.values()])
                
                logging_metrics['regional_avg_rmse'] = regional_rmse
                logging_metrics['regional_avg_mae'] = regional_mae
                logging_metrics['regional_avg_bias'] = regional_bias
                logging_metrics['regional_avg_pearson'] = regional_pearson
        
        # Sea-bin metrics (count of bins with data)
        if 'sea_bin' in metrics:
            sea_bin = metrics['sea_bin']
            logging_metrics['sea_bin_count'] = len(sea_bin)
            
            # Add metrics for each sea bin
            for bin_name, bin_metrics in sea_bin.items():
                for metric_name, metric_value in bin_metrics.items():
                    logging_metrics[f"sea_bin_{bin_name}_{metric_name}"] = metric_value
        
        return logging_metrics
