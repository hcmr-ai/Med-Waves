#!/usr/bin/env python3
"""
Comprehensive evaluation script for WaveBiasCorrector model.
Provides detailed metrics, visualizations, and sea-bin analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))  

# Import your model and dataset classes
from src.classifiers.bu_net import WaveBiasCorrector
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer
from src.pipelines.training.dnn_trainer import DNNConfig, get_file_list, split_files_by_year
from src.commons.dataloaders import CachedWaveDataset, GridPatchWaveDataset
import logging

logger = logging.getLogger(__name__)

def plot_spatial_rmse_map(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    rmse_data: np.ndarray,
    save_path: str,
    title: str,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "YlOrRd",
):
    """
    Plot a spatial RMSE heatmap with coastlines and proper projection.
    
    Parameters
    ----------
    lat_grid : np.ndarray (H, W)
        2D array of latitudes
    lon_grid : np.ndarray (H, W)
        2D array of longitudes
    rmse_data : np.ndarray (H, W)
        2D array of RMSE values
    save_path : str
        File path to save the plot
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits
    cmap : str
        Colormap name
    """
    import cartopy.crs as ccrs
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines and geographic features
    ax.coastlines(resolution='10m', linewidth=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                 linewidth=0.5, alpha=0.5)
    
    # Plot the RMSE heatmap
    im = ax.pcolormesh(
        lon_grid, 
        lat_grid, 
        rmse_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", 
                       label='RMSE (m)', pad=0.05, shrink=0.8)
    
    # Set extent to data bounds
    ax.set_extent([
        np.nanmin(lon_grid),
        np.nanmax(lon_grid),
        np.nanmin(lat_grid),
        np.nanmax(lat_grid)
    ], crs=ccrs.PlateCarree())
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}")

def load_coordinates_from_parquet(file_path, subsample_step=None):
    """Load latitude and longitude coordinates from a parquet file.
    
    Args:
        file_path: Path to parquet file (can be S3 path with or without s3:// prefix)
        subsample_step: Subsampling step size
        
    Returns:
        lat_grid: 2D array of latitudes (H, W)
        lon_grid: 2D array of longitudes (H, W)
    """
    import pyarrow.parquet as pq
    import s3fs
    
    # Check if it's an S3 path (with or without s3:// prefix)
    is_s3 = file_path.startswith("s3://") or not file_path.startswith("/")
    
    if is_s3:
        # Ensure s3:// prefix for s3fs
        if not file_path.startswith("s3://"):
            file_path = f"s3://{file_path}"
        
        # Use s3fs to open the file
        fs = s3fs.S3FileSystem()
        with fs.open(file_path, "rb") as f:
            table = pq.read_table(f)
    else:
        # Local file
        table = pq.read_table(file_path)
    
    # Extract coordinate columns
    lat_data = table.column("latitude").to_numpy()
    lon_data = table.column("longitude").to_numpy()
    
    # Get unique sorted coordinates
    unique_lats = np.unique(lat_data)
    unique_lons = np.unique(lon_data)
    if subsample_step is not None and subsample_step > 1:
        unique_lats = unique_lats[::subsample_step]
        unique_lons = unique_lons[::subsample_step]
    
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(unique_lons, unique_lats)
    
    return lat_grid, lon_grid

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
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.predict_bias = predict_bias
        self.normalizer = normalizer
        self.normalize_target = normalize_target
        # Sea-bin definitions
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
            {"name": "extreme_10_11", "min": 10.0, "max": 11.0, "label": "10.0-11.0m"},
            {"name": "extreme_11_12", "min": 11.0, "max": 12.0, "label": "11.0-12.0m"},
            {"name": "extreme_12_13", "min": 12.0, "max": 13.0, "label": "12.0-13.0m"},
            {"name": "extreme_13_14", "min": 13.0, "max": 14.0, "label": "13.0-14.0m"},
            {"name": "extreme_14_15", "min": 14.0, "max": 15.0, "label": "14.0-15.0m"},
        ]
        self.test_files = test_files
        self.subsample_step = subsample_step
        
        # Add spatial accumulators for RMSE maps
        self.spatial_errors_model = []  # Store (error_map, count_map) for each batch
        self.spatial_errors_baseline = []

        # Initialize accumulators for incremental computation
        self._reset_accumulators()
        
        # Sample storage for plots (optional, limited size)
        self.plot_samples = {
            'y_true': [],
            'y_pred': [],
            'y_uncorrected': [],
        }
    
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
                'count': 0,
                'sum_mae': 0.0,
                'sum_mse': 0.0,
                'sum_bias': 0.0,
                'sum_baseline_mae': 0.0,
                'sum_baseline_mse': 0.0,
                'sum_baseline_bias': 0.0,
            }
            for bin_config in self.sea_bins
        }
        
        # Store error samples for distribution plots
        self.sea_bin_error_samples = {
            bin_config["name"]: {
                'model_errors': [],
                'baseline_errors': [],
            }
            for bin_config in self.sea_bins
        }
        
        self.spatial_rmse_accumulators = {}
    
    def _reconstruct_wave_heights(self, bias: torch.Tensor, vhm0: torch.Tensor) -> torch.Tensor:
        """Reconstruct full wave heights from bias: corrected = vhm0 + bias"""
        return bias + vhm0 
    
    def _process_batch(self, y, mask, vhm0, y_pred):
        """Process a single batch and update accumulators."""
        # Align dimensions
        min_h = min(y_pred.shape[2], y.shape[2])
        min_w = min(y_pred.shape[3], y.shape[3])
        y_pred = y_pred[:, :, :min_h, :min_w]
        y = y[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]
        
        vhm0 = vhm0[:, :, :min_h, :min_w]

        if self.normalize_target and self.normalizer is not None:
            y_pred = self.normalizer.inverse_transform_torch(y_pred)
            y = self.normalizer.inverse_transform_torch(y)
        
        # ========== COMPUTE SPATIAL ERROR MAPS FIRST (using full 4D tensors) ==========
        if self.predict_bias:
            # Reconstruct full wave heights (4D tensors)
            y_pred_full = self._reconstruct_wave_heights(y_pred, vhm0)
            y_true_full = self._reconstruct_wave_heights(y, vhm0)
            y_baseline_full = vhm0  # Baseline is just vhm0
            
            error_map = ((y_pred_full - y_true_full) ** 2).cpu().numpy()  # (N, C, H, W)
            error_map_mae = (y_pred_full - y_true_full).abs().cpu().numpy()  # (N, C, H, W)
            error_map_baseline = ((y_baseline_full - y_true_full) ** 2).cpu().numpy()  # (N, C, H, W)
            error_map_baseline_mae = (y_baseline_full - y_true_full).abs().cpu().numpy()  # (N, C, H, W)
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
        self.spatial_errors_model.append({
            'error_sq': error_map.sum(axis=(0, 1)),  # (H, W)
            'error_sq_mae': error_map_mae.sum(axis=(0, 1)),  # (H, W)
            'count': count_map.sum(axis=(0, 1))  # (H, W)
        })
        
        if error_map_baseline is not None:
            self.spatial_errors_baseline.append({
                'error_sq': error_map_baseline.sum(axis=(0, 1)),  # (H, W)
                'error_sq_mae': error_map_baseline_mae.sum(axis=(0, 1)),  # (H, W)
                'count': count_map.sum(axis=(0, 1))  # (H, W)
            })

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
            self.sum_mse += np.sum(errors ** 2)
            self.sum_bias += np.sum(errors)
            
            # Baseline metrics
            if y_uncorrected is not None:
                y_uncorrected_np = y_uncorrected.cpu().numpy()
                baseline_errors = y_uncorrected_np - y_true_np
                self.sum_baseline_mae += np.sum(np.abs(baseline_errors))
                self.sum_baseline_mse += np.sum(baseline_errors ** 2)
                self.sum_baseline_bias += np.sum(baseline_errors)
            
            # For R² and correlation
            self.sum_y_true += np.sum(y_true_np)
            self.sum_y_true_sq += np.sum(y_true_np ** 2)
            self.sum_y_pred += np.sum(y_pred_np)
            self.sum_y_pred_sq += np.sum(y_pred_np ** 2)
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
                    
                    self.sea_bin_accumulators[bin_name]['count'] += bin_count
                    self.sea_bin_accumulators[bin_name]['sum_mae'] += np.sum(np.abs(bin_errors))
                    self.sea_bin_accumulators[bin_name]['sum_mse'] += np.sum(bin_errors ** 2)
                    self.sea_bin_accumulators[bin_name]['sum_bias'] += np.sum(bin_errors)
                    
                    # Store all error samples
                    self.sea_bin_error_samples[bin_name]['model_errors'].extend(bin_errors.tolist())
                    
                    if y_uncorrected is not None:
                        bin_y_uncorrected = y_uncorrected_np[bin_mask]
                        baseline_bin_errors = bin_y_uncorrected - bin_y_true
                        self.sea_bin_accumulators[bin_name]['sum_baseline_mae'] += np.sum(np.abs(baseline_bin_errors))
                        self.sea_bin_accumulators[bin_name]['sum_baseline_mse'] += np.sum(baseline_bin_errors ** 2)
                        self.sea_bin_accumulators[bin_name]['sum_baseline_bias'] += np.sum(baseline_bin_errors)
                        
                        # Store all baseline error samples
                        self.sea_bin_error_samples[bin_name]['baseline_errors'].extend(baseline_bin_errors.tolist())
            
            # Store samples for plotting (limited)
            self.plot_samples['y_true'].extend(y_true_np)
            self.plot_samples['y_pred'].extend(y_pred_np)
            if y_uncorrected is not None:
                self.plot_samples['y_uncorrected'].extend(y_uncorrected_np)
    
    def run_inference(self):
        """Run model inference and compute metrics incrementally."""
        print("Running inference on test set...")
        self.model.eval()
        self._reset_accumulators()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Processing batches")):
                # Handle batch format based on predict_bias
                if self.predict_bias:
                    X, y, mask, vhm0 = batch
                    vhm0 = vhm0.to(self.device) if vhm0 is not None else None
                else:
                    X, y, mask, vhm0_batch = batch
                    vhm0 = vhm0_batch.to(self.device) if vhm0_batch is not None else None
                
                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                
                # Get predictions
                y_pred = self.model(X)
                
                # Process batch and update accumulators
                self._process_batch(y, mask, vhm0, y_pred)
        
        print(f"Inference complete. Processed {self.total_count} valid pixels.")
    
    def compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall performance metrics from accumulators."""
        if self.total_count == 0:
            return {"error": "No valid data processed"}
        
        # Model metrics
        mae = self.sum_mae / self.total_count
        mse = self.sum_mse / self.total_count
        rmse = np.sqrt(mse)
        bias = self.sum_bias / self.total_count
        
        # Baseline metrics
        baseline_mae = None
        baseline_rmse = None
        baseline_bias = None
        mae_improvement = None
        rmse_improvement = None
        
        if self.sum_baseline_mae > 0:  # Check if baseline data exists
            baseline_mae = self.sum_baseline_mae / self.total_count
            baseline_mse = self.sum_baseline_mse / self.total_count
            baseline_rmse = np.sqrt(baseline_mse)
            baseline_bias = self.sum_baseline_bias / self.total_count
            mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0.0
            rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0.0
        
        # R² score
        mean_y_true = self.sum_y_true / self.total_count
        ss_res = self.sum_mse * self.total_count  # Already sum of squared residuals
        ss_tot = self.sum_y_true_sq - self.total_count * (mean_y_true ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Correlation
        mean_y_pred = self.sum_y_pred / self.total_count
        cov = (self.sum_y_true_y_pred / self.total_count) - (mean_y_true * mean_y_pred)
        std_y_true = np.sqrt((self.sum_y_true_sq / self.total_count) - (mean_y_true ** 2))
        std_y_pred = np.sqrt((self.sum_y_pred_sq / self.total_count) - (mean_y_pred ** 2))
        correlation = cov / (std_y_true * std_y_pred) if (std_y_true * std_y_pred) > 0 else 0.0
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "bias": float(bias),
            "mse": float(mse),
            "r2": float(r2),
            "correlation": float(correlation),
            "baseline_mae": float(baseline_mae) if baseline_mae is not None else None,
            "baseline_rmse": float(baseline_rmse) if baseline_rmse is not None else None,
            "baseline_bias": float(baseline_bias) if baseline_bias is not None else None,
            "mae_improvement_pct": float(mae_improvement) if mae_improvement is not None else None,
            "rmse_improvement_pct": float(rmse_improvement) if rmse_improvement is not None else None,
            "n_samples": int(self.total_count),
            "predict_bias_mode": self.predict_bias
        }
        
        return metrics
    
    def compute_sea_bin_metrics(self) -> Dict[str, Dict]:
        """Compute sea-bin metrics from accumulators."""
        sea_bin_metrics = {}
        
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            bin_data = self.sea_bin_accumulators[bin_name]
            bin_count = bin_data['count']
            
            if bin_count > 0:
                # Model metrics
                mae = bin_data['sum_mae'] / bin_count
                mse = bin_data['sum_mse'] / bin_count
                rmse = np.sqrt(mse)
                bias = bin_data['sum_bias'] / bin_count
                
                # Baseline metrics
                baseline_mae = None
                baseline_rmse = None
                baseline_bias = None
                mae_improvement = None
                rmse_improvement = None
                
                if bin_data['sum_baseline_mae'] > 0:
                    baseline_mae = bin_data['sum_baseline_mae'] / bin_count
                    baseline_mse = bin_data['sum_baseline_mse'] / bin_count
                    baseline_rmse = np.sqrt(baseline_mse)
                    baseline_bias = bin_data['sum_baseline_bias'] / bin_count
                    mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0.0
                    rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0.0
                
                sea_bin_metrics[bin_name] = {
                    "label": bin_config["label"],
                    "count": int(bin_count),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "bias": float(bias),
                    "baseline_mae": float(baseline_mae) if baseline_mae is not None else None,
                    "baseline_rmse": float(baseline_rmse) if baseline_rmse is not None else None,
                    "baseline_bias": float(baseline_bias) if baseline_bias is not None else None,
                    "mae_improvement_pct": float(mae_improvement) if mae_improvement is not None else None,
                    "rmse_improvement_pct": float(rmse_improvement) if rmse_improvement is not None else None
                }
        
        return sea_bin_metrics
    
    def plot_rmse_maps(self):
        """Plot spatial RMSE maps for model and baseline."""
        if not self.test_files or not self.spatial_errors_model:
            logger.warning("No spatial data available for RMSE maps")
            return
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad("white")
        # Load coordinates from first test file
        try:
            lat_grid, lon_grid = load_coordinates_from_parquet(
                "s3://" + self.test_files[0], 
                subsample_step=self.subsample_step
            )
            logger.info(f"Coordinate grid shape: {lat_grid.shape}")
        except Exception as e:
            logger.error(f"Failed to load coordinates: {e}")
            return
        
        # Aggregate spatial errors across all batches
        total_error_sq_model = np.zeros_like(lat_grid)
        total_count = np.zeros_like(lat_grid)
        total_error_sq_mae_model = np.zeros_like(lat_grid)
        
        for i, batch_data in enumerate(self.spatial_errors_model):
            h, w = batch_data['error_sq'].shape
            total_error_sq_model[:h, :w] += batch_data['error_sq']
            total_count[:h, :w] += batch_data['count']
            total_error_sq_mae_model[:h, :w] += batch_data['error_sq_mae']
        # Calculate RMSE (avoid division by zero)
        rmse_model = np.sqrt(total_error_sq_model / np.maximum(total_count, 1))
        rmse_model[total_count == 0] = np.nan
        mae_model = total_error_sq_mae_model / np.maximum(total_count, 1)
        mae_model[total_count == 0] = np.nan
        # Same for baseline if available
        if self.spatial_errors_baseline:
            total_error_sq_baseline = np.zeros_like(lat_grid)
            total_error_sq_mae_baseline = np.zeros_like(lat_grid)
            for batch_data in self.spatial_errors_baseline:
                h, w = batch_data['error_sq'].shape
                total_error_sq_baseline[:h, :w] += batch_data['error_sq']
                total_error_sq_mae_baseline[:h, :w] += batch_data['error_sq_mae']
            rmse_baseline = np.sqrt(total_error_sq_baseline / np.maximum(total_count, 1))
            rmse_baseline[total_count == 0] = np.nan
            mae_baseline = total_error_sq_mae_baseline / np.maximum(total_count, 1)
            mae_baseline[total_count == 0] = np.nan
        else:
            rmse_baseline = None
        
        # Compute appropriate color scale based on actual data
        vmax_model = np.nanpercentile(rmse_model, 98)
        if rmse_baseline is not None:
            vmax_baseline = np.nanpercentile(rmse_baseline, 98)
            vmax_combined = max(vmax_model, vmax_baseline)
        else:
            vmax_combined = vmax_model
        
        logger.info(f"RMSE model - min: {np.nanmin(rmse_model):.3f}, max: {np.nanmax(rmse_model):.3f}, mean: {np.nanmean(rmse_model):.3f}")
        logger.info(f"Using color scale vmax: {vmax_combined:.3f}")
        logger.info(f"Valid pixels: {np.sum(~np.isnan(rmse_model))} / {rmse_model.size}")
        
        # ========== PLOT 1: Model RMSE (separate figure) ==========
        plot_spatial_rmse_map(
            lat_grid, lon_grid, rmse_model,
            save_path=self.output_dir / 'rmse_model.png',
            title='Model RMSE',
            vmin=0, vmax=vmax_combined,
            cmap=cmap
        )
        plot_spatial_rmse_map(
            lat_grid, lon_grid, mae_model,
            save_path=self.output_dir / 'mae_model.png',
            title='Model MAE',
            vmin=0, vmax=max(np.nanpercentile(mae_model, 98), np.nanpercentile(mae_baseline, 98)),
            cmap=cmap
        )
        # fig1, ax1 = plt.subplots(figsize=(10, 6))
        # im1 = ax1.pcolormesh(lon_grid, lat_grid, rmse_model, 
        #                     shading='auto', cmap=cmap, 
        #                     vmin=0, vmax=vmax_combined)
        # ax1.set_title('Model RMSE (m)', fontweight='bold', fontsize=14)
        # ax1.set_xlabel('Longitude', fontsize=12)
        # ax1.set_ylabel('Latitude', fontsize=12)
        # plt.colorbar(im1, ax=ax1, label='RMSE (m)')
        # plt.tight_layout()
        # plt.savefig(self.output_dir / 'rmse_model.png', dpi=300, bbox_inches='tight')
        # plt.close()
        logger.info(f"Saved model RMSE map to {self.output_dir / 'rmse_model.png'}")
        
        # ========== PLOT 2: Reference RMSE (separate figure) ==========
        if rmse_baseline is not None:
            plot_spatial_rmse_map(
                lat_grid, lon_grid, rmse_baseline,
                save_path=self.output_dir / 'rmse_reference.png',
                title='Reference RMSE',
                vmin=0, vmax=vmax_combined,
                cmap=cmap
            )
            plot_spatial_rmse_map(
                lat_grid, lon_grid, mae_baseline,
                save_path=self.output_dir / 'mae_reference.png',
                title='Reference MAE',
                vmin=0, vmax=max(np.nanpercentile(mae_model, 98), np.nanpercentile(mae_baseline, 98)),
                cmap=cmap
            )
            # ========== PLOT 3: Improvement (separate figure) ==========
            improvement = rmse_baseline - rmse_model
            improvement_mae = mae_baseline - mae_model
            # Symmetric scale around zero
            imp_abs_max = np.nanpercentile(np.abs(improvement), 98)
            
            plot_spatial_rmse_map(
                lat_grid, lon_grid, improvement,
                save_path=self.output_dir / 'rmse_improvement.png',
                title='RMSE Improvement (Reference - Model)',
                vmin=-0.06, vmax=0.06,
                cmap=cmap
            )
            plot_spatial_rmse_map(
                lat_grid, lon_grid, improvement_mae,
                save_path=self.output_dir / 'mae_improvement.png',
                title='MAE Improvement (Reference - Model)',
                vmin=-0.06, vmax=0.06,
                cmap=cmap
            )
    
    def plot_sea_bin_metrics(self, sea_bin_metrics: Dict[str, Dict]):
        """Create sea-bin performance metrics plot with baseline comparison."""
        print("Creating sea-bin performance plot...")
        
        # Prepare data for plotting
        bin_names = []
        bin_labels = []
        rmse_values = []
        mae_values = []
        counts = []
        percentages = []
        improvement_mae_values = []
        improvement_rmse_values = []
        # Baseline data
        baseline_rmse_values = []
        baseline_mae_values = []
        
        # Total count for percentage calculation
        total_count = sum(m.get('count', 0) for m in sea_bin_metrics.values())
        
        # Sort bins by their min value
        sorted_bins = sorted(
            self.sea_bins,
            key=lambda x: x["min"]
        )
        
        for bin_config in sorted_bins:
            bin_name = bin_config["name"]
            if bin_name not in sea_bin_metrics:
                continue
                
            metrics = sea_bin_metrics[bin_name]
            if metrics.get('count', 0) == 0:
                continue
            
            bin_labels.append(metrics.get('label', bin_config['label']))
            bin_names.append(bin_name)
            
            rmse_values.append(metrics.get('rmse', 0))
            mae_values.append(metrics.get('mae', 0))
            counts.append(metrics.get('count', 0))
            
            # Calculate percentage
            pct = (metrics.get('count', 0) / total_count * 100) if total_count > 0 else 0
            percentages.append(pct)
            
            # Get baseline metrics
            baseline_rmse = metrics.get('baseline_rmse', 0)
            baseline_mae = metrics.get('baseline_mae', 0)
            baseline_rmse_values.append(baseline_rmse if baseline_rmse is not None else 0)
            baseline_mae_values.append(baseline_mae if baseline_mae is not None else 0)
            improvement_mae_values.append(metrics.get('mae_improvement_pct', 0))
            improvement_rmse_values.append(metrics.get('rmse_improvement_pct', 0))
        
        if not bin_names:
            logger.warning("No sea-bin metrics to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(
            "Sea-Bin Performance Analysis (Model vs Baseline)",
            fontsize=16,
            fontweight="bold",
        )
        
        # Set up bar positions for side-by-side comparison
        x = np.arange(len(bin_names))
        width = 0.35
        
        # Plot 1: RMSE by sea state
        # Draw Reference first (left side), then Model (right side)
        axes[0, 0].bar(
            x - width / 2,
            baseline_rmse_values,
            width,
            label="Reference",
            color="darkblue",
            alpha=0.6,
        )
        axes[0, 0].bar(
            x + width / 2, rmse_values, width, label="Model", color="skyblue", alpha=0.8
        )

        axes[0, 0].set_title("RMSE by Sea State", fontweight="bold")
        axes[0, 0].set_ylabel("RMSE (m)")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(baseline_rmse_values, rmse_values, strict=False)):
            if v1 > 0:
                axes[0, 0].text(
                    i - width / 2,
                    v1 + max(rmse_values) * 0.01,
                    f"{v1:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if v2 > 0:
                axes[0, 0].text(
                    i + width / 2,
                    v2 + max(baseline_rmse_values) * 0.01 if baseline_rmse_values else 0,
                    f"{v2:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        
        # Plot 2: MAE by sea state
        # Draw Reference first (left side), then Model (right side) for consistency
        axes[0, 1].bar(
            x - width / 2,
            baseline_mae_values,
            width,
            label="Reference",
            color="darkred",
            alpha=0.6,
        )
        axes[0, 1].bar(
            x + width / 2,
            mae_values,
            width,
            label="Model",
            color="lightcoral",
            alpha=0.8,
        )

        axes[0, 1].set_title("MAE by Sea State", fontweight="bold")
        axes[0, 1].set_ylabel("MAE (m)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(bin_labels, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(baseline_mae_values, mae_values, strict=False)):
            if v1 > 0:
                axes[0, 1].text(
                    i - width / 2,
                    v1 + max(mae_values) * 0.01,
                    f"{v1:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if v2 > 0:
                axes[0, 1].text(
                    i + width / 2,
                    v2 + max(baseline_mae_values) * 0.01 if baseline_mae_values else 0,
                    f"{v2:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        
        # Plot 3: Improvement percentage (RMSE)
        colors = ['green' if v > 0 else 'red' for v in improvement_rmse_values]
        axes[1, 0].bar(bin_labels, improvement_rmse_values, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title("RMSE Improvement by Sea State", fontweight="bold")
        axes[1, 0].set_ylabel("Improvement (%)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(improvement_rmse_values):
            axes[1, 0].text(
                i, v + (max(improvement_rmse_values) - min(improvement_rmse_values)) * 0.02 if improvement_rmse_values else 0,
                f"{v:.1f}%",
                ha="center",
                va="bottom" if v > 0 else "top",
                fontsize=9
            )
        
        # Plot 4: Sample distribution by sea state
        axes[1, 1].bar(bin_labels, percentages, color="gold", alpha=0.7)
        axes[1, 1].set_title("Sample Distribution by Sea State", fontweight="bold")
        axes[1, 1].set_ylabel("Percentage of Samples (%)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v, cnt) in enumerate(zip(percentages, counts, strict=False)):
            axes[1, 1].text(
                i,
                v + max(percentages) * 0.02 if percentages else 0,
                f"{v:.1f}%\n({cnt:,})",
                ha="center",
                va="bottom",
                fontsize=8
            )
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "sea_bin_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        
        print(f"Saved sea-bin performance plot to {self.output_dir / 'sea_bin_performance.png'}")

    def plot_error_distribution_histograms(self):
        """Plot histogram grid showing error distributions per sea bin."""
        print("Creating error distribution histogram grid...")
        
        # Filter bins with sufficient samples
        bins_to_plot = []
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            if len(self.sea_bin_error_samples[bin_name]['model_errors']) > 0:
                bins_to_plot.append(bin_config)
        
        if not bins_to_plot:
            logger.warning("No error samples available for histogram plots")
            return
        
        # Create grid layout (5 rows x 3 cols for up to 15 bins)
        n_bins = len(bins_to_plot)
        n_cols = 3
        n_rows = int(np.ceil(n_bins / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        fig.suptitle("Error Distribution by Sea State (Model vs Baseline)", 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier iteration
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        for idx, bin_config in enumerate(bins_to_plot):
            ax = axes_flat[idx]
            bin_name = bin_config["name"]
            bin_label = bin_config["label"]
            
            model_errors = np.array(self.sea_bin_error_samples[bin_name]['model_errors'])
            baseline_errors = np.array(self.sea_bin_error_samples[bin_name]['baseline_errors'])
            
            # Determine histogram range
            all_errors = np.concatenate([model_errors, baseline_errors]) if len(baseline_errors) > 0 else model_errors
            error_range = (np.percentile(all_errors, 1), np.percentile(all_errors, 99))
            
            # Plot histograms
            bins = np.linspace(error_range[0], error_range[1], 40)
            
            ax.hist(model_errors, bins=bins, alpha=0.6, color='blue', 
                   label=f'Model (n={len(model_errors):,})', density=True)
            
            if len(baseline_errors) > 0:
                ax.hist(baseline_errors, bins=bins, alpha=0.6, color='red', 
                       label=f'Baseline (n={len(baseline_errors):,})', density=True)
            
            # Add vertical lines for mean
            model_mean = np.mean(model_errors)
            ax.axvline(model_mean, color='blue', linestyle='--', linewidth=2, 
                      label=f'Model μ={model_mean:.3f}m')
            
            if len(baseline_errors) > 0:
                baseline_mean = np.mean(baseline_errors)
                ax.axvline(baseline_mean, color='red', linestyle='--', linewidth=2,
                          label=f'Baseline μ={baseline_mean:.3f}m')
            
            # Add zero line
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            
            # Formatting
            ax.set_title(f'{bin_label}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Error (m)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add statistics box
            model_std = np.std(model_errors)
            stats_text = f'Model: μ={model_mean:.3f}, σ={model_std:.3f}'
            if len(baseline_errors) > 0:
                baseline_std = np.std(baseline_errors)
                stats_text += f'\nBaseline: μ={baseline_mean:.3f}, σ={baseline_std:.3f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(len(bins_to_plot), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_distribution_histograms.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error distribution histograms to {self.output_dir / 'error_distribution_histograms.png'}")

    def plot_error_boxplots(self):
        """Plot box plot comparison of errors across all sea bins."""
        print("Creating error distribution box plots...")
        
        # Prepare data
        bin_labels = []
        model_error_data = []
        baseline_error_data = []
        
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            model_errors = self.sea_bin_error_samples[bin_name]['model_errors']
            baseline_errors = self.sea_bin_error_samples[bin_name]['baseline_errors']
            
            if len(model_errors) > 0:
                bin_labels.append(bin_config["label"])
                model_error_data.append(model_errors)
                baseline_error_data.append(baseline_errors if len(baseline_errors) > 0 else [])
        
        if not bin_labels:
            logger.warning("No error data available for box plots")
            return
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle("Error Distribution Box Plots by Sea State", 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Model errors
        ax1 = axes[0]
        bp1 = ax1.boxplot(model_error_data, labels=bin_labels, patch_artist=True,
                          showmeans=True, meanline=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='blue', linewidth=2),
                          meanprops=dict(color='darkblue', linewidth=2, linestyle='--'),
                          whiskerprops=dict(color='blue'),
                          capprops=dict(color='blue'),
                          flierprops=dict(marker='o', markerfacecolor='blue', 
                                        markersize=3, alpha=0.3))
        
        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_title("Model Errors", fontweight='bold', fontsize=14)
        ax1.set_ylabel("Error (m)", fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add sample counts
        for i, (label, data) in enumerate(zip(bin_labels, model_error_data, strict=False)):
            ax1.text(i+1, ax1.get_ylim()[0], f'n={len(data):,}',
                    ha='center', va='top', fontsize=8, rotation=0)
        
        # Plot 2: Model vs Baseline comparison (side-by-side)
        ax2 = axes[1]
        
        # Prepare positions for side-by-side box plots
        positions_model = np.arange(len(bin_labels)) * 2.5 + 0.6
        positions_baseline = np.arange(len(bin_labels)) * 2.5 + 1.4
        
        bp_model = ax2.boxplot(model_error_data, positions=positions_model,
                               widths=0.6, patch_artist=True, showmeans=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='blue', linewidth=2),
                               meanprops=dict(color='darkblue', linewidth=1.5, linestyle='--'),
                               whiskerprops=dict(color='blue'),
                               capprops=dict(color='blue'),
                               flierprops=dict(marker='o', markerfacecolor='blue', 
                                             markersize=2, alpha=0.3))
        
        # Only plot baseline if data exists
        has_baseline = any(len(d) > 0 for d in baseline_error_data)
        if has_baseline:
            bp_baseline = ax2.boxplot(baseline_error_data, positions=positions_baseline,
                                     widths=0.6, patch_artist=True, showmeans=True,
                                     boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                     medianprops=dict(color='red', linewidth=2),
                                     meanprops=dict(color='darkred', linewidth=1.5, linestyle='--'),
                                     whiskerprops=dict(color='red'),
                                     capprops=dict(color='red'),
                                     flierprops=dict(marker='o', markerfacecolor='red',
                                                   markersize=2, alpha=0.3))
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_title("Model vs Baseline Comparison", fontweight='bold', fontsize=14)
        ax2.set_ylabel("Error (m)", fontsize=12)
        ax2.set_xlabel("Sea State", fontsize=12)
        
        # Set x-ticks at the center of each pair
        ax2.set_xticks((positions_model + positions_baseline) / 2)
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', edgecolor='blue', label='Model')]
        if has_baseline:
            legend_elements.append(Patch(facecolor='lightcoral', edgecolor='red', label='Baseline'))
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_distribution_boxplots.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error distribution box plots to {self.output_dir / 'error_distribution_boxplots.png'}")
    
    def print_summary(self, overall_metrics: Dict, sea_bin_metrics: Dict):
        """Print evaluation summary to console."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print("\nOverall Metrics:")
        print(f"  Samples:              {overall_metrics['n_samples']:,}")
        print(f"  MAE:                  {overall_metrics['mae']:.4f} m")
        print(f"  RMSE:                 {overall_metrics['rmse']:.4f} m")
        print(f"  Bias:                 {overall_metrics['bias']:.4f} m")
        print(f"  R²:                   {overall_metrics['r2']:.4f}")
        print(f"  Correlation:          {overall_metrics['correlation']:.4f}")
        
        if overall_metrics.get('baseline_mae') is not None:
            print("\nBaseline (Uncorrected) Metrics:")
            print(f"  MAE:                  {overall_metrics['baseline_mae']:.4f} m")
            print(f"  RMSE:                 {overall_metrics['baseline_rmse']:.4f} m")
            print(f"  Bias:                 {overall_metrics['baseline_bias']:.4f} m")
            
            print("\nImprovement:")
            if overall_metrics.get('mae_improvement_pct') is not None:
                print(f"  MAE Improvement:      {overall_metrics['mae_improvement_pct']:.2f}%")
            if overall_metrics.get('rmse_improvement_pct') is not None:
                print(f"  RMSE Improvement:     {overall_metrics['rmse_improvement_pct']:.2f}%")
        
        print("\nSea-Bin Metrics:")
        print(f"{'Bin':<20} {'Count':<10} {'MAE':<10} {'RMSE':<10} {'Improvement':<15} {'Improvement RMSE':<15}")
        print("-" * 80)
        
        for _, metrics in sea_bin_metrics.items():
            if metrics['count'] > 0:
                improvement_str = f"{metrics['mae_improvement_pct']:>7.2f}%" if metrics.get('mae_improvement_pct') is not None else "N/A"
                improvement_rmse_str = f"{metrics['rmse_improvement_pct']:>7.2f}%" if metrics.get('rmse_improvement_pct') is not None else "N/A"
                print(f"{metrics['label']:<20} "
                      f"{metrics['count']:<10,} "
                      f"{metrics['mae']:<10.4f} "
                      f"{metrics['rmse']:<10.4f} "
                      f"{improvement_str:>15} "
                      f"{improvement_rmse_str:>15}")
        
        print("="*80 + "\n")
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print("Starting evaluation...")
        
        # Run inference (computes metrics incrementally)
        self.run_inference()
        
        # Compute final metrics from accumulators
        print("Computing final metrics...")
        overall_metrics = self.compute_overall_metrics()
        sea_bin_metrics = self.compute_sea_bin_metrics()
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump({
                "overall": overall_metrics,
                "sea_bins": sea_bin_metrics
                # "spatial": spatial_metrics
            }, f, indent=2)
        
        # Create plots using samples
        print("Creating plots...")
        self.plot_sea_bin_metrics(sea_bin_metrics)
        self.plot_rmse_maps()
        self.plot_error_distribution_histograms()
        self.plot_error_boxplots()
        
        # Print summary
        self.print_summary(overall_metrics, sea_bin_metrics)
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate WaveBiasCorrector model')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default='src/configs/config_dnn.yaml',
                       help='Configuration file')
    args = parser.parse_args()
    
    config = DNNConfig(args.config)

    training_config = config.config["training"]
    data_config = config.config["data"]
    predict_bias = data_config.get("predict_bias", False)
    
    # Get file list (same as training)
    files = get_file_list(
        data_config["data_path"], 
        data_config["file_pattern"], 
        data_config["max_files"]
    )
    _test_files_parq = get_file_list(
        "s3://medwav-dev-data/parquet/hourly/year=2023/", 
        "WAVEAN2023*.parquet", 
    )

    _, _ , test_files_parq = split_files_by_year(
        _test_files_parq,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )
    
    logger.info(f"Found {len(files)} files")
    
    # Split files by year (same as training)
    _, _ , test_files = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )
    
    logger.info(f"Test files: {len(test_files)}")
    
    # Load normalizer (same as training)
    normalizer = WaveNormalizer.load_from_s3("medwav-dev-data", data_config["normalizer_path"])
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")
    
    # Create test dataset (same parameters as training)
    patch_size = tuple(data_config["patch_size"]) if data_config["patch_size"] else None
    excluded_columns = data_config.get(
        "excluded_columns", ["time", "latitude", "longitude", "timestamp"]
    )
    target_column = data_config.get("target_column", "corrected_VHM0")
    subsample_step = data_config.get("subsample_step", None)
    
    if patch_size is not None:
        test_dataset = GridPatchWaveDataset(
            test_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_column=target_column,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            use_cache=False,
            normalize_target=data_config.get("normalize_target", False)
        )
    else: 
        test_dataset = CachedWaveDataset(
            test_files,
            excluded_columns=excluded_columns,
            target_column=target_column,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=False,
            use_cache=False,  # Use cache for evaluation
            normalize_target=data_config.get("normalize_target", False)
        )
    # Create test loader (use training batch size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        persistent_workers=training_config["num_workers"] > 0,
        prefetch_factor=training_config["prefetch_factor"]
    )
    checkpoint = config.config["checkpoint"]["resume_from_checkpoint"]
    logger.info(f"Loading model from {checkpoint}...")
    model = WaveBiasCorrector.load_from_checkpoint(checkpoint)
    logger.info(f"Model loaded. predict_bias={predict_bias}")

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        output_dir=Path(args.output_dir) / (config.config["logging"]["experiment_name"] + "_" + config.config["checkpoint"]["resume_from_checkpoint"].split("/")[-1].split(".")[0]),
        predict_bias=predict_bias,
        device="cuda",
        normalizer=normalizer,
        normalize_target=data_config.get("normalize_target", False),
        test_files=test_files_parq,
        subsample_step=5
    )
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()