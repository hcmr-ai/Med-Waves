#!/usr/bin/env python3
"""
Comprehensive evaluation script for WaveBiasCorrector model.
Provides detailed metrics, visualizations, and sea-bin analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
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
from src.commons.dataloaders import CachedWaveDataset
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(
        self, 
        model: pl.LightningModule, 
        test_loader: DataLoader,
        output_dir: Path,
        predict_bias: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.predict_bias = predict_bias
        
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

        # Initialize accumulators for incremental computation
        self._reset_accumulators()
        
        # Sample storage for plots (optional, limited size)
        self.plot_samples = {
            'y_true': [],
            'y_pred': [],
            'y_uncorrected': [],
            'max_samples': 10000  # Store only samples for visualization
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
        self.spatial_rmse_accumulators = {}
    
    def _reconstruct_wave_heights(self, bias: torch.Tensor, vhm0: torch.Tensor) -> torch.Tensor:
        """Reconstruct full wave heights from bias: corrected = vhm0 + bias"""
        return bias + vhm0 
    
    def _process_batch(self, X, y, mask, vhm0, y_pred, batch_metadata=None):
        """Process a single batch and update accumulators."""
        # Align dimensions
        min_h = min(y_pred.shape[2], y.shape[2])
        min_w = min(y_pred.shape[3], y.shape[3])
        y_pred = y_pred[:, :, :min_h, :min_w]
        y = y[:, :, :min_h, :min_w]
        mask = mask[:, :, :min_h, :min_w]
        
        if vhm0 is not None:
            vhm0 = vhm0[:, :, :min_h, :min_w]
        
        # if self.normalize_target and self.normalizer is not None:
        #     # Denormalize predictions and targets before computing metrics
        #     # Predictions come from model in normalized space
        #     y_pred_denorm = self.normalizer.inverse_transform_torch(y_pred)
        #     y_true_denorm = self.normalizer.inverse_transform_torch(y)
            
        #     # Use denormalized values for metrics
        #     y_pred_for_metrics = y_pred_denorm
        #     y_true_for_metrics = y_true_denorm
        # else:
        #     # Already in original space
        #     y_pred_for_metrics = y_pred
        #     y_true_for_metrics = y_true
        
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
                    
                    if y_uncorrected is not None:
                        bin_y_uncorrected = y_uncorrected_np[bin_mask]
                        baseline_bin_errors = bin_y_uncorrected - bin_y_true
                        self.sea_bin_accumulators[bin_name]['sum_baseline_mae'] += np.sum(np.abs(baseline_bin_errors))
                        self.sea_bin_accumulators[bin_name]['sum_baseline_mse'] += np.sum(baseline_bin_errors ** 2)
                        self.sea_bin_accumulators[bin_name]['sum_baseline_bias'] += np.sum(baseline_bin_errors)
            
            # Store samples for plotting (limited)
            if len(self.plot_samples['y_true']) < self.plot_samples['max_samples']:
                sample_size = min(n, self.plot_samples['max_samples'] - len(self.plot_samples['y_true']))
                indices = np.random.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
                
                self.plot_samples['y_true'].extend(y_true_np[indices])
                self.plot_samples['y_pred'].extend(y_pred_np[indices])
                if y_uncorrected is not None:
                    self.plot_samples['y_uncorrected'].extend(y_uncorrected_np[indices])
            
            # Also accumulate spatial RMSE if metadata available
            if batch_metadata is not None:
                file_idx = batch_metadata.get('file_idx')
                hour_idx = batch_metadata.get('hour_idx')
                
                if file_idx is not None and hour_idx is not None:
                    # For spatial tracking, we need full arrays (not masked) to align with grid indices
                    # Reconstruct full arrays before masking - do computation on GPU first
                    y_full = y.flatten()
                    y_pred_full = y_pred.flatten()
                    mask_full = mask.flatten()
                    
                    # Reconstruct wave heights on full arrays if predicting bias (GPU computation)
                    if self.predict_bias and vhm0 is not None:
                        vhm0_full = vhm0.flatten()
                        # Create a valid mask for reconstruction (where we have both y and vhm0)
                        valid_for_recon = ~torch.isnan(y_full) & ~torch.isnan(vhm0_full)
                        y_true_full = torch.where(valid_for_recon, y_full + vhm0_full, y_full)
                        y_pred_full_waves = torch.where(valid_for_recon, y_pred_full + vhm0_full, y_pred_full)
                    else:
                        y_true_full = y_full
                        y_pred_full_waves = y_pred_full
                    
                    # Get uncorrected for baseline (full array)
                    vhm0_full = vhm0.flatten() if vhm0 is not None else None
                    
                    # Convert to numpy in one batch (move to CPU once)
                    mask_for_spatial = mask_full.cpu().numpy()
                    y_true_for_spatial = y_true_full.cpu().numpy()
                    y_pred_for_spatial = y_pred_full_waves.cpu().numpy()
                    vhm0_for_spatial = vhm0_full.cpu().numpy() if vhm0_full is not None else None
                    
                    self._accumulate_spatial_rmse(
                        y_true_for_spatial, y_pred_for_spatial, 
                        vhm0_for_spatial,
                        mask_for_spatial, file_idx, hour_idx, min_h, min_w
                    )
    
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
                
                # Create batch metadata for spatial tracking
                # Note: For proper spatial tracking, you'd need file/hour indices from dataset
                # Using batch_idx as a simple proxy
                batch_metadata = {'file_idx': batch_idx // 24, 'hour_idx': batch_idx % 24}
                
                # Process batch and update accumulators
                self._process_batch(X, y, mask, vhm0, y_pred, batch_metadata=batch_metadata)
        
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
    
    def compute_spatial_metrics(self) -> Dict[str, Dict]:
        """Compute aggregated spatial metrics from accumulators."""
        if not self.spatial_rmse_accumulators:
            return {}
        
        spatial_metrics = {}
        
        # Aggregate by location (file_idx, hour_idx) or just return cell-level metrics
        # For simplicity, aggregate by (h_idx, w_idx) across all files/hours
        location_aggregators = {}
        
        for (file_idx, hour_idx, h_idx, w_idx), accum in self.spatial_rmse_accumulators.items():
            location_key = (h_idx, w_idx)
            
            if location_key not in location_aggregators:
                location_aggregators[location_key] = {
                    'model': {'sum_sq_error': 0.0, 'count': 0},
                    'baseline': {'sum_sq_error': 0.0, 'count': 0}
                }
            
            location_aggregators[location_key]['model']['sum_sq_error'] += accum['model']['sum_sq_error']
            location_aggregators[location_key]['model']['count'] += accum['model']['count']
            
            if accum['baseline']['count'] > 0:
                location_aggregators[location_key]['baseline']['sum_sq_error'] += accum['baseline']['sum_sq_error']
                location_aggregators[location_key]['baseline']['count'] += accum['baseline']['count']
        
        # Convert to metrics format
        for (h_idx, w_idx), agg in location_aggregators.items():
            model_count = agg['model']['count']
            baseline_count = agg['baseline']['count']
            
            if model_count > 0:
                model_rmse = np.sqrt(agg['model']['sum_sq_error'] / model_count)
                baseline_rmse = np.sqrt(agg['baseline']['sum_sq_error'] / baseline_count) if baseline_count > 0 else None
                rmse_diff = model_rmse - baseline_rmse if baseline_rmse is not None else None
                
                spatial_metrics[f"cell_{h_idx}_{w_idx}"] = {
                    'count': int(model_count),
                    'rmse': float(model_rmse),
                    'baseline_rmse': float(baseline_rmse) if baseline_rmse is not None else None,
                    'rmse_diff': float(rmse_diff) if rmse_diff is not None else None
                }
        
        return spatial_metrics
    
    def _accumulate_spatial_rmse(self, y_true_flat, y_pred_flat, y_uncorrected_flat, 
                                 mask_flat, file_idx, hour_idx, H, W):
        """Accumulate RMSE per grid cell for spatial mapping (vectorized for speed).
        
        Args:
            y_true_flat: Flattened true values (numpy array)
            y_pred_flat: Flattened predicted values (numpy array)
            y_uncorrected_flat: Flattened uncorrected values (numpy array or None)
            mask_flat: Flattened mask (numpy array)
            file_idx: File index for spatial tracking
            hour_idx: Hour index for spatial tracking
            H, W: Height and width of the original grid (before flattening)
        """
        # Ensure all inputs are numpy arrays
        y_true_flat = np.asarray(y_true_flat).flatten()
        y_pred_flat = np.asarray(y_pred_flat).flatten()
        mask_flat = np.asarray(mask_flat).flatten()
        
        if y_uncorrected_flat is not None:
            y_uncorrected_flat = np.asarray(y_uncorrected_flat).flatten()
        
        # Get valid pixels (finite and masked)
        finite_mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        if y_uncorrected_flat is not None:
            finite_mask = finite_mask & np.isfinite(y_uncorrected_flat)
        
        valid_mask = mask_flat.astype(bool) & finite_mask
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return
        
        # Vectorized computation - much faster than loops
        # Convert flat indices to (h, w) grid coordinates
        h_indices = valid_indices // W
        w_indices = valid_indices % W
        
        # Create unique cell identifier (much faster than tuples for grouping)
        # Use a single integer key with large multipliers to avoid collisions
        # Assumes: h_idx < 1000, w_idx < 1000, hour_idx < 100, file_idx < 1000
        cell_ids = file_idx * 100000000 + hour_idx * 1000000 + h_indices * 1000 + w_indices
        
        # Get model errors (fully vectorized)
        model_errors = y_pred_flat[valid_indices] - y_true_flat[valid_indices]
        model_sq_errors = model_errors ** 2
        
        # Get baseline errors if available (fully vectorized)
        baseline_sq_errors = None
        if y_uncorrected_flat is not None:
            baseline_errors = y_uncorrected_flat[valid_indices] - y_true_flat[valid_indices]
            baseline_sq_errors = baseline_errors ** 2
        
        # Use numpy to group and sum - fully vectorized, very fast
        unique_cell_ids, inverse_indices = np.unique(cell_ids, return_inverse=True)
        
        # Accumulate model errors per cell using bincount (vectorized)
        model_sums = np.bincount(inverse_indices, weights=model_sq_errors)
        model_counts = np.bincount(inverse_indices)
        
        # Accumulate baseline errors per cell if available (vectorized)
        baseline_sums = None
        baseline_counts = None
        if baseline_sq_errors is not None:
            baseline_sums = np.bincount(inverse_indices, weights=baseline_sq_errors)
            baseline_counts = np.bincount(inverse_indices)
        
        # Update global accumulators (minimal Python loop, only over unique cells)
        for i, cell_id in enumerate(unique_cell_ids):
            # Reconstruct cell_key from cell_id
            w = int(cell_id % 1000)
            h = int((cell_id // 1000) % 1000)
            hour = int((cell_id // 1000000) % 100)
            file = int(cell_id // 100000000)
            cell_key = (file, hour, h, w)
            
            if cell_key not in self.spatial_rmse_accumulators:
                self.spatial_rmse_accumulators[cell_key] = {
                    'model': {'sum_sq_error': 0.0, 'count': 0},
                    'baseline': {'sum_sq_error': 0.0, 'count': 0}
                }
            
            self.spatial_rmse_accumulators[cell_key]['model']['sum_sq_error'] += float(model_sums[i])
            self.spatial_rmse_accumulators[cell_key]['model']['count'] += int(model_counts[i])
            
            if baseline_sums is not None:
                self.spatial_rmse_accumulators[cell_key]['baseline']['sum_sq_error'] += float(baseline_sums[i])
                self.spatial_rmse_accumulators[cell_key]['baseline']['count'] += int(baseline_counts[i])
    
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
        axes[0, 0].bar(
            x - width / 2, rmse_values, width, label="Model", color="skyblue", alpha=0.8
        )
        axes[0, 0].bar(
            x + width / 2,
            baseline_rmse_values,
            width,
            label="Baseline",
            color="darkblue",
            alpha=0.6,
        )
        axes[0, 0].set_title("RMSE by Sea State", fontweight="bold")
        axes[0, 0].set_ylabel("RMSE (m)")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(rmse_values, baseline_rmse_values, strict=False)):
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
        axes[0, 1].bar(
            x - width / 2,
            mae_values,
            width,
            label="Model",
            color="lightcoral",
            alpha=0.8,
        )
        axes[0, 1].bar(
            x + width / 2,
            baseline_mae_values,
            width,
            label="Baseline",
            color="darkred",
            alpha=0.6,
        )
        axes[0, 1].set_title("MAE by Sea State", fontweight="bold")
        axes[0, 1].set_ylabel("MAE (m)")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(bin_labels, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(mae_values, baseline_mae_values, strict=False)):
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
        improvement_values = []
        for i, (m_rmse, b_rmse) in enumerate(zip(rmse_values, baseline_rmse_values, strict=False)):
            if b_rmse > 0:
                improvement = ((b_rmse - m_rmse) / b_rmse) * 100
            else:
                improvement = 0
            improvement_values.append(improvement)
        
        colors = ['green' if v > 0 else 'red' for v in improvement_values]
        axes[1, 0].bar(bin_labels, improvement_values, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title("RMSE Improvement by Sea State", fontweight="bold")
        axes[1, 0].set_ylabel("Improvement (%)")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(improvement_values):
            axes[1, 0].text(
                i, v + (max(improvement_values) - min(improvement_values)) * 0.02 if improvement_values else 0,
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
    
    def create_rmse_difference_map(self, grid_resolution: float = 0.1):
        """
        Create spatial map showing RMSE difference (model - baseline).
        
        Args:
            grid_resolution: Grid resolution in degrees for spatial binning
        """
        if not self.spatial_rmse_accumulators:
            logger.warning("No spatial RMSE data available. Cannot create difference map.")
            return
        
        print("Creating RMSE difference map...")
        
        # Collect all grid cells with their coordinates
        cells_data = []
        for (file_idx, hour_idx, h_idx, w_idx), accum in self.spatial_rmse_accumulators.items():
            model_count = accum['model']['count']
            baseline_count = accum['baseline']['count']
            
            if model_count > 0:
                model_rmse = np.sqrt(accum['model']['sum_sq_error'] / model_count)
                baseline_rmse = np.sqrt(accum['baseline']['sum_sq_error'] / baseline_count) if baseline_count > 0 else 0.0
                rmse_diff = model_rmse - baseline_rmse  # Positive = model worse, Negative = model better
                
                cells_data.append({
                    'file_idx': file_idx,
                    'hour_idx': hour_idx,
                    'h_idx': h_idx,
                    'w_idx': w_idx,
                    'model_rmse': model_rmse,
                    'baseline_rmse': baseline_rmse,
                    'rmse_diff': rmse_diff,
                    'count': model_count
                })
        
        if not cells_data:
            logger.warning("No valid grid cells for mapping")
            return
        
        # Convert to arrays for easier manipulation
        # Note: Without actual lat/lon, we'll use grid indices
        # For proper mapping, you'd need to load coordinates from original files
        
        # Create aggregated map by binning grid positions
        # Option 1: Simple grid-based visualization (if coordinates not available)
        self._plot_grid_based_rmse_diff(cells_data)
        
        # Option 2: If you have coordinate mapping, use that instead
        # self._plot_coordinate_based_rmse_diff(cells_data, grid_resolution)
    
    def _plot_grid_based_rmse_diff(self, cells_data):
        """Plot RMSE difference using grid indices."""
        import matplotlib.pyplot as plt
        
        # Extract grid positions
        h_indices = [c['h_idx'] for c in cells_data]
        w_indices = [c['w_idx'] for c in cells_data]
        rmse_diffs = [c['rmse_diff'] for c in cells_data]
        model_rmses = [c['model_rmse'] for c in cells_data]
        baseline_rmses = [c['baseline_rmse'] for c in cells_data]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Create grid for visualization
        max_h, max_w = max(h_indices) + 1, max(w_indices) + 1
        model_rmse_grid = np.full((max_h, max_w), np.nan)
        baseline_rmse_grid = np.full((max_h, max_w), np.nan)
        diff_rmse_grid = np.full((max_h, max_w), np.nan)
        
        for c in cells_data:
            h, w = c['h_idx'], c['w_idx']
            model_rmse_grid[h, w] = c['model_rmse']
            baseline_rmse_grid[h, w] = c['baseline_rmse']
            diff_rmse_grid[h, w] = c['rmse_diff']
        
        # Plot 1: Model RMSE
        im1 = axes[0].imshow(model_rmse_grid, aspect='auto', cmap='Reds', origin='lower')
        axes[0].set_title('Model RMSE (m)')
        axes[0].set_xlabel('Longitude Index')
        axes[0].set_ylabel('Latitude Index')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Baseline RMSE
        im2 = axes[1].imshow(baseline_rmse_grid, aspect='auto', cmap='Reds', origin='lower')
        axes[1].set_title('Baseline RMSE (m)')
        axes[1].set_xlabel('Longitude Index')
        axes[1].set_ylabel('Latitude Index')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: RMSE Difference (Model - Baseline)
        # Negative = model better (blue), Positive = model worse (red)
        vmax = np.nanmax(np.abs(diff_rmse_grid))
        im3 = axes[2].imshow(diff_rmse_grid, aspect='auto', cmap='RdBu_r', 
                            origin='lower', vmin=-vmax, vmax=vmax)
        axes[2].set_title('RMSE Difference (Model - Baseline)\nBlue=Better, Red=Worse')
        axes[2].set_xlabel('Longitude Index')
        axes[2].set_ylabel('Latitude Index')
        cbar = plt.colorbar(im3, ax=axes[2])
        cbar.set_label('RMSE Difference (m)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rmse_difference_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved RMSE difference map to {self.output_dir / 'rmse_difference_map.png'}")
    
    def save_metrics(self, overall_metrics: Dict, sea_bin_metrics: Dict):
        """Save metrics to JSON file."""
        results = {
            "overall_metrics": overall_metrics,
            "sea_bin_metrics": sea_bin_metrics
        }
        
        # Save as JSON
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as YAML for readability
        with open(self.output_dir / 'evaluation_metrics.yaml', 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"Saved metrics to {self.output_dir / 'evaluation_metrics.json'}")
        print(f"Saved metrics to {self.output_dir / 'evaluation_metrics.yaml'}")
    
    def print_summary(self, overall_metrics: Dict, sea_bin_metrics: Dict, spatial_metrics: Dict = None):
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
        
        # Print spatial metrics if available
        # if spatial_metrics and len(spatial_metrics) > 0:
        #     print("\nSpatial Metrics:")
        #     print(f"{'Location':<20} {'Count':<10} {'RMSE':<10} {'Baseline RMSE':<15} {'Diff':<10}")
        #     print("-" * 80)
        #     for location, metrics in spatial_metrics.items():
        #         if isinstance(metrics, dict) and metrics.get('count', 0) > 0:
        #             print(f"{str(location):<20} "
        #                   f"{metrics.get('count', 0):<10,} "
        #                   f"{metrics.get('rmse', 0):<10.4f} "
        #                   f"{metrics.get('baseline_rmse', 0):<15.4f} "
        #                   f"{metrics.get('rmse_diff', 0):<10.4f}")
        
        print("\nSea-Bin Metrics:")
        print(f"{'Bin':<20} {'Count':<10} {'MAE':<10} {'RMSE':<10} {'Improvement':<15}")
        print("-" * 80)
        
        for bin_name, metrics in sea_bin_metrics.items():
            if metrics['count'] > 0:
                improvement_str = f"{metrics['mae_improvement_pct']:>7.2f}%" if metrics.get('mae_improvement_pct') is not None else "N/A"
                print(f"{metrics['label']:<20} "
                      f"{metrics['count']:<10,} "
                      f"{metrics['mae']:<10.4f} "
                      f"{metrics['rmse']:<10.4f} "
                      f"{improvement_str:>15}")
        
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
        spatial_metrics = self.compute_spatial_metrics()
        
        # Save metrics
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump({
                "overall": overall_metrics,
                "sea_bins": sea_bin_metrics,
                "spatial": spatial_metrics
            }, f, indent=2)
        
        # Create plots using samples
        print("Creating plots...")
        # self.plot_scatter()
        # self.plot_error_distribution()
        self.plot_sea_bin_metrics(sea_bin_metrics)
        # self.create_rmse_difference_map()
        
        # Print summary
        self.print_summary(overall_metrics, sea_bin_metrics, spatial_metrics)
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate WaveBiasCorrector model')
    parser.add_argument('--checkpoint', type=str, default='/opt/dlami/nvme/preprocessed/checkpoints/epoch=27-val_loss=0.03.ckpt',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
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
    
    logger.info(f"Found {len(files)} files")
    
    # Split files by year (same as training)
    _,test_files, _ = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
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
    
    test_dataset = CachedWaveDataset(
        test_files,
        patch_size=patch_size,
        excluded_columns=excluded_columns,
        target_column=target_column,
        predict_bias=predict_bias,
        subsample_step=subsample_step,
        normalizer=normalizer,
        enable_profiler=False,
        use_cache=False,  # Use cache for evaluation
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

    logger.info(f"Loading model from {args.checkpoint}...")
    model = WaveBiasCorrector.load_from_checkpoint(args.checkpoint)
    logger.info(f"Model loaded. predict_bias={predict_bias}")

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        output_dir=Path(args.output_dir),
        predict_bias=predict_bias,
        device="cuda"
    )
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()