#!/usr/bin/env python3
"""
SNR Analysis Script for Wave Model Evaluation
Computes Signal-to-Noise Ratio per wave height bin and correlates with model error.

High correlation (>0.6) suggests SNR as input feature could condition model on data quality.

USAGE EXAMPLES:
===============

# Analyze 2021 data from S3 (default):
python calculate_snr.py --year 2021

# Analyze 2022 data with limited files:
python calculate_snr.py --year 2022 --max-files 50

# Use custom columns:
python calculate_snr.py --year 2021 --ground-truth-col corrected_VHM0 --reference-col VHM0

# Analyze from local directory:
python calculate_snr.py --input-dir /path/to/data/hourly/ --year 2021

# Custom output path:
python calculate_snr.py --year 2021 --output-path results/snr_2021.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from typing import Dict, Tuple, List, Optional
import warnings
import os
import glob
import pyarrow.parquet as pq
import fsspec
from tqdm import tqdm
import argparse
from collections import defaultdict

warnings.filterwarnings('ignore')

# Set professional plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def compute_local_snr_per_pixel(
    ground_truth: np.ndarray,
    reference: np.ndarray,
    patch_size: int = 5
) -> np.ndarray:
    """
    Compute local SNR for each pixel using neighboring patch variance.
    
    Args:
        ground_truth: Target field [time, lat, lon] or flattened
        reference: Model output [time, lat, lon] or flattened
        patch_size: Size of local patch for variance estimation (spatial only)
    
    Returns:
        SNR per pixel (same shape as input)
    """
    if ground_truth.ndim == 1:
        # Already flattened, compute per-pixel SNR
        # Filter NaN values first
        valid_mask = np.isfinite(ground_truth) & np.isfinite(reference)
        if np.sum(valid_mask) == 0:
            return np.full_like(ground_truth, np.nan)
        
        # Compute per-pixel SNR: signal^2 / noise^2 for each pixel
        # This gives actual variance in SNR across pixels
        signal_squared = ground_truth ** 2
        residuals = ground_truth - reference
        noise_squared = np.maximum(residuals ** 2, 1e-10)
        snr_per_pixel = signal_squared / noise_squared
        
        # Set NaN for invalid pixels
        snr_per_pixel[~valid_mask] = np.nan
        
        return snr_per_pixel
    
    # For 3D data [time, lat, lon], compute local SNR spatially
    t, h, w = ground_truth.shape
    snr_map = np.full((t, h, w), np.nan)  # Initialize with NaN
    residuals = ground_truth - reference
    
    pad = patch_size // 2
    # Use 'constant' mode with NaN to avoid reflecting NaN values
    ground_truth_pad = np.pad(ground_truth, ((0, 0), (pad, pad), (pad, pad)), 
                              mode='constant', constant_values=np.nan)
    residuals_pad = np.pad(residuals, ((0, 0), (pad, pad), (pad, pad)), 
                          mode='constant', constant_values=np.nan)
    
    for i in range(h):
        for j in range(w):
            # Extract local patch across all time steps
            gt_patch = ground_truth_pad[:, i:i+patch_size, j:j+patch_size]
            res_patch = residuals_pad[:, i:i+patch_size, j:j+patch_size]
            
            # Filter NaN values from patch before computing variance
            valid_patch = np.isfinite(gt_patch) & np.isfinite(res_patch)
            if np.sum(valid_patch) < patch_size:  # Need minimum samples
                snr_map[:, i, j] = np.nan
                continue
            
            gt_patch_valid = gt_patch[valid_patch]
            res_patch_valid = res_patch[valid_patch]
            
            signal_var = np.var(gt_patch_valid)
            noise_var = np.var(res_patch_valid)
            
            if np.isfinite(signal_var) and np.isfinite(noise_var) and noise_var > 0:
                snr = signal_var / np.maximum(noise_var, 1e-10)
                snr_map[:, i, j] = snr
            else:
                snr_map[:, i, j] = np.nan
    
    return snr_map


def compute_snr_per_bin(
    ground_truth: np.ndarray,
    reference: np.ndarray,
    bin_edges: List[float],
    min_samples: int = 50
) -> Dict:
    """
    Compute SNR metrics per wave height bin (binned by ground truth).
    
    Args:
        ground_truth: Target values (flattened)
        reference: Model predictions (flattened)
        bin_edges: Wave height bin edges [0, 1, 2, 3, ...]
        min_samples: Minimum samples required per bin
    
    Returns:
        Dictionary with bin metrics
    """
    results = {}
    
    # Ensure all inputs are finite (no NaN/Inf)
    valid_mask = np.isfinite(ground_truth) & np.isfinite(reference)
    ground_truth = ground_truth[valid_mask]
    reference = reference[valid_mask]
    
    for i in range(len(bin_edges) - 1):
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        bin_name = f"{bin_min:.1f}-{bin_max:.1f}m"
        
        # Filter data for this bin (based on ground truth) - already filtered for NaN above
        mask = (ground_truth >= bin_min) & (ground_truth < bin_max)
        n_samples = np.sum(mask)
        
        if n_samples < min_samples:
            continue
        
        gt_bin = ground_truth[mask]
        ref_bin = reference[mask]
        
        # Compute metrics
        residuals = gt_bin - ref_bin
        error_magnitude = np.abs(residuals)
        
        signal_var = np.var(gt_bin)
        noise_var = np.var(residuals)
        
        # SNR calculation
        snr = signal_var / np.maximum(noise_var, 1e-10)
        snr_db = 10 * np.log10(snr) if np.isfinite(snr) else np.inf
        
        # Error metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(error_magnitude)
        bias = np.mean(residuals)
        
        results[bin_name] = {
            'bin_min': bin_min,
            'bin_max': bin_max,
            'n_samples': n_samples,
            'signal_var': signal_var,
            'noise_var': noise_var,
            'snr': snr,
            'snr_db': snr_db,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'error_magnitude': error_magnitude,
            'gt_values': gt_bin,
            'ref_values': ref_bin,
        }
    
    return results


def compute_correlations(
    ground_truth: np.ndarray,
    reference: np.ndarray,
    snr_per_pixel: np.ndarray,
    bin_results: Dict
) -> Dict:
    """
    Compute correlation between error magnitude and SNR.
    
    Returns:
        Dictionary with overall and per-bin correlations
    """
    correlations = {}
    
    # Ensure all inputs are finite (no NaN/Inf) - arrays should already be filtered, but double-check
    # All arrays should have the same length at this point (already filtered in analyze_snr_complete)
    if not (len(ground_truth) == len(reference) == len(snr_per_pixel)):
        # If lengths don't match, filter to common valid indices
        min_len = min(len(ground_truth), len(reference), len(snr_per_pixel))
        ground_truth = ground_truth[:min_len]
        reference = reference[:min_len]
        snr_per_pixel = snr_per_pixel[:min_len]
    
    # Final check for any remaining NaN/Inf values
    valid_mask = np.isfinite(ground_truth) & np.isfinite(reference) & np.isfinite(snr_per_pixel)
    ground_truth = ground_truth[valid_mask]
    reference = reference[valid_mask]
    snr_per_pixel = snr_per_pixel[valid_mask]
    
    # Overall correlation across all pixels
    error_magnitude = np.abs(ground_truth - reference)
    # All values should be finite at this point, but double-check for safety
    valid_mask = np.isfinite(error_magnitude) & np.isfinite(snr_per_pixel)
    
    if np.sum(valid_mask) > 10:
        error_valid = error_magnitude[valid_mask]
        snr_valid = snr_per_pixel[valid_mask]
        
        # Check if SNR has variance (required for correlation)
        snr_var = np.var(snr_valid)
        error_var = np.var(error_valid)
        
        if snr_var < 1e-10 or error_var < 1e-10:
            # No variance in one or both variables, correlation is undefined
            correlations['overall'] = {
                'correlation': np.nan,
                'p_value': np.nan,
                'n_samples': np.sum(valid_mask)
            }
        else:
            overall_corr, overall_pval = pearsonr(error_valid, snr_valid)
            # Ensure correlation is finite
            if not np.isfinite(overall_corr):
                overall_corr = np.nan
            if not np.isfinite(overall_pval):
                overall_pval = np.nan
            correlations['overall'] = {
                'correlation': overall_corr,
                'p_value': overall_pval,
                'n_samples': np.sum(valid_mask)
            }
    else:
        correlations['overall'] = {
            'correlation': np.nan,
            'p_value': np.nan,
            'n_samples': 0
        }
    
    # Per-bin correlations
    correlations['per_bin'] = {}
    for bin_name, metrics in bin_results.items():
        error_mag = metrics['error_magnitude']
        gt_bin = metrics['gt_values']
        ref_bin = metrics['ref_values']
        
        # Filter out NaN/Inf values from bin data
        valid_bin_mask = np.isfinite(error_mag) & np.isfinite(gt_bin) & np.isfinite(ref_bin)
        error_mag = error_mag[valid_bin_mask]
        gt_bin = gt_bin[valid_bin_mask]
        ref_bin = ref_bin[valid_bin_mask]
        
        if len(error_mag) < 10:
            correlations['per_bin'][bin_name] = {
                'correlation': np.nan,
                'p_value': np.nan
            }
            continue
        
        # Compute per-pixel SNR within this bin
        # Use signal strength (gt_bin^2) vs noise (residual^2) as per-pixel SNR proxy
        residuals = gt_bin - ref_bin
        
        # Per-pixel SNR: signal^2 / noise^2
        # This gives actual variance in SNR across pixels in the bin
        signal_squared = gt_bin ** 2
        noise_squared = np.maximum(residuals ** 2, 1e-10)
        local_snr = signal_squared / noise_squared
        
        # Ensure local_snr is finite and positive
        valid_snr_mask = np.isfinite(local_snr) & np.isfinite(error_mag) & (local_snr > 0)
        error_mag = error_mag[valid_snr_mask]
        local_snr = local_snr[valid_snr_mask]
        
        if len(error_mag) < 10:
            correlations['per_bin'][bin_name] = {
                'correlation': np.nan,
                'p_value': np.nan
            }
            continue
        
        # Check if local_snr has variance (required for correlation)
        snr_var = np.var(local_snr)
        error_var = np.var(error_mag)
        
        if snr_var < 1e-10 or error_var < 1e-10:
            # No variance in one or both variables, correlation is undefined
            correlations['per_bin'][bin_name] = {
                'correlation': np.nan,
                'p_value': np.nan
            }
        else:
            try:
                bin_corr, bin_pval = pearsonr(error_mag, local_snr)
                correlations['per_bin'][bin_name] = {
                    'correlation': bin_corr if np.isfinite(bin_corr) else np.nan,
                    'p_value': bin_pval if np.isfinite(bin_pval) else np.nan
                }
            except Exception as e:
                correlations['per_bin'][bin_name] = {
                    'correlation': np.nan,
                    'p_value': np.nan
                }
    
    return correlations


def plot_snr_analysis(
    bin_results: Dict,
    correlations: Dict,
    ground_truth: np.ndarray,
    reference: np.ndarray,
    snr_per_pixel: np.ndarray,
    output_path: str = 'snr_analysis_complete.png'
):
    """
    Generate 4-panel plot with SNR and correlation analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SNR Analysis for Wave Model Evaluation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Extract data for plotting
    bin_names = list(bin_results.keys())
    snr_db_values = [bin_results[b]['snr_db'] for b in bin_names]
    rmse_values = [bin_results[b]['rmse'] for b in bin_names]
    mae_values = [bin_results[b]['mae'] for b in bin_names]
    n_samples = [bin_results[b]['n_samples'] for b in bin_names]
    
    # Get per-bin correlations (handle missing bins)
    bin_corr_values = []
    for bn in bin_names:
        if bn in correlations['per_bin']:
            bin_corr_values.append(correlations['per_bin'][bn]['correlation'])
        else:
            bin_corr_values.append(np.nan)
    
    # --- Plot 1: Mean SNR_dB per wave bin ---
    ax1 = axes[0, 0]
    colors = ['#2E7D32' if snr > 10 else '#F57C00' if snr > 5 else '#C62828' 
              for snr in snr_db_values]
    _ = ax1.bar(range(len(bin_names)), snr_db_values, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (v, n) in enumerate(zip(snr_db_values, n_samples)):
        if np.isfinite(v):
            ax1.text(i, v + 0.5, f'{v:.1f} dB\n(n={n:,})', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Wave Height Bin (m)', fontweight='bold')
    ax1.set_ylabel('Mean SNR (dB)', fontweight='bold')
    ax1.set_title('Plot 1: Mean SNR per Wave Height Bin', fontweight='bold', pad=10)
    ax1.set_xticks(range(len(bin_names)))
    ax1.set_xticklabels(bin_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=10, color='green', linestyle='--', linewidth=1, 
                alpha=0.5, label='Good SNR (>10 dB)')
    ax1.axhline(y=5, color='orange', linestyle='--', linewidth=1, 
                alpha=0.5, label='Moderate SNR (>5 dB)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # --- Plot 2: Scatter - RMSE vs SNR_dB ---
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(rmse_values, snr_db_values, s=[n/100 for n in n_samples], 
                      c=range(len(bin_names)), cmap='viridis', 
                      alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add bin labels to points
    for i, (x, y, bn) in enumerate(zip(rmse_values, snr_db_values, bin_names)):
        if np.isfinite(x) and np.isfinite(y):
            ax2.annotate(bn, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=7, alpha=0.8)
    
    ax2.set_xlabel('RMSE (m)', fontweight='bold')
    ax2.set_ylabel('SNR (dB)', fontweight='bold')
    ax2.set_title('Plot 2: Error vs SNR per Bin', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar2 = plt.colorbar(sc2, ax=ax2, label='Bin Index')
    cbar2.set_label('Bin Index', fontweight='bold')
    
    # Add trend annotation
    valid_idx = [i for i in range(len(rmse_values)) 
                 if np.isfinite(rmse_values[i]) and np.isfinite(snr_db_values[i])]
    if len(valid_idx) > 2:
        rmse_valid = [rmse_values[i] for i in valid_idx]
        snr_valid = [snr_db_values[i] for i in valid_idx]
        corr_rmse_snr, _ = pearsonr(rmse_valid, snr_valid)
        ax2.text(0.05, 0.95, f'Correlation: {corr_rmse_snr:.3f}', 
                transform=ax2.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    
    # --- Plot 3: Correlation coefficient per bin ---
    ax3 = axes[1, 0]
    colors_corr = ['#1976D2' if abs(c) > 0.6 else '#757575' 
                   for c in bin_corr_values if np.isfinite(c)]
    valid_corr_bins = [bn for bn, c in zip(bin_names, bin_corr_values) if np.isfinite(c)]
    valid_corr_values = [c for c in bin_corr_values if np.isfinite(c)]
    
    if len(valid_corr_values) > 0:
        bars3 = ax3.bar(range(len(valid_corr_bins)), valid_corr_values, 
                       color=colors_corr, alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, v in enumerate(valid_corr_values):
            ax3.text(i, v + 0.02 if v >= 0 else v - 0.02, f'{v:.3f}', 
                    ha='center', va='bottom' if v >= 0 else 'top', 
                    fontsize=8, fontweight='bold')
        
        ax3.set_xticks(range(len(valid_corr_bins)))
        ax3.set_xticklabels(valid_corr_bins, rotation=45, ha='right')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(y=0.6, color='blue', linestyle='--', linewidth=1, 
                alpha=0.5, label='Strong correlation (|r|>0.6)')
    ax3.axhline(y=-0.6, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Wave Height Bin (m)', fontweight='bold')
    ax3.set_ylabel('Correlation (Error-SNR)', fontweight='bold')
    ax3.set_title('Plot 3: Per-Bin Error-SNR Correlation', fontweight='bold', pad=10)
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=8)
    
    # Add overall correlation annotation
    overall_corr = correlations['overall']['correlation']
    if np.isfinite(overall_corr):
        corr_str = f"{overall_corr:.3f}"
        if abs(overall_corr) > 0.6:
            interpretation = "→ Strong: SNR feature could help!"
        elif abs(overall_corr) > 0.5:
            interpretation = "→ Moderate: SNR feature may help."
        else:
            interpretation = "→ Weak relationship"
    else:
        corr_str = "N/A"
        interpretation = "→ Cannot compute (no variance in SNR)"
    
    ax3.text(0.05, 0.95, 
            f"Overall correlation: {corr_str}\n{interpretation}",
            transform=ax3.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top')
    
    # --- Plot 4: Scatter - error_magnitude vs SNR (all pixels) ---
    ax4 = axes[1, 1]
    error_magnitude = np.abs(ground_truth - reference)
    
    # Subsample for plotting if too many points
    max_plot_points = 50000
    if len(error_magnitude) > max_plot_points:
        idx = np.random.choice(len(error_magnitude), max_plot_points, replace=False)
        error_plot = error_magnitude[idx]
        snr_plot = snr_per_pixel[idx]
    else:
        error_plot = error_magnitude
        snr_plot = snr_per_pixel
    
    # Filter out infinite/nan values
    valid = np.isfinite(error_plot) & np.isfinite(snr_plot)
    error_plot = error_plot[valid]
    snr_plot = snr_plot[valid]
    
    # Convert SNR to dB for plotting
    snr_plot_db = 10 * np.log10(np.maximum(snr_plot, 1e-10))
    
    # Create hexbin for density
    hexbin = ax4.hexbin(error_plot, snr_plot_db, gridsize=50, cmap='YlOrRd', 
                        mincnt=1, alpha=0.8, edgecolors='black', linewidths=0.2)
    
    ax4.set_xlabel('Error Magnitude (m)', fontweight='bold')
    ax4.set_ylabel('SNR (dB)', fontweight='bold')
    ax4.set_title(f'Plot 4: Error vs SNR (All Pixels, n={len(error_plot):,})', 
                 fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar4 = plt.colorbar(hexbin, ax=ax4, label='Pixel Density')
    cbar4.set_label('Pixel Count', fontweight='bold')
    
    # Add correlation annotation
    if len(error_plot) > 10:
        # Check variance before computing correlation
        error_var = np.var(error_plot)
        snr_var = np.var(snr_plot_db)
        
        if error_var < 1e-10 or snr_var < 1e-10:
            # No variance in one or both variables
            corr_str = "N/A"
            pval_str = "N/A"
            interpretation = "Cannot compute correlation:\nno variance in SNR or error values."
        else:
            overall_pixel_corr, overall_pixel_pval = pearsonr(error_plot, snr_plot_db)
            if np.isfinite(overall_pixel_corr) and np.isfinite(overall_pixel_pval):
                corr_str = f"{overall_pixel_corr:.3f}"
                pval_str = f"{overall_pixel_pval:.2e}"
                interpretation = (
                    "High correlation (>0.6) suggests SNR as input\n"
                    "feature could condition model on data quality."
                    if abs(overall_pixel_corr) > 0.6 else
                    "Moderate/weak correlation suggests other\n"
                    "factors dominate prediction errors."
                )
            else:
                corr_str = "N/A"
                pval_str = "N/A"
                interpretation = "Cannot compute correlation:\ninvalid correlation result."
        
        ax4.text(0.05, 0.95, 
                f'Correlation: {corr_str}\n'
                f'p-value: {pval_str}\n\n'
                f'{interpretation}',
                transform=ax4.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9),
                verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 4-panel plot to: {output_path}")
    plt.close()


def print_detailed_table(bin_results: Dict, correlations: Dict):
    """
    Print comprehensive table with all metrics.
    """
    print("\n" + "="*120)
    print("DETAILED SNR ANALYSIS TABLE")
    print("="*120)
    print(f"{'Bin':<12} {'N_samples':<12} {'Signal_Var':<12} {'Noise_Var':<12} "
          f"{'SNR_dB':<10} {'RMSE':<10} {'MAE':<10} {'Bias':<10} {'Bin_Corr':<12}")
    print("-"*120)
    
    for bin_name, metrics in bin_results.items():
        bin_corr = correlations['per_bin'].get(bin_name, {}).get('correlation', np.nan)
        print(f"{bin_name:<12} "
              f"{metrics['n_samples']:<12,} "
              f"{metrics['signal_var']:<12.4f} "
              f"{metrics['noise_var']:<12.4f} "
              f"{metrics['snr_db']:<10.2f} "
              f"{metrics['rmse']:<10.4f} "
              f"{metrics['mae']:<10.4f} "
              f"{metrics['bias']:<10.4f} "
              f"{bin_corr:<12.3f}")
    
    print("-"*120)
    overall_corr = correlations['overall']['correlation']
    overall_pval = correlations['overall']['p_value']
    overall_n = correlations['overall']['n_samples']
    
    if np.isfinite(overall_corr):
        print(f"\nOVERALL CORRELATION (Error-SNR): {overall_corr:.4f} (p={overall_pval:.2e}, n={overall_n:,})")
        if abs(overall_corr) > 0.7:
            print("→ STRONG RELATIONSHIP: SNR as feature could significantly help TransUNet!")
        elif abs(overall_corr) > 0.5:
            print("→ MODERATE RELATIONSHIP: SNR feature may provide useful signal.")
        else:
            print("→ WEAK RELATIONSHIP: Other factors dominate prediction errors.")
    else:
        print(f"\nOVERALL CORRELATION (Error-SNR): N/A (p=N/A, n={overall_n:,})")
        print("→ CANNOT COMPUTE: No variance in SNR values (all pixels have same SNR)")
    
    print("="*120 + "\n")


def analyze_snr_complete(
    ground_truth: np.ndarray,
    reference: np.ndarray,
    bin_edges: List[float] = None,
    output_path: str = 'snr_analysis_complete.png',
    min_samples: int = 50
):
    """
    Complete SNR analysis pipeline.
    
    Args:
        ground_truth: Target field (flattened or 3D)
        reference: Model output (same shape as ground_truth)
        bin_edges: Wave height bin edges (default: 0 to 15m in 1m steps)
        output_path: Where to save the plot
        min_samples: Minimum samples per bin
    """
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE SNR ANALYSIS")
    print("="*80)
    
    # Default bin edges for wave heights
    if bin_edges is None:
        bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # Flatten if needed
    original_shape = ground_truth.shape
    gt_flat = ground_truth.flatten()
    ref_flat = reference.flatten()
    
    # Remove NaN/Inf values (land pixels and invalid data)
    valid_mask = np.isfinite(gt_flat) & np.isfinite(ref_flat) & (gt_flat >= 0) & (ref_flat >= 0)
    n_removed = np.sum(~valid_mask)
    gt_flat = gt_flat[valid_mask]
    ref_flat = ref_flat[valid_mask]
    
    print(f"Data shape: {original_shape}")
    print(f"Valid samples: {len(gt_flat):,} / {len(ground_truth.flatten()):,}")
    if n_removed > 0:
        print(f"  → Removed {n_removed:,} invalid samples (NaN/Inf/negative values)")
    print(f"Ground truth range: [{gt_flat.min():.2f}, {gt_flat.max():.2f}] m")
    print(f"Reference range: [{ref_flat.min():.2f}, {ref_flat.max():.2f}] m")
    
    # Step 1: Compute local SNR per pixel
    print("\n[1/4] Computing local SNR per pixel...")
    if len(original_shape) == 3:
        # For 3D data, compute SNR (function now handles NaN internally)
        snr_per_pixel_full = compute_local_snr_per_pixel(ground_truth, reference, patch_size=5)
        # Filter to only valid pixels (NaN SNR values will be filtered out)
        snr_per_pixel = snr_per_pixel_full.flatten()[valid_mask]
        # Ensure any remaining NaN in SNR are filtered
        valid_snr_mask = np.isfinite(snr_per_pixel)
        snr_per_pixel = snr_per_pixel[valid_snr_mask]
        # Also filter the corresponding ground truth and reference
        gt_flat = gt_flat[valid_snr_mask]
        ref_flat = ref_flat[valid_snr_mask]
    else:
        snr_per_pixel = compute_local_snr_per_pixel(gt_flat, ref_flat)
    
    # Step 2: Compute SNR per bin
    print("[2/4] Computing SNR per wave height bin...")
    bin_results = compute_snr_per_bin(gt_flat, ref_flat, bin_edges, min_samples)
    print(f"  → Found {len(bin_results)} bins with ≥{min_samples} samples")
    
    # Step 3: Compute correlations
    print("[3/4] Computing error-SNR correlations...")
    correlations = compute_correlations(gt_flat, ref_flat, snr_per_pixel, bin_results)
    
    # Step 4: Generate plots
    print("[4/4] Generating 4-panel visualization...")
    plot_snr_analysis(bin_results, correlations, gt_flat, ref_flat, 
                     snr_per_pixel, output_path)
    
    # Print detailed table
    print_detailed_table(bin_results, correlations)
    
    return bin_results, correlations


def spatial_snr_analysis(
    ground_truth: np.ndarray,
    reference: np.ndarray,
    snr_per_pixel: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    output_path: str = 'snr_spatial_analysis.png',
    grid_resolution: float = 0.5
):
    """
    Perform spatial SNR analysis and generate geographic maps.
    
    Args:
        ground_truth: Target values (flattened)
        reference: Model predictions (flattened)
        snr_per_pixel: SNR values per pixel (flattened)
        lat_coords: Latitude coordinates per pixel
        lon_coords: Longitude coordinates per pixel
        output_path: Where to save the spatial analysis plot
        grid_resolution: Grid cell size in degrees
    """
    print("\n" + "="*80)
    print("SPATIAL SNR ANALYSIS")
    print("="*80)
    
    # Filter valid data
    valid_mask = np.isfinite(ground_truth) & np.isfinite(reference) & np.isfinite(snr_per_pixel)
    valid_mask = valid_mask & np.isfinite(lat_coords) & np.isfinite(lon_coords)
    
    gt_valid = ground_truth[valid_mask]
    ref_valid = reference[valid_mask]
    snr_valid = snr_per_pixel[valid_mask]
    lat_valid = lat_coords[valid_mask]
    lon_valid = lon_coords[valid_mask]
    
    error_magnitude = np.abs(gt_valid - ref_valid)
    snr_db = 10 * np.log10(np.maximum(snr_valid, 1e-10))
    
    print(f"Valid spatial samples: {len(gt_valid):,}")
    print(f"Lat range: [{lat_valid.min():.2f}, {lat_valid.max():.2f}]")
    print(f"Lon range: [{lon_valid.min():.2f}, {lon_valid.max():.2f}]")
    
    # Create spatial grid
    lat_bins = np.arange(lat_valid.min(), lat_valid.max() + grid_resolution, grid_resolution)
    lon_bins = np.arange(lon_valid.min(), lon_valid.max() + grid_resolution, grid_resolution)
    
    print(f"Grid resolution: {grid_resolution}° ({len(lat_bins)-1}x{len(lon_bins)-1} cells)")
    
    # Aggregate data into spatial grid using vectorized operations
    snr_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    error_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    corr_grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    count_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1), dtype=int)
    
    print("Aggregating data into spatial grid (optimized)...")
    
    # Assign each sample to a grid cell using digitize (much faster!)
    lat_indices = np.digitize(lat_valid, lat_bins) - 1
    lon_indices = np.digitize(lon_valid, lon_bins) - 1
    
    # Clip to valid range
    lat_indices = np.clip(lat_indices, 0, len(lat_bins) - 2)
    lon_indices = np.clip(lon_indices, 0, len(lon_bins) - 2)
    
    # Create unique cell IDs
    cell_ids = lat_indices * len(lon_bins) + lon_indices
    unique_cells = np.unique(cell_ids)
    
    print(f"Processing {len(unique_cells)} populated grid cells...")
    
    # Process each unique cell
    for cell_id in tqdm(unique_cells):
        # Find all samples in this cell
        cell_mask = cell_ids == cell_id
        n_samples = np.sum(cell_mask)
        
        if n_samples < 10:  # Need minimum samples
            continue
        
        # Get grid indices
        i = cell_id // len(lon_bins)
        j = cell_id % len(lon_bins)
        
        if i >= len(lat_bins) - 1 or j >= len(lon_bins) - 1:
            continue
        
        cell_snr = snr_db[cell_mask]
        cell_error = error_magnitude[cell_mask]
        
        # Compute statistics for this cell
        snr_grid[i, j] = np.mean(cell_snr)
        error_grid[i, j] = np.mean(cell_error)
        count_grid[i, j] = n_samples
        
        # Compute local correlation if enough variance
        if n_samples > 20 and np.var(cell_snr) > 1e-6 and np.var(cell_error) > 1e-6:
            try:
                corr, _ = pearsonr(cell_error, cell_snr)
                if np.isfinite(corr):
                    corr_grid[i, j] = corr
            except:
                pass
    
    print(f"Grid cells with data: {np.sum(~np.isnan(snr_grid))}")
    
    # Create spatial plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial SNR Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Define extent for plotting
    extent = [lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]]
    
    # Plot 1: Mean SNR map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(snr_grid, extent=extent, origin='lower', cmap='RdYlGn',
                     aspect='auto', vmin=-10, vmax=40)
    ax1.set_xlabel('Longitude (°)', fontweight='bold')
    ax1.set_ylabel('Latitude (°)', fontweight='bold')
    ax1.set_title('Mean SNR (dB) per Region', fontweight='bold', pad=10)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('SNR (dB)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean Error map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(error_grid, extent=extent, origin='lower', cmap='YlOrRd',
                     aspect='auto', vmin=0, vmax=np.nanpercentile(error_grid, 95))
    ax2.set_xlabel('Longitude (°)', fontweight='bold')
    ax2.set_ylabel('Latitude (°)', fontweight='bold')
    ax2.set_title('Mean Error (m) per Region', fontweight='bold', pad=10)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Error (m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Local correlation map
    ax3 = axes[1, 0]
    im3 = ax3.imshow(corr_grid, extent=extent, origin='lower', cmap='RdBu_r',
                     aspect='auto', vmin=-1, vmax=1)
    ax3.set_xlabel('Longitude (°)', fontweight='bold')
    ax3.set_ylabel('Latitude (°)', fontweight='bold')
    ax3.set_title('Local Error-SNR Correlation', fontweight='bold', pad=10)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Correlation', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample count map (log scale)
    ax4 = axes[1, 1]
    count_grid_plot = np.where(count_grid > 0, count_grid, np.nan)
    im4 = ax4.imshow(np.log10(count_grid_plot), extent=extent, origin='lower', 
                     cmap='viridis', aspect='auto')
    ax4.set_xlabel('Longitude (°)', fontweight='bold')
    ax4.set_ylabel('Latitude (°)', fontweight='bold')
    ax4.set_title('Sample Count per Region (log₁₀)', fontweight='bold', pad=10)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('log₁₀(count)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved spatial analysis to: {output_path}")
    plt.close()
    
    # Print spatial statistics
    print("\n" + "="*80)
    print("SPATIAL STATISTICS")
    print("="*80)
    print(f"Mean SNR across regions: {np.nanmean(snr_grid):.2f} dB")
    print(f"Mean error across regions: {np.nanmean(error_grid):.3f} m")
    print(f"Regions with valid correlation: {np.sum(np.isfinite(corr_grid))}")
    if np.sum(np.isfinite(corr_grid)) > 0:
        print(f"Mean local correlation: {np.nanmean(corr_grid):.3f}")
        print(f"Correlation range: [{np.nanmin(corr_grid):.3f}, {np.nanmax(corr_grid):.3f}]")
    print("="*80)


def load_parquet_data(
    input_dir: str,
    ground_truth_col: str = "vhm0_y",
    reference_col: str = "VHM0",
    file_pattern: str = "WAVEAN*.parquet",
    max_files: int = None,
    load_coords: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load ground truth and reference data from parquet files.
    
    Args:
        input_dir: Directory containing parquet files (local or S3)
        ground_truth_col: Column name for ground truth (e.g., 'vhm0_y', 'corrected_VHM0')
        reference_col: Column name for reference/uncorrected data (e.g., 'VHM0')
        file_pattern: Glob pattern for files to load
        max_files: Maximum number of files to process (None = all)
        load_coords: Whether to also load latitude/longitude coordinates
    
    Returns:
        ground_truth, reference, latitude, longitude as flattened numpy arrays
        (latitude and longitude are None if load_coords=False)
    """
    print(f"\n{'='*80}")
    print("LOADING DATA FROM PARQUET FILES")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Ground truth column: {ground_truth_col}")
    print(f"Reference column: {reference_col}")
    print(f"File pattern: {file_pattern}")
    
    # Get file list
    if input_dir.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        files = sorted(fs.glob(os.path.join(input_dir.rstrip("/"), file_pattern)))
        # Ensure S3 scheme
        files = [p if p.startswith("s3://") else f"s3://{p}" for p in files]
        is_s3 = True
    else:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        is_s3 = False
    
    if max_files is not None:
        files = files[:max_files]
    
    print(f"Found {len(files)} files to process")
    
    if len(files) == 0:
        raise ValueError(f"No files found matching pattern {file_pattern} in {input_dir}")
    
    ground_truth_list = []
    reference_list = []
    lat_list = [] if load_coords else None
    lon_list = [] if load_coords else None
    
    # Load data from each file
    for file_path in tqdm(files, desc="Loading parquet files"):
        try:
            # Determine columns to load
            cols_to_load = [ground_truth_col, reference_col]
            if load_coords:
                cols_to_load.extend(['latitude', 'longitude'])
            
            if is_s3:
                with fsspec.open(file_path, "rb") as fh:
                    table = pq.read_table(fh, columns=cols_to_load)
            else:
                table = pq.read_table(file_path, columns=cols_to_load)
            
            # Extract columns as numpy arrays
            gt_data = table.column(ground_truth_col).to_numpy()
            ref_data = table.column(reference_col).to_numpy()
            
            # Filter out NaN/Inf values and negative values (land pixels and invalid data)
            valid_mask = np.isfinite(gt_data) & np.isfinite(ref_data) & (gt_data >= 0) & (ref_data >= 0)
            
            if load_coords:
                lat_data = table.column('latitude').to_numpy()
                lon_data = table.column('longitude').to_numpy()
                valid_mask = valid_mask & np.isfinite(lat_data) & np.isfinite(lon_data)
            
            gt_data = gt_data[valid_mask]
            ref_data = ref_data[valid_mask]
            
            if len(gt_data) > 0:
                ground_truth_list.append(gt_data)
                reference_list.append(ref_data)
                
                if load_coords:
                    lat_list.append(lat_data[valid_mask])
                    lon_list.append(lon_data[valid_mask])
            
            del table, gt_data, ref_data
            if load_coords:
                del lat_data, lon_data
            
        except Exception as e:
            print(f"\n⚠ Warning: Failed to load {os.path.basename(file_path)}: {e}")
            continue
    
    # Concatenate all data
    ground_truth = np.concatenate(ground_truth_list)
    reference = np.concatenate(reference_list)
    latitude = np.concatenate(lat_list) if load_coords else None
    longitude = np.concatenate(lon_list) if load_coords else None
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Total samples: {len(ground_truth):,}")
    print(f"  Ground truth range: [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
    print(f"  Reference range: [{reference.min():.3f}, {reference.max():.3f}]")
    if load_coords:
        print(f"  Latitude range: [{latitude.min():.3f}, {latitude.max():.3f}]")
        print(f"  Longitude range: [{longitude.min():.3f}, {longitude.max():.3f}]")
    print(f"{'='*80}\n")
    
    return ground_truth, reference, latitude, longitude


# Example usage / Integration with evaluate_bunet.py
if __name__ == "__main__":
    """
    Load data from parquet files and run SNR analysis.
    
    Usage:
        # For local files:
        python calculate_snr.py --input-dir /path/to/data --year 2021
        
        # For S3 files:
        python calculate_snr.py --input-dir s3://medwav-dev-data/parquet/hourly/ --year 2021
        
        # With custom columns:
        python calculate_snr.py --input-dir /path/to/data --year 2021 \
            --ground-truth-col corrected_VHM0 --reference-col VHM0
    """
    
    parser = argparse.ArgumentParser(description="SNR Analysis for Wave Model Evaluation")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/tsolis/AI_project/parquet/augmented_with_labels/hourly",
        help="Directory containing parquet files (local or S3 path)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Year to analyze"
    )
    parser.add_argument(
        "--ground-truth-col",
        type=str,
        default="corrected_VHM0",
        help="Column name for ground truth (e.g., vhm0_y, corrected_VHM0)"
    )
    parser.add_argument(
        "--reference-col",
        type=str,
        default="VHM0",
        help="Column name for reference/uncorrected data (e.g., VHM0)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="snr_analysis_complete.png",
        help="Output path for plot"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples per bin"
    )
    parser.add_argument(
        "--spatial-analysis",
        action="store_true",
        help="Perform spatial SNR analysis (generates additional map plots)"
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.5,
        help="Grid resolution in degrees for spatial analysis (default: 0.5°)"
    )
    
    args = parser.parse_args()
    
    # Construct input directory with year
    if args.input_dir.startswith("s3://"):
        input_dir = f"{args.input_dir}year={args.year}/"
    else:
        input_dir = args.input_dir
    
    # Load data from parquet files
    ground_truth, reference, latitude, longitude = load_parquet_data(
        input_dir=input_dir,
        ground_truth_col=args.ground_truth_col,
        reference_col=args.reference_col,
        file_pattern=f"WAVEAN{args.year}*.parquet",
        max_files=args.max_files,
        load_coords=args.spatial_analysis
    )
    
    # Define wave height bins (matching your sea_bins in evaluate_bunet.py)
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # Run complete analysis
    bin_results, correlations = analyze_snr_complete(
        ground_truth=ground_truth,
        reference=reference,
        bin_edges=bin_edges,
        output_path=args.output_path,
        min_samples=args.min_samples
    )
    
    # Perform spatial analysis if requested
    if args.spatial_analysis and latitude is not None and longitude is not None:
        print("\nPreparing spatial analysis...")
        # Compute SNR per pixel for spatial analysis
        snr_per_pixel = compute_local_snr_per_pixel(ground_truth, reference)
        
        spatial_output = args.output_path.replace('.png', '_spatial.png')
        spatial_snr_analysis(
            ground_truth=ground_truth,
            reference=reference,
            snr_per_pixel=snr_per_pixel,
            lat_coords=latitude,
            lon_coords=longitude,
            output_path=spatial_output,
            grid_resolution=args.grid_resolution
        )
    
    print("\n✓ SNR Analysis Complete!")
    print("\nNext steps:")
    print(f"  1. Review the 4-panel plot: {args.output_path}")
    if args.spatial_analysis:
        print(f"  2. Review the spatial analysis: {args.output_path.replace('.png', '_spatial.png')}")
    print("  3. Check correlation values in the table above")
    print("  4. If correlation > 0.6, consider adding SNR as input feature to TransUNet")
    print("  5. Focus on bins with high correlation for local SNR features")
    print("\nTo analyze a different year, run:")
    print(f"  python calculate_snr.py --input-dir {args.input_dir} --year <YEAR>")
    if not args.spatial_analysis:
        print("\nTo include spatial analysis, add: --spatial-analysis")