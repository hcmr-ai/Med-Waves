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
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Storage for predictions
        self.all_predictions = []
        self.all_targets = []
        self.all_masks = []
        self.all_uncorrected = []
        
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
    
    def run_inference(self):
        """Run model inference on test set."""
        print("Running inference on test set...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Processing batches"):
                X, y, mask = batch
                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                
                # Get predictions
                y_pred = self.model(X)
                
                # Align dimensions
                min_h = min(y_pred.shape[2], y.shape[2])
                min_w = min(y_pred.shape[3], y.shape[3])
                y_pred = y_pred[:, :, :min_h, :min_w]
                y = y[:, :, :min_h, :min_w]
                mask = mask[:, :, :min_h, :min_w]
                
                # Extract uncorrected VHM0 (assuming it's in channel 1)
                vhm0_uncorrected = X[:, 1:2, :min_h, :min_w]
                
                # Store results
                self.all_predictions.append(y_pred.cpu().numpy())
                self.all_targets.append(y.cpu().numpy())
                self.all_masks.append(mask.cpu().numpy())
                self.all_uncorrected.append(vhm0_uncorrected.cpu().numpy())
        
        # Concatenate all batches
        self.all_predictions = np.concatenate(self.all_predictions, axis=0)
        self.all_targets = np.concatenate(self.all_targets, axis=0)
        self.all_masks = np.concatenate(self.all_masks, axis=0)
        self.all_uncorrected = np.concatenate(self.all_uncorrected, axis=0)
        
        print(f"Inference complete. Processed {len(self.all_predictions)} samples.")
    
    def run_inference(self):
        print("Running inference on test set...")
        self.model.eval()
        total_mae, total_rmse, total_count = 0, 0, 0

        with torch.no_grad():
            for X, y, mask in tqdm(self.test_loader, desc="Processing batches"):
                X, y, mask = [t.to(self.device) for t in (X, y, mask)]
                y_pred = self.model(X)

                min_h, min_w = min(y_pred.shape[2], y.shape[2]), min(y_pred.shape[3], y.shape[3])
                y_pred, y, mask = y_pred[:, :, :min_h, :min_w], y[:, :, :min_h, :min_w], mask[:, :, :min_h, :min_w]

                diff = (y_pred - y)[mask]
                total_mae += diff.abs().sum().item()
                total_rmse += (diff ** 2).sum().item()
                total_count += mask.sum().item()

        mae = total_mae / total_count
        rmse = (total_rmse / total_count) ** 0.5
        print(f"Inference complete. MAE={mae:.3f}, RMSE={rmse:.3f}, n={total_count}")

    
    def compute_overall_metrics(self) -> Dict[str, float]:
        """Compute overall performance metrics."""
        # Apply mask
        valid_mask = self.all_masks.astype(bool)
        y_true = self.all_targets[valid_mask]
        y_pred = self.all_predictions[valid_mask]
        y_uncorrected = self.all_uncorrected[valid_mask]
        
        # Model metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        bias = np.mean(y_pred - y_true)
        mse = np.mean((y_pred - y_true) ** 2)
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Baseline (uncorrected) metrics
        baseline_mae = np.mean(np.abs(y_uncorrected - y_true))
        baseline_rmse = np.sqrt(np.mean((y_uncorrected - y_true) ** 2))
        baseline_bias = np.mean(y_uncorrected - y_true)
        
        # Improvement
        mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
        rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        
        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "bias": float(bias),
            "mse": float(mse),
            "r2": float(r2),
            "correlation": float(correlation),
            "baseline_mae": float(baseline_mae),
            "baseline_rmse": float(baseline_rmse),
            "baseline_bias": float(baseline_bias),
            "mae_improvement_pct": float(mae_improvement),
            "rmse_improvement_pct": float(rmse_improvement),
            "n_samples": int(len(y_true))
        }
        
        return metrics
    
    def compute_sea_bin_metrics(self) -> Dict[str, Dict]:
        """Compute metrics for each sea state bin."""
        valid_mask = self.all_masks.astype(bool)
        y_true = self.all_targets[valid_mask]
        y_pred = self.all_predictions[valid_mask]
        y_uncorrected = self.all_uncorrected[valid_mask]
        
        sea_bin_metrics = {}
        
        for bin_config in self.sea_bins:
            bin_name = bin_config["name"]
            bin_min = bin_config["min"]
            bin_max = bin_config["max"]
            
            # Filter data for this bin
            bin_mask = (y_true >= bin_min) & (y_true < bin_max)
            bin_count = np.sum(bin_mask)
            
            if bin_count > 0:
                bin_y_true = y_true[bin_mask]
                bin_y_pred = y_pred[bin_mask]
                bin_y_uncorrected = y_uncorrected[bin_mask]
                
                # Model metrics
                mae = np.mean(np.abs(bin_y_pred - bin_y_true))
                rmse = np.sqrt(np.mean((bin_y_pred - bin_y_true) ** 2))
                bias = np.mean(bin_y_pred - bin_y_true)
                
                # Baseline metrics
                baseline_mae = np.mean(np.abs(bin_y_uncorrected - bin_y_true))
                baseline_rmse = np.sqrt(np.mean((bin_y_uncorrected - bin_y_true) ** 2))
                baseline_bias = np.mean(bin_y_uncorrected - bin_y_true)
                
                # Improvement
                mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
                rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
                
                sea_bin_metrics[bin_name] = {
                    "label": bin_config["label"],
                    "count": int(bin_count),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "bias": float(bias),
                    "baseline_mae": float(baseline_mae),
                    "baseline_rmse": float(baseline_rmse),
                    "baseline_bias": float(baseline_bias),
                    "mae_improvement_pct": float(mae_improvement),
                    "rmse_improvement_pct": float(rmse_improvement)
                }
        
        return sea_bin_metrics
    
    def plot_scatter(self):
        """Create scatter plot of predictions vs targets."""
        valid_mask = self.all_masks.astype(bool)
        y_true = self.all_targets[valid_mask]
        y_pred = self.all_predictions[valid_mask]
        y_uncorrected = self.all_uncorrected[valid_mask]
        
        # Sample for visualization if too many points
        max_points = 10000
        if len(y_true) > max_points:
            indices = np.random.choice(len(y_true), max_points, replace=False)
            y_true = y_true[indices]
            y_pred = y_pred[indices]
            y_uncorrected = y_uncorrected[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Model predictions
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.3, s=1)
        ax1.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect fit')
        ax1.set_xlabel('True Wave Height (m)')
        ax1.set_ylabel('Predicted Wave Height (m)')
        ax1.set_title('Model Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Baseline (uncorrected)
        ax2 = axes[1]
        ax2.scatter(y_true, y_uncorrected, alpha=0.3, s=1, color='orange')
        ax2.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect fit')
        ax2.set_xlabel('True Wave Height (m)')
        ax2.set_ylabel('Uncorrected Wave Height (m)')
        ax2.set_title('Baseline (Uncorrected)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved scatter plot to {self.output_dir / 'scatter_plot.png'}")
    
    def plot_error_distribution(self):
        """Plot error distribution."""
        valid_mask = self.all_masks.astype(bool)
        y_true = self.all_targets[valid_mask]
        y_pred = self.all_predictions[valid_mask]
        y_uncorrected = self.all_uncorrected[valid_mask]
        
        model_errors = y_pred - y_true
        baseline_errors = y_uncorrected - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1 = axes[0]
        ax1.hist(model_errors, bins=100, alpha=0.6, label='Model', color='blue')
        ax1.hist(baseline_errors, bins=100, alpha=0.6, label='Baseline', color='orange')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Error (Predicted - True) (m)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        ax2.boxplot([model_errors, baseline_errors], 
                   labels=['Model', 'Baseline'])
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_ylabel('Error (m)')
        ax2.set_title('Error Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error distribution to {self.output_dir / 'error_distribution.png'}")
    
    def plot_sea_bin_metrics(self, sea_bin_metrics: Dict):
        """Plot sea-bin metrics comparison."""
        bins_with_data = [k for k, v in sea_bin_metrics.items() if v['count'] > 0]
        if not bins_with_data:
            print("No sea-bin data available for plotting")
            return
        
        labels = [sea_bin_metrics[b]['label'] for b in bins_with_data]
        
        # Extract metrics
        model_maes = [sea_bin_metrics[b]['mae'] for b in bins_with_data]
        baseline_maes = [sea_bin_metrics[b]['baseline_mae'] for b in bins_with_data]
        model_rmses = [sea_bin_metrics[b]['rmse'] for b in bins_with_data]
        baseline_rmses = [sea_bin_metrics[b]['baseline_rmse'] for b in bins_with_data]
        model_biases = [sea_bin_metrics[b]['bias'] for b in bins_with_data]
        baseline_biases = [sea_bin_metrics[b]['baseline_bias'] for b in bins_with_data]
        counts = [sea_bin_metrics[b]['count'] for b in bins_with_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(labels))
        width = 0.35
        
        # MAE
        ax1 = axes[0, 0]
        ax1.bar(x - width/2, model_maes, width, label='Model', alpha=0.8)
        ax1.bar(x + width/2, baseline_maes, width, label='Baseline', alpha=0.8)
        ax1.set_xlabel('Wave Height Range')
        ax1.set_ylabel('MAE (m)')
        ax1.set_title('Mean Absolute Error by Sea State')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, model_rmses, width, label='Model', alpha=0.8)
        ax2.bar(x + width/2, baseline_rmses, width, label='Baseline', alpha=0.8)
        ax2.set_xlabel('Wave Height Range')
        ax2.set_ylabel('RMSE (m)')
        ax2.set_title('Root Mean Square Error by Sea State')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bias
        ax3 = axes[1, 0]
        ax3.bar(x - width/2, model_biases, width, label='Model', alpha=0.8)
        ax3.bar(x + width/2, baseline_biases, width, label='Baseline', alpha=0.8)
        ax3.set_xlabel('Wave Height Range')
        ax3.set_ylabel('Bias (m)')
        ax3.set_title('Bias by Sea State')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sample counts
        ax4 = axes[1, 1]
        ax4.bar(x, counts, alpha=0.8)
        ax4.set_xlabel('Wave Height Range')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Sample Distribution by Sea State')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax4.text(i, count + max(counts)*0.01, f'{count:,}', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sea_bin_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sea-bin metrics to {self.output_dir / 'sea_bin_metrics.png'}")
    
    def plot_improvement_by_bin(self, sea_bin_metrics: Dict):
        """Plot improvement percentage by sea bin."""
        bins_with_data = [k for k, v in sea_bin_metrics.items() if v['count'] > 0]
        if not bins_with_data:
            return
        
        labels = [sea_bin_metrics[b]['label'] for b in bins_with_data]
        mae_improvements = [sea_bin_metrics[b]['mae_improvement_pct'] for b in bins_with_data]
        rmse_improvements = [sea_bin_metrics[b]['rmse_improvement_pct'] for b in bins_with_data]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, mae_improvements, width, label='MAE Improvement', alpha=0.8)
        ax.bar(x + width/2, rmse_improvements, width, label='RMSE Improvement', alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Wave Height Range')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Model Improvement vs Baseline by Sea State')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (mae_imp, rmse_imp) in enumerate(zip(mae_improvements, rmse_improvements)):
            ax.text(i - width/2, mae_imp + 1, f'{mae_imp:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, rmse_imp + 1, f'{rmse_imp:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_by_bin.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved improvement plot to {self.output_dir / 'improvement_by_bin.png'}")
    
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
        
        print("\nBaseline (Uncorrected) Metrics:")
        print(f"  MAE:                  {overall_metrics['baseline_mae']:.4f} m")
        print(f"  RMSE:                 {overall_metrics['baseline_rmse']:.4f} m")
        print(f"  Bias:                 {overall_metrics['baseline_bias']:.4f} m")
        
        print("\nImprovement:")
        print(f"  MAE Improvement:      {overall_metrics['mae_improvement_pct']:.2f}%")
        print(f"  RMSE Improvement:     {overall_metrics['rmse_improvement_pct']:.2f}%")
        
        print("\nSea-Bin Metrics:")
        print(f"{'Bin':<20} {'Count':<10} {'MAE':<10} {'RMSE':<10} {'Improvement':<15}")
        print("-" * 80)
        
        for bin_name, metrics in sea_bin_metrics.items():
            if metrics['count'] > 0:
                print(f"{metrics['label']:<20} "
                      f"{metrics['count']:<10,} "
                      f"{metrics['mae']:<10.4f} "
                      f"{metrics['rmse']:<10.4f} "
                      f"{metrics['mae_improvement_pct']:>7.2f}%")
        
        print("="*80 + "\n")
    
    def evaluate(self):
        """Run complete evaluation pipeline."""
        # Run inference
        self.run_inference()
        
        # Compute metrics
        print("\nComputing overall metrics...")
        overall_metrics = self.compute_overall_metrics()
        
        print("Computing sea-bin metrics...")
        sea_bin_metrics = self.compute_sea_bin_metrics()
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_scatter()
        self.plot_error_distribution()
        self.plot_sea_bin_metrics(sea_bin_metrics)
        self.plot_improvement_by_bin(sea_bin_metrics)
        
        # Save results
        print("\nSaving results...")
        self.save_metrics(overall_metrics, sea_bin_metrics)
        
        # Print summary
        self.print_summary(overall_metrics, sea_bin_metrics)
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate WaveBiasCorrector model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--config', type=str, default='./configs/config_dnn.yaml',
                       help='Configuration file')
    args = parser.parse_args()
    config = DNNConfig(args.config)
    model_config = config.config["model"]
    training_config = config.config["training"]
    data_config = config.config["data"]

    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )

    logger.info(f"Found {len(files)} files")

    _, _, test_files = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
    )

    logger.info(f"Test files: {len(test_files)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = WaveBiasCorrector.load_from_checkpoint(args.checkpoint)
    
    # For now, placeholder:
    print("Please implement your dataset loading in the main() function")
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Normalizer stats: {normalizer.stats_}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")

    normalizer = WaveNormalizer.load_from_s3("medwav-dev-data",data_config["normalizer_path"])
    
    test_dataset = CachedWaveDataset(
        test_files,
        # index_map=train_index_map,
        # fs=fs,
        patch_size=data_config["patch_size"],
        excluded_columns=data_config["excluded_columns"],
        target_column=data_config["target_column"],
        predict_bias=data_config["predict_bias"],
        subsample_step=data_config["subsample_step"],
        normalizer=normalizer,
    )

    # Create data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1
    )

    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        output_dir=Path(args.output_dir),
        device=args.device
    )
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()