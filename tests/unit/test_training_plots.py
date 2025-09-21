#!/usr/bin/env python3
"""
Test Training Plots

This test verifies that the training and evaluation plots are correctly
created and saved.
"""

import sys
import os
sys.path.append('.')

import numpy as np
from pathlib import Path
from src.analytics.plots.training_plots import (
    plot_sea_bin_metrics, plot_regional_comparison, 
    plot_prediction_vs_actual, plot_error_analysis,
    plot_training_progress, create_comprehensive_plots
)

def create_test_metrics():
    """Create test metrics for plotting."""
    # Sea-bin metrics
    sea_bin_metrics = {
        "calm_sea": {
            "rmse": 0.0821,
            "mae": 0.0609,
            "pearson": 0.9958,
            "count": 2427696,
            "percentage": 55.2
        },
        "slight_sea": {
            "rmse": 0.0943,
            "mae": 0.0701,
            "pearson": 0.9885,
            "count": 1234567,
            "percentage": 28.1
        },
        "moderate_sea": {
            "rmse": 0.1241,
            "mae": 0.0957,
            "pearson": 0.9723,
            "count": 456789,
            "percentage": 10.4
        },
        "rough_sea": {
            "rmse": 0.1876,
            "mae": 0.1345,
            "pearson": 0.9876,
            "count": 234567,
            "percentage": 5.3
        },
        "high_sea": {
            "rmse": 0.3048,
            "mae": 0.2277,
            "pearson": 0.9697,
            "count": 45678,
            "percentage": 1.0
        }
    }
    
    # Regional metrics
    regional_metrics = {
        "atlantic": {
            "rmse": 0.3048,
            "mae": 0.2277,
            "pearson": 0.9697
        },
        "mediterranean": {
            "rmse": 0.1876,
            "mae": 0.1345,
            "pearson": 0.9876
        },
        "eastern_med": {
            "rmse": 0.2034,
            "mae": 0.1456,
            "pearson": 0.9854
        }
    }
    
    # Training history
    training_history = {
        "train_loss": [0.5, 0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20],
        "val_loss": [0.6, 0.5, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]
    }
    
    return sea_bin_metrics, regional_metrics, training_history

def create_test_data():
    """Create test data for plotting."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic wave height data
    y_true = np.random.gamma(2, 0.5, n_samples)  # Gamma distribution for wave heights
    y_pred = y_true + np.random.normal(0, 0.1, n_samples)  # Add some noise
    
    # Create regions
    regions = np.random.choice(["atlantic", "mediterranean", "eastern_med"], n_samples)
    
    return y_true, y_pred, regions

def test_sea_bin_metrics_plot():
    """Test sea-bin metrics plotting."""
    sea_bin_metrics, _, _ = create_test_metrics()
    
    # Test plotting
    plot_sea_bin_metrics(
        sea_bin_metrics,
        title="Test Sea-Bin Metrics",
        save_path="test_sea_bin_metrics.png"
    )
    
    # Check if file was created
    assert Path("test_sea_bin_metrics.png").exists(), "Sea-bin metrics plot should be created"
    print("✅ Sea-bin metrics plot test passed")

def test_regional_comparison_plot():
    """Test regional comparison plotting."""
    _, regional_metrics, _ = create_test_metrics()
    
    # Test plotting
    plot_regional_comparison(
        regional_metrics,
        title="Test Regional Comparison",
        save_path="test_regional_comparison.png"
    )
    
    # Check if file was created
    assert Path("test_regional_comparison.png").exists(), "Regional comparison plot should be created"
    print("✅ Regional comparison plot test passed")

def test_prediction_vs_actual_plot():
    """Test predictions vs actual plotting."""
    y_true, y_pred, regions = create_test_data()
    
    # Test plotting
    plot_prediction_vs_actual(
        y_true, y_pred, regions,
        title="Test Predictions vs Actual",
        save_path="test_predictions_vs_actual.png"
    )
    
    # Check if file was created
    assert Path("test_predictions_vs_actual.png").exists(), "Predictions vs actual plot should be created"
    print("✅ Predictions vs actual plot test passed")

def test_error_analysis_plot():
    """Test error analysis plotting."""
    y_true, y_pred, regions = create_test_data()
    
    # Test plotting
    plot_error_analysis(
        y_true, y_pred, regions,
        title="Test Error Analysis",
        save_path="test_error_analysis.png"
    )
    
    # Check if file was created
    assert Path("test_error_analysis.png").exists(), "Error analysis plot should be created"
    print("✅ Error analysis plot test passed")

def test_training_progress_plot():
    """Test training progress plotting."""
    _, _, training_history = create_test_metrics()
    
    # Test plotting
    plot_training_progress(
        training_history,
        title="Test Training Progress",
        save_path="test_training_progress.png"
    )
    
    # Check if file was created
    assert Path("test_training_progress.png").exists(), "Training progress plot should be created"
    print("✅ Training progress plot test passed")

def test_comprehensive_plots():
    """Test comprehensive plotting function."""
    sea_bin_metrics, regional_metrics, training_history = create_test_metrics()
    y_true, y_pred, regions = create_test_data()
    
    # Test comprehensive plotting
    create_comprehensive_plots(
        y_true=y_true,
        y_pred=y_pred,
        regions=regions,
        sea_bin_metrics=sea_bin_metrics,
        regional_metrics=regional_metrics,
        training_history=training_history,
        output_dir="test_plots"
    )
    
    # Check if directory was created
    plots_dir = Path("test_plots")
    assert plots_dir.exists(), "Plots directory should be created"
    
    # Check if expected files exist
    expected_files = [
        "sea_bin_metrics.png",
        "regional_comparison.png", 
        "predictions_vs_actual.png",
        "error_analysis.png",
        "training_progress.png"
    ]
    
    for file in expected_files:
        assert (plots_dir / file).exists(), f"Plot file {file} should be created"
    
    print("✅ Comprehensive plots test passed")

def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_sea_bin_metrics.png",
        "test_regional_comparison.png",
        "test_predictions_vs_actual.png", 
        "test_error_analysis.png",
        "test_training_progress.png"
    ]
    
    for file in test_files:
        if Path(file).exists():
            Path(file).unlink()
    
    # Remove test plots directory
    test_plots_dir = Path("test_plots")
    if test_plots_dir.exists():
        for file in test_plots_dir.glob("*.png"):
            file.unlink()
        test_plots_dir.rmdir()
    
    print("✅ Test files cleaned up")

if __name__ == "__main__":
    # Run tests
    test_sea_bin_metrics_plot()
    test_regional_comparison_plot()
    test_prediction_vs_actual_plot()
    test_error_analysis_plot()
    test_training_progress_plot()
    test_comprehensive_plots()
    
    # Clean up
    cleanup_test_files()
    
    print("All training plots tests passed! ✅")
