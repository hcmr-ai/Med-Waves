#!/usr/bin/env python3
"""
Test Extended DiagnosticPlotter

This test verifies that the extended DiagnosticPlotter correctly creates
regional analysis, sea-bin analysis, and spatial maps.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import polars as pl
from pathlib import Path
from src.evaluation.diagnostic_plotter import DiagnosticPlotter

def create_mock_trainer():
    """Create a mock trainer object for testing."""
    class MockTrainer:
        def __init__(self):
            # Basic data
            self.y_test = np.random.gamma(2, 0.5, 1000)
            self.y_test_coords = np.column_stack([
                np.random.uniform(30, 50, 1000),  # lat
                np.random.uniform(-10, 40, 1000)  # lon
            ])
            
            # Regional data
            self.regions_test = np.random.choice(["atlantic", "mediterranean", "eastern_med"], 1000)
            
            # Regional metrics
            self.regional_test_metrics = {
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
            
            # Sea-bin metrics
            self.sea_bin_test_metrics = {
                "calm_sea": {
                    "rmse": 0.0821,
                    "mae": 0.0609,
                    "pearson": 0.9958,
                    "count": 550,
                    "percentage": 55.0
                },
                "slight_sea": {
                    "rmse": 0.0943,
                    "mae": 0.0701,
                    "pearson": 0.9885,
                    "count": 250,
                    "percentage": 25.0
                },
                "moderate_sea": {
                    "rmse": 0.1241,
                    "mae": 0.0957,
                    "pearson": 0.9723,
                    "count": 150,
                    "percentage": 15.0
                },
                "rough_sea": {
                    "rmse": 0.1876,
                    "mae": 0.1345,
                    "pearson": 0.9876,
                    "count": 50,
                    "percentage": 5.0
                }
            }
            
            # Configuration
            self.config = {
                "feature_block": {
                    "sea_bin_metrics": {
                        "enabled": True,
                        "bins": [
                            {"name": "calm_sea", "min": 0.0, "max": 0.5},
                            {"name": "slight_sea", "min": 0.5, "max": 1.25},
                            {"name": "moderate_sea", "min": 1.25, "max": 2.5},
                            {"name": "rough_sea", "min": 2.5, "max": 4.0}
                        ]
                    }
                }
            }
            
            # Training history
            self.training_history = {
                "train_loss": [0.5, 0.4, 0.35, 0.32, 0.30],
                "val_loss": [0.6, 0.5, 0.45, 0.42, 0.40]
            }
            
            # Current metrics
            self.current_train_metrics = {"snr": 10.5, "snr_db": 10.2}
            self.current_test_metrics = {"snr": 8.3, "snr_db": 9.2}
    
    return MockTrainer()

def test_regional_analysis_plots():
    """Test regional analysis plotting functionality."""
    config = {
        "diagnostics": {
            "plots_save_path": "test_diagnostic_plots",
            "create_regional_analysis": True
        }
    }
    
    plotter = DiagnosticPlotter(config)
    trainer = create_mock_trainer()
    test_predictions = trainer.y_test + np.random.normal(0, 0.1, len(trainer.y_test))
    
    # Test regional comparison plot
    plotter._create_regional_comparison_plot(trainer, Path("test_diagnostic_plots"))
    
    # Test regional predictions plots
    plotter._create_regional_predictions_plots(trainer, test_predictions, Path("test_diagnostic_plots"))
    
    # Test regional error analysis
    plotter._create_regional_error_analysis(trainer, test_predictions, Path("test_diagnostic_plots"))
    
    # Check if files were created
    expected_files = [
        "test_diagnostic_plots/regional_comparison.png",
        "test_diagnostic_plots/regional_predictions_vs_actual.png",
        "test_diagnostic_plots/regional_error_analysis.png"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Regional analysis plot {file_path} should be created"
    
    print("✅ Regional analysis plots test passed")

def test_sea_bin_analysis_plots():
    """Test sea-bin analysis plotting functionality."""
    config = {
        "diagnostics": {
            "plots_save_path": "test_diagnostic_plots",
            "create_sea_bin_analysis": True
        }
    }
    
    plotter = DiagnosticPlotter(config)
    trainer = create_mock_trainer()
    test_predictions = trainer.y_test + np.random.normal(0, 0.1, len(trainer.y_test))
    
    # Test sea-bin performance plot
    plotter._create_sea_bin_performance_plot(trainer, Path("test_diagnostic_plots"))
    
    # Test sea-bin predictions plots
    plotter._create_sea_bin_predictions_plots(trainer, test_predictions, Path("test_diagnostic_plots"))
    
    # Check if files were created
    expected_files = [
        "test_diagnostic_plots/sea_bin_performance.png",
        "test_diagnostic_plots/sea_bin_predictions_vs_actual.png"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Sea-bin analysis plot {file_path} should be created"
    
    print("✅ Sea-bin analysis plots test passed")

def test_spatial_maps():
    """Test spatial maps functionality."""
    config = {
        "diagnostics": {
            "plots_save_path": "test_diagnostic_plots",
            "create_spatial_maps": True
        }
    }
    
    plotter = DiagnosticPlotter(config)
    trainer = create_mock_trainer()
    test_predictions = trainer.y_test + np.random.normal(0, 0.1, len(trainer.y_test))
    
    # Test spatial metrics calculation
    spatial_metrics = plotter._calculate_spatial_metrics(trainer, test_predictions)
    
    if spatial_metrics is not None:
        # Test spatial map creation
        spatial_dir = Path("test_diagnostic_plots/spatial_maps")
        spatial_dir.mkdir(exist_ok=True)
        
        for metric in ['rmse', 'mae', 'bias', 'pearson']:
            if metric in spatial_metrics.columns:
                plotter._create_spatial_map(spatial_metrics, metric, spatial_dir)
        
        # Check if spatial maps directory was created
        assert spatial_dir.exists(), "Spatial maps directory should be created"
        
        # Check if spatial map files exist
        spatial_files = list(spatial_dir.glob("*.png"))
        assert len(spatial_files) > 0, "At least one spatial map should be created"
    
    print("✅ Spatial maps test passed")

def test_full_diagnostic_plotter():
    """Test the full diagnostic plotter functionality."""
    config = {
        "diagnostics": {
            "plots_save_path": "test_diagnostic_plots",
            "create_regional_analysis": True,
            "create_sea_bin_analysis": True,
            "create_spatial_maps": True
        }
    }
    
    plotter = DiagnosticPlotter(config)
    trainer = create_mock_trainer()
    test_predictions = trainer.y_test + np.random.normal(0, 0.1, len(trainer.y_test))
    
    # Test full diagnostic plotting
    plotter.create_diagnostic_plots(trainer, test_predictions)
    
    # Check if main diagnostic plots were created
    expected_files = [
        "test_diagnostic_plots/test_predictions_vs_actual.png",
        "test_diagnostic_plots/train_predictions_vs_actual.png",
        "test_diagnostic_plots/test_residuals_vs_predicted.png",
        "test_diagnostic_plots/train_residuals_vs_predicted.png",
        "test_diagnostic_plots/learning_curves.png",
        "test_diagnostic_plots/snr_comparison_dual_scale.png",
        "test_diagnostic_plots/regional_comparison.png",
        "test_diagnostic_plots/regional_predictions_vs_actual.png",
        "test_diagnostic_plots/regional_error_analysis.png",
        "test_diagnostic_plots/sea_bin_performance.png",
        "test_diagnostic_plots/sea_bin_predictions_vs_actual.png"
    ]
    
    for file_path in expected_files:
        assert Path(file_path).exists(), f"Diagnostic plot {file_path} should be created"
    
    print("✅ Full diagnostic plotter test passed")

def cleanup_test_files():
    """Clean up test files."""
    test_dir = Path("test_diagnostic_plots")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    print("✅ Test files cleaned up")

if __name__ == "__main__":
    # Run tests
    test_regional_analysis_plots()
    test_sea_bin_analysis_plots()
    test_spatial_maps()
    test_full_diagnostic_plotter()
    
    # Clean up
    cleanup_test_files()
    
    print("All extended DiagnosticPlotter tests passed! ✅")
