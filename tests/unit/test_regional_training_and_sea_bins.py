#!/usr/bin/env python3
"""
Test Regional Training and Sea-Bin Metrics

This test verifies that regional training and sea-bin metrics
are correctly implemented and working.
"""

import sys
import os
sys.path.append('.')

import polars as pl
import numpy as np
from src.classifiers.full_dataset_trainer import FullDatasetTrainer

def create_test_data():
    """Create test data with temporal structure and different regions."""
    # Create test data with multiple time steps per location
    n_locations = 3
    n_timesteps = 10
    
    data = []
    for lat in range(n_locations):
        for lon_idx in range(n_locations):
            for t in range(n_timesteps):
                # Create different regions based on longitude
                if lon_idx == 0:  # Atlantic region (lon < -5)
                    lon = -6.0
                    region = "atlantic"
                    # Higher wave heights for Atlantic (more challenging conditions)
                    vhm0_x = 2.0 + 0.2 * t + 0.1 * lat
                    vhm0_y = 2.2 + 0.2 * t + 0.1 * lat
                elif lon_idx == 2:  # Eastern Med region (lon > 30)
                    lon = 31.0
                    region = "eastern_med"
                    # Medium wave heights
                    vhm0_x = 1.0 + 0.1 * t + 0.05 * lat
                    vhm0_y = 1.1 + 0.1 * t + 0.05 * lat
                else:  # Mediterranean region (-5 <= lon <= 30)
                    lon = 15.0
                    region = "mediterranean"
                    # Lower wave heights (calm conditions)
                    vhm0_x = 0.5 + 0.05 * t + 0.02 * lat
                    vhm0_y = 0.6 + 0.05 * t + 0.02 * lat
                
                data.append({
                    'lat': lat,
                    'lon': lon,
                    'time': t,
                    'vhm0_x': vhm0_x,
                    'wspd': 10.0 + 0.5 * t + 0.1 * lat,
                    'vtm02': 5.0 + 0.2 * t,
                    'wdir': 180.0 + 10 * t,
                    'vmdr': 90.0 + 5 * t,
                    'vhm0_y': vhm0_y,
                })
    
    return pl.DataFrame(data)

def test_regional_training():
    """Test that regional training filters data correctly."""
    # Create test data
    df = create_test_data()
    print(f"Original data shape: {df.shape}")
    
    # Create config with regional training enabled
    config = {
        "feature_block": {
            "regional_training": {
                "enabled": True,
                "training_regions": ["atlantic"]
            },
            "features_to_exclude": ["time", "lat", "lon"]
        },
        "model": {"type": "xgb"},
        "data": {"max_samples_per_file": 1000}
    }
    
    # Initialize trainer
    trainer = FullDatasetTrainer(config)
    
    # Test feature preparation
    X, y, regions = trainer._prepare_features(df, "vhm0_y")
    
    # Check that only Atlantic data is used
    unique_regions = np.unique(regions)
    print(f"Regions after filtering: {unique_regions}")
    
    assert "atlantic" in unique_regions, "Atlantic region should be present"
    assert len(unique_regions) == 1, f"Should only have Atlantic region, got: {unique_regions}"
    assert X.shape[0] > 0, "Should have some data after filtering"
    
    print("✅ Regional training test passed")

def test_sea_bin_metrics():
    """Test that sea-bin metrics are calculated correctly."""
    # Create test data
    df = create_test_data()
    
    # Create config with sea-bin metrics enabled
    config = {
        "feature_block": {
            "sea_bin_metrics": {
                "enabled": True,
                "bins": [
                    {"name": "calm_sea", "min": 0.0, "max": 1.0, "description": "Calm conditions"},
                    {"name": "moderate_sea", "min": 1.0, "max": 2.0, "description": "Moderate conditions"},
                    {"name": "rough_sea", "min": 2.0, "max": 999.0, "description": "Rough conditions"}
                ]
            },
            "features_to_exclude": ["time", "lat", "lon"]
        },
        "model": {"type": "xgb"},
        "data": {"max_samples_per_file": 1000}
    }
    
    # Initialize trainer
    trainer = FullDatasetTrainer(config)
    
    # Test feature preparation
    X, y, regions = trainer._prepare_features(df, "vhm0_y")
    
    # Create dummy predictions for testing
    y_pred = y + np.random.normal(0, 0.1, len(y))  # Add some noise
    
    # Test sea-bin metrics calculation
    sea_bin_metrics = trainer._calculate_sea_bin_metrics(y, y_pred)
    
    # Check that sea-bin metrics were calculated
    assert len(sea_bin_metrics) > 0, "Should have sea-bin metrics"
    
    # Check that each bin has the expected structure
    for bin_name, metrics in sea_bin_metrics.items():
        assert "rmse" in metrics, f"Bin {bin_name} should have RMSE"
        assert "mae" in metrics, f"Bin {bin_name} should have MAE"
        assert "pearson" in metrics, f"Bin {bin_name} should have Pearson"
        assert "count" in metrics, f"Bin {bin_name} should have count"
        assert "percentage" in metrics, f"Bin {bin_name} should have percentage"
        assert metrics["count"] > 0, f"Bin {bin_name} should have some samples"
    
    print("✅ Sea-bin metrics test passed")
    print(f"Sea-bin metrics: {list(sea_bin_metrics.keys())}")

def test_regional_training_disabled():
    """Test that regional training works when disabled."""
    # Create test data
    df = create_test_data()
    
    # Create config with regional training disabled
    config = {
        "feature_block": {
            "regional_training": {
                "enabled": False,
                "training_regions": ["atlantic"]
            },
            "features_to_exclude": ["time", "lat", "lon"]
        },
        "model": {"type": "xgb"},
        "data": {"max_samples_per_file": 1000}
    }
    
    # Initialize trainer
    trainer = FullDatasetTrainer(config)
    
    # Test feature preparation
    X, y, regions = trainer._prepare_features(df, "vhm0_y")
    
    # Check that all regions are present when disabled
    unique_regions = np.unique(regions)
    print(f"Regions when disabled: {unique_regions}")
    
    assert len(unique_regions) > 1, "Should have multiple regions when disabled"
    assert "atlantic" in unique_regions, "Atlantic should be present"
    assert "mediterranean" in unique_regions, "Mediterranean should be present"
    
    print("✅ Regional training disabled test passed")

def test_sea_bin_metrics_disabled():
    """Test that sea-bin metrics work when disabled."""
    # Create test data
    df = create_test_data()
    
    # Create config with sea-bin metrics disabled
    config = {
        "feature_block": {
            "sea_bin_metrics": {
                "enabled": False,
                "bins": []
            },
            "features_to_exclude": ["time", "lat", "lon"]
        },
        "model": {"type": "xgb"},
        "data": {"max_samples_per_file": 1000}
    }
    
    # Initialize trainer
    trainer = FullDatasetTrainer(config)
    
    # Test feature preparation
    X, y, regions = trainer._prepare_features(df, "vhm0_y")
    
    # Create dummy predictions for testing
    y_pred = y + np.random.normal(0, 0.1, len(y))
    
    # Test sea-bin metrics calculation
    sea_bin_metrics = trainer._calculate_sea_bin_metrics(y, y_pred)
    
    # Check that no sea-bin metrics were calculated
    assert len(sea_bin_metrics) == 0, "Should have no sea-bin metrics when disabled"
    
    print("✅ Sea-bin metrics disabled test passed")

if __name__ == "__main__":
    # Run tests
    test_regional_training()
    test_sea_bin_metrics()
    test_regional_training_disabled()
    test_sea_bin_metrics_disabled()
    print("All regional training and sea-bin metrics tests passed! ✅")
