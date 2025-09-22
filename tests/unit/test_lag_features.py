#!/usr/bin/env python3
"""
Test Lag Features Functionality

This test verifies that lag features are correctly created and integrated
into the feature set without causing data leakage.
"""

import sys
import os
sys.path.append('.')

import polars as pl
import numpy as np
from src.classifiers.full_dataset_trainer import FullDatasetTrainer

def create_test_data():
    """Create test data with temporal structure."""
    # Create test data with multiple time steps per location
    n_locations = 3
    n_timesteps = 10
    
    data = []
    for lat in range(n_locations):
        for lon in range(n_locations):
            for t in range(n_timesteps):
                data.append({
                    'lat': lat,
                    'lon': lon,
                    'time': t,
                    'vhm0_x': 1.0 + 0.1 * t + 0.01 * lat + 0.01 * lon,  # Degraded wave height
                    'wspd': 10.0 + 0.5 * t + 0.1 * lat,  # Wind speed
                    'vtm02': 5.0 + 0.2 * t,  # Wave period
                    'wdir': 180.0 + 10 * t,  # Wind direction
                    'vmdr': 90.0 + 5 * t,  # Wave direction
                    'vhm0_y': 1.2 + 0.1 * t + 0.01 * lat + 0.01 * lon,  # Corrected wave height (target)
                })
    
    return pl.DataFrame(data)

def test_lag_features_creation():
    """Test that lag features are correctly created."""
    # Create test data
    df = create_test_data()
    
    # Create config with lag features enabled
    config = {
        "feature_block": {
            "lag_features": {
                "enabled": True,
                "lags": {
                    "vhm0_x": [1, 2],
                    "wspd": [1],
                    "vtm02": [1]
                }
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
    
    # Check that lag features were added
    expected_lag_features = [
        "vhm0_x_lag_1h", "vhm0_x_lag_2h",
        "wspd_lag_1h",
        "vtm02_lag_1h"
    ]
    
    for lag_feature in expected_lag_features:
        assert lag_feature in trainer.feature_names, f"Lag feature {lag_feature} not found in feature names"
    
    # Check that we have the expected number of features
    base_features = ["vhm0_x", "wspd", "vtm02", "wdir", "vmdr"]
    # Basin feature is added by default (include_basin: true)
    expected_total_features = len(base_features) + len(expected_lag_features) + 1  # +1 for basin
    print(f"Expected features: {base_features + expected_lag_features + ['basin']}")
    print(f"Actual features: {trainer.feature_names}")
    print(f"Expected total: {expected_total_features}, Actual total: {len(trainer.feature_names)}")
    assert len(trainer.feature_names) == expected_total_features, f"Expected {expected_total_features} features, got {len(trainer.feature_names)}"

def test_lag_features_no_temporal_data():
    """Test that lag features are skipped when no temporal data is available."""
    # Create test data without time column
    df = create_test_data().drop("time")
    
    # Create config with lag features enabled
    config = {
        "feature_block": {
            "lag_features": {
                "enabled": True,
                "lags": {
                    "vhm0_x": [1, 2],
                    "wspd": [1]
                }
            },
            "features_to_exclude": ["lat", "lon"]
        },
        "model": {"type": "xgb"},
        "data": {"max_samples_per_file": 1000}
    }
    
    # Initialize trainer
    trainer = FullDatasetTrainer(config)
    
    # Test feature preparation
    X, y, regions = trainer._prepare_features(df, "vhm0_y")
    
    # Check that no lag features were added
    lag_features = [f for f in trainer.feature_names if "_lag_" in f]
    assert len(lag_features) == 0, f"Expected no lag features, but found: {lag_features}"

def test_lag_features_disabled():
    """Test that lag features are not created when disabled."""
    # Create test data
    df = create_test_data()
    
    # Create config with lag features disabled
    config = {
        "feature_block": {
            "lag_features": {
                "enabled": False,
                "lags": {
                    "vhm0_x": [1, 2],
                    "wspd": [1]
                }
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
    
    # Check that no lag features were added
    lag_features = [f for f in trainer.feature_names if "_lag_" in f]
    assert len(lag_features) == 0, f"Expected no lag features, but found: {lag_features}"

def test_lag_features_data_integrity():
    """Test that lag features maintain data integrity and don't cause leakage."""
    # Create test data
    df = create_test_data()
    
    # Create config with lag features enabled
    config = {
        "feature_block": {
            "lag_features": {
                "enabled": True,
                "lags": {
                    "vhm0_x": [1, 2],
                    "wspd": [1]
                }
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
    
    # Check that target variable is not used for lag features
    target_lag_features = [f for f in trainer.feature_names if "vhm0_y_lag_" in f]
    assert len(target_lag_features) == 0, f"Target variable lag features found: {target_lag_features} - this would be data leakage!"
    
    # Check that we have the expected shape
    assert X.shape[1] == len(trainer.feature_names), "Feature matrix shape doesn't match feature names"
    assert X.shape[0] == y.shape[0], "Feature matrix and target vector have different number of samples"

if __name__ == "__main__":
    # Run tests
    test_lag_features_creation()
    test_lag_features_no_temporal_data()
    test_lag_features_disabled()
    test_lag_features_data_integrity()
    print("All lag features tests passed! ✅")
