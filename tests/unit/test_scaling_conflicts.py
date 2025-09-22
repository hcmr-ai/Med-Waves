#!/usr/bin/env python3
"""
Test Scaling Configuration Conflicts

This script demonstrates the validation logic for scaling configurations.
"""

import sys
import os
sys.path.append('.')

import yaml
from src.classifiers.full_dataset_trainer import FullDatasetTrainer

def test_scaling_configurations():
    """Test different scaling configuration combinations."""
    
    # Base configuration
    base_config = {
        "model": {"type": "xgb", "n_estimators": 10},
        "data": {"max_samples_per_file": 1000},
        "preprocessing": {
            "scaler": "standard",
            "regional_scaling": {"enabled": False}
        },
        "feature_block": {
            "features_to_exclude": ["time", "lat", "lon"]
        }
    }
    
    test_cases = [
        {
            "name": "Valid: Standard scaling",
            "config": {
                "preprocessing": {
                    "scaler": "standard",
                    "regional_scaling": {"enabled": False}
                }
            }
        },
        {
            "name": "Valid: Regional scaling with standard base",
            "config": {
                "preprocessing": {
                    "scaler": "standard",
                    "regional_scaling": {"enabled": True}
                }
            }
        },
        {
            "name": "Valid: No scaling",
            "config": {
                "preprocessing": {
                    "scaler": "null",
                    "regional_scaling": {"enabled": False}
                }
            }
        },
        {
            "name": "Invalid: Null scaler with regional scaling",
            "config": {
                "preprocessing": {
                    "scaler": "null",
                    "regional_scaling": {"enabled": True}
                }
            }
        }
    ]
    
    print("Testing Scaling Configuration Validation")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 40)
        
        # Merge base config with test config
        config = base_config.copy()
        config["preprocessing"].update(test_case["config"]["preprocessing"])
        
        try:
            # Create trainer (this will trigger validation)
            trainer = FullDatasetTrainer(config)
            print("✅ Configuration is valid")
            print(f"   Scaler: {trainer.scaler}")
            print(f"   Regional Scaler: {trainer.regional_scaler}")
            
        except Exception as e:
            print(f"❌ Configuration error: {e}")
    
    print("\n" + "=" * 60)
    print("Configuration Rules:")
    print("1. scaler: 'null' → No scaling applied")
    print("2. regional_scaling.enabled: true → Requires valid scaler")
    print("3. Cannot combine scaler: 'null' with regional_scaling: true")
    print("4. Valid combinations:")
    print("   - scaler: 'standard/robust/minmax' + regional_scaling: false")
    print("   - scaler: 'standard/robust/minmax' + regional_scaling: true")
    print("   - scaler: 'null' + regional_scaling: false")

if __name__ == "__main__":
    test_scaling_configurations()
