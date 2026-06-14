#!/usr/bin/env python3
"""
Script to manually create a new Comet ML experiment for DiffCorrector evaluation
"""

from datetime import datetime
from pathlib import Path
import sys

from comet_ml import Experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.commons.comet_utils import resolve_comet_settings

def create_manual_experiment():
    """
    Create a new Comet ML experiment manually
    """
    # Create a new experiment
    comet_kwargs = resolve_comet_settings(require_api_key=True)
    comet_kwargs.pop("project", None)
    experiment = Experiment(**comet_kwargs)
    
    # Set experiment name and description
    experiment_name = f"DiffCorrector_manual_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment.set_name(experiment_name)
    
    # Add tags
    experiment.add_tag("manual")
    experiment.add_tag("evaluation")
    experiment.add_tag("diff_corrector")
    
    # Log some basic parameters
    experiment.log_parameter("evaluation_type", "manual")
    experiment.log_parameter("corrector", "DiffCorrector")
    experiment.log_parameter("created_at", datetime.now().isoformat())
    
    print(f"✅ Created new experiment: {experiment_name}")
    print(f"🔗 Experiment URL: {experiment.url}")
    print(f"🔑 Experiment Key: {experiment.get_key()}")
    
    return experiment

if __name__ == "__main__":
    experiment = create_manual_experiment()
    print("\n📋 You can now use this experiment key in your baseline plotter script!")
