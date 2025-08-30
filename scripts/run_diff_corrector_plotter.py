#!/usr/bin/env python3
"""
Script to run baseline plotter for DiffCorrector evaluation
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.baseline_correctors_plotter import main
from comet_ml import Experiment
from datetime import datetime

def run_diff_corrector_evaluation():
    """
    Run DiffCorrector evaluation with a new experiment
    """
    # Create a new experiment
    experiment = Experiment(
        api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
        project_name="hcmr-ai",
        workspace="ioannisgkinis"
    )
    
    # Set experiment name
    experiment_name = f"DiffCorrector_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}"
    experiment.set_name(experiment_name)
    
    # Add tags
    experiment.add_tag("diff_corrector")
    experiment.add_tag("evaluation")
    experiment.add_tag("baseline_plotter")
    
    # Log parameters
    experiment.log_parameter("corrector", "DiffCorrector")
    experiment.log_parameter("evaluation_type", "baseline_plotter")
    
    print(f"✅ Created experiment: {experiment_name}")
    print(f"🔑 Experiment Key: {experiment.get_key()}")
    
    # Set the prediction directory path
    prediction_dir = "/data/tsolis/AI_project/output/experiments/DiffCorrector/run_diff_v1"
    
    print(f"📂 Using prediction directory: {prediction_dir}")
    
    # Import and run the plotter
    from evaluation.baseline_correctors_plotter import PredictionPlotter
    
    plotter = PredictionPlotter(
        prediction_dir=prediction_dir,
        comet_exp=experiment
    )
    
    # Run the analysis
    print("🚀 Starting DiffCorrector evaluation...")
    plotter.run_all()
    
    print(f"✅ Evaluation complete! Check the experiment at: {experiment.url}")

if __name__ == "__main__":
    run_diff_corrector_evaluation()
