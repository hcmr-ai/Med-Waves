import json
from pathlib import Path
from typing import List, Type

import joblib
import numpy as np
import polars as pl
from comet_ml import Experiment
from tqdm import tqdm

from src.classifiers.delta_corrector import DeltaCorrector
from src.classifiers.edcdf_corrector import EDCDFCorrector


def save_corrector_model(corrector, output_dir: Path, corrector_name: str, run_id: str):
    """Save the trained corrector model and its parameters to disk."""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the model object
    model_path = model_dir / f"{corrector_name}_{run_id}.joblib"
    joblib.dump(corrector, model_path)
    print(f"💾 Saved model to: {model_path}")

    # Save model parameters as JSON for easy inspection
    params_path = model_dir / f"{corrector_name}_{run_id}_params.json"

    if isinstance(corrector, DeltaCorrector):
        params = {
            "corrector_type": "DeltaCorrector",
            "bias_per_variable": corrector.bias_per_variable,
            "run_id": run_id
        }
    elif isinstance(corrector, EDCDFCorrector):
        # For EDCDF, we can't easily serialize the interpolators, but we can save metadata
        params = {
            "corrector_type": "EDCDFCorrector",
            "variables": list(corrector.cdf_models.keys()),
            "run_id": run_id,
            "note": "CDF interpolators saved in joblib file"
        }
    else:
        params = {
            "corrector_type": str(type(corrector)),
            "run_id": run_id
        }

    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2, default=str)
    print(f"💾 Saved parameters to: {params_path}")

    return model_path, params_path

def incremental_fit_delta_corrector(
    train_files: List[Path],
    variables: list[str],
    corrected_suffix: str = "corrected_",
    batch_size: int = 5
) -> DeltaCorrector:
    """Incrementally fit DeltaCorrector by computing running means across batches."""
    print("🔄 Incrementally fitting DeltaCorrector...")

    # Initialize running statistics
    running_stats = {var: {"model_sum": 0.0, "obs_sum": 0.0, "count": 0} for var in variables}

    # Process training files in batches
    for i in tqdm(range(0, len(train_files), batch_size), desc="Training batches", unit="batch"):
        batch_files = train_files[i:i + batch_size]

        # Load and concatenate batch files
        batch_dfs = []
        for file_path in batch_files:
            try:
                df = pl.read_parquet(file_path)
                batch_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue

        if batch_dfs:
            batch_df = pl.concat(batch_dfs)

            # Update running statistics for each variable
            for var in variables:
                model_col = var
                obs_col = corrected_suffix + var

                if model_col in batch_df.columns and obs_col in batch_df.columns:
                    # Drop nulls for this variable
                    valid_data = batch_df.drop_nulls(subset=[model_col, obs_col])

                    if len(valid_data) > 0:
                        model_sum = valid_data[model_col].sum()
                        obs_sum = valid_data[obs_col].sum()
                        count = len(valid_data)

                        running_stats[var]["model_sum"] += model_sum
                        running_stats[var]["obs_sum"] += obs_sum
                        running_stats[var]["count"] += count

    # Create and fit the corrector with computed means
    corrector = DeltaCorrector()
    for var in variables:
        if running_stats[var]["count"] > 0:
            model_mean = running_stats[var]["model_sum"] / running_stats[var]["count"]
            obs_mean = running_stats[var]["obs_sum"] / running_stats[var]["count"]
            corrector.bias_per_variable[var] = model_mean - obs_mean
            print(f"   {var}: bias = {corrector.bias_per_variable[var]:.6f}")

    return corrector

def incremental_fit_edcdf_corrector(
    train_files: List[Path],
    variables: list[str],
    corrected_suffix: str = "corrected_",
    batch_size: int = 5,
    quantile_resolution: int = 10000  # Number of quantiles for CDF approximation
) -> EDCDFCorrector:
    """Memory-efficient EDCDF corrector using quantile-based CDF approximation."""
    print("🔄 Incrementally fitting EDCDFCorrector (quantile-based approach)...")

    # Initialize quantile estimators for each variable

    # Store quantiles instead of all data
    quantiles = {var: {"model": [], "obs": []} for var in variables}
    total_samples = {var: 0 for var in variables}

    # Process training files in batches
    for i in tqdm(range(0, len(train_files), batch_size), desc="Training batches", unit="batch"):
        batch_files = train_files[i:i + batch_size]

        # Load and concatenate batch files
        batch_dfs = []
        for file_path in batch_files:
            try:
                df = pl.read_parquet(file_path)
                batch_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue

        if batch_dfs:
            batch_df = pl.concat(batch_dfs)

            # Process each variable
            for var in variables:
                model_col = var
                obs_col = corrected_suffix + var

                if model_col in batch_df.columns and obs_col in batch_df.columns:
                    # Drop nulls for this variable
                    valid_data = batch_df.drop_nulls(subset=[model_col, obs_col])

                    if len(valid_data) > 0:
                        model_values = valid_data[model_col].to_numpy()
                        obs_values = valid_data[obs_col].to_numpy()

                        # Update quantiles incrementally
                        if total_samples[var] == 0:
                            # First batch: compute initial quantiles
                            quantiles[var]["model"] = np.percentile(model_values, np.linspace(0, 100, quantile_resolution))
                            quantiles[var]["obs"] = np.percentile(obs_values, np.linspace(0, 100, quantile_resolution))
                        else:
                            # Subsequent batches: update quantiles using weighted approach
                            # Combine current quantiles with new data quantiles
                            new_model_quantiles = np.percentile(model_values, np.linspace(0, 100, quantile_resolution))
                            new_obs_quantiles = np.percentile(obs_values, np.linspace(0, 100, quantile_resolution))

                            # Weighted average (more weight to larger datasets)
                            weight_old = total_samples[var] / (total_samples[var] + len(model_values))
                            weight_new = len(model_values) / (total_samples[var] + len(model_values))

                            quantiles[var]["model"] = weight_old * quantiles[var]["model"] + weight_new * new_model_quantiles
                            quantiles[var]["obs"] = weight_old * quantiles[var]["obs"] + weight_new * new_obs_quantiles

                        total_samples[var] += len(model_values)

    # Create and fit the corrector using quantile-based CDFs
    corrector = EDCDFCorrector()
    for var in variables:
        if total_samples[var] > 0:
            # Create CDF interpolators from quantiles
            model_quantiles = quantiles[var]["model"]
            obs_quantiles = quantiles[var]["obs"]

            # Create CDF values (0 to 1)
            cdf_values = np.linspace(0, 1, quantile_resolution)

            # Create interpolators
            from scipy.interpolate import interp1d
            f_model_inv = interp1d(cdf_values, model_quantiles, bounds_error=False, fill_value="extrapolate")
            f_obs_inv = interp1d(cdf_values, obs_quantiles, bounds_error=False, fill_value="extrapolate")
            f_model_cdf = interp1d(model_quantiles, cdf_values, bounds_error=False, fill_value=(0,1))

            # Store in corrector
            corrector.cdf_models[var] = (f_model_inv, f_obs_inv, f_model_cdf)

            print(f"   {var}: CDF computed with {total_samples[var]:,} total samples using {quantile_resolution} quantiles")

    return corrector

def train_and_evaluate_corrector(
    corrector_class: Type,               # DeltaCorrector or EDCDFCorrector
    train_files: List[Path],             # List of training file paths
    test_files: List[Path],              # List of test file paths
    comet_experiment: Experiment,
    variables: list[str],
    run_id: str,
    corrected_suffix: str = "corrected_",
    batch_size: int = 5                  # Number of files to process in each batch
):
    print(f"📈 Training corrector: {corrector_class.__name__}")
    print(f"   Training files: {len(train_files)}")
    print(f"   Test files: {len(test_files)}")
    print(f"   Batch size: {batch_size}")

    # Use incremental fitting based on corrector type
    if corrector_class == DeltaCorrector:
        corrector = incremental_fit_delta_corrector(
            train_files, variables, corrected_suffix, batch_size
        )
    elif corrector_class == EDCDFCorrector:
        corrector = incremental_fit_edcdf_corrector(
            train_files, variables, corrected_suffix, batch_size, quantile_resolution=10000
        )
    else:
        raise ValueError(f"Unsupported corrector class: {corrector_class}")

    # Create output directory structure
    output_dir = Path(f"/data/tsolis/AI_project/output/experiments/{corrector_class.__name__}/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions and save individual files
    print("📤 Generating predictions and saving individual files...")
    individual_predictions_dir = output_dir / "individual_predictions"
    individual_predictions_dir.mkdir(exist_ok=True)

    total_predictions = 0

    for i in tqdm(range(0, len(test_files), batch_size), desc="Prediction batches", unit="batch"):
        batch_files = test_files[i:i + batch_size]

        for file_path in batch_files:
            try:
                # Load original test file
                df_test = pl.read_parquet(file_path)

                # Generate predictions for this file
                df_pred_single = corrector.predict(df_test)

                # Save individual prediction file with original filename
                pred_path = individual_predictions_dir / file_path.name
                df_pred_single.write_parquet(pred_path)

                total_predictions += len(df_pred_single)

            except Exception as e:
                print(f"⚠️ Error processing predictions for {file_path}: {e}")
                continue

    print(f"💾 Saved {total_predictions} total predictions to: {individual_predictions_dir}")

    # Save model and parameters
    model_path, params_path = save_corrector_model(
        corrector, output_dir, corrector_class.__name__, run_id
    )

    # Save metadata about the run
    metadata = {
        "run_id": run_id,
        "corrector_class": corrector_class.__name__,
        "variables": variables,
        "train_files_count": len(train_files),
        "test_files_count": len(test_files),
        "batch_size": batch_size,
        "corrected_suffix": corrected_suffix,
        "total_predictions": total_predictions,
        "model_path": str(model_path),
        "params_path": str(params_path),
        "individual_predictions_dir": str(individual_predictions_dir)
    }

    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"💾 Saved metadata to: {metadata_path}")

    # Log run metadata to Comet
    comet_experiment.set_name(f"{corrector_class.__name__}__{run_id}")
    comet_experiment.log_parameter("corrector", corrector_class.__name__)
    comet_experiment.log_parameter("variables", ",".join(variables))
    comet_experiment.log_parameter("run_id", run_id)
    comet_experiment.log_parameter("train_files", len(train_files))
    comet_experiment.log_parameter("test_files", len(test_files))
    comet_experiment.log_parameter("batch_size", batch_size)

    # Log bias values for DeltaCorrector
    if isinstance(corrector, DeltaCorrector):
        for var, bias in corrector.bias_per_variable.items():
            comet_experiment.log_parameter(f"bias_{var}", bias)


    print(f"✅ Run complete for {corrector_class.__name__} [Run ID: {run_id}]")
    print(f"📁 All outputs saved to: {output_dir}")


experiment = Experiment(
                api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
                project_name="hcmr-ai",
                workspace="ioannisgkinis"
            )

# Define data directory and patterns
train_dir = Path("/data/tsolis/AI_project/parquet/augmented_with_labels/hourly")
train_patterns = ["WAVEAN2021", "WAVEAN2022"]
test_patterns = ["WAVEAN2023"]

experiment.log_parameter("train_file_patterns", ", ".join(train_patterns))
experiment.log_parameter("test_file_patterns", ", ".join(test_patterns))

# Get training files
train_files = []
for pattern in train_patterns:
    pattern_files = sorted(train_dir.glob(f"{pattern}*.parquet"))
    train_files.extend(pattern_files)

# Get test files
test_files = []
for pattern in test_patterns:
    pattern_files = sorted(train_dir.glob(f"{pattern}*.parquet"))
    test_files.extend(pattern_files)

print(f"📊 Found {len(train_files)} training files and {len(test_files)} test files")

# Run Delta Corrector
# train_and_evaluate_corrector(
#     corrector_class=DeltaCorrector,
#     train_files=train_files,
#     test_files=test_files,
#     comet_experiment=experiment,
#     variables=["VHM0", "VTM02"],
#     run_id="run_delta_v1",
#     batch_size=5
# )

# Run EDCDF Corrector
train_and_evaluate_corrector(
    corrector_class=EDCDFCorrector,
    train_files=train_files,
    test_files=test_files,
    comet_experiment=experiment,
    variables=["VHM0", "VTM02"],
    run_id="run_edcdf_v1",
    batch_size=5
)
