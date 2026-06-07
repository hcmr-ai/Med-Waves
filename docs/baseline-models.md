# Baseline Models

This document covers the baseline and alternative model paths in the repo. These are useful for comparison, ablations, and revisiting older research directions.

## Status Summary

- DNN training and evaluation documented in [`training-dnn.md`](training-dnn.md) and [`evaluation_dnn.md`](evaluation_dnn.md)
- Baseline paths with current entrypoints in `src/`: EDCDF, full-dataset trainer, model-per-point
- Diagnostic baseline path in `scripts/`: static bias map evaluation
- Legacy previous-researcher material: `scripts/trainers/`

## Static Bias Map Baseline

Script:
- [`scripts/static_baseline_evaluation.py`](../scripts/static_baseline_evaluation.py)

What it does:
- computes a per-pixel mean bias map from training years if `--bias_map` is not provided
- saves the resulting `static_bias_map_<region>.npy`
- evaluates the raw field and static-map-corrected field against `corrected_VHM0`

Expected input data:
- preprocessed `.pt` files under a directory such as `/mnt/blobstorage/preprocessed_extended_subsampled_step_5/`
- each `.pt` file is expected to contain `tensor` and `feature_cols`
- required feature columns include at least `VHM0`, `corrected_VHM0`, `latitude`, and `longitude`

Important behavior:
- if `--bias_map /path/to/static_bias_map.npy` is provided, the script loads that artifact and skips recomputing the map
- if `--bias_map` is omitted, the script builds the map directly from the `--train_years` files

Typical command:

```bash
PYTHONUNBUFFERED=1 poetry run python scripts/static_baseline_evaluation.py \
  --data_path /mnt/blobstorage/preprocessed_extended_subsampled_step_5/ \
  --train_years 2018 2019 2020 2021 \
  --eval_years 2017 2018 2019 2020 2021 2022 2023 \
  --region mediterranean
```

This is a useful baseline because any learned model should beat the static map consistently.

## EDCDF Regional Baseline

Training:
- [`src/pipelines/training/train_edcdf_regional.py`](../src/pipelines/training/train_edcdf_regional.py)

Evaluation:
- [`src/pipelines/evaluation/evaluate_edcdf.py`](../src/pipelines/evaluation/evaluate_edcdf.py)

What it does:
- fits a regional EDCDF corrector on parquet data
- supports Atlantic and Mediterranean region splits aligned with the main dataset logic
- evaluates corrected predictions with a plot suite similar to the DNN evaluation path
- saves the fitted `.joblib` model under `models/<run_id>.joblib` inside the configured output base

Expected input data:
- parquet files, in the documented workflow under `/mnt/blobstorage/parquet/hourly_extra_features`
- the training script expects raw variables such as `VHM0` and `VTM02` plus corrected targets such as `corrected_VHM0` and `corrected_VTM02`
- the evaluation script expects prediction parquet outputs produced by `train_edcdf_regional.py`

Documented artifact layout:
- output base: `/mnt/blobstorage/diagnostics/edcdf_regional`
- per-region run directory: `/mnt/blobstorage/diagnostics/edcdf_regional/<region>/<run_id>/`
- model artifact: `/mnt/blobstorage/diagnostics/edcdf_regional/<region>/<run_id>/models/<run_id>.joblib`

This matches the `edcdf_model_path` pattern used in [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml).

Training example:

```bash
poetry run python -m src.pipelines.training.train_edcdf_regional \
  --train-years 2018 2019 2020 2021 \
  --test-years 2017 2022 2023 \
  --regions mediterranean \
  --variables VHM0 VTM02 \
  --parquet-base /mnt/blobstorage/parquet/hourly_extra_features \
  --output-base /mnt/blobstorage/diagnostics/edcdf_regional \
  --no-s3
```

Evaluation example:

```bash
poetry run python -m src.pipelines.evaluation.evaluate_edcdf \
  --predictions-dir data/edcdf_regional/mediterranean/edcdf_mediterranean_train_2018-2019-2020-2021_test_2022-2023/predictions \
  --variable VHM0
```

Notes:
- `train_edcdf_regional.py` still contains optional S3 input and upload support.
- The documented Med-WAV operator path uses mounted blob storage under `/mnt/blobstorage/...`, not direct S3 access.
- If someone revisits the script internals, they should treat the S3 code paths as optional compatibility logic rather than the primary documented storage path.

## MLP Through The Standard DNN Trainer

Main path:
- [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py)
- [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)

What it is:
- an MLP baseline that uses the same config-driven trainer surface as the DNN family
- useful when you want a simpler neural baseline without switching to a separate pipeline

Expected input data:
- the same preprocessed `.pt` tensor dataset used by the DNN trainer
- a scaler artifact compatible with the config, if scaling is enabled
- the same mounted storage conventions as the standard DNN config unless paths are overridden

How to use it:
- derive a config from `config_dnn.yaml`
- set the model type to the MLP variant supported by the model factory
- keep the rest of the training and evaluation flow aligned with the documented DNN path

Status:
- current and reachable through the standard trainer surface
- useful as a simpler neural baseline within the same config-driven setup

## Full-Dataset Trainer Baselines

Training entrypoint:
- [`src/pipelines/training/train_full_dataset.py`](../src/pipelines/training/train_full_dataset.py)

Trainer:
- [`src/classifiers/full_dataset_trainer.py`](../src/classifiers/full_dataset_trainer.py)

Example config:
- [`src/configs/config_full_dataset.yaml`](../src/configs/config_full_dataset.yaml)

What it does:
- loads the full tabularized dataset into memory
- trains classical ML baselines such as XGBoost, Random Forest, linear models, and simple correctors
- performs integrated train/evaluate/save inside the same training script

Expected input data:
- parquet files, either local or on S3
- each row is treated as a tabular sample rather than as part of a 2D spatial tensor
- expected columns are controlled by the config and typically include engineered features plus the configured target column

Typical command:

```bash
poetry run python src/pipelines/training/train_full_dataset.py \
  --config src/configs/config_full_dataset.yaml
```

Status:
- current baseline path in `src/`
- useful for comparisons against non-spatial models trained on flattened feature tables

## Model-Per-Point

Current `src/` path:
- [`src/pipelines/training/train_model_per_point.py`](../src/pipelines/training/train_model_per_point.py)
- [`src/classifiers/model_per_point.py`](../src/classifiers/model_per_point.py)
- [`src/configs/config_model_per_point.yaml`](../src/configs/config_model_per_point.yaml)

Legacy previous-researcher material:
- [`scripts/trainers/Polynomial_ModelPerPoint.py`](../scripts/trainers/Polynomial_ModelPerPoint.py)
- [`scripts/trainers/Polynomial_modelPerPoint_Linear_neptune.py`](../scripts/trainers/Polynomial_modelPerPoint_Linear_neptune.py)
- [`scripts/trainers/Polynomial_modelPerPoint_Linear_onlyModels_neptune.py`](../scripts/trainers/Polynomial_modelPerPoint_Linear_onlyModels_neptune.py)

What it appears to be:
- a per-point training path for bias-correction research
- the `src/` implementation is more structured and config-driven than the old `scripts/trainers/` versions
- the legacy scripts are strongly tied to older researcher-specific environments and should not be treated as canonical

Expected input data:
- parquet files for the current `src/` path, loaded through the config-driven data loader
- the legacy `scripts/trainers/` variants appear to work directly from older researcher-specific raw/preprocessed paths and should be treated separately from the current `src/` workflow

Confidence note:
- low confidence as a recommended handover path
- there is a current `src/` implementation, but this area is less curated than some other documented paths in the repo
- if someone wants to revisit it, they should inspect both the `src/` trainer and the legacy `scripts/trainers/` history before treating it as production-like

Typical command:

```bash
poetry run python src/pipelines/training/train_model_per_point.py \
  --config src/configs/config_model_per_point.yaml
```

## Guidance

- Use this page for baseline comparisons and alternative model workflows.
- Treat `scripts/trainers/` as historical reference, not as the default implementation surface.

## Legacy Corrector Experiments

The repo also contains older corrector experiment entrypoints:
- [`src/pipelines/training/train_evaluator.py`](../src/pipelines/training/train_evaluator.py)
- [`src/pipelines/training/train_random_regressor.py`](../src/pipelines/training/train_random_regressor.py)

Status:
- both should be treated as legacy research scripts
- `train_evaluator.py` uses incremental or batchwise fitting for correctors such as Delta, EDCDF, and EQM
- both scripts are tied to older assumptions around Comet logging, local path layout, and output naming
- they are kept for historical comparison or reproduction, not as clean current training entrypoints
