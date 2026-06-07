# Preprocessing Pipeline

This document describes the preprocessing path that was actually used for the current Med-WAV workflow.

It covers:
- raw `.nc` inputs
- `.nc` to parquet conversion and feature augmentation
- parquet to `.pt` tensor conversion
- spatial subsampling
- scaler fitting from `.pt`

Feature definitions and current model-input status live in:
- [`docs/features.md`](features.md)

It does **not** cover:
- `scripts/helpers/`
- `notebooks/`

Those are legacy artifacts from a previous researcher and are not part of the documented handover path.

## Canonical Preprocessing Path

The confirmed preprocessing path is:

1. raw degraded `.nc` + corrected `.nc`
2. [`src/data_engineering/aws/netcdf_to_parquet_features.py`](../src/data_engineering/aws/netcdf_to_parquet_features.py)
3. parquet outputs with corrected targets and engineered features
4. [`src/pipelines/preprocessing/preprocessed_files.py`](../src/pipelines/preprocessing/preprocessed_files.py)
5. daily `.pt` tensors with spatial subsampling
6. [`src/pipelines/preprocessing/fit_scalers_from_tensors.py`](../src/pipelines/preprocessing/fit_scalers_from_tensors.py)
7. scaler `.pkl` artifacts

## Stage 1: Raw `.nc` Inputs

Inputs:
- degraded raw `.nc`
- corrected raw `.nc`

Expected naming convention:
- `WAVEANYYYYMMDD.nc`

Operational note:
- the documented path starts from these `.nc` files
- any older helper scripts under `scripts/helpers/` are intentionally excluded from this handover document

## Stage 2: `.nc` To Parquet With Corrected Targets And Features

Script:
- [`src/data_engineering/aws/netcdf_to_parquet_features.py`](../src/data_engineering/aws/netcdf_to_parquet_features.py)

What it does:
- reads degraded inputs
- reads corrected inputs
- converts hourly data into flat parquet
- appends corrected targets:
  - `corrected_VHM0`
  - `corrected_VTM02`
- applies feature augmentation via `add_features_lazy(...)`

Relevant implementation:
- `convert_netcdf_to_parquet_hourly(...)`
- `process_all_lazy(...)`

CLI form from the script:

```bash
poetry run python -m src.data_engineering.aws.netcdf_to_parquet_features \
  --degraded-dir <degraded_nc_dir> \
  --corrected-dir <corrected_nc_dir> \
  --output-dir <parquet_output_dir> \
  --concurrency 4
```

Notes:
- the script supports both local paths and `s3://...`
- it can also process already-existing parquet inputs, but the documented path here is the `.nc` path
- output files are parquet, one output per input day/file

Expected output:
- parquet files derived from `WAVEANYYYYMMDD.nc`
- output names become `WAVEANYYYYMMDD.parquet`

## Stage 3: Parquet To `.pt` Tensor Files

Script:
- [`src/pipelines/preprocessing/preprocessed_files.py`](../src/pipelines/preprocessing/preprocessed_files.py)

What it does:
- reads `WAVEAN*.parquet`
- reconstructs a dense tensor with shape `(T, H, W, C)`
- stores:
  - `tensor`
  - `feature_cols`
- writes `.pt`

Relevant implementation:
- `load_parquet_as_tensor(...)`
- `process_file(...)`

Important behavior:
- each daily parquet file becomes one daily `.pt` file
- the daily `.pt` file convention is:
  - `WAVEANYYYYMMDD.pt`
- in the checked-in script, `SAVE_HOURLY = False`, so the documented path is daily `.pt`, not hourly `.pt`

Documented script settings used in this path:
- `SAVE_HOURLY = False`
- `SUBSAMPLE_STEP = 5`

Operational note:
- this script is currently configured by editing module-level variables rather than a CLI
- the key variables at the top of the script are:
  - `INPUT_DIR`
  - `OUTPUT_DIR`
  - `SAVE_HOURLY`
  - `SUBSAMPLE_STEP`

How it was run:
- configure the paths and settings in the script
- then run it directly

Command form:

```bash
poetry run python src/pipelines/preprocessing/preprocessed_files.py
```

Expected output:
- daily `.pt` files such as `WAVEAN20200101.pt`

## Stage 4: Spatial Subsampling

Subsampling happens inside:
- [`src/pipelines/preprocessing/preprocessed_files.py`](../src/pipelines/preprocessing/preprocessed_files.py)

It is not a separate post-processing step.

How it works:

```python
tensor = tensor[:, ::subsample_step, ::subsample_step, :].clone()
```

Confirmed setting used:
- `SUBSAMPLE_STEP = 5`

Meaning:
- keep every 5th pixel in latitude
- keep every 5th pixel in longitude
- keep all timesteps and all channels

Why it matters:
- reduces tensor size
- reduces storage and training cost
- the resulting dataset naming commonly reflects this, for example `preprocessed_extended_subsampled_step_5`

## Stage 5: Scaler Fitting From `.pt`

Current script used:
- [`src/pipelines/preprocessing/fit_scalers_from_tensors.py`](../src/pipelines/preprocessing/fit_scalers_from_tensors.py)

Legacy script not used for the current documented path:
- [`src/pipelines/preprocessing/fit_scalers.py`](../src/pipelines/preprocessing/fit_scalers.py)

Why `fit_scalers_from_tensors.py` is the current path:
- works from final `.pt` tensors
- computes derived tensor-level channels before fitting:
  - `dVHM0`
  - `dWSPD`
  - `grad_mag`
- supports `--streaming` low-memory fitting

Why `fit_scalers.py` is treated as legacy:
- parquet-based
- loads all data into memory
- less memory efficient
- does not reflect the final `.pt`-based tensor pipeline as directly

### Derived Features Added Before Fitting

Inside [`fit_scalers_from_tensors.py`](../src/pipelines/preprocessing/fit_scalers_from_tensors.py), the scaler fit path first augments tensors with:
- `dVHM0`
- `dWSPD`
- `grad_mag`

This happens in:
- `compute_tensor_features(...)`

### Command Form

CLI from the script:

```bash
poetry run python -m src.pipelines.preprocessing.fit_scalers_from_tensors \
  --data-dirs <pt_dir_1> [<pt_dir_2> ...] \
  --years 2018 2019 2020 2021 \
  --scaler-name <scaler_name> \
  --mode zscore \
  --target-feature corrected_VHM0 \
  --streaming
```

Important options:
- `--data-dirs`
- `--years`
- `--region-filter`
- `--max-files`
- `--scaler-name`
- `--mode`
- `--target-feature`
- `--streaming`
- `--no-s3`

Recommended operational mode for large datasets:
- use `--streaming`

### Outputs

The script saves:
- local scaler file:
  - `data/scalers/<scaler_name>.pkl`
- optional S3 upload unless `--no-s3` is passed

## Stage 6: Time Mapping Support For `.pt`

Support script:
- [`scripts/build_pt_timestamp_map.py`](../scripts/build_pt_timestamp_map.py)

This is not part of `.nc -> .pt` preprocessing itself, but it is tightly related to how daily `.pt` files encode time.

What it does:
- reads only the parquet `time` column
- reconstructs the mapping:
  - `pt_stem`
  - `hour_idx`
  - `timestamp`

Useful command form:

```bash
poetry run python scripts/build_pt_timestamp_map.py \
  --parquet-dir <parquet_dir> \
  --output-csv <output_csv> \
  --years 2022 2023
```

Use this when downstream evaluation needs correct timestamps for `.pt` hour indices.

## Final Artifacts

After the full preprocessing path, the main artifacts are:

- parquet files with corrected targets and engineered features
- daily `.pt` tensor files with:
  - `tensor`
  - `feature_cols`
- scaler `.pkl` files fit from `.pt`
- optional timestamp CSV for evaluation

## Validation Checklist

After parquet generation:
- confirm expected parquet file count
- confirm corrected target columns exist
- confirm engineered feature columns exist

After `.pt` generation:
- load one sample file with `torch.load(...)`
- confirm it contains:
  - `tensor`
  - `feature_cols`
- confirm tensor shape is `(T, H, W, C)`
- confirm the spatial shape reflects `SUBSAMPLE_STEP = 5`

After scaler fitting:
- confirm the `.pkl` file exists
- confirm the scaler reports a valid `feature_order_`
- confirm `target_feature_name_` is correct
- confirm downstream training can load the scaler path referenced in the config

## Handover Notes

- This document intentionally excludes `scripts/helpers/` and `notebooks/`.
- The documented scaler-fitting path is `fit_scalers_from_tensors.py`, not `fit_scalers.py`.
- The documented subsampling path is inside `preprocessed_files.py` with `SUBSAMPLE_STEP = 5`.
- The preprocessing pipeline is partly script-configured rather than fully CLI-driven, so script-level path variables are part of the operational procedure.
