# Data Locations

This document explains where Med-WAV data lives across the environments referenced by the repo, what each folder is used for, and how the current Azure-mounted layout is accessed.

It separates:
- current Azure-mounted paths used by the documented workflow
- Pegasus server paths shared during the original data handover
- Neptune internal-server paths that still appear in legacy or older pipeline code

Where a location is inferred from code rather than confirmed by a current operator workflow, that is stated explicitly.

## Summary

### Current documented training workflow

Current training and evaluation are centered on Azure-mounted paths:
- `/mnt/blobstorage`
- `/mnt/blobstorage-scalers`
- `/mnt/local_datasets`

Access to the mounted containers is provisioned by:
- [`scripts/azure/setup_blobfuse_mounts.sh`](../scripts/azure/setup_blobfuse_mounts.sh)

Default mount points from that script:
- `/mnt/blobstorage`
- `/mnt/blobstorage-scalers`

### Older internal-server workflow

Older raw-data and merged-data locations also exist on Pegasus:
- `/data2/ntsolis/merged`
- `/data2/ocean2/RAN_ML/VAL`
- `/data2/ocean2/REANALYSIS/VAL`

Older preprocessing and baseline code references Neptune-local paths such as:
- `/data/tsolis/AI_project/...`
- `/home/n.tsolis/AI_project/...`

These should be treated as older internal-server layouts unless explicitly revived.

## Azure Layout

The documented Azure environment uses blobfuse mounts created by [`scripts/azure/setup_blobfuse_mounts.sh`](../scripts/azure/setup_blobfuse_mounts.sh).

Storage account:
- `medwavdatastorageneu`

Default mount points:
- `/mnt/blobstorage`
- `/mnt/blobstorage-scalers`

Default containers:
- `medwav-data` mounted at `/mnt/blobstorage`
- `scalers` mounted at `/mnt/blobstorage-scalers`

Blob-to-mount mapping:
- storage account `medwavdatastorageneu`, container `medwav-data` -> `/mnt/blobstorage`
- storage account `medwavdatastorageneu`, container `scalers` -> `/mnt/blobstorage-scalers`

When reading this repo, interpret mounted paths like:
- `/mnt/blobstorage/parquet/hourly_extra_features/...`
as blob paths like:
- account `medwavdatastorageneu`
- container `medwav-data`
- blob prefix `parquet/hourly_extra_features/...`

And interpret mounted paths like:
- `/mnt/blobstorage-scalers/scalers/...`
as blob paths like:
- account `medwavdatastorageneu`
- container `scalers`
- blob prefix `scalers/...`

Verification:

```bash
mountpoint -q /mnt/blobstorage && echo ok
mountpoint -q /mnt/blobstorage-scalers && echo ok
```

### `/mnt/blobstorage`

Purpose:
- main mounted container for datasets, diagnostics, checkpoints, and logs

Important subpaths evidenced in the repo:

#### `/mnt/blobstorage/parquet/hourly_extra_features`

Purpose:
- parquet dataset with corrected targets and engineered features
- used by EDCDF training and by timestamp/diagnostic utilities

Expected structure:
- `/mnt/blobstorage/parquet/hourly_extra_features/year=YYYY/WAVEANYYYYMMDD.parquet`

Underlying blob location:
- storage account: `medwavdatastorageneu`
- container: `medwav-data`
- blob prefix: `parquet/hourly_extra_features/year=YYYY/WAVEANYYYYMMDD.parquet`

Evidence:
- [`src/pipelines/training/train_edcdf_regional.py`](../src/pipelines/training/train_edcdf_regional.py)
- [`scripts/build_pt_timestamp_map.py`](../scripts/build_pt_timestamp_map.py)
- [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py)

#### `/mnt/blobstorage/preprocessed_extended_subsampled_step_5`

Purpose:
- canonical blob-backed store of daily `.pt` tensor files after parquet-to-tensor conversion and spatial subsampling

Expected structure:
- flat directory of files such as `WAVEANYYYYMMDD.pt`

Underlying blob location:
- storage account: `medwavdatastorageneu`
- container: `medwav-data`
- blob prefix: `preprocessed_extended_subsampled_step_5/WAVEANYYYYMMDD.pt`

Operational note:
- this is usually not read directly during training for every batch
- the common workflow is to copy it once to `/mnt/local_datasets/...` and train from local disk

Evidence:
- [`docs/environment-setup.md`](environment-setup.md)
- [`docs/training-dnn.md`](training-dnn.md)
- diagnostics scripts such as [`scripts/static_baseline_evaluation.py`](../scripts/static_baseline_evaluation.py)

#### `/mnt/blobstorage/diagnostics`

Purpose:
- outputs from analysis, baselines, evaluation helpers, and support artifacts

Important known subfolders:

- `/mnt/blobstorage/diagnostics/static_baseline`
  - static bias map baseline outputs
  - includes region-specific static bias map `.npy` artifacts
  - blob prefix: `diagnostics/static_baseline/...`

- `/mnt/blobstorage/diagnostics/edcdf_regional`
  - EDCDF regional baseline outputs
  - expected structure:
    - `/mnt/blobstorage/diagnostics/edcdf_regional/<region>/<run_id>/models/<run_id>.joblib`
    - `/mnt/blobstorage/diagnostics/edcdf_regional/<region>/<run_id>/predictions/...`
    - `/mnt/blobstorage/diagnostics/edcdf_regional/<region>/<run_id>/metadata.json`
  - blob prefix: `diagnostics/edcdf_regional/<region>/<run_id>/...`

- `/mnt/blobstorage/diagnostics/pt_timestamp_map.csv`
  - support CSV mapping `.pt` day stems and hour indices back to timestamps
  - blob path: `diagnostics/pt_timestamp_map.csv`

- other diagnostic outputs referenced by the repo:
  - `bias_distribution`
  - `bias_stationarity`
  - `amplitude_proxy`

#### `/mnt/blobstorage/checkpoints`

Purpose:
- Lightning checkpoint storage for DNN experiments

Expected structure:
- one subdirectory per experiment or run naming convention
- inside each, `.ckpt` files such as `epoch=XX-val_loss=YY.ckpt`

Underlying blob location:
- storage account: `medwavdatastorageneu`
- container: `medwav-data`
- blob prefix: `checkpoints/...`

Evidence:
- [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)
- [`scripts/azure/download_checkpoint.py`](../scripts/azure/download_checkpoint.py)

#### `/mnt/blobstorage/logs_*`

Purpose:
- training logs and run-specific output directories

Operational note:
- naming is experiment-specific and currently verbose
- these are coupled to checkpoint naming in the config

Underlying blob location:
- storage account: `medwavdatastorageneu`
- container: `medwav-data`
- blob prefix: `logs_...`

### `/mnt/blobstorage-scalers`

Purpose:
- mounted scaler container used for reusable normalization artifacts

Important subpath:
- `/mnt/blobstorage-scalers/scalers`

Expected contents:
- `.pkl` normalizer artifacts such as `BU24h_zscore_18-21_med_extended.pkl`

Underlying blob location:
- storage account: `medwavdatastorageneu`
- container: `scalers`
- blob prefix: `scalers/<normalizer>.pkl`

Evidence:
- [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)

### `/mnt/local_datasets`

Purpose:
- local on-VM working copy of preprocessed tensor datasets
- used to avoid repeated blobfuse-backed reads during training

Important subpath:
- `/mnt/local_datasets/preprocessed_extended_subsampled_step_5`

Expected structure:
- flat directory of `WAVEANYYYYMMDD.pt` files

Why it exists:
- faster local reads
- more stable dataloader performance
- lower repeated remote-access overhead during long training runs

Bootstrap command:

```bash
mkdir -p /mnt/local_datasets/
rsync -a --info=progress2 "/mnt/blobstorage/preprocessed_extended_subsampled_step_5/" "/mnt/local_datasets/preprocessed_extended_subsampled_step_5/"
```

## Pegasus Layout

The original handover also included paths on an internal server named `pegasus`.

Operator note from the earlier handover:
- host `10.6.3.5` also runs service workloads
- if someone runs heavy processing there, it should be assigned to CPUs `100+`
- if this server is still in active use, confirm the current policy with the infrastructure owner before running large jobs

### `/data2/ntsolis/merged`

Purpose:
- merged dataset directory shared during the original handover

Status:
- confirmed handover path
- exact internal file layout still needs verification on the server

### `/data2/ocean2/RAN_ML/VAL`

Purpose:
- uncorrected raw inputs before Med-WAV preprocessing

Status:
- confirmed handover path
- expected to contain the uncorrected source data used before any project-side transformations

### `/data2/ocean2/REANALYSIS/VAL`

Purpose:
- ground-truth or reference data before Med-WAV preprocessing

Status:
- confirmed handover path
- expected to contain the reference data paired with the uncorrected source set

### Recommended interpretation

For handover purposes, Pegasus should be described as:
- an upstream raw-data and merged-data host
- separate from the Azure training/storage workflow
- separate from the older Neptune-local preprocessing layout embedded in parts of the codebase

## Neptune Layout

The repo still contains many paths that appear to come from an internal Neptune server or an older internal Linux environment.

These are not the current documented Azure operator paths, but they are important for understanding historical data organization.

### Main Neptune prefixes seen in code

- `/data/tsolis/AI_project/...`
- `/home/n.tsolis/AI_project/...`

The `/data/tsolis/...` tree appears to be the more structured internal-server working area.
The `/home/n.tsolis/...` tree appears in older scripts and should be treated as legacy or user-home-specific.

### Raw NetCDF inputs on Neptune

Repo evidence suggests the older raw `.nc` layout used:

- `/data/tsolis/AI_project/without_reduced`
- `/data/tsolis/AI_project/with_reduced`

Likely meaning, based on current preprocessing conventions:
- `without_reduced`: degraded/raw model data
- `with_reduced`: corrected/reference data used to derive `corrected_*` targets

This mapping is inferred from:
- [`src/data_engineering/aws/netcdf_to_parquet_features.py`](../src/data_engineering/aws/netcdf_to_parquet_features.py)
- [`src/data_engineering/neptune/convert.py`](../src/data_engineering/neptune/convert.py)
- [`src/data_engineering/neptune/feature_augmentation.py`](../src/data_engineering/neptune/feature_augmentation.py)

Expected file naming:
- `WAVEANYYYYMMDD.nc`

### Parquet on Neptune

Older Neptune-oriented preprocessing code references:

- `/data/tsolis/AI_project/parquet/with_reduced/hourly`
- `/data/tsolis/AI_project/parquet/augmented_with_labels/hourly`

Purpose:
- hourly parquet derived from NetCDF
- augmented parquet with corrected labels and engineered features

Evidence:
- [`src/data_engineering/neptune/convert.py`](../src/data_engineering/neptune/convert.py)
- [`src/data_engineering/neptune/aggregate.py`](../src/data_engineering/neptune/aggregate.py)
- [`src/data_engineering/data_loader.py`](../src/data_engineering/data_loader.py)

Additional derived folders on Neptune:

- `/data/tsolis/AI_project/parquet/augmented_with_labels/hourly_mean`
  - hourly means aggregated by timestamp

- `/data/tsolis/AI_project/parquet/augmented_with_labels/monthly_spatial_stats`
  - monthly per-grid spatial summary statistics

### Preprocessed tensors on Neptune

Older sync code references:

- `/data/tsolis/AI_project/preprocessed_subsampled_step_5`

Purpose:
- local Neptune copy of preprocessed `.pt` files synced from object storage

Evidence:
- [`src/data_engineering/aws/s3_download.py`](../src/data_engineering/aws/s3_download.py)

This appears to be an older equivalent of the newer Azure-side `/mnt/local_datasets/...` training copy pattern.

### Older home-directory layouts

Very old scripts reference paths such as:

- `/home/n.tsolis/AI_project/without_reduced/_reduced_For_Testing_Grid_5step_TEST`
- `/home/n.tsolis/AI_project/without_reduced/_reduced_For_Testing_Grid_5step_TRAIN`
- `/home/n.tsolis/AI_project/with_reduced/_reduced_For_Testing_Grid_5step_TEST`
- `/home/n.tsolis/AI_project/with_reduced/_reduced_For_Testing_Grid_5step_TRAIN`
- `/home/n.tsolis/AI_project/Output`

These show up in legacy per-point scripts under [`scripts/trainers/`](../scripts/trainers).

Purpose:
- older researcher-local train/test splits and outputs

Status:
- legacy
- not part of the documented current workflow

## Recommended Handover Interpretation

For handover, the clean mental model is:

### Current Azure workflow

- canonical blob-backed store:
  - `/mnt/blobstorage/...`
- scaler container:
  - `/mnt/blobstorage-scalers/...`
- local fast training copy:
  - `/mnt/local_datasets/...`

### Older Neptune/internal workflow

- raw and parquet staging:
  - `/data/tsolis/AI_project/...`
- older user-home experiments:
  - `/home/n.tsolis/AI_project/...`

### Legacy caveat

Neptune and older home-directory paths should be treated as historical layout references unless someone explicitly confirms they are still actively maintained on the internal server.
