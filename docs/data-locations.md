# Data Locations

This document explains where Med-WAV data and related artifacts live across the environments referenced by the repo.

The three environments to keep in mind are:
- Azure: current mounted training and artifact workflow
- Pegasus: internal server from the original handover by Nikos Tsolis
- Neptune: internal server with older preprocessing outputs, parquet stores, experiments, and support artifacts

## Quick Map

### Azure

Use Azure for the current mounted workflow:
- datasets and diagnostics: `/mnt/blobstorage`
- scalers: `/mnt/blobstorage-scalers`
- fast local training copy: `/mnt/local_datasets`

Azure storage mapping:
- storage account: `medwavdatastorageneu`
- container `medwav-data` -> `/mnt/blobstorage`
- container `scalers` -> `/mnt/blobstorage-scalers`

Mounts are provisioned by:
- [`scripts/azure/setup_blobfuse_mounts.sh`](../scripts/azure/setup_blobfuse_mounts.sh)

### Pegasus

Pegasus is also an internal server.

The original handover from Nikos Tsolis identified these important Pegasus paths:
- merged data: `/data2/ntsolis/merged`
- preprocessed data without spatial subsampling: `/data2/ntsolis/preprocessed`
- raw uncorrected data: `/data2/ocean2/RAN_ML/VAL`
- raw ground truth or reference data: `/data2/ocean2/REANALYSIS/VAL`

Operator note from the original handover:
- host `10.6.3.5` also runs service workloads
- if heavy processing is run there, it should use CPUs `100+`
- if Pegasus is still in active use, confirm the current policy with the infrastructure owner before running large jobs

### Neptune

Neptune is an internal server with older project data and artifacts under:
- `/data/tsolis/AI_project`
- legacy user-home paths under `/home/n.tsolis/AI_project`

It contains:
- raw `.nc` data for years `2021` to `2023`
- a large parquet tree
- subsampled `.pt` tensors
- dashboard outputs
- experiment outputs
- old scalers
- old model checkpoints

## Azure

### Mounts and containers

Mounted paths on the VM:
- `/mnt/blobstorage`
- `/mnt/blobstorage-scalers`
- `/mnt/local_datasets`

Blob mapping:
- `/mnt/blobstorage` -> account `medwavdatastorageneu`, container `medwav-data`
- `/mnt/blobstorage-scalers` -> account `medwavdatastorageneu`, container `scalers`

Verification:

```bash
mountpoint -q /mnt/blobstorage && echo ok
mountpoint -q /mnt/blobstorage-scalers && echo ok
```

### Main Azure data folders

`/mnt/blobstorage/parquet/hourly_extra_features`
- Purpose: parquet with corrected targets and engineered features
- Structure: `year=YYYY/WAVEANYYYYMMDD.parquet`
- Blob prefix: `parquet/hourly_extra_features/...`

`/mnt/blobstorage/preprocessed_extended_subsampled_step_5`
- Purpose: canonical blob-backed store of daily `.pt` tensors after preprocessing and spatial subsampling
- Structure: flat directory of `WAVEANYYYYMMDD.pt`
- Blob prefix: `preprocessed_extended_subsampled_step_5/...`

`/mnt/blobstorage/diagnostics`
- Purpose: baseline outputs, evaluation helpers, and analysis artifacts
- Important subfolders:
  - `static_baseline`
  - `edcdf_regional`
  - `pt_timestamp_map.csv`
  - other diagnostic outputs such as `bias_distribution`, `bias_stationarity`, and `amplitude_proxy`
- Blob prefix: `diagnostics/...`

`/mnt/blobstorage/checkpoints`
- Purpose: DNN checkpoint storage
- Structure: run-specific directories with `.ckpt` files
- Blob prefix: `checkpoints/...`

`/mnt/blobstorage/logs_*`
- Purpose: training log and run-output directories
- Blob prefix: `logs_...`

`/mnt/blobstorage-scalers/scalers`
- Purpose: reusable normalizer `.pkl` artifacts
- Blob prefix: `scalers/...`

### Local Azure training copy

`/mnt/local_datasets/preprocessed_extended_subsampled_step_5`
- Purpose: fast local copy of the preprocessed tensor dataset
- Why it exists: local reads are faster and more stable than repeated blobfuse reads

Bootstrap command:

```bash
mkdir -p /mnt/local_datasets/
rsync -a --info=progress2 "/mnt/blobstorage/preprocessed_extended_subsampled_step_5/" "/mnt/local_datasets/preprocessed_extended_subsampled_step_5/"
```

## Pegasus

### Raw upstream data

`/data2/ocean2/RAN_ML/VAL`
- Purpose: raw uncorrected inputs before Med-WAV preprocessing

`/data2/ocean2/REANALYSIS/VAL`
- Purpose: raw ground-truth or reference data before Med-WAV preprocessing

These are the raw-data locations for the years that are not stored on Neptune.

### Merged data

`/data2/ntsolis/merged`
- Purpose: merged dataset directory from the original handover
- Note: the exact internal layout still needs to be checked directly on Pegasus

### Preprocessed data

`/data2/ntsolis/preprocessed`
- Purpose: preprocessed tensors before any spatial subsampling step
- Role: non-subsampled counterpart to:
  - Neptune `/data/tsolis/AI_project/preprocessed_subsampled_step_5`
  - Azure `/mnt/blobstorage/preprocessed_extended_subsampled_step_5`
- Note: the exact internal layout still needs to be checked directly on Pegasus

## Neptune

Neptune paths in this repo mainly live under `/data/tsolis/AI_project`.

### Raw `.nc` data

`/data/tsolis/AI_project/without_reduced`
- Purpose: degraded or uncorrected raw model data

`/data/tsolis/AI_project/with_reduced`
- Purpose: corrected or reference raw data used to derive `corrected_*` targets

Coverage:
- Neptune holds raw data for years `2021` to `2023`
- the remaining raw years live on Pegasus

Expected file naming:
- `WAVEANYYYYMMDD.nc`

### Parquet store

`/data/tsolis/AI_project/parquet`
- Size: about `555G`
- Role: main parquet area on Neptune

Important subtrees:

`augmented_with_labels` (`~410G`, `2,562` files)
- `hourly`: main augmented modeling table with features and corrected labels
- `monthly_spatial_stats`: monthly or yearly aggregated spatial statistics
- `hourly_mean`: compact yearly mean summaries
- `hourly_extra_features`: currently empty

`with_reduced` (`~73G`, `1,098` files)
- `hourly`: daily or hourly parquet slices for the reduced variant
- `hourly_mean`: small yearly mean summaries
- `monthly_spatial_stats`: currently empty

`without_reduced` (`~71G`, `1,100` files)
- `hourly`: daily or hourly parquet slices for the non-reduced variant
- `hourly_mean`: small yearly mean summaries
- `monthly_spatial_stats`: monthly aggregates

`hourly` (`~281M`, `4` files)
- small staging or export branch
- contains `with_reduced`

Takeaway:
- `augmented_with_labels/hourly` appears to be the main engineered modeling table
- `with_reduced` and `without_reduced` look like the base parquet variants

### Preprocessed tensors

`/data/tsolis/AI_project/preprocessed_subsampled_step_5`
- Size: about `107G`
- Structure: flat folder, no subfolders
- Contents: about `2,556` `.pt` files named like `WAVEANYYYYMMDD.pt`
- Role: one preprocessed tensor file per day, already spatially subsampled at step `5`

### Other Neptune project folders

`/data/tsolis/AI_project/output`
- Purpose: output data consumed by [`dashboard/med_wav.py`](../dashboard/med_wav.py)

`/data/tsolis/AI_project/experiments`
- Purpose: experiment outputs from the older `full_training.py` workflow

`/data/tsolis/AI_project/venvs`
- Purpose: local virtual environments
- Type: infrastructure or support, not dataset content

`/data/tsolis/AI_project/scalers`
- Purpose: older scaler artifacts

`/data/tsolis/AI_project/model_checkpoints`
- Purpose: historical model checkpoints, including models trained on AWS and kept later in the Neptune tree

### Legacy Neptune paths

Older user-home paths also appear in legacy scripts:
- `/home/n.tsolis/AI_project/without_reduced/...`
- `/home/n.tsolis/AI_project/with_reduced/...`
- `/home/n.tsolis/AI_project/Output`

These should be treated as historical researcher-local paths, not current documented operator paths.

## Handover Interpretation

Use this mental model:
- Azure is the current mounted training and artifact environment.
- Pegasus is the upstream raw-data, merged-data, and non-subsampled preprocessed-data host from the original handover.
- Neptune is the older internal project workspace holding raw 2021-2023 `.nc` files, parquet stores, subsampled tensors, dashboard outputs, experiments, scalers, and old checkpoints.
