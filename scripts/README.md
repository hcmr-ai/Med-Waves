# Scripts Guide

This directory contains a mix of:

- current operator scripts
- diagnostics and support utilities
- legacy scripts from previous research work

This document only highlights the scripts a new developer or operator is expected to use during handover.

## Recommended Scripts

### Experiment Management

- [`run_residual_penalty_sweep.sh`](run_residual_penalty_sweep.sh)
  - runs sequential DNN experiments for multiple `residual_penalty_lambda` values
  - generates per-run configs and isolates checkpoint/log destinations

### Azure / Storage

- [`azure/setup_blobfuse_mounts.sh`](azure/setup_blobfuse_mounts.sh)
  - installs blobfuse2 if missing
  - mounts the default Med-WAV Azure blob containers

- [`azure/install_nvidia_part1.sh`](azure/install_nvidia_part1.sh)
  - manual NVIDIA setup, part 1
  - prepares the pinned kernel and reboots

- [`azure/install_nvidia_part2.sh`](azure/install_nvidia_part2.sh)
  - manual NVIDIA setup, part 2
  - installs the pinned GRID driver and does basic machine bootstrap

- [`azure/download_checkpoint.py`](azure/download_checkpoint.py)
  - downloads checkpoint folders from Azure Blob Storage

### Evaluation Support

- [`build_pt_timestamp_map.py`](build_pt_timestamp_map.py)
  - builds the `pt_stem × hour_idx → timestamp` CSV used by evaluation workflows that need correct timestamps for `.pt` files

## Diagnostic / Analysis Scripts

This directory also contains diagnostics such as:

- `bias_distribution_diagnostic.py`
- `bias_stationarity_diagnostic.py`
- `amplitude_proxy_analysis.py`
- `static_baseline_evaluation.py`

These can be useful, but they are not the default handover workflow. Treat them as analysis utilities rather than setup-critical scripts.

In particular:
- [`static_baseline_evaluation.py`](static_baseline_evaluation.py) computes a static bias map from training years if no precomputed map is passed, then evaluates that baseline against `corrected_VHM0`

## Destructive Scripts

The following are operational cleanup utilities and should be treated as destructive:

- [`azure/delete_a_blob.py`](azure/delete_a_blob.py)
- [`azure/delete_soft_deleted_blobs.py`](azure/delete_soft_deleted_blobs.py)

They require Azure storage credentials and should not be run casually.

## Legacy Scripts

These areas are **not** part of the documented current workflow and should be treated as legacy material from a previous researcher:

- `scripts/helpers/`
- `scripts/trainers/`

Do not use them as the default handover path unless there is a specific reason and the workflow is being intentionally revived.

## Rule Of Thumb

If a task is part of the current handover path, prefer:

- `scripts/azure/...`
- `run_residual_penalty_sweep.sh`
- `build_pt_timestamp_map.py`

If a script is not referenced by the main repo docs and looks exploratory, diagnostic, or historical, treat it as non-canonical until confirmed.
