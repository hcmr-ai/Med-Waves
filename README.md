# Med-WAV

Med-WAV is a research and operations repository for wave reanalysis correction. It contains machine learning workflows that correct wave variables such as `VHM0` and `VTM02` from gridded datasets, along with baseline models, diagnostics, plotting utilities, and storage/bootstrap scripts for the data environment.

This repository is configuration-driven. The current DNN workflow uses [`src/configs/config_dnn.yaml`](src/configs/config_dnn.yaml) with training in [`src/pipelines/training/dnn_trainer.py`](src/pipelines/training/dnn_trainer.py) and evaluation in [`src/pipelines/evaluation/evaluate_bunet.py`](src/pipelines/evaluation/evaluate_bunet.py).

The repo also uses Comet ML for experiment tracking in multiple training workflows.

## Status

- Active path: DNN training and evaluation under `src/pipelines/training` and `src/pipelines/evaluation`
- Secondary baseline paths exist for EDCDF, static-map correction, full-dataset models, and model-per-point
- Legacy or less-curated paths exist in the repo; treat them carefully before relying on them operationally

## Repo Map

- [`src/configs`](src/configs): YAML configs for DNN, full-dataset, evaluation, and per-point workflows
- [`src/pipelines/training`](src/pipelines/training): train entrypoints for DNN and other model variants
- [`src/pipelines/evaluation`](src/pipelines/evaluation): evaluation entrypoints and orchestration scripts
- [`src/classifiers`](src/classifiers): Lightning modules and model code
- [`src/commons`](src/commons): shared datasets, losses, callbacks, helpers, AWS/S3 utilities, preprocessing, postprocessing
- [`src/evaluation`](src/evaluation): plotting and reporting helpers used after evaluation
- [`scripts`](scripts): operator scripts, utilities, diagnostics, skill installers
- [`skills`](skills): repo-local skills that can be installed into agent environments

## Environment

- Python: `>=3.11,<3.15`
- Package manager: Poetry
- Typical runtime: Linux/Ubuntu, GPU-enabled machine for DNN training
- Typical storage assumptions:
  - local or mounted datasets under `/mnt/...`
  - Azure Blob mounts at `/mnt/blobstorage` and `/mnt/blobstorage-scalers`
  - some workflows can read from S3, but the current DNN config is wired to mounted storage paths

Install dependencies:

```bash
poetry install
```

## Start Here

Mount Azure blob storage if needed:

```bash
./scripts/azure/setup_blobfuse_mounts.sh
```

Train a DNN workflow:

```bash
poetry run python src/pipelines/training/dnn_trainer.py --config src/configs/config_dnn.yaml
```

Run the main evaluation workflow:

```bash
poetry run python src/pipelines/evaluation/evaluate_bunet.py \
  --config src/configs/config_dnn.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-dir ./evaluation_results
```

Run the current evaluation orchestration script:

```bash
./src/pipelines/evaluation/full_evaluation.sh
```

Sweep `residual_penalty_lambda` values sequentially:

```bash
./scripts/run_residual_penalty_sweep.sh 0 0.01 0.05 0.1
```

Install repo-local skills into Codex, Claude, and Cursor:

```bash
./scripts/install_repo_skills.sh
```

## Repo Skills

This repo ships with repo-local agent skills under [`skills/`](skills).

Current skills:
- [`medwav-blobfuse-mounts`](skills/medwav-blobfuse-mounts/SKILL.md): operator guidance for Azure blobfuse2 setup using [`scripts/azure/setup_blobfuse_mounts.sh`](scripts/azure/setup_blobfuse_mounts.sh)
- [`medwav-poetry-full-install`](skills/medwav-poetry-full-install/SKILL.md): full Poetry-based environment setup for Med-WAV, including Poetry bootstrap, dependency install, and torch/CUDA verification

Install repo skills into supported agent environments with:

```bash
./scripts/install_repo_skills.sh
```

Supported targets:
- Codex
- Claude
- Cursor

Use the skill when the task is to:
- mount the Med-WAV Azure Blob containers
- explain or troubleshoot blobfuse setup
- adapt the mount workflow for different storage account, container, cache, or mount-path overrides
- bootstrap Poetry and install the full Med-WAV environment from scratch
- verify that the Poetry virtualenv and torch/CUDA setup are working

## Documentation Index

- [`AGENTS.md`](AGENTS.md): repo-wide instructions for coding agents
- [`CLAUDE.md`](CLAUDE.md): thin pointer for Claude-based agents
- [`docs/environment-setup.md`](docs/environment-setup.md): local environment and dependencies
- [`docs/azure-training-vm.md`](docs/azure-training-vm.md): setup for the Azure `Standard_NV12ads_A10_v5` training VM
- [`docs/data-locations.md`](docs/data-locations.md): storage mounts plus Azure and Neptune data layout and folder purposes
- [`docs/features.md`](docs/features.md): feature dictionary, engineered features, and current model inputs
- [`docs/preprocessing-pipeline.md`](docs/preprocessing-pipeline.md): raw `.nc` to final `.pt` preprocessing, subsampling, and scaler fitting
- [`docs/training-architecture.md`](docs/training-architecture.md): end-to-end DNN training stack from tensors to losses, priors, and optimization
- [`scripts/README.md`](scripts/README.md): curated guide to the `scripts/` directory
- [`docs/training-dnn.md`](docs/training-dnn.md): DNN training workflow
- [`docs/evaluation_dnn.md`](docs/evaluation_dnn.md): DNN evaluation entrypoints and outputs
- [`docs/baseline-models.md`](docs/baseline-models.md): EDCDF, static-map, full-dataset, MLP, and model-per-point baseline paths
- [`docs/config-reference.md`](docs/config-reference.md): practical guide to `config_dnn.yaml`

## Known Handover Caveats

- Many default paths are hardcoded to `/mnt/...` locations and assume mounted cloud-backed storage.
- The current DNN config contains an explicit `resume_from_checkpoint` and experiment-specific logging/checkpoint paths; do not assume it is safe for a fresh experiment without editing or generating a derived config.
- Some evaluation scripts are large and evolved organically. Prefer the main documented path before using older or parallel variants.
- Some repo areas contain exploratory or historical code. Treat the docs above as the source of truth for the documented workflows.
