# DNN Training

## Main Entrypoint

The primary DNN trainer is:

- [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py)

Primary config:

- [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)

## Standard Command

```bash
poetry run python src/pipelines/training/dnn_trainer.py --config src/configs/config_dnn.yaml
```

Useful CLI overrides:

```bash
poetry run python src/pipelines/training/dnn_trainer.py \
  --config src/configs/config_dnn.yaml \
  --batch_size 16 \
  --max_epochs 5 \
  --learning_rate 1e-4 \
  --deterministic
```

Supported direct overrides from the trainer:
- `--resume`
- `--data_path`
- `--batch_size`
- `--max_epochs`
- `--learning_rate`
- `--deterministic`

## Recommended Pre-Run Checks

- Confirm `data.data_path` exists
- Confirm `checkpoint.checkpoint_dir` and `logging.log_dir` point to intended destinations
- Inspect `checkpoint.resume_from_checkpoint`
- Confirm `data.predict_bias` and `data.predict_residual_to_prior` match the intended target semantics
- Confirm the experiment name is not reusing an old run unintentionally

## Local Data Copy

The documented training path uses a local on-VM copy of the preprocessed tensors rather than reading them directly from blobfuse for every batch.

Typical preparation step:

```bash
mkdir -p /mnt/local_datasets/
rsync -a --info=progress2 "/mnt/blobstorage/preprocessed_extended_subsampled_step_5/" "/mnt/local_datasets/preprocessed_extended_subsampled_step_5/"
```

Why:
- local disk reads are faster and more stable than repeated blobfuse-backed reads
- dataloader throughput is more predictable
- repeated remote storage access can be reduced during long training runs

## Current Config Caveat

The checked-in [`config_dnn.yaml`](../src/configs/config_dnn.yaml) is not a neutral starter config. It is wired to:
- a specific checkpoint resume path
- a specific checkpoint directory
- a specific experiment name and log directory
- a specific `residual_penalty_lambda`

For new experiments, derive a fresh config rather than mutating the base file casually.

## Experiment Tracking

The repo uses Comet ML as an experiment tracker.

For the DNN path, Comet is controlled from the `logging` section of [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml), especially:
- `use_comet`
- `comet_tags`
- `comet_notes`
- `experiment_name`

In practice:
- `experiment_name` is the run name shown in Comet
- training metrics, artifacts, and some metadata are logged from [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py)
- the trainer can fall back to TensorBoard if Comet is disabled

## Residual Penalty Sweeps

Use the repo wrapper:

```bash
./scripts/run_residual_penalty_sweep.sh 0 0.01 0.05 0.1
```

What it does:
- generates per-run configs
- rewrites `residual_penalty_lambda`
- rewrites checkpoint and logging destinations
- runs experiments sequentially
- defaults to fresh runs unless `--resume-base` is used

## Outputs

The trainer writes:
- checkpoints under `checkpoint.checkpoint_dir`
- logs under `logging.log_dir`
- Comet runs using `logging.experiment_name`

## Main Risks During Handover

- Reusing the checked-in experiment paths accidentally
- Running without mounted blob storage when the config expects it
- Misunderstanding target mode:
  - `predict_bias: true`
  - `predict_residual_to_prior: false`
- Assuming the residual penalty comment in YAML is fully accurate without checking the code path
