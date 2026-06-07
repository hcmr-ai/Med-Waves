# Config Reference

This document is a practical guide to [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml). It is not a full field-by-field schema; it focuses on the settings that matter operationally during handover.

## Data Section

Important fields:
- `data_path`: root path for training/evaluation data
- `file_pattern`: expected file pattern, currently `WAVEAN*.pt`
- `train_year`, `val_year`, `test_year`: year-based split definition
- `target_columns`: active prediction targets
- `excluded_columns`: features removed from model inputs
- `predict_bias`: train model to predict correction bias rather than the corrected variable directly
- `predict_residual_to_prior`: residual-learning mode relative to a prior bias
- `region_filter`: restricts the operational region

Operational note:
- `predict_bias` and `predict_residual_to_prior` are central to understanding what the model output means. Do not change them casually.

## Model Section

Important fields:
- `model_type`
- `in_channels`
- `learning_rate`
- `loss_type`
- `weight_decay`
- `optimizer_type`
- `residual_penalty_lambda`

Current active architecture in the checked-in config:
- `model_type: "moe_transunet"`

## Residual Penalty

Current config:
- `residual_penalty_lambda: 0.1`

Practical interpretation in the current code path:
- this acts as an extra penalty term on predicted bias/residual magnitude
- it is in addition to optimizer `weight_decay`

Handover note:
- the YAML comment says this is active only in residual mode, but the current trainer/loss path also applies it when `predict_bias: true`

## MoE Auxiliary Weights

These knobs exist:
- `gate_entropy_weight`
- `gate_balance_weight`
- `gate_prior_weight`
- `expert_diversity_weight`

In the checked-in config they are all `0.0`, so they are effectively disabled.

## Training Section

Important fields:
- `batch_size`
- `max_epochs`
- `num_workers`
- `accelerator`
- `devices`
- `precision`
- `early_stopping_patience`
- `monitor`

Operational note:
- The checked-in values are tuned for a specific environment. If memory pressure changes, start with `batch_size`, `num_workers`, and data caching settings.

## Checkpoint And Logging Sections

These fields are tightly coupled:
- `resume_from_checkpoint`
- `checkpoint_dir`
- `log_dir`
- `experiment_name`

Do not update only one of them for a new experiment. For fresh runs, create a derived config or use a wrapper that rewrites all experiment-specific destinations consistently.

## Recommended Practice For New Runs

For new experiments:
1. keep the base config as a reference
2. generate a derived config for the run
3. update:
   - `residual_penalty_lambda` or other experimental knobs
   - `resume_from_checkpoint`
   - `checkpoint_dir`
   - `log_dir`
   - `experiment_name`

The residual penalty sweep wrapper already follows this pattern:
- [`scripts/run_residual_penalty_sweep.sh`](../scripts/run_residual_penalty_sweep.sh)
