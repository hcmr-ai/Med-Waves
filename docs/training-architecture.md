# Training Architecture

This document is the high-level overview of the Med-WAV DNN training stack. It connects preprocessing, dataset loading, model selection, trainer behavior, losses, optimization, and evaluation-facing priors.

If a new developer only reads one technical training doc before changing experiments, this should be the one.

## Main Control Surface

The DNN training stack is driven by:
- config: [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)
- entrypoint: [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py)
- Lightning module: [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py)
- model factory: [`src/classifiers/model_factory.py`](../src/classifiers/model_factory.py)
- dataloaders: [`src/commons/dataloaders.py`](../src/commons/dataloaders.py)
- loss factory: [`src/commons/losses_factory.py`](../src/commons/losses_factory.py)

## End-To-End Flow

The documented DNN path is:

1. raw `.nc` files are converted into parquet and then `.pt` tensors
2. scalers are fit and saved as `WaveNormalizer` artifacts
3. `dnn_trainer.py` loads `config_dnn.yaml`
4. files are split into train / val / test
5. datasets produce input tensors, targets, masks, and auxiliary fields
6. `WaveBiasCorrector` builds the configured network
7. the trainer computes predictions, applies the configured loss, and logs metrics
8. validation and evaluation use the same structural pipeline, with optional static and EDCDF priors available downstream

## Data Contract

The DNN training path expects:
- preprocessed `.pt` files such as `WAVEANYYYYMMDD.pt`
- each file contains `tensor` and `feature_cols`
- a compatible normalizer `.pkl`

The current documented training setup uses:
- mounted tensors under a `/mnt/...` path
- a mounted scaler under `/mnt/blobstorage-scalers/...`

The preprocessed tensors are already spatially subsampled in the documented workflow, so runtime `subsample_step` is usually left `null`.

## File Splitting

`dnn_trainer.py` delegates data splitting through [`create_data_loaders(...)`](../src/commons/dataloaders.py), which uses year-based file selection.

Current config split:
- train: `2018, 2019, 2020, 2021`
- val: `2022`
- test: `2023`

Month-level filters can further constrain validation and test coverage.

## Dataset Layer

The standard full-grid path uses:
- [`CachedWaveDataset`](../src/commons/datasets/cache_wave_dataset.py)

The alternative patch path uses:
- [`TimestepPatchWaveDataset`](../src/commons/datasets/time_step_patch_dataset.py)

Current documented config uses the full-grid path because:
- `data.use_patch_sampling: false`

The dataset layer is responsible for:
- reading `.pt` tensors
- applying region filtering
- constructing targets from corrected/raw variables
- returning valid-pixel masks
- applying optional residual-to-prior logic
- applying feature normalization
- appending optional sea-mask or domain-mean channels

### Dataset Types

The standard full-grid path uses:
- [`CachedWaveDataset`](../src/commons/datasets/cache_wave_dataset.py)

It is used when:
- `data.use_patch_sampling: false`

It supports:
- single-task and multi-task targets
- bias prediction
- residual-to-prior prediction
- region filtering
- optional sea-mask channel
- optional domain-mean `VHM0` channel
- optional random patch crops when `patch_size` is set

Typical outputs:
- single-task: `X, y, mask, vhm0`
- single-task residual mode: `X, y, mask, vhm0, prior_bias`
- multi-task: `X, targets_dict, mask, vhm0`
- multi-task residual mode: `X, targets_dict, mask, vhm0, prior_bias`

Returned tensor shapes after channel-first conversion:
- `X`: `(C_in, H, W)`
- `y`: `(C_out, H, W)` or dict of task tensors
- `mask`: `(C_out, H, W)`
- `vhm0`: `(1, H, W)`

The alternative patch path uses:
- [`TimestepPatchWaveDataset`](../src/commons/datasets/time_step_patch_dataset.py)

It is used when:
- `data.use_patch_sampling: true`

Important constraint:
- `predict_residual_to_prior` is currently rejected for the patch-sampling path

## Input Tensor Semantics

The dataset returns channel-first tensors:
- `X`: model inputs
- `y` or `targets_dict`: training targets
- `mask`: valid sea pixels
- `vhm0`: raw uncorrected `VHM0`, used by several losses and diagnostics
- optional `prior_bias`: static prior field for residual-to-prior mode

Current config:
- `model.in_channels: 16`

That corresponds to:
- 15 active engineered/raw feature channels
- plus 1 sea-mask channel

## Training Target Modes

The DNN stack supports three mutually exclusive target modes:
- `predict_bias`
- `predict_residual_to_prior`
- `normalize_target`

This is validated in [`_validate_training_mode_config(...)`](../src/pipelines/training/dnn_trainer.py).

### Bias Mode

If `predict_bias: true`:
- the target is `corrected - raw`

This is the current config behavior.

### Residual-To-Prior Mode

If `predict_residual_to_prior: true`:
- the dataset first constructs a bias target
- then subtracts a static prior bias map
- the trainer later reconstructs the full bias for metrics and logging

Current limitations:
- only static prior is supported
- only `residual_prior_task='vhm0'` is supported

### Direct Corrected-Value Mode

If neither of the above is enabled:
- the model predicts corrected values directly

## Hidden Dataset Behavior

### Valid-pixel masks

The dataset always returns a boolean mask describing valid sea pixels.

This is important because:
- land pixels are represented with `NaN` targets
- region filtering can still leave masked pixels inside a cropped rectangle
- downstream losses should not treat those invalid pixels as real zeros

## Region Filtering

Region filtering happens inside the dataset rather than through separate datasets or file partitions.

Current supported region filters:
- `atlantic`
- `mediterranean`
- `aegean`
- `None`

Current config:
- `region_filter: mediterranean`

Behavior:
- the dataset infers latitude/longitude from a sample tensor
- crops the spatial domain to the selected region
- can still mask out non-region pixels inside that cropped rectangle

This is why the valid-pixel mask remains important even after region cropping.

## Normalization

Feature normalization is handled by [`WaveNormalizer`](../src/commons/preprocessing/bu_net_preprocessing.py), loaded in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

The normalizer is applied in the dataset layer, not in the trainer.

Important details:
- feature normalization happens before the optional sea-mask channel is appended
- target normalization is optional and mutually exclusive with bias/residual target modes

### Sea-mask channel

If `add_sea_mask_channel: true`:
- the dataset appends a binary sea-mask channel after normalization
- sea is defined as `~isnan(vhm0)`

Current config:
- `add_sea_mask_channel: true`
- `transformer_sea_mask_channel_index: 15`

### Domain-mean VHM0 channel

If `add_domain_mean_vhm0_channel: true`:
- the dataset computes the snapshot-wide mean `VHM0`
- broadcasts it as a constant spatial channel
- optionally normalizes it with `VHM0` stats

Current config:
- disabled

### Runtime subsampling and patch cropping

The documented preprocessing pipeline already writes subsampled `.pt` tensors, so:
- `data.subsample_step` is typically `null`

`CachedWaveDataset` still supports runtime subsampling, but that is not the documented current path.

If `patch_size` is set in `CachedWaveDataset`:
- a random crop is taken from each loaded sample

Current config:
- `patch_size: null`

So the documented path uses full preprocessed tensors rather than runtime crops.

## Model Layer

The trainer constructs the network through [`create_model(...)`](../src/classifiers/model_factory.py).

Supported DNN-family model types include:
- `nick`
- `geo`
- `enhanced`
- `transunet`
- `moe_transunet`
- `mlp`
- `swinunet`
- `transunet_gan`

Current config uses:
- `model_type: "moe_transunet"`

The model factory is the architecture selection point. This keeps `dnn_trainer.py` and `WaveBiasCorrector` independent of the concrete network class.

## Lightning Module Responsibilities

[`WaveBiasCorrector`](../src/classifiers/lightning_trainer.py) owns:
- model construction
- forward behavior
- task configuration
- optimizer and scheduler setup
- loss dispatch
- metric computation
- validation/eval logging

It is the main place where configuration choices become runtime behavior.

Important architectural role:
- the dataset decides what target representation to emit
- the Lightning module decides how to interpret predictions, compute loss, and reconstruct bias/residual forms for metrics

## Loss Layer

Loss selection is centralized in [`compute_loss(...)`](../src/commons/losses_factory.py).

The current config uses:
- `bin_balanced_smooth_l1`

That loss:
- uses the valid-pixel mask
- uses raw `VHM0` to assign sea-state bins
- averages non-empty bin losses so rare regimes are not dominated by common ones

The same loss factory also adds the residual penalty term when configured.

Current important nuance:
- `residual_penalty_lambda` is active not only in residual-to-prior mode, but also in bias mode because the trainer passes `residual_pred=y_pred` when `predict_bias: true`

### Supported loss families

The loss factory currently supports:
- `mse`
- `mse_with_calm_shrink`
- `smooth_l1`
- `weighted_mse`
- `multi_bin_weighted_smooth_l1`
- `pixel_switch_mse`
- `mse_perceptual`
- `mse_ssim`
- `mse_ssim_perceptual`
- `mse_mdn`
- `mdn`
- `mse_gan`
- `huber`
- `huber_classical`
- `mse_huber_tail`
- `multi_bin_weighted_mse`
- `bin_balanced_smooth_l1`
- `atlantic_low_bin_balanced_smooth_l1`

### Masked loss behavior

Almost every DNN loss in this repo is mask-aware:
- tensors are shape-aligned by cropping to the minimum shared `H x W`
- `NaN` targets are replaced with zeros only after mask logic is established
- the scalar loss is computed only over valid masked pixels

That behavior is one of the key reasons the dataset-layer mask is part of the architecture rather than just a convenience output.

### Current active loss behavior

With the checked-in config:
- base loss is bin-balanced Smooth L1
- loss is evaluated only on valid pixels
- raw `VHM0` is used for sea-state bin assignment
- an additional L2 penalty is applied to the predicted bias field

## Multi-Task Structure

The trainer supports both single-task and multi-task operation.

Current config is multi-task:
- `vhm0`
- `vtm02`

The multi-task loss path:
- reads `tasks_config`
- computes one loss per task
- applies task weights
- sums them into the total loss

This is handled in [`compute_multi_task_loss(...)`](../src/classifiers/lightning_trainer.py).

## Priors And Post-Processing Hooks

The training and evaluation stack can interact with two important external priors:
- static bias map
- EDCDF corrector

Static bias map:
- path from `data.static_bias_map_path`
- used directly in residual-to-prior training
- also used in evaluation/post-processing flows

EDCDF prior:
- path from `data.edcdf_model_path`
- loaded during evaluation flows rather than training
- used for optional blending or fallback behavior in DNN evaluation

So:
- static prior is part of the training architecture when residual mode is active
- EDCDF is mainly part of the evaluation-side architecture

## Optimization Layer

`WaveBiasCorrector` sets up:
- optimizer choice
- weight decay
- scheduler behavior

Current config uses:
- optimizer: `AdamW`
- learning rate: `3e-5`
- weight decay: `1e-3`
- scheduler: `CosineAnnealingLR`

The training entrypoint also adds callbacks for:
- checkpoints
- early stopping
- LR monitoring
- optional SWA
- optional EMA
- optional pixel-switch threshold updates
- optional layer freezing

## Evaluation Coupling

Training is not isolated from evaluation assumptions.

The following configuration fields affect both experiment behavior and downstream evaluation compatibility:
- target mode
- region filter
- feature count / channel count
- sea-mask channel usage
- prior paths
- experiment naming

This matters because the evaluation stack expects:
- compatible checkpoint outputs
- compatible input feature layout
- compatible prior artifacts such as static maps and EDCDF joblib models

## What To Read Next

Use this document as the overview, then go deeper only where needed:
- [`training-dnn.md`](training-dnn.md) for the operator run path
- [`features.md`](features.md) for active input channels
- [`config-reference.md`](config-reference.md) for practical config editing
