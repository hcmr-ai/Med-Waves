# `config_dnn.yaml` Detailed Reference

This document explains all active variables currently defined in [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml), what they control, and how they interact.

It is intended as the practical runbook for DNN tensor training.

## Scope And Structure

The config is organized into:
- `data`: dataset source, splitting, target semantics, feature filtering, and optional post-processing priors
- `model`: architecture, optimization, loss, scheduler, and task-weight setup
- `training`: runtime behavior, precision, validation cadence, and stabilization options
- `checkpoint`: resume behavior and checkpoint/S3 sync
- `logging`: experiment identity and telemetry backends

---

## `data`

### Where `data.*` Is Used In The Codebase

- Main ingestion and split wiring is in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py) via `create_data_loaders(...)`.
- Dataset-level behavior is implemented in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py) and (for patch sampling mode) [`src/commons/datasets/time_step_patch_dataset.py`](../src/commons/datasets/time_step_patch_dataset.py).
- Training-mode validation for mutually exclusive target modes is in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py) (`_validate_training_mode_config`).
- Evaluation-time postprocessing/prior blending (`blend_sigma`, EDCDF, low-bin affine, etc.) is consumed in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).
- `data.handle_nan` is currently documented in config, but not actively consumed in the DNN training/evaluation paths.

### Dataset Location And Split

- `data.data_path` (string): root path containing preprocessed `.pt` tensors.
  - Example in file: `/mnt/local_datasets/preprocessed_extended_subsampled_step_5/`
  - Must match the expected tensor format for the DNN loader.

- `data.file_pattern` (string): filename pattern matched under `data_path`.
  - Default in file: `WAVEAN*.pt`

- `data.train_year` (list[int]): years used for training split.
  - Current: `[2018, 2019, 2020, 2021]`

- `data.val_year` (list[int]): years used for validation split.
  - Current: `[2022]`

- `data.val_months` (list[int]): validation months to include from `val_year`.
  - Current: all months `1..12`

- `data.test_year` (list[int]): years used for test split.
  - Current: `[2023]`

- `data.test_months` (list[int]): test months to include from `test_year`.
  - Current: all months `1..12`
  - Code usage: file listing/splitting is handled by `get_file_list(...)` + `split_files_by_year(...)` in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

### Spatial Sampling And Filtering

- `data.patch_size` (list[int] | null): patch size for sliding-window extraction.
  - `null` means no runtime patch cropping from this setting (or full-grid behavior, depending on loader path).
  - Code usage: passed from loader into dataset constructors in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.stride` (list[int] | null): step size between sampled patches.
  - `null` disables explicit stride-based patch traversal.
  - Code usage note: not consumed by the current DNN loader path; `stride` exists in [`src/commons/datasets/grid_patched_dataset.py`](../src/commons/datasets/grid_patched_dataset.py) but is not wired through `create_data_loaders(...)`.

- `data.min_valid_pixels` (float): minimum fraction of valid sea pixels required for a patch to be kept.
  - Current: `0.3` (30%)
  - Code usage: used in patch-sampling config construction in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.max_files` (int | null): cap on number of files loaded.
  - `null` means use all matching files.
  - Useful for quick smoke runs.
  - Code usage: passed to `get_file_list(...)` in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.random_seed` (int): random seed for deterministic sampling/splitting behavior where applicable.
  - Current: `42`
  - Code usage: passed to patch dataset seed in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.subsample_step` (int | null): additional runtime downsampling factor.
  - `null` means no extra runtime subsampling.
  - Code usage: passed into `CachedWaveDataset(...)` in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py); also used in evaluation setup in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.use_cache` (bool): enables in-memory file cache.
  - Current: `false` to reduce host RAM pressure.
  - Code usage: forwarded into dataset constructors in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py); cache behavior implemented in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py).

- `data.max_cache_size` (int): max cached file entries when `use_cache=true`.
  - Current: `8`
  - Code usage: mapped to `max_cache_size`/`max_cache_files` in dataset constructors from [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.use_patch_sampling` (bool): enables explicit patch sampling mode.
  - Current: `false`
  - Code usage: branch switch in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py) and forwarded into model module from [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py).

- `data.sampling_mode` (string): patch sampling strategy.
  - Current: `exhaustive`
  - Keep aligned with `use_patch_sampling`.
  - Code usage: selects exhaustive vs balanced-bin sampler path in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.region_filter` (string | null): regional geographic filter.
  - Current: `mediterranean`
  - Supported note in config comments: `"atlantic"`, `"mediterranean"`, or `null`.
  - Code usage: region masking/cropping logic lives in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py) and is also used by evaluation in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

### Input Channels And Prediction Semantics

- `data.add_sea_mask_channel` (bool): appends sea-mask as an input channel.
  - Current: `true`
  - If enabled, `model.in_channels` and mask index settings must stay consistent.
  - Code usage: forwarded through loader/dataset path in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py) and respected by evaluation dataset creation in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.add_domain_mean_vhm0_channel` (bool): appends domain-mean VHM0 feature channel.
  - Current: `false`
  - Code usage: forwarded to `CachedWaveDataset(...)` from [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.predict_log_correction` (bool): predicts correction in log-space.
  - Current: `false`
  - Target becomes `log(corrected+eps) - log(raw+eps)` when enabled.
  - Code usage: forwarded to patch dataset path in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.predict_bias` (bool): target semantics switch.
  - `true`: model predicts bias (`corrected - raw`)
  - `false`: model predicts corrected value directly
  - Current: `true`
  - Code usage: validated in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py), used by datasets in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py), and used by evaluator initialization in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.predict_residual_to_prior` (bool): residual learning against a prior bias.
  - Current: `false`
  - When enabled, target is residual relative to configured prior.
  - Code usage: validated in trainer (`_validate_training_mode_config`) and enforced in dataset logic (including constraints) in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py).

- `data.prior_source` (string): prior source used in residual mode.
  - Current: `"static"`
  - Commented options currently indicate `"static"` or `"none"` (residual mode support is constrained).
  - Code usage: checked in `CachedWaveDataset` for residual mode support in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py).

- `data.residual_prior_task` (string | null): optional explicit task used as prior reference.
  - Current: `null` (auto-inferred for compatible single-task setups).
  - Code usage: auto-inferred/validated in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py) and revalidated in dataset initialization.

- `data.normalize_target` (bool): applies target normalization logic.
  - Current: `false`
  - Code usage: validated for mode exclusivity in trainer and passed to datasets/model in training pipeline.

- `data.bin_sampling_weights` (list[number] | null): optional per-bin sampling weights.
  - Current: `null`
  - Code usage: consumed by `BalancedBinBatchSampler` setup in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

### Static And Dynamic Prior Blending (Evaluation/Post-processing)

- `data.static_bias_map_path` (string): path to static bias map (`.npy`) used in optional blending.
  - Current points to Mediterranean static baseline diagnostics path.
  - Code usage: required by residual-prior dataset mode in [`src/commons/datasets/cache_wave_dataset.py`](../src/commons/datasets/cache_wave_dataset.py) and also used by evaluation postprocessing in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.blend_sigma` (float | null): Gaussian trust sigma for blending DNN bias with static bias map.
  - Smaller values increase pull toward the static map.
  - `null` disables this blend.
  - Code usage: evaluated in postprocessing blend logic in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.uncertainty_blend_sigma` (float | null): uncertainty-aware blend sigma variant.
  - `null` disables.
  - Code usage: used in uncertainty-aware blending path in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.domain_mean_recalibration` (bool): recalibrates DNN domain-mean to static-map level per timestep.
  - Current: `false`
  - Code usage: applied in evaluator recalibration routine in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.edcdf_model_path` (string): path to fitted EDCDF model used as dynamic prior.
  - Code usage: model loading and blend/fallback setup in evaluator (`_load_edcdf_corrector`) in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.edcdf_blend_sigma` (float | null): sigma for soft blend between DNN bias and EDCDF-implied bias.
  - `null` disables.
  - Code usage: soft blend trust computation in evaluator postprocessing path.

- `data.edcdf_hard_fallback_bins` (list[object] | null): optional wave bins where EDCDF bias replaces DNN bias 100%.
  - `null` disables hard fallback.
  - Code usage: hard-fallback bin routing in evaluator postprocessing path.

- `data.edcdf_fallback_bin_source` (string): source for selecting fallback bins.
  - Current: `"raw"`
  - Commented options: `"raw"`, `"edcdf"`, `"true"` (`"true"` is typically eval-only diagnostic mode).
  - Code usage: source selector in evaluator fallback path (`raw`/`edcdf`/`true`) in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.low_bin_affine_source` (string): source used for low-bin affine calibration routing.
  - Current: `"raw"`
  - Code usage: affine source selection in evaluator postprocessing.

- `data.low_bin_affine_params` (list[object] | null): optional affine calibration rules by bin (`b' = a*b + c`).
  - `null` means disabled.
  - When enabled, each item typically defines `min`, `max`, `a`, `c`.
  - Code usage: parsed/applied in evaluator low-bin affine branch in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

### Targets, Features, And Missing Values

- `data.target_columns` (map[string,string]): prediction tasks and their target column names.
  - Current:
    - `vhm0: "corrected_VHM0"`
    - `vtm02: "corrected_VTM02"`
  - Task names should align with `model.tasks_config`.
  - Code usage: propagated into datasets (`CachedWaveDataset`), trainer validation, and evaluation task selection (`--eval-task`) in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `data.excluded_columns` (list[string]): columns removed from model inputs.
  - Current list excludes metadata/time fields and target columns, plus directional/storm-related engineered fields.
  - Adjusting this list changes effective input channels.
  - Code usage: feature-channel filtering in dataset classes via loader wiring in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).

- `data.handle_nan` (bool): enables NaN handling/masking in preprocessing pipeline.
  - Current: `true`
  - Code usage note: currently not referenced in DNN training/evaluation code paths.

- `data.normalizer_path` (string): path to scaler/normalizer artifact used by data pipeline.
  - Current points to `BU24h_zscore_18-21_med_extended.pkl`.
  - Code usage: loaded in loader/evaluation setup (`WaveNormalizer.load(...)` or S3 path handling) in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py) and [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

---

## `model`

### Where `model.*` Is Used In The Codebase

- `model.*` is injected into `WaveBiasCorrector(...)` / `WaveBiasCorrector.load_from_checkpoint(...)` in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py).
- Architecture creation is delegated to [`src/classifiers/model_factory.py`](../src/classifiers/model_factory.py).
- Training-step loss/optimizer/scheduler/gating behavior is implemented in [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py).
- Scheduler object construction is centralized in [`src/commons/scheduler_factory.py`](../src/commons/scheduler_factory.py).

### Core Model And Optimization

- `model.model_type` (string): architecture family.
  - Current: `"moe_transunet"`
  - Comments mention supported values such as `nick`, `geo`, `enhanced`, `transunet`, `moe_transunet`, `mlp`, `swinunet`, `transunet_gan`.
  - Code usage: selected in trainer and resolved by model factory in [`src/classifiers/model_factory.py`](../src/classifiers/model_factory.py).

- `model.in_channels` (int): number of model input channels after feature filtering and channel augmentation.
  - Current: `16`
  - Must match actual tensor channel construction.

- `model.learning_rate` (float): base optimizer learning rate.
  - Current: `3e-5`

- `model.loss_type` (string): active loss function family.
  - Current: `"bin_balanced_smooth_l1"`
  - Code usage: drives criterion setup and training branches in [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py) and loss dispatch in [`src/commons/losses_factory.py`](../src/commons/losses_factory.py).

- `model.filters` (list[int]): encoder/decoder feature widths (for compatible architectures).
  - Current: `[32, 64, 128]`

- `model.dropout` (float): dropout probability in relevant model blocks.
  - Current: `0`

- `model.add_vhm0_residual` (bool): enables additive residual connection using VHM0 channel (architecture-dependent).
  - Current: `false`

- `model.vhm0_channel_index` (int): index of VHM0 input channel for residual/additive wiring.
  - Current: `0`

- `model.weight_decay` (float): optimizer weight decay.
  - Current: `1e-3`

- `model.upsample_mode` (string): decoder upsampling implementation.
  - Current: `"nearest"`
  - Comment notes `"nearest"` or `"transpose"` logic.

- `model.optimizer_type` (string): optimizer backend.
  - Current: `"AdamW"`
  - Commented options: `Adam`, `AdamW`, `SGD`.
  - Code usage: optimizer selection in `configure_optimizers()` inside [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py).

### Probabilistic/GAN/Residual Penalties

- `model.use_mdn` (bool): enables MDN behavior where supported.
  - Current: `false`
  - Code usage: model/head and training branch behavior in Lightning module and network implementations (for example [`src/classifiers/networks/trans_unet.py`](../src/classifiers/networks/trans_unet.py)).

- `model.lambda_adv` (float): adversarial-loss weighting in GAN-like training modes.
  - Current: `0.1`
  - Code usage: generator total loss weighting in GAN path of [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py).

- `model.n_discriminator_updates` (int): discriminator update count per generator step in GAN mode.
  - Current: `3`
  - Code usage: discriminator update loop count in GAN training branch.

- `model.discriminator_lr_multiplier` (float): learning-rate multiplier for discriminator relative to generator LR.
  - Current: `2.0`
  - Code usage: discriminator optimizer LR derivation in Lightning `configure_optimizers()`.

- `model.residual_penalty_lambda` (float): L2 regularization weight applied to predicted residual magnitude (residual mode).
  - Current: `0.1`
  - Code usage: residual penalty term in Lightning training loss path.

### TransUNet-Specific Parameters

- `model.transunet_base_channels` (int): base channel count in TransUNet backbone.
  - Current: `16`

- `model.transunet_bottleneck_dim` (int): transformer bottleneck embedding width.
  - Current: `128`

- `model.transunet_patch_size` (int): transformer token patch size.
  - Current: `15`

- `model.transunet_num_layers` (int): number of transformer layers.
  - Current: `1`

- `model.transunet_num_heads` (int): attention heads per transformer layer.
  - Current: `4`

- `model.transformer_dropout` (float): transformer dropout.
  - Current: `0.2`

- `model.transformer_use_coord_pos_enc` (bool): enables coordinate positional encoding.
  - Current: `true`

- `model.transformer_sea_mask_channel_index` (int): channel index of sea mask for transformer logic when mask channel is present.
  - Current: `15`
  - Code usage for all TransUNet knobs above: passed from trainer to model factory and consumed by TransUNet modules in [`src/classifiers/networks/trans_unet.py`](../src/classifiers/networks/trans_unet.py).

### MoE TransUNet Parameters

- `model.num_experts` (int): number of expert heads.
  - Current: `3`

- `model.gate_temperature` (float): softmax temperature for gate assignments.
  - Current: `1.0`

- `model.gate_input_mode` (string): gate input source.
  - Current: `"features"`
  - Commented options include `"features"` and `"input_channels"`.

- `model.gate_input_channels` (list[int]): specific input-channel indices used by gate when `gate_input_mode="input_channels"`.
  - Current: `[0]`

- `model.gate_entropy_weight` (float): regularization weight encouraging gate entropy.
  - Current: `0.0`

- `model.gate_balance_weight` (float): regularization weight encouraging balanced expert usage.
  - Current: `0.0`

- `model.gate_prior_weight` (float): prior-guided gate regularization weight.
  - Current: `0.0`

- `model.gate_bin_edges` (list[float]): wave bins used for gate-prior partitioning.
  - Current: `[1.0, 3.0]`

- `model.expert_diversity_weight` (float): diversity penalty weight across experts (cosine-similarity discouragement).
  - Current: `0.0`

- `model.expert_dropout` (float): dropout applied at expert level.
  - Current: `0.0`

- `model.return_gate_maps` (bool): emits gate maps for diagnostics/analysis.
  - Current: `true`
  - Code usage for all MoE knobs above: consumed by MoE model construction and gating/loss regularization branches in [`src/classifiers/networks/trans_unet.py`](../src/classifiers/networks/trans_unet.py) and [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py).

### Scheduler (`model.lr_scheduler`)

- `model.lr_scheduler.type` (string): scheduler class/strategy.
  - Current: `CosineAnnealingLR`
  - Comments mention: `ReduceLROnPlateau`, `CosineAnnealingLR`, `StepLR`, `ExponentialLR`, `none`, `CosineAnnealingWarmupRestarts`, `LambdaLR`.

- `model.lr_scheduler.monitor` (string): monitored metric for monitor-based schedulers.
  - Current: `val_loss`

- `model.lr_scheduler.mode` (string): optimization direction for monitored metric.
  - Current: `min`

- `model.lr_scheduler.factor` (float): reduction factor for compatible schedulers.
  - Current: `0.5`

- `model.lr_scheduler.patience` (int): patience before LR reduction for compatible schedulers.
  - Current: `5`

- `model.lr_scheduler.min_lr` (float): floor learning rate.
  - Current: `1e-7`

- `model.lr_scheduler.T_max` (int): total optimizer-step horizon for cosine annealing schedule.
  - Current: `5475`

- `model.lr_scheduler.T_max_epochs` (int): epoch-horizon variant for scheduler logic where supported.
  - Current: `5`

- `model.lr_scheduler.eta_min` (float): minimum LR in cosine scheduling.
  - Current: `1e-6`

- `model.lr_scheduler.warmup_steps` (float | int): warmup amount for warmup-enabled scheduling paths.
  - Current: `0.01`
  - Verify expected unit (fraction vs absolute steps) in trainer implementation before changing.

- `model.lr_scheduler.step_size` (int): step interval for `StepLR`.
  - Current: `10`

- `model.lr_scheduler.gamma` (float): multiplicative LR decay for `StepLR`/`ExponentialLR`.
  - Current: `0.1`
  - Code usage for scheduler block: `lr_scheduler_config` is passed from trainer into Lightning module and resolved by [`src/commons/scheduler_factory.py`](../src/commons/scheduler_factory.py).

### Task Weights (`model.tasks_config`)

- `model.tasks_config` (list[object]): task names and scalar weights used in multi-task aggregation.
  - Current entries:
    - `{ name: "vhm0", weight: 1.0 }`
    - `{ name: "vtm02", weight: 1.0 }`
  - `name` values should match keys in `data.target_columns`.
  - Code usage: task list and per-task weighting/loss setup in [`src/classifiers/lightning_trainer.py`](../src/classifiers/lightning_trainer.py).

---

## `training`

### Where `training.*` Is Used In The Codebase

- Trainer runtime args (`accelerator`, `devices`, `precision`, clipping, validation cadence, accumulation) are passed into `Trainer(...)` in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py).
- Callback-related training keys (`early_stopping_patience`, `save_top_k`, SWA/EMA/freezing, grad-norm logging toggle) are wired in `create_callbacks(...)` in the same file.
- DataLoader behavior (`batch_size`, workers, `persistent_workers`, `prefetch_factor`) is consumed in [`src/commons/dataloaders.py`](../src/commons/dataloaders.py).
- `log_train_sea_bin_metrics` is forwarded to the Lightning module and used in metric-logging branches.

- `training.batch_size` (int): per-step batch size.
  - Current: `32`

- `training.accumulate_grad_batches` (int): gradient accumulation steps.
  - Current: `1`

- `training.max_epochs` (int): max training epochs.
  - Current: `50`

- `training.num_workers` (int): DataLoader workers.
  - Current: `6`

- `training.persistent_workers` (bool): keeps workers alive between epochs.
  - Current: `true`

- `training.pin_memory` (bool): enables pinned memory in DataLoader.
  - Current: `true`

- `training.accelerator` (string): device backend selection.
  - Current: `"gpu"`

- `training.devices` (int | list): number of devices (or explicit device list in compatible frameworks).
  - Current: `1`

- `training.precision` (string | int): numeric precision mode.
  - Current: `"32"`

- `training.log_every_n_steps` (int): training log cadence.
  - Current: `20`

- `training.early_stopping_patience` (int): epochs without improvement before early stop.
  - Current: `10`

- `training.save_top_k` (int): number of best checkpoints to keep.
  - Current: `50`

- `training.monitor` (string): metric used for early stopping and top-k checkpoint ranking.
  - Current: `val_loss`

- `training.mode` (string): optimization direction for `monitor`.
  - Current: `min`

- `training.fast_dev_run` (bool): quick debug run mode.
  - Current: `false`

- `training.check_val_every_n_epoch` (int): validation frequency by epoch interval.
  - Current: `1`

- `training.run_eval_each_epoch` (bool): runs additional evaluation routine each epoch.
  - Current: `true`

- `training.num_sanity_val_steps` (int): sanity validation steps before first training epoch.
  - Current: `0`

- `training.benchmark` (bool): enables backend benchmarking/autotune (framework-dependent).
  - Current: `true`

- `training.prefetch_factor` (int): batches prefetched per DataLoader worker.
  - Current: `2`

- `training.val_check_interval` (float | int | null): intra-epoch validation frequency override.
  - Current: `null` (epoch-based checks apply).

- `training.use_swa` (bool): enables stochastic weight averaging path.
  - Current: `false`

- `training.use_ema` (bool): enables exponential moving average of model weights.
  - Current: `false`

- `training.finetune_model` (bool): toggles fine-tuning mode assumptions.
  - Current: `false`

- `training.freeze_encoder_layers` (bool): freezes encoder layers for fine-tune scenarios.
  - Current: `false`

- `training.pixel_switch_threshold_m` (float): threshold used by pixel-switch loss/logic modes (meters).
  - Current: `0.45`
  - Code usage note: consumed from `model.pixel_switch_threshold_m` in trainer/lightning paths.

- `training.gradient_clip_val` (float | null): gradient clipping magnitude.
  - Current: `null` (disabled)

- `training.gradient_clip_algorithm` (string): clipping algorithm when clipping enabled.
  - Current: `"norm"`

- `training.aggressive_freeze` (bool): stronger parameter freezing mode.
  - Current: `false`

- `training.log_train_sea_bin_metrics` (bool): per-step sea-bin metric logging during training.
  - Current: `false` (lower sync overhead)

- `training.log_param_grad_norms` (bool): per-parameter gradient-norm logging.
  - Current: `false` (reduces telemetry overhead)

---

## `checkpoint`

### Where `checkpoint.*` Is Used In The Codebase

- Checkpoint callback and optional S3 sync wiring are in `create_callbacks(...)` in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py).
- Resume resolution is handled in trainer main flow (`--resume` override, config fallback, optional S3 download) in the same file.
- `low_wave_ckpt` and `high_wave_ckpt` are consumed by evaluation specialization logic in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `checkpoint.resume_from_checkpoint` (string | null): checkpoint path for resuming interrupted or staged training.
  - Current: `null`

- `checkpoint.checkpoint_dir` (string): local/attached storage path where checkpoints are written.
  - Current placeholder: `/mnt/blobstorage/checkpoints/<fill_me_checkpoint_dir>`

- `checkpoint.s3_sync_dir` (string | null): optional S3 URI for checkpoint backup/sync.
  - Current: `null`

- `checkpoint.save_last` (bool): always keep/update a `last` checkpoint artifact.
  - Current: `true`

- `checkpoint.sync_frequency` (int): epoch interval for S3 sync when enabled.
  - Current: `2`

- `checkpoint.high_wave_ckpt` (string | null): optional specialized checkpoint slot for high-wave model staging.
  - Current: `null`

- `checkpoint.low_wave_ckpt` (string | null): optional specialized checkpoint slot for low-wave model staging.
  - Current: `null`

---

## `logging`

### Where `logging.*` Is Used In The Codebase

- Logger creation is centralized in `create_experiment_loggers(...)` in [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py).
- Comet visualization callback is enabled from logging/training switches in `create_callbacks(...)`.
- Evaluation output directories also reference `logging.experiment_name` in [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py).

- `logging.log_dir` (string): run output/log root.
  - Current placeholder: `/mnt/blobstorage/<fill_me_log_dir>`

- `logging.experiment_name` (string): run identifier used across logging/checkpoint naming.
  - Current placeholder: `<fill_me_experiment_name>`

- `logging.use_comet` (bool): enables Comet logging backend.
  - Current: `true`

- `logging.use_tensorboard` (bool): enables TensorBoard logging backend.
  - Current: `false`

- `logging.comet_tags` (list[string]): tags attached to Comet experiment.
  - Current tags reflect architecture, objective, region, and scheduler setup.

- `logging.comet_notes` (string): free-form run notes shown in Comet.
  - Current note summarizes MoE/TransUNet and regularization settings.

---

## Consistency Checks Before Running

For reliable runs, verify these groups together:

- **Channels:** `data.excluded_columns`, `data.add_sea_mask_channel`, `data.add_domain_mean_vhm0_channel`, `model.in_channels`, `model.transformer_sea_mask_channel_index`
- **Task mapping:** `data.target_columns` keys match every `model.tasks_config[].name`
- **Target semantics:** `data.predict_bias`, `data.predict_log_correction`, `data.predict_residual_to_prior`, `data.prior_source`, `model.residual_penalty_lambda`
- **Scheduler intent:** `model.learning_rate` and `model.lr_scheduler.*` must be coherent for step-based vs epoch-based tuning
- **Run identity:** `checkpoint.checkpoint_dir`, `logging.log_dir`, `logging.experiment_name`, and optional `checkpoint.s3_sync_dir`

## Common Safe Edit Workflow

1. Copy `config_dnn.yaml` to an experiment-specific file.
2. Update split years/months and region first.
3. Confirm target semantics (`predict_bias` vs corrected-direct vs residual).
4. Recompute/verify `in_channels` if you changed input feature composition.
5. Set checkpoint/logging paths before launch.
6. Run a short smoke test (`max_files`, or reduced epochs) before full training.
