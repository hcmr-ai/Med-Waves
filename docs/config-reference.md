# Config Reference

This is a practical guide to the main config files under [`src/configs`](../src/configs). It is not a full schema. The goal is to make it easy to identify which config to start from and which fields are operationally important.

## Quick Map

`config_dnn.yaml`
- DNN training on preprocessed `.pt` tensors
- used by [`src/pipelines/training/dnn_trainer.py`](../src/pipelines/training/dnn_trainer.py)

`config_full_dataset.yaml`
- full-dataset or classical-model training on parquet data
- used by the full-dataset training pipeline

`config_evaluation.yaml`
- evaluation for the full-dataset or parquet-based model family

`config_model_per_point.yaml`
- per-point training on parquet data
- used by [`src/pipelines/training/train_model_per_point.py`](../src/pipelines/training/train_model_per_point.py)
- if this path is revisited, also check the legacy per-point implementation under [`scripts/trainers/`](../scripts/trainers)

## Shared Patterns

Most configs follow the same logic:
- `data`: where files live and how splits are defined
- `model`: which estimator or architecture to use
- `feature_block` or equivalent: what target is learned and which features are excluded
- `training`, `evaluation`, `output`, or `checkpoint`: runtime behavior and artifact destinations

The most important distinction across configs is the data format:
- DNN configs use preprocessed `.pt` tensors
- model-per-point and full-dataset configs use parquet tables

## `config_dnn.yaml`

Use this for:
- UNet-style, TransUNet-style, MoE, and related tensor models
- full variable-by-variable details in [`config-dnn-reference.md`](config-dnn-reference.md)

Key areas:
- `data.data_path`: points to the `.pt` tensor dataset
- `data.file_pattern`: expected file pattern, usually `WAVEAN*.pt`
- `data.train_year`, `data.val_year`, `data.test_year`: year-based split
- `data.target_columns`: prediction targets
- `data.excluded_columns`: features removed from model inputs
- `data.predict_bias`: predict correction bias instead of corrected value directly
- `data.predict_residual_to_prior`: residual-learning mode relative to a prior
- `model.model_type`: network family
- `model.in_channels`: must match the active feature set
- `model.loss_type`: active loss family
- `model.residual_penalty_lambda`: extra penalty on predicted bias or residual magnitude
- `training.batch_size`, `training.num_workers`, `training.precision`: main memory and throughput knobs
- `checkpoint.resume_from_checkpoint`, `checkpoint.checkpoint_dir`, `logging.log_dir`, `logging.experiment_name`: tightly coupled run-identity fields

Important operational note:
- for a fresh run, do not change only `experiment_name` or only `checkpoint_dir`
- update the checkpoint and logging fields together

## `config_model_per_point.yaml`

Use this for:
- per-point models trained from parquet rather than tensor grids

Handover note:
- before relying on this path, also inspect the legacy per-point code under [`scripts/trainers/`](../scripts/trainers)
- the repo contains both current `src/` and older previous-researcher per-point surfaces

Key areas:
- `data.data_path`: parquet root
- `data.file_pattern`: usually `*.parquet`
- `data.split`: split strategy and years or months
- `feature_block.predict_bias`: output semantics
- `feature_block.features_to_exclude`: feature filtering
- `feature_block.sampling_strategy`: how rows are sampled
- `feature_block.regional_training`: optional region filtering
- `model`: estimator settings for the per-point trainer
- `evaluation` and output-related sections: result generation and artifact saving

Important operational note:
- this config is row-based, not grid-based
- changes to sampling strategy can materially change the training set size and class balance

## `config_full_dataset.yaml`

Use this for:
- classical ML or full-dataset experiments on parquet data

Typical model types in this config family:
- `xgb`
- `rf`
- `elasticnet`
- `lasso`
- `ridge`
- `eqm`
- `delta`

Key areas:
- `model.type`: estimator family
- `model.*`: estimator-specific hyperparameters
- `data.data_path`: parquet root
- `data.split`: year-based or other split mode
- `feature_block.predict_bias`: target semantics
- `feature_block.features_to_exclude`: feature filtering
- `feature_block.max_samples_per_file` and `sampling_strategy`: main sampling controls
- `feature_block.regional_training`: optional region filtering
- `feature_block.scaler`: tabular scaling mode

Important operational note:
- this config is for parquet-based training, not `.pt` tensors
- it is easier to compare tabular baselines here than in the DNN config

## `config_evaluation.yaml`

Use this for:
- evaluation of the full-dataset or parquet-based model family

Key areas:
- `data.model_path`: trained model location
- `data.data_path`: parquet dataset location
- `feature_block`: must stay aligned with the training-time feature logic
- `evaluation.year` and related settings: what period is evaluated
- `output.output_dir`: local output directory
- `output.s3`: optional remote result destination
- `diagnostics`: which plots and reports to create

Important operational note:
- this config is not the main DNN evaluation surface
- DNN evaluation is documented separately in [`evaluation_dnn.md`](evaluation_dnn.md)

## Which Config To Start From

If the task is:
- tensor DNN training: start from `config_dnn.yaml`
- parquet per-point training: start from `config_model_per_point.yaml`
- classical ML or full-dataset training: start from `config_full_dataset.yaml`
- parquet-based model evaluation: start from `config_evaluation.yaml`

## Safe Editing Rules

- Confirm whether the pipeline expects `.pt` or parquet before changing `data_path`.
- Keep target semantics consistent: `predict_bias`, residual modes, and target columns should not be changed casually.
- Treat artifact paths as part of experiment identity.
- For new experiments, prefer a derived config instead of rewriting the checked-in base config in place.
