# Feature Engineering Experiments

This document describes the extended IncrementalTrainer that supports various feature engineering experiments while maintaining streaming capabilities.

## Overview

The IncrementalTrainer has been extended to support:
- Baseline SGD (no polynomial features)
- ElasticNet SGD with feature selection
- Polynomial features on subset or all features
- Feature selection via SelectFromModel
- Dimension reduction (IncrementalPCA, TruncatedSVD, RandomProjection)
- Comprehensive logging and artifact saving
- **Proper time-based data splitting** for temporal evaluation
- **Parquet file support** for efficient data loading

## Configuration

All experiments are configured via the `feature_block` section in YAML config files:

```yaml
feature_block:
  base_features: ["vhm0_x", "wspd", "lat", "lon"]
  use_poly: false
  poly_degree: 2
  poly_scope: "subset"   # "subset" | "all"
  use_selector: false
  selector_type: "elasticnet"  # "lasso" | "elasticnet"
  selector_alpha: 1e-3
  selector_l1_ratio: 0.1
  warmup_days: 7
  use_dimred: false
  dimred_type: "ipca"     # "ipca" | "tsvd" | "rproj"
  dimred_components: 64
```

### Configuration Parameters

- **base_features**: List of base features to use for polynomial expansion when `poly_scope="subset"`
- **use_poly**: Enable polynomial feature generation
- **poly_degree**: Degree of polynomial features (default: 2)
- **poly_scope**: 
  - `"subset"`: Apply polynomial features only to base_features
  - `"all"`: Apply polynomial features to all available features
- **use_selector**: Enable feature selection
- **selector_type**: Type of selector (`"lasso"` or `"elasticnet"`)
- **selector_alpha**: Regularization strength for selector
- **selector_l1_ratio**: L1 ratio for ElasticNet selector
- **warmup_days**: Number of days to use for warmup stage
- **use_dimred**: Enable dimension reduction
- **dimred_type**: Type of dimension reduction:
  - `"ipca"`: IncrementalPCA (for dense features)
  - `"tsvd"`: TruncatedSVD (for sparse features)
  - `"rproj"`: GaussianRandomProjection
- **dimred_components**: Number of components for dimension reduction

## Experiment Types

### 1. Baseline SGD
**Config**: `config_baseline_sgd.yaml`
- No polynomial features
- No feature selection
- No dimension reduction
- Uses only base features: `["vhm0_x", "wspd", "lat", "lon"]`

### 2. ElasticNet SGD
**Config**: `config_elasticnet_sgd.yaml`
- No polynomial features
- Feature selection with ElasticNet
- L1 ratio: 0.05 (close to Lasso)
- Alpha: 0.05

### 3. Poly-lite
**Config**: `config_poly_lite.yaml`
- Polynomial features (degree=2) on subset only
- No feature selection
- No dimension reduction
- Features: `["vhm0_x", "wspd", "lat", "lon"]` → polynomial expansion

### 4. Poly + Selection
**Config**: `config_poly_selection.yaml`
- Polynomial features (degree=2) on all features
- Feature selection with ElasticNet
- L1 ratio: 0.2
- Alpha: 1e-3

### 5. Poly + Selection + DimRed
**Config**: `config_poly_dimred.yaml`
- Polynomial features (degree=2) on all features
- Feature selection with ElasticNet
- Dimension reduction with IncrementalPCA
- Components: 64

### 6. Debug Configuration
**Config**: `config_debug.yaml`
- Minimal configuration for quick testing
- Reduced warmup days (1 day)
- Comet logging disabled
- Perfect for `--debug` mode runs

## Running Experiments

### Using the Main Training Script

```bash
# Normal runs (full dataset)
poetry run python src/pipelines/training/train_incremental.py --config config_baseline_sgd
poetry run python src/pipelines/training/train_incremental.py --config config_elasticnet_sgd
poetry run python src/pipelines/training/train_incremental.py --config config_poly_lite
poetry run python src/pipelines/training/train_incremental.py --config config_poly_selection
poetry run python src/pipelines/training/train_incremental.py --config config_poly_dimred

# Debug runs (quick testing with minimal data)
poetry run python src/pipelines/training/train_incremental.py --config config_debug --debug
poetry run python src/pipelines/training/train_incremental.py --config config_baseline_sgd --debug --debug-train-days 2 --debug-test-days 1
poetry run python src/pipelines/training/train_incremental.py --config config_poly_lite --debug --debug-train-days 3 --debug-test-days 2
```

### Using the Trainer Directly

```python
import yaml
from src.pipelines.training.train_incremental_global_model import IncrementalTrainer

# Load config
with open("src/configs/config_poly_lite.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize trainer
trainer = IncrementalTrainer(config)

# Run experiment
trainer.train(x_train_files, y_train_files)
trainer.evaluate(x_test_files, y_test_files)
```

## Architecture

### Data Splitting
- **Time-based split**: Uses `time_based_split()` function for proper temporal evaluation
- **Training period**: 2021-2022 (inclusive)
- **Testing period**: 2023 (inclusive)
- **File format**: Parquet files with naming pattern `WAVEAN20231231.parquet`
- **No data leakage**: Ensures no future information is used during training

### Warmup Stage
1. Streams first `warmup_days` of training files
2. Fits transforms in sequence:
   - Scaler
   - PolynomialFeatures (if enabled)
   - FeatureSelector (if enabled)
   - Dimension reduction (if enabled)
3. Freezes all transforms

### Training Stage
1. Streams all training files (2021-2022)
2. Applies frozen transforms to each batch
3. Updates SGDRegressor with `partial_fit`

### Evaluation Stage
1. Streams test files (2023)
2. Applies same frozen transforms
3. Evaluates model performance

## Logging

### Comet ML Integration
- **Run naming**: Automatically generated based on configuration
  - Example: `sgd_poly2_subset_elasticnet0.001_l10.2_ipca64`
- **Feature reports**: Logs feature counts at each transformation step
- **Performance metrics**: RMSE, MAE for training and evaluation
- **Explained variance**: For dimension reduction methods
- **Selector coefficients**: Bar chart of feature importance

### Artifacts Saved
- `scaler.joblib`: Fitted StandardScaler
- `poly.joblib`: Fitted PolynomialFeatures (if used)
- `selector.joblib`: Fitted FeatureSelector (if used)
- `dimred.joblib`: Fitted dimension reduction (if used)
- `{model_name}.joblib`: Trained SGDRegressor

## Key Features

### Streaming Architecture
- No concatenation of full-year data
- Processes data in hourly/day chunks
- Memory-efficient for large datasets

### Feature Selection
- Custom `FeatureSelector` class preserves indices for streaming
- Fits on warmup sample only
- Applies selection during training/evaluation

### Dimension Reduction
- **IncrementalPCA**: Fits on warmup, transforms during streaming
- **TruncatedSVD**: Fits on warmup, transforms during streaming
- **RandomProjection**: Fits on warmup, transforms during streaming

### Error Handling
- Robust NaN handling
- Graceful handling of empty batches
- Comprehensive logging of errors

## Performance Considerations

1. **Warmup size**: Larger warmup_days provide better transform fitting but use more memory
2. **Batch size**: Smaller batches use less memory but may be slower
3. **Feature selection**: Can significantly reduce feature count and training time
4. **Dimension reduction**: Can reduce memory usage and training time

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch_size or warmup_days
2. **Feature mismatch**: Ensure consistent feature names across files
3. **Transform errors**: Check that warmup stage completed successfully

### Debugging
- Enable `log_batch_metrics: true` for detailed training progress
- Check Comet ML logs for feature transformation reports
- Verify artifact files are saved correctly
