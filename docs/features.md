# Features

This document describes the Med-WAV feature set after preprocessing and feature engineering.

It separates:
- features produced by preprocessing
- targets
- tensor-derived features used during scaler fitting
- features actually used by the current DNN config

Code anchors:
- feature augmentation:
  - [`src/data_engineering/neptune/feature_augmentation.py`](../src/data_engineering/neptune/feature_augmentation.py)
- `.nc` to parquet pipeline:
  - [`src/data_engineering/aws/netcdf_to_parquet_features.py`](../src/data_engineering/aws/netcdf_to_parquet_features.py)
- tensor-derived features for scaler fitting:
  - [`src/pipelines/preprocessing/fit_scalers_from_tensors.py`](../src/pipelines/preprocessing/fit_scalers_from_tensors.py)
- current model input selection:
  - [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml)

## Feature Groups

The feature set falls into four groups:

1. raw source variables from the marine model fields
2. engineered features added during parquet generation
3. corrected target variables
4. tensor-derived features added later during scaler fitting

## Meteorological Variables

| Feature | Description | Status |
| --- | --- | --- |
| `WSPD` | Wind speed at 10 meters | produced and used |
| `WDIR` | Wind direction in degrees | produced, excluded from current DNN inputs |
| `U10` | Zonal wind component: `WSPD * sin(WDIR)` | produced and used |
| `V10` | Meridional wind component: `WSPD * cos(WDIR)` | produced and used |
| `wind_dir_sin` | Sine encoding of wind direction | produced, excluded from current DNN inputs |
| `wind_dir_cos` | Cosine encoding of wind direction | produced, excluded from current DNN inputs |

## Wave Parameters

| Feature | Description | Status |
| --- | --- | --- |
| `VHM0` | Significant wave height from degraded model | produced and used |
| `corrected_VHM0` | Bias-corrected significant wave height | produced, target |
| `VTM02` | Mean wave period from degraded model | produced and used |
| `corrected_VTM02` | Bias-corrected mean wave period | produced, target |
| `VMDR` | Mean wave direction in degrees | produced, excluded from current DNN inputs |
| `wave_dir_sin` | Sine encoding of wave direction | produced and used |
| `wave_dir_cos` | Cosine encoding of wave direction | produced and used |

## Temporal Encoding Features

These are created during feature augmentation to handle periodic structure without discontinuities.

| Feature | Description | Status |
| --- | --- | --- |
| `sin_hour` | Sine encoding of hour-of-day | produced and used |
| `cos_hour` | Cosine encoding of hour-of-day | produced and used |
| `sin_doy` | Sine encoding of day-of-year | produced and used |
| `cos_doy` | Cosine encoding of day-of-year | produced and used |
| `sin_month` | Sine encoding of month-of-year | produced and used |
| `cos_month` | Cosine encoding of month-of-year | produced and used |

## Geospatial Features

| Feature | Description | Status |
| --- | --- | --- |
| `latitude` | Latitude of grid point | produced, excluded from current DNN inputs |
| `longitude` | Longitude of grid point | produced, excluded from current DNN inputs |
| `lat_norm` | Min-max normalized latitude | produced and used |
| `lon_norm` | Min-max normalized longitude | produced and used |

## Circular / Alignment Features

These are engineered in feature augmentation to encode circular variables and wind-wave alignment.

| Feature | Based on | Description | Status |
| --- | --- | --- | --- |
| `sin_delta` | `WDIR - VMDR` | Sine of wind-wave directional difference | produced, excluded from current DNN inputs |
| `cos_delta` | `WDIR - VMDR` | Cosine of wind-wave directional difference | produced, excluded from current DNN inputs |
| `alongwind` | `WSPD * cos_delta` | Wind projection aligned with wave direction | produced, excluded from current DNN inputs |
| `crosswind` | `WSPD * sin_delta` | Wind projection orthogonal to wave direction | produced, excluded from current DNN inputs |

## Storm-Regime Features

| Feature | Description | Status |
| --- | --- | --- |
| `storm_regime` | `log(1 + VHM0)` | produced, excluded from current DNN inputs |
| `storm_regime_sig` | sigmoid transform centered near `VHM0 = 3.0` | produced, excluded from current DNN inputs |

## Timestamp / Metadata Columns

| Feature | Description | Status |
| --- | --- | --- |
| `time` | original time column from source data | produced, excluded from current DNN inputs |
| `timestamp` | explicit datetime cast created during feature augmentation | produced, excluded from current DNN inputs |

## Tensor-Derived Features For Scaler Fitting

These are not created during parquet feature augmentation. They are added later from the full `(T, H, W)` tensor representation in [`fit_scalers_from_tensors.py`](../src/pipelines/preprocessing/fit_scalers_from_tensors.py).

| Feature | Description | Status |
| --- | --- | --- |
| `dVHM0` | Temporal difference of `VHM0` along the time axis | derived during scaler fitting |
| `dWSPD` | Temporal difference of `WSPD` along the time axis | derived during scaler fitting |
| `grad_mag` | 2D spatial gradient magnitude of `VHM0` | derived during scaler fitting |

Important note:
- these features require the dense tensor layout and are computed after parquet-to-`.pt` conversion
- they are part of the scaler-fitting path, not the current DNN input feature set in `config_dnn.yaml`

## Why These Engineered Features Exist

### Circular variables

The following are produced to avoid discontinuities such as `0°` vs `360°`:
- `U10`, `V10`
- `wind_dir_sin`, `wind_dir_cos`
- `wave_dir_sin`, `wave_dir_cos`
- `sin_hour`, `cos_hour`
- `sin_doy`, `cos_doy`
- `sin_month`, `cos_month`

### Spatial awareness

The following allow the model to learn geographic structure while keeping magnitude scales controlled:
- `lat_norm`
- `lon_norm`

### Wind-wave interaction

These capture directional coupling between wind and waves:
- `sin_delta`
- `cos_delta`
- `alongwind`
- `crosswind`

### Storm intensity regime

These provide a coarse nonlinear representation of sea-state intensity:
- `storm_regime`
- `storm_regime_sig`

## Current DNN Targets

From the current config:

```yaml
target_columns:
  vhm0: corrected_VHM0
  vtm02: corrected_VTM02
```

So the current multi-task targets are:
- `corrected_VHM0`
- `corrected_VTM02`

The current config also uses:
- `predict_bias: true`

That means the training target semantics are bias-style rather than direct corrected-value prediction, even though the target columns themselves are the corrected variables.

## Features Excluded By The Current DNN Config

From [`src/configs/config_dnn.yaml`](../src/configs/config_dnn.yaml), the current excluded columns are:

- `time`
- `latitude`
- `longitude`
- `timestamp`
- `corrected_VHM0`
- `corrected_VTM02`
- `WDIR`
- `VMDR`
- `wind_dir_sin`
- `wind_dir_cos`
- `sin_delta`
- `cos_delta`
- `alongwind`
- `crosswind`
- `storm_regime`
- `storm_regime_sig`

## Features Used By The Current DNN Config

Given the current `excluded_columns`, `target_columns`, and `add_sea_mask_channel: true`, the active learned inputs are effectively:

- `VHM0`
- `WSPD`
- `VTM02`
- `U10`
- `V10`
- `wave_dir_sin`
- `wave_dir_cos`
- `sin_hour`
- `cos_hour`
- `sin_doy`
- `cos_doy`
- `sin_month`
- `cos_month`
- `lat_norm`
- `lon_norm`
- sea-mask channel

That matches:
- `in_channels: 16`

## Notes On Feature Status

- A feature being “produced” does not mean it is currently used by the model.
- Some features are present for experimentation, diagnostics, or legacy model variants.
- The current DNN config is the source of truth for what is actually used in the active training path.

## Validation Tips

To inspect one processed `.pt` file:

```bash
poetry run python - <<'PY'
import torch
data = torch.load("/path/to/WAVEAN20200101.pt", map_location="cpu", weights_only=False)
print(data["feature_cols"])
print(len(data["feature_cols"]))
print(data["tensor"].shape)
PY
```

To verify the current config’s excluded features:

```bash
rg -n "excluded_columns|target_columns|in_channels" src/configs/config_dnn.yaml
```
