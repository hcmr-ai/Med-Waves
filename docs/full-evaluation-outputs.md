# `full_evaluation.sh` Output Guide

This document explains what each step in [`src/pipelines/evaluation/full_evaluation.sh`](../src/pipelines/evaluation/full_evaluation.sh) writes to disk, and how artifacts flow between steps.

## What This Script Orchestrates

`full_evaluation.sh` runs, in order:
1. `evaluate_bunet.py`
2. `plot_maps.py`
3. `global_evaluation.py`
4. `plot_points.py`
5. `native_plots_and_summary.py`

The script assumes fixed infrastructure paths (`/mnt/Med-WAV`, `/mnt/blobstorage`) and region-specific sampled points.

## Output Root And Run Directory

Two paths matter:

- **Base output root** (`OUTPUT_BASE` in the shell script):
  - `/mnt/Med-WAV/evaluation_results/${REGION}${SUFFIX}`

- **Per-run directory** (`RUN_DIR`, computed after step 1):
  - `OUTPUT_BASE / <config.logging.experiment_name> / <config.data.test_year[0]> / <checkpoint_stem>`
  - For a single checkpoint file `epoch=04-val_loss=0.06.ckpt`, `checkpoint_stem` is `epoch=04-val_loss=0.06`

All downstream steps write under `RUN_DIR`, except one side-effect file noted below.

---

## Step 0: Preflight Checks (No New Artifacts)

Before evaluation starts, the script validates:
- `poetry` is available
- sampled points CSV exists at `SAMPLED_POINTS_CSV`

If either fails, the script exits early and writes no run artifacts.

---

## Step 1: `evaluate_bunet.py` (Primary Evaluation Artifacts)

Command shape:
- `poetry run python src/pipelines/evaluation/evaluate_bunet.py ... --output-dir "$OUTPUT_BASE" --save-predictions ...`

The evaluator internally appends experiment/test-year/checkpoint, so files land in `RUN_DIR`.

### Files `full_evaluation.sh` depends on

These are required by later steps and explicitly checked by the shell script:
- `RUN_DIR/plot_samples.npz`
- `RUN_DIR/grid_point_timeseries.csv`

### What those two files contain

- `plot_samples.npz`
  - arrays: `y_true`, `y_pred`, `vhm0`, `lat`, `lon`
  - consumed by `plot_maps.py` and `global_evaluation.py`

- `grid_point_timeseries.csv`
  - per sampled grid point and timestamp:
  - columns: `timestamp`, `batch_idx`, `sample_in_batch`, `region`, `requested_lat`, `requested_lon`, `grid_lat`, `grid_lon`, `reference`, `uncorrected`, `corrected`
  - consumed by `plot_points.py` and `native_plots_and_summary.py`

### Additional artifacts created by evaluator

Common outputs in `RUN_DIR` include:
- `metrics.json` (overall + sea-bin + category + per-point-improvement summaries)
- `heatmap_rmse_improvement_all_points.png`
- `heatmap_rmse_improvement_all_points.pdf`
- `heatmap_rmse_improvement_all_points_values.csv`
- `heatmap_rmse_improvement_all_points_counts.csv`
- RMSE/MAE map plots from `evaluation_plots`:
  - `rmse_model.png`, `mae_model.png`
  - `rmse_reference.png`, `mae_reference.png` (when baseline/reference is available)
  - `rmse_improvement_symmetric.png`, `rmse_improvement_binary.png`, `mae_improvement_binary.png`
- Sea-bin summary plots:
  - `sea_bin_performance.png`
  - `model_better_percentage.png`
  - `model_worse_percentage.png`
- Scatter diagnostic:
  - `scatter_ref_minus_uncorrected_vs_ref_minus_corrected.png`

Depending on config/flags, evaluator may also write optional directories (for example `low_bin_spatial_maps` or `bin_spatial_rmse`).

---

## Step 2: `RUN_DIR` Resolution Block (No New Artifacts)

The inline Python block in `full_evaluation.sh` computes `RUN_DIR` from config + checkpoint stem.

No files are written in this step; it only determines where subsequent commands read/write.

---

## Step 3: Required Hand-off Validation (No New Artifacts)

The shell script hard-checks:
- `RUN_DIR/plot_samples.npz`
- `RUN_DIR/grid_point_timeseries.csv`

If either is missing, execution stops before map/global/point plotting begins.

---

## Step 4: `plot_maps.py` -> `RUN_DIR/plot_maps`

Command:
- `poetry run python plot_maps.py --input-npz "$RUN_DIR/plot_samples.npz" --output-dir "$RUN_DIR/plot_maps" --metric impr --region "$REGION"`

### Main outputs

Inside `RUN_DIR/plot_maps`:
- `statistics.txt` (detailed textual diagnostics)
- map PDFs by subset/mode, pattern:
  - `rmse_<metric>_<subset_tag>_map.pdf`
  - with this script call: `<metric>` is `impr`
- distribution/scatter diagnostics, patterns:
  - `<subset_tag>_distr.pdf`
  - `<subset_tag>_scatter_per_bin.png`
  - denoised and filtered-out variants like:
    - `<subset_tag>_denoised_0.05_distr.pdf`
    - `<subset_tag>_denoised_0.05_scatter_per_bin.png`
    - `<subset_tag>_filtered_out_0.05_distr.pdf`
    - `<subset_tag>_filtered_out_0.05_scatter_per_bin.png`

### Side-effect file outside `RUN_DIR`

`plot_maps.py` also saves:
- `sea_grid_cells_5798.pdf`

This is written to the **current working directory** of that step (`/mnt/Med-WAV/src/evaluation` in this script), not to `RUN_DIR/plot_maps`.

---

## Step 5: `global_evaluation.py` -> `RUN_DIR/global_evaluation`

Command:
- `poetry run python global_evaluation.py --input-npz "$RUN_DIR/plot_samples.npz" --output-dir "$RUN_DIR/global_evaluation" --threshold-cm 5.0`

### Top-level outputs in `RUN_DIR/global_evaluation`

- `metrics_summary.csv`
- `failure_summaries.json`
- `statistics_report.txt`
- `metrics_glossary.txt`

### Regime subdirectories

The script evaluates regimes `overall`, `noise`, and `clean`, creating:
- `RUN_DIR/global_evaluation/overall/`
- `RUN_DIR/global_evaluation/noise/`
- `RUN_DIR/global_evaluation/clean/`

Each regime directory typically includes:
- `metrics_by_wave_bin.csv`
- `metrics_by_wave_bin.pdf`
- `distributions.pdf`
- `scatter_per_bin.png`
- `residual_hist.pdf`
- `d_error_hist.pdf`

### Failure-analysis subdirectories (noise/clean)

For regimes with failure analysis:
- `RUN_DIR/global_evaluation/noise_failure_analysis/`
- `RUN_DIR/global_evaluation/clean_failure_analysis/`

Typical files there:
- `summary.json`
- `failure_samples.csv` (if failures exist)
- `failure_locations.csv` (if failures exist)
- `failure_rate_by_location.csv` (if failures exist)
- `failure_rate_map.pdf` (if failures exist)
- `failure_count_map.pdf` (if failures exist)
- `failure_severity.pdf` (if failures exist)
- `fail_vs_ok.pdf` (if failures exist)

---

## Step 6: `plot_points.py` -> `RUN_DIR/plots_300/grid_point_timeseries`

Command:
- `poetry run python plot_points.py --csv "$RUN_DIR/grid_point_timeseries.csv" --region "$REGION" --output-dir "$RUN_DIR/plots_300/grid_point_timeseries"`

### Run-level outputs

In `RUN_DIR/plots_300/grid_point_timeseries`:
- `map_overview.png`
- `heatmap_rmse_improvement_cyprus.png`
- `heatmap_rmse_improvement_cyprus.pdf`
- `degraded_scatter.png`
- `improved_scatter.png`
- `degraded_pdf.png`
- `improved_pdf.png`

### Per-point folders

For each selected point: one folder like `latXXpXXXXX_lonYYpYYYYY/` containing:
- timeseries CSV exports:
  - `timeseries_native.csv`
  - `timeseries_7d.csv`
  - `timeseries_15d.csv`
  - `timeseries_15_days.csv`
  - `timeseries_monthly.csv`
- absolute-difference CSV exports:
  - `timeseries_native_absdiff.csv`
  - `timeseries_7d_absdiff.csv`
  - `timeseries_15d_absdiff.csv`
  - `timeseries_15_days_absdiff.csv`
  - `timeseries_monthly_absdiff.csv`
- plots:
  - `plot_native.png`, `plot_7d.png`, `plot_15d.png`, `plot_monthly.png`
  - `plot_abs_errors_native.png`, `plot_abs_errors_7d.png`, `plot_abs_errors_15d.png`, `plot_abs_errors_monthly.png`
  - `map_location.png`

---

## Step 7: `native_plots_and_summary.py` -> `RUN_DIR/plots_300_native/grid_point_timeseries`

Command:
- `poetry run python native_plots_and_summary.py --csv "$RUN_DIR/grid_point_timeseries.csv" --region "$REGION" --n-points 300 --output-dir "$RUN_DIR/plots_300_native/grid_point_timeseries"`

This creates a lighter per-point diagnostic tree:

- per point folder:
  - `plot_abs_errors_native.png`
  - `plot_abs_errors_native.png.pdf`
  - `plot_abs_errors_native_trend.png`
  - `plot_abs_errors_native_trend.png.pdf`

This script is primarily plot-oriented and prints summary stats to stdout; it does not write a consolidated summary CSV/JSON by default.

---

## Practical Artifact Flow

- `evaluate_bunet.py` is the producer for both hand-off files.
- `plot_maps.py` and `global_evaluation.py` consume `plot_samples.npz`.
- `plot_points.py` and `native_plots_and_summary.py` consume `grid_point_timeseries.csv`.
- If either producer artifact is missing, the wrapper exits before downstream plotting steps.
