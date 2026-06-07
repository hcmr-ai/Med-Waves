# DNN Evaluation

## Main Entrypoints

Recommended DNN evaluation path:
- [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py)

Memory-efficient variant:
- [`src/pipelines/evaluation/evaluate_bunet_lowmem.py`](../src/pipelines/evaluation/evaluate_bunet_lowmem.py)
  - use this when full-grid evaluation would otherwise consume too much RAM
  - intended as a drop-in replacement for `evaluate_bunet.py`

Existing orchestration wrapper:
- [`src/pipelines/evaluation/full_evaluation.sh`](../src/pipelines/evaluation/full_evaluation.sh)

Additional older evaluation variants also exist in the repo. Use `evaluate_bunet.py` or `evaluate_bunet_lowmem.py` unless there is a specific reason not to.

## Standard Command

```bash
poetry run python src/pipelines/evaluation/evaluate_bunet.py \
  --config src/configs/config_dnn.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-dir ./evaluation_results
```

Useful options:
- `--eval-task`
- `--region-filter`
- `--sampled-points-csv`
- `--timestamps-csv`
- `--save-predictions`
- `--denoise-abs-threshold`

## Low-Memory Command

```bash
poetry run python src/pipelines/evaluation/evaluate_bunet_lowmem.py \
  --config src/configs/config_dnn.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --output-dir ./evaluation_results \
  --max-plot-samples 10000000
```

Why it exists:
- avoids the worst RAM spikes during full-grid evaluation
- keeps spatial-error accumulation in running sums instead of unbounded per-batch lists
- caps stored plot samples through random subsampling
- is meant to produce the same metrics and outputs as `evaluate_bunet.py`

## Full Evaluation Script

[`full_evaluation.sh`](../src/pipelines/evaluation/full_evaluation.sh) runs:
1. `evaluate_bunet.py`
2. `plot_maps.py`
3. `global_evaluation.py`
4. `plot_points.py`
5. `native_plots_and_summary.py`

This script is tightly coupled to:
- `/mnt/Med-WAV`
- `/mnt/blobstorage`
- a region-specific sampled-points CSV
- a specific experiment/checkpoint naming convention

Treat it as an operator script, not a generic evaluation interface.

## Typical Outputs

Depending on flags and downstream scripts:
- `plot_samples.npz`
- `grid_point_timeseries.csv`
- map plots
- global summary outputs
- pointwise timeseries plots

## Before Running

- Verify the checkpoint path exists
- Verify `sampled_points_csv` and `timestamps_csv` exist if using pointwise exports
- Verify the output directory is intentional and writable
- Confirm the evaluation region and task align with the config targets

## Related Docs

- [`training-dnn.md`](training-dnn.md)
- [`config-reference.md`](config-reference.md)
- [`baseline-models.md`](baseline-models.md)
