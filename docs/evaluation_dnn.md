# DNN Evaluation

## Main Entrypoints

Recommended DNN evaluation path:
- [`src/pipelines/evaluation/evaluate_bunet.py`](../src/pipelines/evaluation/evaluate_bunet.py)

Existing orchestration wrapper:
- [`src/pipelines/evaluation/full_evaluation.sh`](../src/pipelines/evaluation/full_evaluation.sh)

Additional paths exist in the repo, including low-memory and older evaluation variants. Use the main path above unless there is a specific reason not to.

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
