#!/usr/bin/env bash
set -euo pipefail

cd /mnt/Med-WAV

if ! command -v poetry >/dev/null 2>&1; then
  echo "'poetry' is not available in PATH." >&2
  exit 1
fi
#checkpoints_subsampled_step_5_100_val_22_test_23_moe_transunet_pos_enc_sea_mask_18-21_bin_balanced_smooth_l1_32_weight_decay_1e-3_cosine_annealing_lr_bias_correction_vhm0_vtm02_atlantic_16_128_1layer_dropout_0.2_beta_0.05/
CONFIG_PATH="src/configs/config_dnn.yaml"
SUFFIX=""
REGION="mediterranean"
OUTPUT_BASE="/mnt/Med-WAV/evaluation_results/${REGION}${SUFFIX}"
ROOT_DIR="/mnt/blobstorage/checkpoints/"
EXP_NAME="checkpoints_subsampled_step_5_100_val_22_test_23_moe_transunet_pos_enc_sea_mask_18-21_bin_masked_smooth_l1_32_weight_decay_1e-3_cosine_annealing_lr_bias_correction_vhm0_mediterranean_16_128_1layer_dropout_0.2_features_gate_residual_penalty_0.01"
CHECKPOINT_STEM="epoch=08-val_loss=0.05.ckpt"
CKPT_DIR="${ROOT_DIR}/${EXP_NAME}/${CHECKPOINT_STEM}"
TIMESTAMPS_CSV="/mnt/blobstorage/diagnostics/pt_timestamp_map.csv"
DENOISE_ABS_THRESHOLD=null
export CONFIG_PATH OUTPUT_BASE CKPT_DIR

# Region-specific sampled points file (atlantic/mediterranean/aegean)
SAMPLED_POINTS_CSV="/mnt/blobstorage/diagnostics/sampled_grid_points_300/sampled_grid_points_mediterranean.csv"
if [[ ! -f "$SAMPLED_POINTS_CSV" ]]; then
  echo "Sampled points file not found: $SAMPLED_POINTS_CSV" >&2
  exit 1
fi

EVAL_CMD=(
  poetry run python src/pipelines/evaluation/evaluate_bunet.py
  --config "$CONFIG_PATH"
  --checkpoint "$CKPT_DIR"
  --output-dir "$OUTPUT_BASE"
  --save-predictions
  --region-filter "$REGION"
#   --apply-geographic-filtering
  --sampled-points-csv "$SAMPLED_POINTS_CSV"
  --timestamps-csv "$TIMESTAMPS_CSV"
  # --denoise-abs-threshold "$DENOISE_ABS_THRESHOLD"
)
"${EVAL_CMD[@]}"

RUN_DIR="$(poetry run python - <<'PY'
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["CONFIG_PATH"])
output_base = Path(os.environ["OUTPUT_BASE"])
checkpoint = Path(os.environ["CKPT_DIR"])

cfg = yaml.safe_load(config_path.read_text())
exp_name = cfg["logging"]["experiment_name"]
test_year = cfg["data"]["test_year"]
if isinstance(test_year, list):
    test_year = test_year[0]

print(output_base / exp_name / str(test_year) / checkpoint.stem)
PY
)"

if [[ ! -f "$RUN_DIR/plot_samples.npz" ]]; then
  echo "Expected file not found: $RUN_DIR/plot_samples.npz" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/grid_point_timeseries.csv" ]]; then
  echo "Expected file not found: $RUN_DIR/grid_point_timeseries.csv" >&2
  exit 1
fi

cd /mnt/Med-WAV/src/evaluation
PLOT_MAPS_CMD=(
  poetry run python plot_maps.py
  --input-npz "$RUN_DIR/plot_samples.npz"
  --output-dir "$RUN_DIR/plot_maps"
  --metric impr
  --region "$REGION"
)
"${PLOT_MAPS_CMD[@]}"

GLOBAL_EVAL_CMD=(
  poetry run python global_evaluation.py
  --input-npz "$RUN_DIR/plot_samples.npz"
  --output-dir "$RUN_DIR/global_evaluation"
  --threshold-cm 5.0
)
"${GLOBAL_EVAL_CMD[@]}"

PLOT_POINTS_CMD=(
  poetry run python plot_points.py
  --csv "$RUN_DIR/grid_point_timeseries.csv"
  --region "$REGION"
  --output-dir "$RUN_DIR/plots_300/grid_point_timeseries"
)
"${PLOT_POINTS_CMD[@]}"

NATIVE_SUMMARY_CMD=(
  poetry run python native_plots_and_summary.py
  --csv "$RUN_DIR/grid_point_timeseries.csv"
  --region "$REGION"
  --n-points 300
  --output-dir "$RUN_DIR/plots_300_native/grid_point_timeseries"
)
"${NATIVE_SUMMARY_CMD[@]}"