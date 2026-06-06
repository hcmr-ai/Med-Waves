#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_residual_penalty_sweep.sh [options] <lambda1> [<lambda2> ...]

Options:
  --config PATH        Base YAML config. Default: src/configs/config_dnn.yaml
  --output-dir PATH    Generated configs directory.
                       Default: tmp/generated_sweeps/residual_penalty
  --trainer PATH       Trainer entrypoint.
                       Default: src/pipelines/training/dnn_trainer.py
  --resume-base        Keep base resume_from_checkpoint for all runs.
  --overwrite          Allow overwriting generated configs and reusing existing
                       checkpoint/log directories.
  -h, --help           Show this help.

Examples:
  scripts/run_residual_penalty_sweep.sh 0 0.01 0.05 0.1
  scripts/run_residual_penalty_sweep.sh --resume-base --config src/configs/config_dnn.yaml 0.05 0.1
EOF
}

CONFIG_PATH="src/configs/config_dnn.yaml"
OUTPUT_DIR="tmp/generated_sweeps/residual_penalty"
TRAINER_PATH="src/pipelines/training/dnn_trainer.py"
RESUME_BASE=0
OVERWRITE=0
LAMBDA_VALUES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Missing value for --config" >&2; exit 1; }
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --output-dir" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --trainer)
      [[ $# -ge 2 ]] || { echo "Missing value for --trainer" >&2; exit 1; }
      TRAINER_PATH="$2"
      shift 2
      ;;
    --resume-base)
      RESUME_BASE=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      LAMBDA_VALUES+=("$1")
      shift
      ;;
  esac
done

[[ -f "$CONFIG_PATH" ]] || { echo "Base config not found: $CONFIG_PATH" >&2; exit 1; }
[[ -f "$TRAINER_PATH" ]] || { echo "Trainer entrypoint not found: $TRAINER_PATH" >&2; exit 1; }
[[ ${#LAMBDA_VALUES[@]} -gt 0 ]] || { echo "Provide at least one residual_penalty_lambda value." >&2; usage >&2; exit 1; }
command -v poetry >/dev/null 2>&1 || { echo "'poetry' is required but not found in PATH." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "'python3' is required but not found in PATH." >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH_ABS="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
TRAINER_PATH_ABS="$(cd "$(dirname "$TRAINER_PATH")" && pwd)/$(basename "$TRAINER_PATH")"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

for lambda_value in "${LAMBDA_VALUES[@]}"; do
  echo "Preparing residual_penalty_lambda=${lambda_value}"

  generated_config="$OUTPUT_DIR_ABS/config_residual_penalty_${lambda_value//./p}.yaml"

  python3 - "$CONFIG_PATH_ABS" "$generated_config" "$lambda_value" "$RESUME_BASE" "$OVERWRITE" <<'PY'
import re
import sys
from pathlib import Path

import yaml

base_config_path = Path(sys.argv[1])
generated_config_path = Path(sys.argv[2])
lambda_raw = sys.argv[3]
resume_base = bool(int(sys.argv[4]))
overwrite = bool(int(sys.argv[5]))


def strip_penalty_suffix(value: str) -> str:
    return re.sub(r"_residual_absolute_penalty_[A-Za-z0-9.\-]+$", "", value)


def ensure_writable_target(path_str: str, overwrite_allowed: bool, label: str) -> None:
    path = Path(path_str)
    if not path.exists():
        return
    if overwrite_allowed:
        return
    if path.is_file():
        raise SystemExit(f"{label} already exists: {path}")
    try:
        next(path.iterdir())
    except StopIteration:
        return
    except OSError:
        return
    raise SystemExit(f"{label} already exists and is not empty: {path}")


try:
    lambda_float = float(lambda_raw)
except ValueError as exc:
    raise SystemExit(f"Invalid lambda value '{lambda_raw}': {exc}") from exc

with base_config_path.open("r", encoding="utf-8") as fh:
    config = yaml.safe_load(fh)

model_cfg = config.setdefault("model", {})
checkpoint_cfg = config.setdefault("checkpoint", {})
logging_cfg = config.setdefault("logging", {})

lambda_tag = format(lambda_float, "g")
suffix = f"residual_absolute_penalty_{lambda_tag}"

base_experiment_name = strip_penalty_suffix(logging_cfg.get("experiment_name", "dnn_wave_correction"))
experiment_name = f"{base_experiment_name}_{suffix}"

base_checkpoint_dir = strip_penalty_suffix(checkpoint_cfg.get("checkpoint_dir", "checkpoints"))
base_log_dir = strip_penalty_suffix(logging_cfg.get("log_dir", "logs"))

checkpoint_dir = f"{base_checkpoint_dir}_{suffix}"
log_dir = f"{base_log_dir}_{suffix}"

model_cfg["residual_penalty_lambda"] = lambda_float
checkpoint_cfg["resume_from_checkpoint"] = checkpoint_cfg.get("resume_from_checkpoint") if resume_base else None
checkpoint_cfg["checkpoint_dir"] = checkpoint_dir
logging_cfg["log_dir"] = log_dir
logging_cfg["experiment_name"] = experiment_name

if generated_config_path.exists() and not overwrite:
    raise SystemExit(f"Generated config already exists: {generated_config_path}")

ensure_writable_target(checkpoint_dir, overwrite, "Checkpoint directory")
ensure_writable_target(log_dir, overwrite, "Log directory")

generated_config_path.parent.mkdir(parents=True, exist_ok=True)
with generated_config_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)

print(experiment_name)
print(checkpoint_dir)
print(log_dir)
PY

  echo "Launching training with config: $generated_config"
  poetry run python "$TRAINER_PATH_ABS" --config "$generated_config"
done
