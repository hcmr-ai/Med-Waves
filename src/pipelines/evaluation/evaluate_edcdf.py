"""
Evaluate EDCDF corrector predictions using the same plot suite as evaluate_bunet.

Loads prediction parquets produced by train_edcdf_regional.py, computes sea-bin
metrics, error distributions, and generates all standard evaluation plots.

Usage:
    poetry run python -m src.pipelines.evaluation.evaluate_edcdf \
        --predictions-dir data/edcdf_regional/mediterranean/edcdf_mediterranean_train_2018-2019-2020-2021_test_2022-2023/predictions \
        --variable VHM0

    # Or point at an S3 output:
    poetry run python -m src.pipelines.evaluation.evaluate_edcdf \
        --predictions-dir data/edcdf_regional/atlantic/edcdf_atlantic_train_2018-2019-2020-2021_test_2022-2023/predictions \
        --variable VHM0 --variable VTM02
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from tqdm import tqdm

from src.evaluation.evaluation_plots import (
    plot_error_boxplots,
    plot_error_cdfs,
    plot_error_distribution_histograms,
    plot_error_violins,
    plot_model_better_percentage,
    plot_sea_bin_metrics,
    plot_vhm0_distributions,
)

SEA_BINS_VHM0 = [
    {"name": "calm", "min": 0.0, "max": 1.0, "label": "0.0-1.0m"},
    {"name": "light", "min": 1.0, "max": 2.0, "label": "1.0-2.0m"},
    {"name": "moderate", "min": 2.0, "max": 3.0, "label": "2.0-3.0m"},
    {"name": "rough", "min": 3.0, "max": 4.0, "label": "3.0-4.0m"},
    {"name": "very_rough", "min": 4.0, "max": 5.0, "label": "4.0-5.0m"},
    {"name": "extreme_5_6", "min": 5.0, "max": 6.0, "label": "5.0-6.0m"},
    {"name": "extreme_6_7", "min": 6.0, "max": 7.0, "label": "6.0-7.0m"},
    {"name": "extreme_7_8", "min": 7.0, "max": 8.0, "label": "7.0-8.0m"},
    {"name": "extreme_8_9", "min": 8.0, "max": 9.0, "label": "8.0-9.0m"},
    {"name": "extreme_9_10", "min": 9.0, "max": 10.0, "label": "9.0-10.0m"},
    {"name": "extreme_10_11", "min": 10.0, "max": 11.0, "label": "10.0-11.0m"},
    {"name": "extreme_11_12", "min": 11.0, "max": 12.0, "label": "11.0-12.0m"},
    {"name": "extreme_12_13", "min": 12.0, "max": 13.0, "label": "12.0-13.0m"},
    {"name": "extreme_13_14", "min": 13.0, "max": 14.0, "label": "13.0-14.0m"},
    {"name": "extreme_14_15", "min": 14.0, "max": 15.0, "label": "14.0-15.0m"},
]

SEA_BINS_VTM02 = [
    {"name": f"bin_{i}_{i+1}", "min": float(i), "max": float(i + 1), "label": f"{i}-{i+1}s"}
    for i in range(0, 20)
]

VAR_CONFIG = {
    "VHM0": {
        "target_column": "corrected_VHM0",
        "var_name": "VHM0",
        "var_name_full": "Significant Wave Height",
        "unit": "m",
        "sea_bins": SEA_BINS_VHM0,
    },
    "VTM02": {
        "target_column": "corrected_VTM02",
        "var_name": "VTM02",
        "var_name_full": "Wave Period",
        "unit": "s",
        "sea_bins": SEA_BINS_VTM02,
    },
}


def load_predictions(predictions_dir: Path) -> pl.DataFrame:
    """Load all WAVEAN*.parquet prediction files from a directory."""
    files = sorted(predictions_dir.glob("WAVEAN2022*.parquet"))
    if not files:
        raise FileNotFoundError(f"No WAVEAN*.parquet files in {predictions_dir}")

    print(f"Loading {len(files)} prediction files from {predictions_dir}")
    dfs = []
    for f in tqdm(files, desc="Loading predictions"):
        try:
            dfs.append(pl.read_parquet(f))
        except Exception as e:
            print(f"  Warning: skipping {f.name}: {e}")
    df = pl.concat(dfs)
    print(f"  Total rows: {len(df):,}")
    return df


def build_sea_bin_data(
    df: pl.DataFrame,
    var: str,
    sea_bins: List[Dict],
    corrected_suffix: str = "corrected_",
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Build sea_bin_metrics and sea_bin_error_samples from prediction DataFrame.

    Returns:
        (sea_bin_metrics, sea_bin_error_samples) matching the format
        expected by the evaluation_plots functions.
    """
    pred_col = f"predicted_{var}"
    target_col = f"{corrected_suffix}{var}"
    uncorrected_col = var

    required = [pred_col, target_col, uncorrected_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in predictions DataFrame")

    pred = df[pred_col].to_numpy().astype(float)
    target = df[target_col].to_numpy().astype(float)
    uncorrected = df[uncorrected_col].to_numpy().astype(float)

    finite = np.isfinite(pred) & np.isfinite(target) & np.isfinite(uncorrected)
    pred, target, uncorrected = pred[finite], target[finite], uncorrected[finite]

    model_errors = pred - target
    baseline_errors = uncorrected - target

    sea_bin_metrics = {}
    sea_bin_error_samples = {}

    for b in sea_bins:
        bin_name = b["name"]
        mask = (uncorrected >= b["min"]) & (uncorrected < b["max"])
        count = int(mask.sum())

        sea_bin_error_samples[bin_name] = {
            "model_errors": model_errors[mask].tolist() if count > 0 else [],
            "baseline_errors": baseline_errors[mask].tolist() if count > 0 else [],
        }

        if count == 0:
            sea_bin_metrics[bin_name] = {
                "count": 0,
                "label": b["label"],
            }
            continue

        m_err = model_errors[mask]
        b_err = baseline_errors[mask]

        rmse = float(np.sqrt(np.mean(m_err**2)))
        mae = float(np.mean(np.abs(m_err)))
        bias = float(np.mean(m_err))
        baseline_rmse = float(np.sqrt(np.mean(b_err**2)))
        baseline_mae = float(np.mean(np.abs(b_err)))
        baseline_bias = float(np.mean(b_err))

        rmse_imp = ((baseline_rmse - rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0.0
        mae_imp = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0

        count_model_better = int(np.sum(np.abs(m_err) < np.abs(b_err)))
        count_model_worse = int(np.sum(np.abs(m_err) > np.abs(b_err)))
        pct_model_better = (count_model_better / count * 100) if count > 0 else 0.0
        pct_model_worse = (count_model_worse / count * 100) if count > 0 else 0.0

        sea_bin_metrics[bin_name] = {
            "count": count,
            "label": b["label"],
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "baseline_rmse": baseline_rmse,
            "baseline_mae": baseline_mae,
            "baseline_bias": baseline_bias,
            "rmse_improvement_pct": rmse_imp,
            "mae_improvement_pct": mae_imp,
            "count_model_better": count_model_better,
            "count_model_worse": count_model_worse,
            "pct_model_better": pct_model_better,
            "pct_model_worse": pct_model_worse,
        }

    return sea_bin_metrics, sea_bin_error_samples


def build_plot_samples(
    df: pl.DataFrame,
    var: str,
    corrected_suffix: str = "corrected_",
    max_samples: int = 5_000_000,
) -> Dict[str, list]:
    """Build plot_samples dict for distribution plots."""
    pred_col = f"predicted_{var}"
    target_col = f"{corrected_suffix}{var}"

    nan_cols = [pred_col, target_col, var]
    df_clean = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in nan_cols
    ]).drop_nulls(subset=nan_cols)

    if len(df_clean) > max_samples:
        df_clean = df_clean.sample(n=max_samples, seed=42)

    return {
        "y_true": df_clean[target_col].to_numpy().tolist(),
        "y_pred": df_clean[pred_col].to_numpy().tolist(),
        "y_uncorrected": df_clean[var].to_numpy().tolist(),
        "vhm0": df_clean[var].to_numpy().tolist(),
    }


def compute_overall_metrics(
    df: pl.DataFrame,
    var: str,
    corrected_suffix: str = "corrected_",
) -> Dict[str, float]:
    """Compute overall metrics for a variable."""
    pred = df[f"predicted_{var}"].to_numpy().astype(float)
    target = df[f"{corrected_suffix}{var}"].to_numpy().astype(float)
    uncorrected = df[var].to_numpy().astype(float)

    finite = np.isfinite(pred) & np.isfinite(target) & np.isfinite(uncorrected)
    pred, target, uncorrected = pred[finite], target[finite], uncorrected[finite]

    residuals = pred - target
    baseline_residuals = uncorrected - target

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    corr, _ = pearsonr(target, pred)

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    baseline_rmse = float(np.sqrt(np.mean(baseline_residuals**2)))
    baseline_mae = float(np.mean(np.abs(baseline_residuals)))
    baseline_bias = float(np.mean(baseline_residuals))

    rmse_imp = ((baseline_rmse - rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0.0
    mae_imp = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "correlation": float(corr),
        "r2": r2,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "baseline_bias": baseline_bias,
        "rmse_improvement_pct": rmse_imp,
        "mae_improvement_pct": mae_imp,
        "n_samples": len(pred),
    }


def print_summary(overall_metrics: Dict[str, Dict], sea_bin_metrics_all: Dict[str, Dict]):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for var, metrics in overall_metrics.items():
        print(f"\n  {var}:")
        print(f"    Samples:          {metrics['n_samples']:,}")
        print(f"    RMSE:             {metrics['rmse']:.4f}  (baseline: {metrics['baseline_rmse']:.4f}, improvement: {metrics['rmse_improvement_pct']:+.1f}%)")
        print(f"    MAE:              {metrics['mae']:.4f}  (baseline: {metrics['baseline_mae']:.4f}, improvement: {metrics['mae_improvement_pct']:+.1f}%)")
        print(f"    Bias:             {metrics['bias']:.4f}  (baseline: {metrics['baseline_bias']:.4f})")
        print(f"    Correlation:      {metrics['correlation']:.4f}")
        print(f"    R2:               {metrics['r2']:.4f}")

    for var, sbm in sea_bin_metrics_all.items():
        print(f"\n  {var} — Sea-bin breakdown:")
        print(f"    {'Bin':<14s} {'Count':>10s} {'RMSE':>8s} {'Base RMSE':>10s} {'Impr%':>8s} {'Better%':>8s}")
        for b_name, m in sbm.items():
            if m.get("count", 0) == 0:
                continue
            print(
                f"    {m.get('label', b_name):<14s} "
                f"{m['count']:>10,d} "
                f"{m.get('rmse', 0):>8.4f} "
                f"{m.get('baseline_rmse', 0):>10.4f} "
                f"{m.get('rmse_improvement_pct', 0):>+7.1f}% "
                f"{m.get('pct_model_better', 0):>7.1f}%"
            )

    print("=" * 80)


def plot_bias_per_bin(
    sea_bin_metrics: Dict[str, Dict],
    sea_bins: List[Dict],
    target_column: str,
    unit: str,
    output_dir: Path,
):
    """Plot bias (mean error) per sea bin for both model and baseline."""
    sorted_bins = sorted(sea_bins, key=lambda x: x["min"])

    bin_labels = []
    model_bias = []
    baseline_bias = []

    for b in sorted_bins:
        m = sea_bin_metrics.get(b["name"], {})
        if m.get("count", 0) == 0:
            continue
        bin_labels.append(m.get("label", b["label"]))
        model_bias.append(m.get("bias", 0))
        baseline_bias.append(m.get("baseline_bias", 0))

    if not bin_labels:
        return

    x = np.arange(len(bin_labels))
    width = 0.35

    is_period = target_column == "corrected_VTM02"
    title_kind = "Period Range" if is_period else "Sea State"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, baseline_bias, width, label="Baseline (uncorrected)", color="darkblue", alpha=0.6)
    ax.bar(x + width / 2, model_bias, width, label="EDCDF", color="skyblue", alpha=0.8)

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_title(f"Mean Bias by {title_kind} (Prediction − Reference)", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"{title_kind} Bin", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Bias ({unit})", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    for i, (b_val, m_val) in enumerate(zip(baseline_bias, model_bias, strict=False)):
        ax.text(i - width / 2, b_val, f"{b_val:+.3f}", ha="center",
                va="bottom" if b_val >= 0 else "top", fontsize=8)
        ax.text(i + width / 2, m_val, f"{m_val:+.3f}", ha="center",
                va="bottom" if m_val >= 0 else "top", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "bias_per_bin.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved bias per bin plot to {output_dir / 'bias_per_bin.png'}")


def evaluate_variable(
    df: pl.DataFrame,
    var: str,
    output_dir: Path,
    corrected_suffix: str = "corrected_",
):
    """Run full evaluation for one variable."""
    cfg = VAR_CONFIG[var]
    sea_bins = cfg["sea_bins"]
    var_output = output_dir / var
    var_output.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Evaluating {var} ---")

    overall = compute_overall_metrics(df, var, corrected_suffix)
    sea_bin_metrics, sea_bin_error_samples = build_sea_bin_data(
        df, var, sea_bins, corrected_suffix
    )
    plot_samples = build_plot_samples(df, var, corrected_suffix)

    # Sea-bin bar charts (RMSE/MAE/improvement/distribution)
    print("Plotting sea-bin metrics...")
    plot_sea_bin_metrics(
        sea_bin_metrics=sea_bin_metrics,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # Bias per bin
    print("Plotting bias per bin...")
    plot_bias_per_bin(
        sea_bin_metrics=sea_bin_metrics,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # Model better/worse percentage
    print("Plotting model-better percentage...")
    plot_model_better_percentage(
        sea_bin_metrics=sea_bin_metrics,
        sea_bins=sea_bins,
        var_name_full=cfg["var_name_full"],
        output_dir=var_output,
    )

    # Error histograms per bin
    print("Plotting error distribution histograms...")
    plot_error_distribution_histograms(
        sea_bin_error_samples=sea_bin_error_samples,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # Error boxplots
    print("Plotting error boxplots...")
    plot_error_boxplots(
        sea_bin_error_samples=sea_bin_error_samples,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # Error violins
    print("Plotting error violins...")
    plot_error_violins(
        sea_bin_error_samples=sea_bin_error_samples,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # Error CDFs
    print("Plotting error CDFs...")
    plot_error_cdfs(
        sea_bin_error_samples=sea_bin_error_samples,
        sea_bins=sea_bins,
        target_column=cfg["target_column"],
        unit=cfg["unit"],
        output_dir=var_output,
    )

    # VHM0 distribution KDEs
    print("Plotting VHM0 distributions...")
    plot_vhm0_distributions(
        plot_samples=plot_samples,
        var_name=cfg["var_name"],
        var_name_full=cfg["var_name_full"],
        unit=cfg["unit"],
        corrected_label="Corrected (Reference)",
        model_label="EDCDF Prediction",
        uncorrected_label="Uncorrected",
        output_dir=var_output,
    )

    # Save metrics JSON
    metrics_out = {
        "overall": overall,
        "sea_bins": sea_bin_metrics,
    }
    with open(var_output / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"Saved metrics to {var_output / 'metrics.json'}")

    return overall, sea_bin_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate EDCDF predictions")
    parser.add_argument(
        "--predictions-dir", type=str, required=True,
        help="Directory containing WAVEAN*.parquet prediction files from train_edcdf_regional.py",
    )
    parser.add_argument(
        "--variable", dest="variables", action="append", default=None,
        help="Variable(s) to evaluate (default: all available). Can be repeated: --variable VHM0 --variable VTM02",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots and metrics (default: <predictions-dir>/../evaluation)",
    )
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = predictions_dir.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("EDCDF Evaluation")
    print("=" * 80)
    print(f"  Predictions: {predictions_dir}")
    print(f"  Output:      {output_dir}")

    df = load_predictions(predictions_dir)

    # Auto-detect available variables if not specified
    if args.variables is None:
        variables = []
        for var in ["VHM0", "VTM02"]:
            if f"predicted_{var}" in df.columns:
                variables.append(var)
        if not variables:
            raise ValueError("No predicted_* columns found in prediction files")
    else:
        variables = args.variables

    print(f"  Variables:   {variables}")
    print()

    all_overall = {}
    all_sea_bin = {}

    for var in variables:
        if var not in VAR_CONFIG:
            print(f"  Warning: no config for variable '{var}', skipping")
            continue
        overall, sea_bin_metrics = evaluate_variable(df, var, output_dir)
        all_overall[var] = overall
        all_sea_bin[var] = sea_bin_metrics

    print_summary(all_overall, all_sea_bin)

    # Save combined metrics
    combined = {"variables": {}}
    for var in variables:
        if var in all_overall:
            combined["variables"][var] = {
                "overall": all_overall[var],
                "sea_bins": all_sea_bin[var],
            }
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nAll plots and metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
