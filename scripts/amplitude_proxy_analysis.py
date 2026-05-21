"""
Amplitude Proxy Analysis
========================
Tests whether the 90th percentile of uncorrected VHM0 (per year) is a
reliable proxy for bias amplitude — i.e. whether the P90 ranking matches
the bias-amplitude ranking seen in the stationarity analysis.

If P90 ↔ bias_amplitude ranking matches, the amplitude proxy approach is
valid: we can fit a linear calibration on training years and scale model
outputs at inference to fix the 2017 over-correction.

Outputs (saved to --output_dir):
  01_p90_vs_bias_amplitude.png   — scatter + bar chart, ranking comparison
  02_calibration_fit.png         — linear fit on train years, 2017/2023 predicted
  amplitude_proxy_stats.csv      — per-year P90 and bias amplitude values

Usage (on the machine with data access):
  python scripts/amplitude_proxy_analysis.py \
      --data_path /mnt/local_datasets/preprocessed_extended_subsampled_step_5/ \
      --output_dir /mnt/blobstorage/diagnostics/amplitude_proxy \
      --region mediterranean \
      --train_years 2018 2019 2020 2021 \
      --test_years 2017 2022 2023
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_year_month(filename):
    name = Path(filename).stem
    marker = "WAVEAN"
    idx = name.find(marker)
    if idx != -1 and len(name) >= idx + 12:
        return int(name[idx + 6:idx + 10]), int(name[idx + 10:idx + 12])
    return None, None


def get_region_mask(lon_grid, lat_grid, region):
    """Return boolean mask (H, W) — True = keep pixel for this region."""
    if region == "mediterranean":
        return lon_grid >= -5.5
    elif region == "atlantic":
        return lon_grid < -5.5
    else:
        return np.ones_like(lon_grid, dtype=bool)


def load_vhm0_fields(file_path, vhm0_idx, corrected_vhm0_idx):
    """Load one .pt file.

    Returns:
        raw_vhm0:  (24, H, W)  — uncorrected WAM VHM0
        bias:      (24, H, W)  — corrected - raw  (NaN on land)
    """
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    tensor = data["tensor"]  # (24, H, W, C)
    raw_vhm0 = tensor[..., vhm0_idx].numpy().astype(np.float32)
    corrected = tensor[..., corrected_vhm0_idx].numpy().astype(np.float32)
    bias = corrected - raw_vhm0
    # land pixels are 0 in both → bias = 0; raw_vhm0 = 0 on land
    # mask land with NaN using raw == 0 heuristic (same as stationarity script)
    land = raw_vhm0 == 0.0
    raw_vhm0 = np.where(land, np.nan, raw_vhm0)
    bias = np.where(land, np.nan, bias)
    return raw_vhm0, bias


# ── per-year accumulation ─────────────────────────────────────────────────────

def compute_year_stats(file_list, vhm0_idx, corrected_vhm0_idx,
                       region_mask, max_files=None):
    """Return (p90_raw_vhm0, mean_abs_bias) scalars for one year.

    Accumulates across all files without loading everything into RAM:
    - Keeps a reservoir of sea-pixel raw VHM0 values to compute P90
    - Accumulates sum/count for mean abs bias
    """
    if max_files:
        file_list = file_list[:max_files]

    all_raw = []    # list of 1-D arrays (sampled sea pixels)
    bias_sum = 0.0
    bias_count = 0

    for fpath in file_list:
        try:
            raw, bias = load_vhm0_fields(fpath, vhm0_idx, corrected_vhm0_idx)
        except Exception as e:
            print(f"  SKIP {Path(fpath).name}: {e}")
            continue

        # raw: (24, H, W)
        for t in range(raw.shape[0]):
            raw_t = raw[t]          # (H, W)
            bias_t = bias[t]        # (H, W)
            sea = region_mask & ~np.isnan(raw_t)
            if not sea.any():
                continue
            # subsample to keep memory manageable (every 4th pixel)
            raw_sea = raw_t[sea][::4]
            all_raw.append(raw_sea)
            abs_bias = np.abs(bias_t[sea])
            bias_sum += abs_bias.sum()
            bias_count += abs_bias.size

    if not all_raw or bias_count == 0:
        return np.nan, np.nan

    p90 = float(np.percentile(np.concatenate(all_raw), 90))
    mean_abs_bias = float(bias_sum / bias_count)
    return p90, mean_abs_bias


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_ranking_comparison(years, p90_vals, bias_vals, train_years, output_dir):
    """Bar chart + scatter comparing P90 ranking vs bias amplitude ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#2196F3" if y in train_years else "#FF5722" for y in years]
    x = np.arange(len(years))

    # Left: side-by-side bars normalised to [0,1]
    ax = axes[0]
    p90_norm = (p90_vals - np.nanmin(p90_vals)) / (np.nanmax(p90_vals) - np.nanmin(p90_vals) + 1e-9)
    bias_norm = (bias_vals - np.nanmin(bias_vals)) / (np.nanmax(bias_vals) - np.nanmin(bias_vals) + 1e-9)
    w = 0.35
    bars1 = ax.bar(x - w/2, p90_norm, w, label="P90 raw VHM0 (proxy)", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, bias_norm, w, label="Mean |bias| (target)", color=colors, alpha=0.45, edgecolor="k", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Normalised value (0–1)")
    ax.set_title("P90 raw VHM0  vs  Bias amplitude\n(normalised, per year)")
    ax.legend()

    # Ranking labels
    p90_rank = np.argsort(np.argsort(p90_vals)) + 1
    bias_rank = np.argsort(np.argsort(bias_vals)) + 1
    for i, yr in enumerate(years):
        ax.text(x[i] - w/2, p90_norm[i] + 0.02, f"#{p90_rank[i]}", ha="center", fontsize=8, color="navy")
        ax.text(x[i] + w/2, bias_norm[i] + 0.02, f"#{bias_rank[i]}", ha="center", fontsize=8, color="saddlebrown")

    # Legend for train/test colours
    from matplotlib.patches import Patch
    legend_els = [Patch(fc="#2196F3", label="Train year"),
                  Patch(fc="#FF5722", label="Test year")]
    ax.legend(handles=legend_els + ax.get_legend_handles_labels()[0][:2], fontsize=8)

    # Right: scatter P90 vs bias amplitude
    ax2 = axes[1]
    for i, yr in enumerate(years):
        ax2.scatter(p90_vals[i], bias_vals[i], color=colors[i], s=80, zorder=3)
        ax2.annotate(str(yr), (p90_vals[i], bias_vals[i]),
                     textcoords="offset points", xytext=(5, 4), fontsize=9)

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(p90_vals, bias_vals)
    ax2.set_xlabel("P90 raw VHM0 (m)")
    ax2.set_ylabel("Mean |bias| (m)")
    ax2.set_title(f"Scatter: proxy vs bias amplitude\nSpearman ρ = {rho:.3f}  (p = {pval:.3f})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "01_p90_vs_bias_amplitude.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_calibration(years, p90_vals, bias_vals, train_years, test_years, output_dir):
    """Fit linear calibration on train years, predict test years."""
    train_mask = np.array([y in train_years for y in years])
    test_mask = np.array([y in test_years for y in years])

    p90_tr = p90_vals[train_mask]
    bias_tr = bias_vals[train_mask]

    # Linear fit
    coeffs = np.polyfit(p90_tr, bias_tr, 1)
    alpha, beta = coeffs
    fit_line_x = np.linspace(p90_vals.min() * 0.95, p90_vals.max() * 1.05, 100)
    fit_line_y = alpha * fit_line_x + beta

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fit_line_x, fit_line_y, "k--", alpha=0.5, label=f"Linear fit (train): y={alpha:.4f}x+{beta:.4f}")

    for i, yr in enumerate(years):
        col = "#2196F3" if yr in train_years else "#FF5722"
        marker = "o" if yr in train_years else "^"
        ax.scatter(p90_vals[i], bias_vals[i], color=col, marker=marker, s=90, zorder=3)
        # predicted for test
        if yr in test_years:
            pred = alpha * p90_vals[i] + beta
            ax.scatter(p90_vals[i], pred, color=col, marker="x", s=120, linewidths=2,
                       zorder=4, label=f"{yr} predicted={pred:.4f}m  actual={bias_vals[i]:.4f}m")
        ax.annotate(str(yr), (p90_vals[i], bias_vals[i]),
                    textcoords="offset points", xytext=(5, 4), fontsize=9)

    from matplotlib.patches import Patch
    legend_els = [Patch(fc="#2196F3", label="Train year"),
                  Patch(fc="#FF5722", label="Test year")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_els + handles, fontsize=8)
    ax.set_xlabel("P90 raw VHM0 (m)")
    ax.set_ylabel("Mean |bias| (m)")
    ax.set_title("Linear calibration: P90 → bias amplitude\n(train years fit, test years predicted)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "02_calibration_fit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print calibration table
    print("\n── Calibration results ──")
    print(f"  Fit:  bias_amplitude = {alpha:.5f} × P90  +  {beta:.5f}")
    print(f"\n  {'Year':<8} {'P90 (m)':<12} {'Actual bias (m)':<18} {'Predicted bias (m)':<20} {'Error (m)'}")
    for i, yr in enumerate(years):
        pred = alpha * p90_vals[i] + beta
        tag = "[TRAIN]" if yr in train_years else "[TEST] "
        print(f"  {yr} {tag}  P90={p90_vals[i]:.4f}   actual={bias_vals[i]:.4f}   "
              f"predicted={pred:.4f}   err={abs(pred - bias_vals[i]):.4f}")

    return alpha, beta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amplitude proxy analysis for wave bias correction")
    parser.add_argument("--data_path", required=True,
                        help="Path to preprocessed .pt files, e.g. /mnt/local_datasets/preprocessed_extended_subsampled_step_5/")
    parser.add_argument("--file_pattern", default="WAVEAN*.pt")
    parser.add_argument("--output_dir", default="./amplitude_proxy_output")
    parser.add_argument("--region", default="mediterranean",
                        choices=["mediterranean", "atlantic", "all"])
    parser.add_argument("--train_years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    parser.add_argument("--test_years", nargs="+", type=int, default=[2017, 2022, 2023])
    parser.add_argument("--max_files_per_year", type=int, default=None,
                        help="Limit files per year for a quick test run (None = use all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.region
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover files ────────────────────────────────────────────────────────
    all_files = sorted(glob.glob(f"{args.data_path}/{args.file_pattern}"))
    print(f"Found {len(all_files)} total .pt files")

    files_by_year = defaultdict(list)
    for f in all_files:
        year, month = parse_year_month(f)
        if year:
            files_by_year[year].append(f)

    all_years = sorted(files_by_year.keys())
    print(f"Years available: {all_years}")
    for y in all_years:
        print(f"  {y}: {len(files_by_year[y])} files")

    # ── get feature indices from first file ───────────────────────────────────
    first_file = sorted(all_files)[0]
    data0 = torch.load(first_file, map_location="cpu", weights_only=False)
    feature_cols = data0["feature_cols"]
    tensor0 = data0["tensor"]  # (24, H, W, C)

    print(f"\nFeature columns: {feature_cols}")
    vhm0_idx = feature_cols.index("VHM0")
    corrected_vhm0_idx = feature_cols.index("corrected_VHM0")
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    print(f"  VHM0 idx: {vhm0_idx},  corrected_VHM0 idx: {corrected_vhm0_idx}")

    # ── build region mask from first file ─────────────────────────────────────
    lat_grid = tensor0[0, ..., lat_idx].numpy()   # (H, W)
    lon_grid = tensor0[0, ..., lon_idx].numpy()   # (H, W)
    region_mask = get_region_mask(lon_grid, lat_grid, args.region)
    print(f"\nRegion: {args.region}  |  sea pixels in mask: {region_mask.sum()}")

    # ── compute per-year stats ────────────────────────────────────────────────
    years_to_process = [y for y in all_years if y in args.train_years + args.test_years]
    print(f"\nProcessing years: {years_to_process}")

    results = {}
    for year in years_to_process:
        flist = files_by_year[year]
        print(f"\n  Year {year}: {len(flist)} files ...", flush=True)
        p90, mean_bias = compute_year_stats(
            flist, vhm0_idx, corrected_vhm0_idx,
            region_mask, max_files=args.max_files_per_year
        )
        results[year] = {"p90": p90, "mean_abs_bias": mean_bias}
        print(f"    P90 raw VHM0 = {p90:.4f} m   |   Mean |bias| = {mean_bias:.4f} m")

    # ── assemble arrays ───────────────────────────────────────────────────────
    years_arr = np.array(sorted(results.keys()))
    p90_arr = np.array([results[y]["p90"] for y in years_arr])
    bias_arr = np.array([results[y]["mean_abs_bias"] for y in years_arr])

    # ── save CSV ──────────────────────────────────────────────────────────────
    import csv
    csv_path = output_dir / "amplitude_proxy_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "p90_raw_vhm0_m", "mean_abs_bias_m", "split"])
        for y, p90, bias in zip(years_arr, p90_arr, bias_arr):
            split = "train" if y in args.train_years else "test"
            writer.writerow([y, f"{p90:.5f}", f"{bias:.5f}", split])
    print(f"\nSaved CSV: {csv_path}")

    # ── ranking check ─────────────────────────────────────────────────────────
    p90_rank = np.argsort(np.argsort(p90_arr)) + 1
    bias_rank = np.argsort(np.argsort(bias_arr)) + 1
    print("\n── Ranking comparison ──")
    print(f"  {'Year':<8} {'P90 rank':<12} {'Bias rank':<12} {'Match?'}")
    for y, pr, br in zip(years_arr, p90_rank, bias_rank):
        match = "✓" if pr == br else f"✗ (diff={abs(int(pr)-int(br))})"
        print(f"  {y:<8} {pr:<12} {br:<12} {match}")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_ranking_comparison(years_arr, p90_arr, bias_arr, args.train_years, output_dir)
    alpha, beta = plot_calibration(
        years_arr, p90_arr, bias_arr,
        args.train_years, args.test_years, output_dir
    )

    # ── print scale factors for inference ────────────────────────────────────
    train_bias_mean = np.mean([results[y]["mean_abs_bias"] for y in args.train_years if y in results])
    print("\n── Inference scale factors  (predicted_bias / train_mean_bias) ──")
    print(f"  Train mean bias amplitude: {train_bias_mean:.4f} m")
    for y in args.test_years:
        if y not in results:
            continue
        pred_bias = alpha * results[y]["p90"] + beta
        scale = pred_bias / train_bias_mean
        print(f"  {y}: P90={results[y]['p90']:.4f}m → predicted_bias={pred_bias:.4f}m → scale={scale:.4f}")

    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
