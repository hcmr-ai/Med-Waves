"""
Bias Distribution Diagnostic
==============================
For each data split (train, test-2017, test-2023) compute and plot the
distribution of the **uncorrected absolute bias**:

    bias = |corrected_VHM0 - VHM0|

Answers three questions:
  1. What is the bias distribution in train vs test?  (mean, std, quantiles)
  2. Is there data leakage? (compare raw-VHM0 distributions across splits)
  3. Does the test set contain cases the model learned?
     (overlap of VHM0 / bias ranges between train and each test split)

All accumulation is streaming (histogram-based) so no large in-memory arrays.

Usage:
  poetry run python scripts/bias_distribution_diagnostic.py \
    --data_path /mnt/blobstorage/preprocessed_extended_subsampled_step_5/ \
    --train_years 2018 2019 2020 2021 \
    --test_years 2017 2023 \
    --region mediterranean \
    --output_dir /mnt/blobstorage/diagnostics/bias_distribution
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy import stats as _scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GIBRALTAR_LON = -5.5
BISCAY_LAT    = 43.0
BISCAY_LON    = 0.0

BIAS_HIST_BINS   = np.linspace(0, 6, 601)     # 0–6 m, 0.01 m bins
VHM0_HIST_BINS   = np.linspace(0, 15, 301)    # 0–15 m, 0.05 m bins
_BIAS_CENTERS    = 0.5 * (BIAS_HIST_BINS[:-1] + BIAS_HIST_BINS[1:])
_VHM0_CENTERS    = 0.5 * (VHM0_HIST_BINS[:-1] + VHM0_HIST_BINS[1:])

WAVE_HEIGHT_BINS = [
    (0.0, 1.0, "calm"),
    (1.0, 2.0, "light"),
    (2.0, 3.0, "moderate"),
    (3.0, 5.0, "rough"),
    (5.0, 15.0, "extreme"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_year(filename: str) -> int | None:
    m = re.search(r"WAVEAN(\d{4})", Path(filename).stem)
    return int(m.group(1)) if m else None


def build_region_mask(lat: np.ndarray, lon: np.ndarray, region: str) -> np.ndarray:
    biscay = (lat > BISCAY_LAT) & (lon < BISCAY_LON)
    if region == "atlantic":
        return (lon < GIBRALTAR_LON) | biscay
    if region == "mediterranean":
        return (lon >= GIBRALTAR_LON) & ~biscay
    return np.ones_like(lat, dtype=bool)  # "all"


def _hist_kde(hist_counts: np.ndarray, bin_centers: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Build a smooth KDE from histogram counts."""
    w = hist_counts.astype(np.float64)
    total = w.sum()
    if total == 0 or not HAS_SCIPY:
        return np.zeros_like(x_grid)
    w_norm = w / total
    std_est = float(np.sqrt(np.sum(w_norm * (bin_centers - np.sum(w_norm * bin_centers)) ** 2)))
    bw = max(0.5 * std_est * total ** (-1 / 5), 1e-3) if std_est > 0 else 0.1
    kde = _scipy_stats.gaussian_kde(bin_centers, weights=w + 1e-12, bw_method=bw)
    return kde(x_grid)


def weighted_stats(hist: np.ndarray, centers: np.ndarray) -> dict:
    """Compute mean, std, and quantiles from a histogram."""
    total = hist.sum()
    if total == 0:
        return {k: np.nan for k in ("mean", "std", "p25", "p50", "p75", "p90", "p95", "p99", "n")}
    w = hist / total
    mean = float(np.sum(w * centers))
    std  = float(np.sqrt(np.sum(w * (centers - mean) ** 2)))
    cdf  = np.cumsum(hist) / total
    def quantile(p):
        idx = np.searchsorted(cdf, p)
        return float(centers[min(idx, len(centers) - 1)])
    return {
        "n":   int(total),
        "mean": mean,
        "std":  std,
        "p25":  quantile(0.25),
        "p50":  quantile(0.50),
        "p75":  quantile(0.75),
        "p90":  quantile(0.90),
        "p95":  quantile(0.95),
        "p99":  quantile(0.99),
    }


def kl_divergence(p_hist: np.ndarray, q_hist: np.ndarray, eps: float = 1e-10) -> float:
    """KL(P||Q) from two unnormalised histograms (same bins)."""
    p = (p_hist + eps) / (p_hist.sum() + eps * len(p_hist))
    q = (q_hist + eps) / (q_hist.sum() + eps * len(q_hist))
    return float(np.sum(p * np.log(p / q)))


def histogram_overlap(p_hist: np.ndarray, q_hist: np.ndarray) -> float:
    """Bhattacharyya overlap coefficient in [0, 1]."""
    p = p_hist / (p_hist.sum() + 1e-30)
    q = q_hist / (q_hist.sum() + 1e-30)
    return float(np.sum(np.sqrt(p * q)))

# ---------------------------------------------------------------------------
# Core accumulator
# ---------------------------------------------------------------------------

class SplitAccumulator:
    """Streaming accumulator for one data split."""

    def __init__(self, label: str):
        self.label = label
        self.bias_hist   = np.zeros(len(BIAS_HIST_BINS) - 1, dtype=np.float64)
        self.vhm0_hist   = np.zeros(len(VHM0_HIST_BINS) - 1, dtype=np.float64)
        self.n_pixels    = np.int64(0)
        # Per-wave-height-bin bias histograms (for "overlap" / leakage check)
        self.wbin_bias_hist: dict[str, np.ndarray] = {
            name: np.zeros(len(BIAS_HIST_BINS) - 1, dtype=np.float64)
            for *_, name in WAVE_HEIGHT_BINS
        }
        self.wbin_vhm0_hist: dict[str, np.ndarray] = {
            name: np.zeros(len(VHM0_HIST_BINS) - 1, dtype=np.float64)
            for *_, name in WAVE_HEIGHT_BINS
        }

    def update(self, vhm0: np.ndarray, corrected: np.ndarray, valid: np.ndarray):
        """Accept 1-D arrays of valid pixels."""
        bias = np.abs(corrected[valid] - vhm0[valid])
        raw  = vhm0[valid]
        self.bias_hist += np.histogram(bias, bins=BIAS_HIST_BINS)[0]
        self.vhm0_hist += np.histogram(raw,  bins=VHM0_HIST_BINS)[0]
        self.n_pixels  += int(valid.sum())
        for lo, hi, name in WAVE_HEIGHT_BINS:
            mask = (raw >= lo) & (raw < hi)
            if mask.any():
                self.wbin_bias_hist[name] += np.histogram(bias[mask], bins=BIAS_HIST_BINS)[0]
                self.wbin_vhm0_hist[name] += np.histogram(raw[mask],  bins=VHM0_HIST_BINS)[0]

    def stats(self) -> dict:
        return weighted_stats(self.bias_hist, _BIAS_CENTERS)

    def vhm0_stats(self) -> dict:
        return weighted_stats(self.vhm0_hist, _VHM0_CENTERS)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
COLORS = {
    "train":  "#2ca02c",   # green
    "test_2017": "#1f77b4",  # blue
    "test_2023": "#d62728",  # red
}

def _kde_plot(ax, acc: SplitAccumulator, hist: np.ndarray, centers: np.ndarray,
              x_grid: np.ndarray, color: str, label: str):
    if hist.sum() == 0:
        return
    kde = _hist_kde(hist, centers, x_grid)
    ax.plot(x_grid, kde, color=color, lw=2, label=label)
    ax.fill_between(x_grid, kde, alpha=0.12, color=color)


def _stats_table_text(stats: dict) -> str:
    lines = [
        f"n = {stats['n']:,}",
        f"mean = {stats['mean']:.3f} m",
        f"std  = {stats['std']:.3f} m",
        f"p25  = {stats['p25']:.3f} m",
        f"p50  = {stats['p50']:.3f} m",
        f"p75  = {stats['p75']:.3f} m",
        f"p90  = {stats['p90']:.3f} m",
        f"p95  = {stats['p95']:.3f} m",
        f"p99  = {stats['p99']:.3f} m",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", required=True,
                   help="Root directory with WAVEAN*.pt files")
    p.add_argument("--file_pattern", default="WAVEAN*.pt")
    p.add_argument("--train_years", nargs="+", type=int,
                   default=[2018, 2019, 2020, 2021])
    p.add_argument("--test_years", nargs="+", type=int,
                   default=[2017, 2023])
    p.add_argument("--region", default="mediterranean",
                   choices=["all", "atlantic", "mediterranean"])
    p.add_argument("--output_dir",
                   default="/mnt/blobstorage/diagnostics/bias_distribution")
    p.add_argument("--max_files_per_year", type=int, default=None,
                   help="Limit files per year (for quick test runs)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir) / args.region
    out.mkdir(parents=True, exist_ok=True)

    all_files = sorted(glob.glob(f"{args.data_path}/{args.file_pattern}"))
    if not all_files:
        print(f"ERROR: no files found at {args.data_path}/{args.file_pattern}", flush=True)
        sys.exit(1)
    print(f"Found {len(all_files)} total files", flush=True)

    files_by_year: dict[int, list[str]] = defaultdict(list)
    for f in all_files:
        y = parse_year(f)
        if y:
            files_by_year[y].append(f)

    years_present = sorted(files_by_year)
    print(f"Years present: {years_present}", flush=True)

    # Read grid shape and feature indices from first file
    data0 = torch.load(all_files[0], map_location="cpu", weights_only=False)
    feature_cols = data0["feature_cols"]
    H, W = data0["tensor"].shape[1], data0["tensor"].shape[2]
    vhm0_idx      = feature_cols.index("VHM0")
    corrected_idx = feature_cols.index("corrected_VHM0")
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")

    # Build region mask (constant across files)
    lat_grid = data0["tensor"][0, :, :, lat_idx].numpy()
    lon_grid = data0["tensor"][0, :, :, lon_idx].numpy()
    del data0

    region_mask = build_region_mask(lat_grid, lon_grid, args.region)
    sea_mask = region_mask & ~np.isnan(lat_grid) & ~np.isnan(lon_grid)
    print(f"Region '{args.region}': {sea_mask.sum():,} / {sea_mask.size:,} pixels kept "
          f"({100 * sea_mask.mean():.1f}%)", flush=True)

    # ------------------------------------------------------------------ #
    # Accumulate histograms per split                                      #
    # ------------------------------------------------------------------ #
    train_acc = SplitAccumulator("train")
    test_accs: dict[int, SplitAccumulator] = {
        y: SplitAccumulator(f"test_{y}") for y in args.test_years
    }
    all_years = set(args.train_years) | set(args.test_years)

    for year in sorted(all_years):
        if year not in files_by_year:
            print(f"  WARNING: year {year} not found, skipping.", flush=True)
            continue

        is_train = year in args.train_years
        acc = train_acc if is_train else test_accs[year]
        tag = "TRAIN" if is_train else f"TEST-{year}"
        file_list = files_by_year[year]
        if args.max_files_per_year:
            file_list = file_list[:args.max_files_per_year]

        print(f"\n[{tag}] {year}: processing {len(file_list)} files …", flush=True)
        for fi, fpath in enumerate(file_list):
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            tensor = data["tensor"]  # (24, H, W, C)
            raw       = tensor[:, :, :, vhm0_idx].numpy()      # (24, H, W)
            corrected = tensor[:, :, :, corrected_idx].numpy()  # (24, H, W)
            del data, tensor

            for hour in range(24):
                valid_mask = (
                    sea_mask &
                    ~np.isnan(raw[hour]) &
                    ~np.isnan(corrected[hour])
                )
                if not valid_mask.any():
                    continue
                acc.update(raw[hour], corrected[hour], valid_mask)

            if (fi + 1) % 20 == 0 or fi == len(file_list) - 1:
                print(f"  {year}: {fi+1}/{len(file_list)} files, "
                      f"pixels so far: {acc.n_pixels:,}", flush=True)

    # ------------------------------------------------------------------ #
    # Print statistics                                                     #
    # ------------------------------------------------------------------ #
    all_accs: dict[str, SplitAccumulator] = {"train": train_acc}
    for y, acc in test_accs.items():
        all_accs[f"test_{y}"] = acc

    print("\n" + "=" * 70, flush=True)
    print("BIAS STATISTICS  (|corrected_VHM0 − VHM0|)", flush=True)
    print("=" * 70, flush=True)
    header = f"  {'Split':<12} {'n':>12} {'mean':>7} {'std':>7} "
    header += f"{'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'p95':>7} {'p99':>7}"
    print(header)
    print("  " + "-" * 80)
    for key, acc in all_accs.items():
        s = acc.stats()
        print(f"  {key:<12} {s['n']:>12,} {s['mean']:>7.4f} {s['std']:>7.4f} "
              f"{s['p25']:>7.4f} {s['p50']:>7.4f} {s['p75']:>7.4f} "
              f"{s['p90']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f}")

    print("\n" + "=" * 70, flush=True)
    print("RAW VHM0 STATISTICS  (input to the model)", flush=True)
    print("=" * 70, flush=True)
    print(header.replace("BIAS", "VHM0"))
    print("  " + "-" * 80)
    for key, acc in all_accs.items():
        s = acc.vhm0_stats()
        print(f"  {key:<12} {s['n']:>12,} {s['mean']:>7.4f} {s['std']:>7.4f} "
              f"{s['p25']:>7.4f} {s['p50']:>7.4f} {s['p75']:>7.4f} "
              f"{s['p90']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f}")

    # ------------------------------------------------------------------ #
    # Leakage / overlap metrics                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70, flush=True)
    print("DATA LEAKAGE CHECK", flush=True)
    print("=" * 70, flush=True)
    print("  (High overlap / low KL → distributions are similar → no leakage signal)")
    for test_key, test_acc in [(k, v) for k, v in all_accs.items() if k != "train"]:
        kl_vhm0  = kl_divergence(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ov_vhm0  = histogram_overlap(train_acc.vhm0_hist, test_acc.vhm0_hist)
        kl_bias  = kl_divergence(train_acc.bias_hist, test_acc.bias_hist)
        ov_bias  = histogram_overlap(train_acc.bias_hist, test_acc.bias_hist)
        print(f"\n  train vs {test_key}:")
        print(f"    VHM0  — KL(train||test) = {kl_vhm0:.4f},  overlap = {ov_vhm0:.4f}")
        print(f"    bias  — KL(train||test) = {kl_bias:.4f},  overlap = {ov_bias:.4f}")

        # Per wave-height bin: does each test bin appear in train?
        print(f"    Per sea-state coverage (test bias within train VHM0 range):")
        for lo, hi, name in WAVE_HEIGHT_BINS:
            train_n = int(train_acc.wbin_bias_hist[name].sum())
            test_n  = int(test_acc.wbin_bias_hist[name].sum())
            ov = histogram_overlap(
                train_acc.wbin_bias_hist[name], test_acc.wbin_bias_hist[name]
            )
            print(f"      [{lo:.0f}–{hi:.0f}m] {name:<10}: "
                  f"train n={train_n:>12,}  test n={test_n:>12,}  bias overlap={ov:.4f}")

    # ------------------------------------------------------------------ #
    # PLOTS                                                                #
    # ------------------------------------------------------------------ #
    if not HAS_SCIPY:
        print("\nWARNING: scipy not available — skipping KDE plots.", flush=True)
        return

    x_bias = np.linspace(0, 4, 400)
    x_vhm0 = np.linspace(0, 15, 300)

    color_map = {"train": COLORS["train"]}
    for y in args.test_years:
        color_map[f"test_{y}"] = COLORS.get(f"test_{y}", f"C{y % 10}")

    # ------------------------------------------------------------------
    # Plot 1: Overall bias distribution (all splits overlaid)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in all_accs.items():
        _kde_plot(ax, acc, acc.bias_hist, _BIAS_CENTERS, x_bias,
                  color_map[key], label=f"{key} (n={acc.n_pixels:,})")
    ax.set_xlabel("|corrected_VHM0 − VHM0|  [m]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Uncorrected Bias Distribution — train vs test  [{args.region}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01_bias_distribution_overall.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out / '01_bias_distribution_overall.png'}")

    # ------------------------------------------------------------------
    # Plot 2: Statistics table (text panel per split)
    # ------------------------------------------------------------------
    n_splits = len(all_accs)
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5))
    if n_splits == 1:
        axes = [axes]
    for ax, (key, acc) in zip(axes, all_accs.items()):
        s = acc.stats()
        txt = f"{key.upper()}\n\n" + _stats_table_text(s)
        ax.text(0.5, 0.5, txt, transform=ax.transAxes,
                ha="center", va="center", fontsize=12,
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=color_map[key], alpha=0.15))
        ax.set_title(f"Bias statistics — {key}", fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Bias |corrected_VHM0 − VHM0| statistics  [{args.region}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "02_bias_statistics_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '02_bias_statistics_table.png'}")

    # ------------------------------------------------------------------
    # Plot 3: Raw VHM0 distribution (data leakage check)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in all_accs.items():
        _kde_plot(ax, acc, acc.vhm0_hist, _VHM0_CENTERS, x_vhm0,
                  color_map[key], label=f"{key} (n={acc.n_pixels:,})")
    ax.set_xlabel("Raw VHM0 [m]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Raw VHM0 Distribution — data leakage check  [{args.region}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    note = ("Leakage indicator: if train and test VHM0 distributions are nearly "
            "identical,\nthe split may have leaked. Large shift = genuine temporal gap.")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out / "03_vhm0_distribution_leakage_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '03_vhm0_distribution_leakage_check.png'}")

    # ------------------------------------------------------------------
    # Plot 4: Bias distribution per wave-height bin (train vs each test)
    # ------------------------------------------------------------------
    for test_key, test_acc in [(k, v) for k, v in all_accs.items() if k != "train"]:
        n_bins = len(WAVE_HEIGHT_BINS)
        fig, axes = plt.subplots(1, n_bins, figsize=(4 * n_bins, 5), sharey=False)
        for ax, (lo, hi, name) in zip(axes, WAVE_HEIGHT_BINS):
            tr_h = train_acc.wbin_bias_hist[name]
            te_h = test_acc.wbin_bias_hist[name]
            if tr_h.sum() > 0:
                _kde_plot(ax, None, tr_h, _BIAS_CENTERS, x_bias,
                          color_map["train"], label=f"train (n={int(tr_h.sum()):,})")
            if te_h.sum() > 0:
                _kde_plot(ax, None, te_h, _BIAS_CENTERS, x_bias,
                          color_map[test_key], label=f"{test_key} (n={int(te_h.sum()):,})")
            ov = histogram_overlap(tr_h, te_h)
            ax.set_title(f"{name}\n[{lo:.0f}–{hi:.0f}m]\noverlap={ov:.3f}", fontsize=9)
            ax.set_xlabel("|bias| [m]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Bias distribution per sea-state bin: train vs {test_key}  [{args.region}]\n"
            "overlap→1: test covers same conditions as train  (model has \"seen\" these cases)",
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout()
        fname = f"04_bias_per_seabin_train_vs_{test_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

    # ------------------------------------------------------------------
    # Plot 5: Raw VHM0 per wave-height bin (coverage check)
    # ------------------------------------------------------------------
    for test_key, test_acc in [(k, v) for k, v in all_accs.items() if k != "train"]:
        n_bins = len(WAVE_HEIGHT_BINS)
        fig, axes = plt.subplots(1, n_bins, figsize=(4 * n_bins, 5), sharey=False)
        for ax, (lo, hi, name) in zip(axes, WAVE_HEIGHT_BINS):
            tr_h = train_acc.wbin_vhm0_hist[name]
            te_h = test_acc.wbin_vhm0_hist[name]
            x_sub = np.linspace(lo, hi, 100)
            if tr_h.sum() > 0:
                _kde_plot(ax, None, tr_h, _VHM0_CENTERS, x_sub,
                          color_map["train"], label=f"train (n={int(tr_h.sum()):,})")
            if te_h.sum() > 0:
                _kde_plot(ax, None, te_h, _VHM0_CENTERS, x_sub,
                          color_map[test_key], label=f"{test_key} (n={int(te_h.sum()):,})")
            ov = histogram_overlap(tr_h, te_h)
            ax.set_title(f"{name}\n[{lo:.0f}–{hi:.0f}m]\noverlap={ov:.3f}", fontsize=9)
            ax.set_xlabel("VHM0 [m]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"VHM0 distribution per sea-state bin: train vs {test_key}  [{args.region}]\n"
            "overlap→1: test VHM0 values fall within the range the model trained on",
            fontsize=11, fontweight="bold"
        )
        fig.tight_layout()
        fname = f"05_vhm0_per_seabin_train_vs_{test_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

    # ------------------------------------------------------------------
    # Plot 6: Side-by-side comparison panels for each test year
    # ------------------------------------------------------------------
    for test_key, test_acc in [(k, v) for k, v in all_accs.items() if k != "train"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: bias KDE overlaid
        ax = axes[0]
        _kde_plot(ax, None, train_acc.bias_hist, _BIAS_CENTERS, x_bias,
                  color_map["train"], label=f"train  (n={train_acc.n_pixels:,})")
        _kde_plot(ax, None, test_acc.bias_hist, _BIAS_CENTERS, x_bias,
                  color_map[test_key], label=f"{test_key}  (n={test_acc.n_pixels:,})")
        ax.set_xlabel("|bias| [m]")
        ax.set_ylabel("Density")
        ax.set_title("Bias distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: stats comparison
        ax = axes[1]
        s_tr = train_acc.stats()
        s_te = test_acc.stats()
        keys_show = ["mean", "std", "p25", "p50", "p75", "p90", "p95", "p99"]
        y_pos = np.arange(len(keys_show))
        tr_vals = [s_tr[k] for k in keys_show]
        te_vals = [s_te[k] for k in keys_show]
        ax.barh(y_pos + 0.2, tr_vals, height=0.35, color=color_map["train"],
                alpha=0.7, label="train")
        ax.barh(y_pos - 0.2, te_vals, height=0.35, color=color_map[test_key],
                alpha=0.7, label=test_key)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys_show)
        ax.set_xlabel("Bias [m]")
        ax.set_title("Statistics comparison")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

        # Panel 3: VHM0 KDE (leakage check)
        ax = axes[2]
        _kde_plot(ax, None, train_acc.vhm0_hist, _VHM0_CENTERS, x_vhm0,
                  color_map["train"], label=f"train  (n={train_acc.n_pixels:,})")
        _kde_plot(ax, None, test_acc.vhm0_hist, _VHM0_CENTERS, x_vhm0,
                  color_map[test_key], label=f"{test_key}  (n={test_acc.n_pixels:,})")
        kl_v = kl_divergence(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ov_v = histogram_overlap(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ax.set_xlabel("Raw VHM0 [m]")
        ax.set_ylabel("Density")
        ax.set_title(f"VHM0 distribution (leakage check)\nKL={kl_v:.3f}  overlap={ov_v:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Bias distribution summary: train vs {test_key}  [{args.region}]",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()
        fname = f"06_summary_train_vs_{test_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

    print(f"\nAll outputs written to {out}", flush=True)
    print("\nINTERPRETATION GUIDE", flush=True)
    print("-" * 60, flush=True)
    print("  Leakage check (plot 03 / plot 06 panel 3):", flush=True)
    print("    overlap ≈ 1.0 and KL ≈ 0 → train/test see same VHM0 values", flush=True)
    print("    (expected for holdout-by-year; some overlap is fine)", flush=True)
    print("    Very low overlap → test has a very different sea-state regime", flush=True)
    print("  Coverage check (plot 04, plot 05):", flush=True)
    print("    Per-bin overlap ≈ 1.0 → test bias is drawn from the same", flush=True)
    print("    distribution the model learned in that sea-state bucket.", flush=True)
    print("    Low per-bin overlap → test is out-of-distribution for that bin.", flush=True)


if __name__ == "__main__":
    main()
