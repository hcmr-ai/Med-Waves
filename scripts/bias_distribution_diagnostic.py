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
     (per-sea-state bias overlap between train and each test split)

All accumulation is streaming (histogram-based). Files are processed in
parallel with ProcessPoolExecutor so the full multi-year dataset finishes
in reasonable wall-clock time.

Usage:
  poetry run python scripts/bias_distribution_diagnostic.py \
    --data_path /mnt/blobstorage/preprocessed_extended_subsampled_step_5/ \
    --train_years 2018 2019 2020 2021 \
    --test_years 2017 2023 \
    --region mediterranean \
    --workers 8 \
    --output_dir /mnt/blobstorage/diagnostics/bias_distribution
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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

BIAS_HIST_BINS = np.linspace(0, 6, 601)     # 0–6 m, 0.01 m bins
VHM0_HIST_BINS = np.linspace(0, 15, 301)    # 0–15 m, 0.05 m bins
_BIAS_CENTERS  = 0.5 * (BIAS_HIST_BINS[:-1] + BIAS_HIST_BINS[1:])
_VHM0_CENTERS  = 0.5 * (VHM0_HIST_BINS[:-1] + VHM0_HIST_BINS[1:])

WAVE_HEIGHT_BINS = [
    (0.0,  1.0,  "calm"),
    (1.0,  2.0,  "light"),
    (2.0,  3.0,  "moderate"),
    (3.0,  5.0,  "rough"),
    (5.0,  6.0, "extreme_5_6"),
    (6.0,  7.0, "extreme_6_7"),
    (7.0,  8.0, "extreme_7_8"),
    (8.0,  9.0, "extreme_8_9"),
    (9.0,  10.0, "extreme_9_10"),
    (10.0,  11.0, "extreme_10_11"),
    (11.0,  12.0, "extreme_11_12"),
    (12.0,  13.0, "extreme_12_13"),
    (13.0,  14.0, "extreme_13_14"),
]
WAVE_BIN_NAMES = [name for *_, name in WAVE_HEIGHT_BINS]

COLORS = {
    "train":    "#2ca02c",
    "test_2017": "#1f77b4",
    "test_2023": "#d62728",
}

# ---------------------------------------------------------------------------
# Utilities
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
    return np.ones_like(lat, dtype=bool)


def weighted_stats(hist: np.ndarray, centers: np.ndarray) -> dict:
    total = hist.sum()
    if total == 0:
        return {
            k: np.nan
            for k in ("mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99", "n")
        }
    w = hist / total
    mean = float(np.sum(w * centers))
    std  = float(np.sqrt(np.sum(w * (centers - mean) ** 2)))
    nonzero_idx = np.flatnonzero(hist)
    min_v = float(centers[nonzero_idx[0]])
    max_v = float(centers[nonzero_idx[-1]])
    cdf  = np.cumsum(hist) / total
    def pct(p):
        idx = np.searchsorted(cdf, p)
        return float(centers[min(idx, len(centers) - 1)])
    return dict(n=int(total), mean=mean, std=std, min=min_v, max=max_v,
                p25=pct(0.25), p50=pct(0.50), p75=pct(0.75),
                p90=pct(0.90), p95=pct(0.95), p99=pct(0.99))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    pn = (p + eps) / (p.sum() + eps * len(p))
    qn = (q + eps) / (q.sum() + eps * len(q))
    return float(np.sum(pn * np.log(pn / qn)))


def bhattacharyya_overlap(p: np.ndarray, q: np.ndarray) -> float:
    pn = p / (p.sum() + 1e-30)
    qn = q / (q.sum() + 1e-30)
    return float(np.sum(np.sqrt(pn * qn)))


def _hist_kde(hist: np.ndarray, centers: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    w = hist.astype(np.float64)
    total = w.sum()
    if total == 0 or not HAS_SCIPY:
        return np.zeros_like(x_grid)
    wn = w / total
    std_est = float(np.sqrt(np.sum(wn * (centers - np.sum(wn * centers)) ** 2)))
    bw = max(0.5 * std_est * total ** (-1 / 5), 1e-3) if std_est > 0 else 0.1
    kde = _scipy_stats.gaussian_kde(centers, weights=w + 1e-12, bw_method=bw)
    return kde(x_grid)

# ---------------------------------------------------------------------------
# Per-file worker  (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_file(args: tuple) -> dict | None:
    """
    Load one .pt file and return histogram arrays.
    Returns None on error (printed to stderr, not raised).
    """
    fpath, vhm0_idx, corrected_idx, sea_mask = args

    try:
        data   = torch.load(fpath, map_location="cpu", weights_only=False)
        tensor = data["tensor"]          # (24, H, W, C)
        raw       = tensor[:, :, :, vhm0_idx].numpy()      # (24, H, W)
        corrected = tensor[:, :, :, corrected_idx].numpy()  # (24, H, W)
        del data, tensor

        n_bias   = len(BIAS_HIST_BINS) - 1
        n_vhm0   = len(VHM0_HIST_BINS) - 1
        n_wbins  = len(WAVE_HEIGHT_BINS)

        bias_hist         = np.zeros(n_bias,              dtype=np.float64)
        vhm0_hist         = np.zeros(n_vhm0,              dtype=np.float64)
        wbin_bias_hist    = np.zeros((n_wbins, n_bias),   dtype=np.float64)
        wbin_vhm0_hist    = np.zeros((n_wbins, n_vhm0),   dtype=np.float64)
        n_pixels          = np.int64(0)

        for hour in range(raw.shape[0]):
            valid = sea_mask & ~np.isnan(raw[hour]) & ~np.isnan(corrected[hour])
            if not valid.any():
                continue
            r = raw[hour][valid]
            c = corrected[hour][valid]
            b = np.abs(c - r)

            bias_hist += np.histogram(b, bins=BIAS_HIST_BINS)[0]
            vhm0_hist += np.histogram(r, bins=VHM0_HIST_BINS)[0]
            n_pixels  += valid.sum()

            for wi, (lo, hi, _) in enumerate(WAVE_HEIGHT_BINS):
                mask = (r >= lo) & (r < hi)
                if mask.any():
                    wbin_bias_hist[wi] += np.histogram(b[mask], bins=BIAS_HIST_BINS)[0]
                    wbin_vhm0_hist[wi] += np.histogram(r[mask], bins=VHM0_HIST_BINS)[0]

        return dict(
            bias_hist=bias_hist,
            vhm0_hist=vhm0_hist,
            wbin_bias_hist=wbin_bias_hist,
            wbin_vhm0_hist=wbin_vhm0_hist,
            n_pixels=int(n_pixels),
        )

    except Exception as exc:
        print(f"  ERROR {Path(fpath).name}: {exc}", file=sys.stderr, flush=True)
        return None

# ---------------------------------------------------------------------------
# Result accumulator (main process)
# ---------------------------------------------------------------------------

class SplitAccumulator:
    def __init__(self, label: str):
        self.label         = label
        self.bias_hist     = np.zeros(len(BIAS_HIST_BINS) - 1, dtype=np.float64)
        self.vhm0_hist     = np.zeros(len(VHM0_HIST_BINS) - 1, dtype=np.float64)
        self.wbin_bias_hist = np.zeros((len(WAVE_HEIGHT_BINS), len(BIAS_HIST_BINS) - 1), dtype=np.float64)
        self.wbin_vhm0_hist = np.zeros((len(WAVE_HEIGHT_BINS), len(VHM0_HIST_BINS) - 1), dtype=np.float64)
        self.n_pixels      = 0

    def add(self, result: dict):
        self.bias_hist      += result["bias_hist"]
        self.vhm0_hist      += result["vhm0_hist"]
        self.wbin_bias_hist += result["wbin_bias_hist"]
        self.wbin_vhm0_hist += result["wbin_vhm0_hist"]
        self.n_pixels       += result["n_pixels"]

    def stats(self)      -> dict: return weighted_stats(self.bias_hist,  _BIAS_CENTERS)
    def vhm0_stats(self) -> dict: return weighted_stats(self.vhm0_hist,  _VHM0_CENTERS)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _kde_plot(ax, hist, centers, x_grid, color, label):
    if hist.sum() == 0:
        return
    kde = _hist_kde(hist, centers, x_grid)
    ax.plot(x_grid, kde, color=color, lw=2, label=label)
    ax.fill_between(x_grid, kde, alpha=0.12, color=color)


def _hist_density_plot(ax, hist, bins, color, label):
    total = hist.sum()
    if total == 0:
        return
    widths = np.diff(bins)
    density = hist / (total * widths)
    ax.step(bins[:-1], density, where="post", color=color, lw=1.8, label=label)
    ax.fill_between(bins[:-1], density, step="post", alpha=0.10, color=color)


def _stats_text(s: dict) -> str:
    return "\n".join([
        f"n    = {s['n']:,}",
        f"mean = {s['mean']:.3f} m",
        f"std  = {s['std']:.3f} m",
        f"min  = {s['min']:.3f} m",
        f"max  = {s['max']:.3f} m",
        f"p25  = {s['p25']:.3f} m",
        f"p50  = {s['p50']:.3f} m",
        f"p75  = {s['p75']:.3f} m",
        f"p90  = {s['p90']:.3f} m",
        f"p95  = {s['p95']:.3f} m",
        f"p99  = {s['p99']:.3f} m",
    ])


def _split_color(key: str) -> str:
    return COLORS.get(key, "#ff7f0e")


def _fmt_bin_edge(value: float) -> str:
    return f"{value:g}"


def _format_stats_row(split: str, s: dict) -> str:
    return (
        f"  {split:<12} {s['n']:>14,} {s['mean']:>7.4f} {s['std']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f} "
        f"{s['p25']:>7.4f} {s['p50']:>7.4f} {s['p75']:>7.4f} "
        f"{s['p90']:>7.4f} {s['p95']:>7.4f} {s['p99']:>7.4f}"
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--file_pattern", default="WAVEAN*.pt")
    p.add_argument("--train_years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    p.add_argument("--test_years",  nargs="+", type=int, default=[2017, 2023])
    p.add_argument("--region", default="mediterranean",
                   choices=["all", "atlantic", "mediterranean"])
    p.add_argument("--output_dir",
                   default="/mnt/blobstorage/diagnostics/bias_distribution")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes")
    p.add_argument("--simple_hist_only", action="store_true",
                   help="Generate only a simple per-split bias histogram (actual counts)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir) / args.region
    out.mkdir(parents=True, exist_ok=True)

    all_files = sorted(glob.glob(f"{args.data_path}/{args.file_pattern}"))
    if not all_files:
        sys.exit(f"ERROR: no files found at {args.data_path}/{args.file_pattern}")
    print(f"Found {len(all_files)} total files", flush=True)

    files_by_year: dict[int, list[str]] = defaultdict(list)
    for f in all_files:
        y = parse_year(f)
        if y:
            files_by_year[y].append(f)

    years_present = sorted(files_by_year)
    print(f"Years present: {years_present}", flush=True)

    # Read grid metadata from first file
    data0        = torch.load(all_files[0], map_location="cpu", weights_only=False)
    feature_cols = data0["feature_cols"]
    vhm0_idx      = feature_cols.index("VHM0")
    corrected_idx = feature_cols.index("corrected_VHM0")
    lat_idx       = feature_cols.index("latitude")
    lon_idx       = feature_cols.index("longitude")
    lat_grid = data0["tensor"][0, :, :, lat_idx].numpy()
    lon_grid = data0["tensor"][0, :, :, lon_idx].numpy()
    del data0

    sea_mask = (
        build_region_mask(lat_grid, lon_grid, args.region)
        & ~np.isnan(lat_grid)
        & ~np.isnan(lon_grid)
    )
    print(f"Region '{args.region}': {sea_mask.sum():,} / {sea_mask.size:,} pixels "
          f"({100 * sea_mask.mean():.1f}%)", flush=True)

    # ------------------------------------------------------------------ #
    # Build work list                                                       #
    # ------------------------------------------------------------------ #
    train_years = set(args.train_years)
    test_years  = set(args.test_years)
    all_years   = train_years | test_years

    work: list[tuple[str, int]] = []   # (fpath, year)
    for year in sorted(all_years):
        if year not in files_by_year:
            print(f"  WARNING: year {year} not found, skipping.", flush=True)
            continue
        for fpath in files_by_year[year]:
            work.append((fpath, year))

    print(f"\nTotal files to process: {len(work)}", flush=True)
    for year in sorted(all_years):
        n = len(files_by_year.get(year, []))
        tag = "TRAIN" if year in train_years else "TEST"
        print(f"  {year} [{tag}]: {n} files", flush=True)

    # ------------------------------------------------------------------ #
    # Parallel accumulation                                                 #
    # ------------------------------------------------------------------ #
    accumulators: dict[str, SplitAccumulator] = {"train": SplitAccumulator("train")}
    for y in args.test_years:
        accumulators[f"test_{y}"] = SplitAccumulator(f"test_{y}")

    worker_args = [
        (fpath, vhm0_idx, corrected_idx, sea_mask)
        for fpath, _ in work
    ]
    year_for_file = {fpath: year for fpath, year in work}

    print(f"\nProcessing with {args.workers} workers …", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_file, wa): wa[0]
            for wa in worker_args
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="files"):
            fpath  = futures[future]
            result = future.result()
            if result is None:
                continue
            year = year_for_file[fpath]
            key  = "train" if year in train_years else f"test_{year}"
            accumulators[key].add(result)

    # ------------------------------------------------------------------ #
    # Print statistics                                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("BIAS STATISTICS  |corrected_VHM0 − VHM0|")
    print("=" * 70)
    hdr = (f"  {'Split':<12} {'n':>14} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} "
           f"{'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7} {'p95':>7} {'p99':>7}")
    table_sep = "  " + "-" * 101
    stats_report_lines = [
        "=" * 70,
        "BIAS STATISTICS  |corrected_VHM0 - VHM0|",
        "=" * 70,
        hdr,
        table_sep,
    ]
    print(hdr)
    print(table_sep)
    for key, acc in accumulators.items():
        s = acc.stats()
        row = _format_stats_row(key, s)
        print(row)
        stats_report_lines.append(row)

    print("\n" + "=" * 70)
    print("RAW VHM0 STATISTICS")
    print("=" * 70)
    print(hdr)
    print(table_sep)
    stats_report_lines.extend([
        "",
        "=" * 70,
        "RAW VHM0 STATISTICS",
        "=" * 70,
        hdr,
        table_sep,
    ])
    for key, acc in accumulators.items():
        s = acc.vhm0_stats()
        row = _format_stats_row(key, s)
        print(row)
        stats_report_lines.append(row)

    print("\n" + "=" * 70)
    print("DATA LEAKAGE CHECK")
    print("=" * 70)
    stats_report_lines.extend([
        "",
        "=" * 70,
        "DATA LEAKAGE CHECK",
        "=" * 70,
    ])
    train_acc = accumulators["train"]
    for test_key, test_acc in ((k, v) for k, v in accumulators.items() if k != "train"):
        kl_v = kl_divergence(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ov_v = bhattacharyya_overlap(train_acc.vhm0_hist, test_acc.vhm0_hist)
        kl_b = kl_divergence(train_acc.bias_hist, test_acc.bias_hist)
        ov_b = bhattacharyya_overlap(train_acc.bias_hist, test_acc.bias_hist)
        tr_v = train_acc.vhm0_stats()
        te_v = test_acc.vhm0_stats()
        tr_b = train_acc.stats()
        te_b = test_acc.stats()
        vhm0_line = (
            f"    VHM0  KL(train||test)={kl_v:.4f}  overlap={ov_v:.4f}  "
            f"train[min={tr_v['min']:.4f}, max={tr_v['max']:.4f}, p90={tr_v['p90']:.4f}]  "
            f"test[min={te_v['min']:.4f}, max={te_v['max']:.4f}, p90={te_v['p90']:.4f}]"
        )
        bias_line = (
            f"    bias  KL(train||test)={kl_b:.4f}  overlap={ov_b:.4f}  "
            f"train[min={tr_b['min']:.4f}, max={tr_b['max']:.4f}, p90={tr_b['p90']:.4f}]  "
            f"test[min={te_b['min']:.4f}, max={te_b['max']:.4f}, p90={te_b['p90']:.4f}]"
        )
        print(f"\n  train vs {test_key}:")
        print(vhm0_line)
        print(bias_line)
        print("    Per sea-state bias stats (model coverage):")
        per_bin_hdr = (
            "      bin             train_n      test_n   overlap"
            "   tr_mean   te_mean    tr_std    te_std    tr_p50    te_p50"
        )
        print(per_bin_hdr)
        stats_report_lines.extend([
            "",
            f"  train vs {test_key}:",
            vhm0_line,
            bias_line,
            "    Per sea-state bias stats (model coverage):",
            per_bin_hdr,
        ])
        for wi, (lo, hi, name) in enumerate(WAVE_HEIGHT_BINS):
            tr_n = int(train_acc.wbin_bias_hist[wi].sum())
            te_n = int(test_acc.wbin_bias_hist[wi].sum())
            ov   = bhattacharyya_overlap(
                train_acc.wbin_bias_hist[wi], test_acc.wbin_bias_hist[wi])
            tr_s = weighted_stats(train_acc.wbin_bias_hist[wi], _BIAS_CENTERS)
            te_s = weighted_stats(test_acc.wbin_bias_hist[wi], _BIAS_CENTERS)
            bin_label = f"[{_fmt_bin_edge(lo)}-{_fmt_bin_edge(hi)}m] {name}"
            line = (
                f"      {bin_label:<15} {tr_n:>12,} {te_n:>11,} {ov:>8.4f}"
                f" {tr_s['mean']:>9.4f} {te_s['mean']:>9.4f}"
                f" {tr_s['std']:>9.4f} {te_s['std']:>9.4f}"
                f" {tr_s['p50']:>9.4f} {te_s['p50']:>9.4f}"
            )
            print(line)
            stats_report_lines.append(line)

    interpretation_lines = [
        "",
        "INTERPRETATION GUIDE",
        "-" * 60,
        "  Plot 03 / plot 06C (VHM0 distribution):",
        "    overlap ~= 1, KL ~= 0  -> same sea-state regime in train & test",
        "    A slight shift is EXPECTED and healthy for year-holdout splits.",
        "    Identical distributions -> potential leakage.",
        "  Per-bin bias overlap table (in this report):",
        "    overlap -> 1 per bin: test bias values fall within the range the",
        "    model saw during training -> model has learned these conditions.",
        "    Low overlap in a bin -> that sea state is out-of-distribution.",
    ]
    stats_report_path = out / "00_bias_statistics_report.txt"

    if args.simple_hist_only:
        n_splits = len(accumulators)

        def _save_simple_hist_per_split(
            values_getter,
            bins: np.ndarray,
            xlabel: str,
            suptitle: str,
            filename: str,
            use_log_y: bool = False,
            xlim: tuple[float, float] | None = None,
        ) -> None:
            fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5), sharey=True)
            if n_splits == 1:
                axes = [axes]
            widths = np.diff(bins)
            for ax, (key, acc) in zip(axes, accumulators.items()):
                counts = values_getter(acc)
                ax.bar(
                    bins[:-1],
                    counts,
                    width=widths,
                    align="edge",
                    color=_split_color(key),
                    alpha=0.75,
                    edgecolor="none",
                )
                ax.set_title(f"{key} (n={int(counts.sum()):,})", fontsize=11, fontweight="bold")
                ax.set_xlabel(xlabel)
                ax.grid(True, alpha=0.3)
                if use_log_y:
                    ax.set_yscale("log")
                if xlim is not None:
                    ax.set_xlim(*xlim)
            axes[0].set_ylabel("Count")
            fig.suptitle(f"{suptitle}  [{args.region}]", fontsize=13, fontweight="bold")
            fig.tight_layout()
            path = out / filename
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved → {path}")

        # Keep the current simple plot unchanged.
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.bias_hist,
            bins=BIAS_HIST_BINS,
            xlabel="|corrected_VHM0 - VHM0| [m]",
            suptitle="Simple Bias Histogram Per Split (actual counts)",
            filename="01_simple_bias_hist_per_split.png",
        )

        # Additional bias views: log-y and x-zoom.
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.bias_hist,
            bins=BIAS_HIST_BINS,
            xlabel="|corrected_VHM0 - VHM0| [m]",
            suptitle="Simple Bias Histogram Per Split (actual counts, log-y)",
            filename="01c_simple_bias_hist_per_split_logy.png",
            use_log_y=True,
        )
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.bias_hist,
            bins=BIAS_HIST_BINS,
            xlabel="|corrected_VHM0 - VHM0| [m]",
            suptitle="Simple Bias Histogram Per Split (actual counts, x-zoom 0-2m)",
            filename="01d_simple_bias_hist_per_split_xzoom_0_2m.png",
            xlim=(0.0, 2.0),
        )

        # Same set for raw VHM0.
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.vhm0_hist,
            bins=VHM0_HIST_BINS,
            xlabel="Raw VHM0 [m]",
            suptitle="Simple Raw VHM0 Histogram Per Split (actual counts)",
            filename="03c_simple_vhm0_hist_per_split.png",
        )
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.vhm0_hist,
            bins=VHM0_HIST_BINS,
            xlabel="Raw VHM0 [m]",
            suptitle="Simple Raw VHM0 Histogram Per Split (actual counts, log-y)",
            filename="03d_simple_vhm0_hist_per_split_logy.png",
            use_log_y=True,
        )
        _save_simple_hist_per_split(
            values_getter=lambda acc: acc.vhm0_hist,
            bins=VHM0_HIST_BINS,
            xlabel="Raw VHM0 [m]",
            suptitle="Simple Raw VHM0 Histogram Per Split (actual counts, x-zoom 0-6m)",
            filename="03e_simple_vhm0_hist_per_split_xzoom_0_6m.png",
            xlim=(0.0, 6.0),
        )

        print(f"\nAll outputs → {out}")
        print("\nINTERPRETATION GUIDE")
        print("-" * 60)
        print("  Plot 03 / plot 06C (VHM0 distribution):")
        print("    overlap ≈ 1, KL ≈ 0  → same sea-state regime in train & test")
        print("    A slight shift is EXPECTED and healthy for year-holdout splits.")
        print("    Identical distributions → potential leakage.")
        print("  Per-bin bias overlap table (in this report):")
        print("    overlap → 1 per bin: test bias values fall within the range the")
        print("    model saw during training → model has learned these conditions.")
        print("    Low overlap in a bin → that sea state is out-of-distribution.")
        stats_report_lines.extend(interpretation_lines)
        stats_report_path.write_text("\n".join(stats_report_lines) + "\n", encoding="utf-8")
        print(f"Saved → {stats_report_path}")
        return

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    if not HAS_SCIPY:
        print("\nWARNING: scipy not available — skipping KDE plots.")
        print(f"\nAll outputs → {out}")
        print("\nINTERPRETATION GUIDE")
        print("-" * 60)
        print("  Plot 03 / plot 06C (VHM0 distribution):")
        print("    overlap ≈ 1, KL ≈ 0  → same sea-state regime in train & test")
        print("    A slight shift is EXPECTED and healthy for year-holdout splits.")
        print("    Identical distributions → potential leakage.")
        print("  Per-bin bias overlap table (in this report):")
        print("    overlap → 1 per bin: test bias values fall within the range the")
        print("    model saw during training → model has learned these conditions.")
        print("    Low overlap in a bin → that sea state is out-of-distribution.")
        stats_report_lines.extend(interpretation_lines)
        stats_report_path.write_text("\n".join(stats_report_lines) + "\n", encoding="utf-8")
        print(f"Saved → {stats_report_path}")
        return

    x_bias = np.linspace(0, 4,  400)
    x_vhm0 = np.linspace(0, 15, 300)

    # ------------------------------------------------------------------
    # Plot 1: Overall bias distribution — all splits overlaid
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in accumulators.items():
        _kde_plot(ax, acc.bias_hist, _BIAS_CENTERS, x_bias,
                  _split_color(key), f"{key}  (n={acc.n_pixels:,})")
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
    # Plot 1B: Overall bias distribution — empirical histogram density
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in accumulators.items():
        _hist_density_plot(
            ax, acc.bias_hist, BIAS_HIST_BINS, _split_color(key), f"{key}  (n={acc.n_pixels:,})"
        )
    ax.set_xlabel("|corrected_VHM0 - VHM0|  [m]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Uncorrected Bias Distribution (empirical) — train vs test  [{args.region}]",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01b_bias_distribution_overall_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '01b_bias_distribution_overall_hist.png'}")

    # ------------------------------------------------------------------
    # Plot 2: Statistics panels
    # ------------------------------------------------------------------
    n_splits = len(accumulators)
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5))
    if n_splits == 1:
        axes = [axes]
    for ax, (key, acc) in zip(axes, accumulators.items()):
        txt = f"{key.upper()}\n\n" + _stats_text(acc.stats())
        ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=_split_color(key), alpha=0.15))
        ax.set_title(key, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"Bias |corrected_VHM0 − VHM0| statistics  [{args.region}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "02_bias_statistics_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '02_bias_statistics_table.png'}")

    # ------------------------------------------------------------------
    # Plot 3: Raw VHM0 distribution — data leakage check
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in accumulators.items():
        _kde_plot(ax, acc.vhm0_hist, _VHM0_CENTERS, x_vhm0,
                  _split_color(key), f"{key}  (n={acc.n_pixels:,})")
    ax.set_xlabel("Raw VHM0 [m]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Raw VHM0 Distribution — leakage check  [{args.region}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    note = ("Leakage signal: if train ≈ test here the split may have leaked.\n"
            "A distribution shift is expected and healthy for year-holdout splits.")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out / "03_vhm0_distribution_leakage_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '03_vhm0_distribution_leakage_check.png'}")

    # ------------------------------------------------------------------
    # Plot 3B: Raw VHM0 distribution — empirical histogram density
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, acc in accumulators.items():
        _hist_density_plot(
            ax, acc.vhm0_hist, VHM0_HIST_BINS, _split_color(key), f"{key}  (n={acc.n_pixels:,})"
        )
    ax.set_xlabel("Raw VHM0 [m]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Raw VHM0 Distribution (empirical) — leakage check  [{args.region}]",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    note = ("Leakage signal: if train ≈ test here the split may have leaked.\n"
            "A distribution shift is expected and healthy for year-holdout splits.")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out / "03b_vhm0_distribution_leakage_check_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '03b_vhm0_distribution_leakage_check_hist.png'}")

    # ------------------------------------------------------------------
    # Plot 4 & 5: Per sea-state bin (empirical only, one figure per split)
    # ------------------------------------------------------------------
    n_wb = len(WAVE_HEIGHT_BINS)
    for split_key, split_acc in accumulators.items():
        # 4: bias empirical density per bin for one split
        fig, axes = plt.subplots(1, n_wb, figsize=(4 * n_wb, 5), sharey=False)
        if n_wb == 1:
            axes = [axes]
        for ax, (wi, (lo, hi, name)) in zip(axes, enumerate(WAVE_HEIGHT_BINS)):
            h = split_acc.wbin_bias_hist[wi]
            _hist_density_plot(
                ax, h, BIAS_HIST_BINS, _split_color(split_key), f"{split_key} (n={int(h.sum()):,})"
            )
            ax.set_title(f"{name}\n[{_fmt_bin_edge(lo)}–{_fmt_bin_edge(hi)}m]", fontsize=9)
            ax.set_xlabel("|bias| [m]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Bias distribution per sea-state (empirical) — {split_key}  [{args.region}]",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        fname = f"04_bias_per_seabin_hist_{split_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

        # 5: VHM0 empirical density per bin for one split
        fig, axes = plt.subplots(1, n_wb, figsize=(4 * n_wb, 5), sharey=False)
        if n_wb == 1:
            axes = [axes]
        for ax, (wi, (lo, hi, name)) in zip(axes, enumerate(WAVE_HEIGHT_BINS)):
            h = split_acc.wbin_vhm0_hist[wi]
            _hist_density_plot(
                ax, h, VHM0_HIST_BINS, _split_color(split_key), f"{split_key} (n={int(h.sum()):,})"
            )
            ax.set_title(f"{name}\n[{_fmt_bin_edge(lo)}–{_fmt_bin_edge(hi)}m]", fontsize=9)
            ax.set_xlabel("VHM0 [m]", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"VHM0 distribution per sea-state (empirical) — {split_key}  [{args.region}]",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        fname = f"05_vhm0_per_seabin_hist_{split_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

    # ------------------------------------------------------------------
    # Plot 6B: Absolute bias per sea-state bin across splits
    # (stationarity-style bar chart, adapted to train/test splits)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_wb), 6))
    x = np.arange(n_wb)
    split_items = list(accumulators.items())
    bar_w = 0.8 / max(1, len(split_items))
    for si, (split_key, split_acc) in enumerate(split_items):
        vals = []
        for wi in range(n_wb):
            h = split_acc.wbin_bias_hist[wi]
            if h.sum() == 0:
                vals.append(np.nan)
            else:
                vals.append(weighted_stats(h, _BIAS_CENTERS)["mean"])
        offset = (si - (len(split_items) - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            vals,
            width=bar_w,
            label=split_key,
            color=_split_color(split_key),
            alpha=0.80,
        )
    bin_labels = [f"{_fmt_bin_edge(lo)}-{_fmt_bin_edge(hi)}m" for lo, hi, _ in WAVE_HEIGHT_BINS]
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right")
    ax.set_ylabel("Mean absolute bias [m]")
    ax.set_xlabel("Sea-state bin (raw VHM0)")
    ax.set_title(
        f"Absolute bias per sea-state bin across splits  [{args.region}]",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06b_abs_bias_per_bin_by_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '06b_abs_bias_per_bin_by_split.png'}")

    # ------------------------------------------------------------------
    # Plot 6C: Raw VHM0 per sea-state bin across splits
    # (same grouped-bar style as 06B)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * n_wb), 6))
    x = np.arange(n_wb)
    split_items = list(accumulators.items())
    bar_w = 0.8 / max(1, len(split_items))
    for si, (split_key, split_acc) in enumerate(split_items):
        vals = []
        for wi in range(n_wb):
            h = split_acc.wbin_vhm0_hist[wi]
            if h.sum() == 0:
                vals.append(np.nan)
            else:
                vals.append(weighted_stats(h, _VHM0_CENTERS)["mean"])
        offset = (si - (len(split_items) - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            vals,
            width=bar_w,
            label=split_key,
            color=_split_color(split_key),
            alpha=0.80,
        )
    bin_labels = [f"{_fmt_bin_edge(lo)}-{_fmt_bin_edge(hi)}m" for lo, hi, _ in WAVE_HEIGHT_BINS]
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right")
    ax.set_ylabel("Mean raw VHM0 [m]")
    ax.set_xlabel("Sea-state bin (raw VHM0)")
    ax.set_title(
        f"Raw VHM0 per sea-state bin across splits  [{args.region}]",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "06c_vhm0_per_bin_by_split.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '06c_vhm0_per_bin_by_split.png'}")

    # ------------------------------------------------------------------
    # Plot 06D: Statistics comparison across all splits (bias + raw VHM0)
    # ------------------------------------------------------------------
    stat_keys_all = ["mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99"]
    split_items = list(accumulators.items())
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, title, stats_getter, x_label in [
        (axes[0], "Bias statistics", lambda acc: acc.stats(), "Bias [m]"),
        (axes[1], "Raw VHM0 statistics", lambda acc: acc.vhm0_stats(), "Raw VHM0 [m]"),
    ]:
        y = np.arange(len(stat_keys_all))
        bar_h = 0.8 / max(1, len(split_items))
        for si, (split_key, split_acc) in enumerate(split_items):
            s = stats_getter(split_acc)
            offset = (si - (len(split_items) - 1) / 2) * bar_h
            ax.barh(
                y + offset,
                [s[k] for k in stat_keys_all],
                height=bar_h,
                color=_split_color(split_key),
                alpha=0.75,
                label=split_key,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(stat_keys_all)
        ax.set_xlabel(x_label)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Statistics comparison across all splits  [{args.region}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "06d_statistics_comparison_all_splits.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out / '06d_statistics_comparison_all_splits.png'}")

    # ------------------------------------------------------------------
    # Plot 6: 3-panel summary per test year
    # ------------------------------------------------------------------
    for test_key, test_acc in ((k, v) for k, v in accumulators.items() if k != "train"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel A: bias KDE
        ax = axes[0]
        _kde_plot(ax, train_acc.bias_hist, _BIAS_CENTERS, x_bias,
                  _split_color("train"),    f"train  (n={train_acc.n_pixels:,})")
        _kde_plot(ax, test_acc.bias_hist,  _BIAS_CENTERS, x_bias,
                  _split_color(test_key), f"{test_key}  (n={test_acc.n_pixels:,})")
        ax.set_xlabel("|bias| [m]")
        ax.set_ylabel("Density")
        ax.set_title("Bias distribution")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Panel B: stats horizontal bar chart
        ax = axes[1]
        s_tr = train_acc.stats()
        s_te = test_acc.stats()
        stat_keys = ["mean", "std", "p25", "p50", "p75", "p90", "p95", "p99"]
        y_pos = np.arange(len(stat_keys))
        ax.barh(y_pos + 0.2, [s_tr[k] for k in stat_keys], height=0.35,
                color=_split_color("train"),    alpha=0.7, label="train")
        ax.barh(y_pos - 0.2, [s_te[k] for k in stat_keys], height=0.35,
                color=_split_color(test_key), alpha=0.7, label=test_key)
        ax.set_yticks(y_pos); ax.set_yticklabels(stat_keys)
        ax.set_xlabel("Bias [m]")
        ax.set_title("Statistics comparison")
        ax.legend(fontsize=9); ax.grid(True, axis="x", alpha=0.3)

        # Panel C: VHM0 KDE (leakage check)
        ax = axes[2]
        _kde_plot(ax, train_acc.vhm0_hist, _VHM0_CENTERS, x_vhm0,
                  _split_color("train"),    f"train  (n={train_acc.n_pixels:,})")
        _kde_plot(ax, test_acc.vhm0_hist,  _VHM0_CENTERS, x_vhm0,
                  _split_color(test_key), f"{test_key}  (n={test_acc.n_pixels:,})")
        kl_v = kl_divergence(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ov_v = bhattacharyya_overlap(train_acc.vhm0_hist, test_acc.vhm0_hist)
        ax.set_xlabel("Raw VHM0 [m]")
        ax.set_ylabel("Density")
        ax.set_title(f"VHM0 distribution (leakage check)\nKL={kl_v:.3f}  overlap={ov_v:.3f}")
        ax.legend(); ax.grid(True, alpha=0.3)

        fig.suptitle(f"Summary: train vs {test_key}  [{args.region}]",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fname = f"06_summary_train_vs_{test_key}.png"
        fig.savefig(out / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out / fname}")

    # ------------------------------------------------------------------
    # Print interpretation guide
    # ------------------------------------------------------------------
    print(f"\nAll outputs → {out}")
    print("\nINTERPRETATION GUIDE")
    print("-" * 60)
    print("  Plot 03 / plot 06C (VHM0 distribution):")
    print("    overlap ≈ 1, KL ≈ 0  → same sea-state regime in train & test")
    print("    A slight shift is EXPECTED and healthy for year-holdout splits.")
    print("    Identical distributions → potential leakage.")
    print("  Per-bin bias overlap table (in this report):")
    print("    overlap → 1 per bin: test bias values fall within the range the")
    print("    model saw during training → model has learned these conditions.")
    print("    Low overlap in a bin → that sea state is out-of-distribution.")
    stats_report_lines.extend(interpretation_lines)
    stats_report_path.write_text("\n".join(stats_report_lines) + "\n", encoding="utf-8")
    print(f"Saved → {stats_report_path}")


if __name__ == "__main__":
    main()
