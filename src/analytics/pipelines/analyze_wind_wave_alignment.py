#!/usr/bin/env python3
"""
Wind-Wave Alignment Correlation Analysis

Analyzes cos(θ−φ) (θ: wind direction, φ: wave direction) as a potential bias-correction
feature. This term captures the alignment between wind and waves:
  - cos(θ−φ) ~ +1: wind and waves aligned (wind sea, active growth) — more common in Mediterranean
  - cos(θ−φ) ~ 0:  mixed/transitional sea state
  - cos(θ−φ) ~ −1: wind opposing wave propagation (swell/decay) — more common in Atlantic

Usage:
    python analyze_wind_wave_alignment.py --year 2018
    python analyze_wind_wave_alignment.py --year 2018 --max-files 100 --output-dir outputs/wind_wave_alignment
"""

import argparse
import glob
import os
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12

GIBRALTAR_LON = -5.5


# Features to load for correlation analysis (model input + raw + corrected)
# Raw parquet has: VHM0, WSPD, VTM02, latitude, longitude
# Augmented parquet adds: U10, V10, wave_dir_sin/cos, sin/cos hour/doy/month, lat_norm, lon_norm
FEATURE_COLS = [
    "VHM0",
    "WSPD",
    "VTM02",
    "U10",
    "V10",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
    "sin_month",
    "cos_month",
    "lat_norm",
    "lon_norm",
    "wave_dir_sin",
    "wave_dir_cos",
    "latitude",
    "longitude",
    "corrected_VHM0",
    "corrected_VTM02",
]


def load_wind_wave_data(
    input_dir: str,
    year: int,
    max_files: int | None = None,
    load_all_features: bool = False,
    subsample_rows: float | None = None,
    min_vhm0: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]:
    """
    Load WDIR, VMDR, VHM0, corrected_VHM0, latitude, longitude from S3 parquet files.
    Optionally load all features for correlation analysis.

    min_vhm0: Exclude pixels with VHM0 < min_vhm0 (likely land). Default 0.01 m.

    Returns:
        cos_alignment, bias, latitude, longitude, region (1=Med, 0=Atlantic),
        features_dict (or None if load_all_features=False)
    """
    file_pattern = f"WAVEAN{year}*.parquet"

    if input_dir.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        files = sorted(fs.glob(f"{input_dir.rstrip('/')}/{file_pattern}"))
        files = [p if p.startswith("s3://") else f"s3://{p}" for p in files]
        is_s3 = True
    else:
        files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        is_s3 = False

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise ValueError(f"No files found: {input_dir}/{file_pattern}")

    print(f"Loading {len(files)} parquet files from {input_dir}")
    print(f"  Keeping only sea pixels (VHM0 >= {min_vhm0} m)")
    if subsample_rows is not None and subsample_rows < 1.0:
        print(f"  Subsampling {subsample_rows*100:.1f}% of sea pixels per file (for memory)")

    cos_list, bias_list, lat_list, lon_list = [], [], [], []
    features_list: dict[str, list] = {f: [] for f in FEATURE_COLS} if load_all_features else {}

    for file_path in tqdm(files, desc="Loading"):
        try:
            if is_s3:
                with fsspec.open(file_path, "rb") as fh:
                    table = pq.read_table(fh)
            else:
                table = pq.read_table(file_path)

            schema_cols = set(table.column_names)
            # Compute sin_month, cos_month from time if missing (parquet may lack augmentation)
            time_col = "timestamp" if "timestamp" in schema_cols else "time"
            if time_col in schema_cols and ("sin_month" not in schema_cols or "cos_month" not in schema_cols):
                time_arr = table.column(time_col)
                # Extract month: handle datetime-like or numpy datetime64
                def _month(t):
                    if hasattr(t, "month"):
                        return t.month
                    s = str(np.datetime64(t))
                    return int(s[5:7]) if len(s) >= 7 else 1

                months = np.array([_month(t) for t in time_arr], dtype=np.float32)
                sin_month_arr = np.sin(2 * np.pi * months / 12.0).astype(np.float32)
                cos_month_arr = np.cos(2 * np.pi * months / 12.0).astype(np.float32)
                if "sin_month" not in schema_cols:
                    table = table.append_column("sin_month", pa.array(sin_month_arr))
                if "cos_month" not in schema_cols:
                    table = table.append_column("cos_month", pa.array(cos_month_arr))
                schema_cols = set(table.column_names)
            wdir = np.deg2rad(table.column("WDIR").to_numpy())
            vmdr = np.deg2rad(table.column("VMDR").to_numpy())
            vhm0 = table.column("VHM0").to_numpy()
            corrected = table.column("corrected_VHM0").to_numpy()
            lat = table.column("latitude").to_numpy()
            lon = table.column("longitude").to_numpy()

            # cos(θ−φ) = cos(θ)cos(φ) + sin(θ)sin(φ)
            cos_align = np.cos(wdir) * np.cos(vmdr) + np.sin(wdir) * np.sin(vmdr)
            bias = corrected - vhm0

            valid = (
                np.isfinite(cos_align)
                & np.isfinite(bias)
                & np.isfinite(lat)
                & np.isfinite(lon)
                & (vhm0 >= min_vhm0)
                & (corrected >= min_vhm0)
            )

            idx = np.where(valid)[0]
            if subsample_rows is not None and subsample_rows < 1.0 and len(idx) > 0:
                rng = np.random.default_rng(42)
                n_keep = max(1, int(len(idx) * subsample_rows))
                idx = rng.choice(idx, size=n_keep, replace=False)

            cos_list.append(cos_align[idx])
            bias_list.append(bias[idx])
            lat_list.append(lat[idx])
            lon_list.append(lon[idx])

            if load_all_features:
                for f in FEATURE_COLS:
                    if f in schema_cols:
                        arr = table.column(f).to_numpy()[idx]
                        features_list[f].append(arr)

        except Exception as e:
            print(f"  ⚠ Skipped {os.path.basename(file_path)}: {e}")

    cos_align = np.concatenate(cos_list)
    bias = np.concatenate(bias_list)
    lat = np.concatenate(lat_list)
    lon = np.concatenate(lon_list)

    region = (lon >= GIBRALTAR_LON).astype(int)  # 1=Med, 0=Atlantic

    features_dict = None
    if load_all_features:
        features_dict = {}
        for f in FEATURE_COLS:
            if features_list[f]:
                features_dict[f] = np.concatenate(features_list[f])
        if not features_dict:
            features_dict = None

    print(f"  Loaded {len(cos_align):,} valid samples")
    if features_dict:
        print(f"  Features for correlation: {list(features_dict.keys())}")
    return cos_align, bias, lat, lon, region, features_dict


def plot_hexbin_scatter(
    cos_align: np.ndarray,
    bias: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
):
    """Hexbin scatter of cos(θ−φ) vs bias with correlation."""
    r_pearson, p_pearson = pearsonr(cos_align, bias)
    r_spearman, p_spearman = spearmanr(cos_align, bias)

    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(
        cos_align, bias, gridsize=80, cmap="viridis", mincnt=1, edgecolors="none"
    )
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel(r"$\cos(\theta - \varphi)$ (wind-wave alignment)")
    ax.set_ylabel("Bias (corrected VHM0 − VHM0) [m]")
    ax.set_title(
        f"Wind-Wave Alignment vs Bias{title_suffix}\n"
        f"Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), "
        f"Spearman ρ={r_spearman:.3f} (p={p_spearman:.2e})"
    )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_bias_by_regime(
    cos_align: np.ndarray,
    bias: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
):
    """Box/violin plot of bias by cos(θ−φ) regime."""
    bins = [-1.01, -0.5, 0, 0.5, 1.01]
    labels = [
        "Opposing\n(swell/decay)",
        "Cross-swell",
        "Mixed",
        "Aligned\n(wind sea)",
    ]
    bin_idx = np.digitize(cos_align, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)

    # Collect bias arrays per regime
    data_by_regime = [bias[bin_idx == i] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(
        data_by_regime,
        positions=range(len(labels)),
        showmeans=True,
        showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_xlabel(r"$\cos(\theta - \varphi)$ regime")
    ax.set_ylabel("Bias [m]")
    ax.set_title(f"Bias Distribution by Wind-Wave Regime{title_suffix}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_regional_comparison(
    cos_align: np.ndarray,
    bias: np.ndarray,
    region: np.ndarray,
    output_path: Path,
):
    """Side-by-side hexbin for Mediterranean vs Atlantic."""
    med_mask = region == 1
    atl_mask = region == 0

    r_med, p_med = pearsonr(cos_align[med_mask], bias[med_mask]) if med_mask.sum() > 10 else (np.nan, np.nan)
    r_atl, p_atl = pearsonr(cos_align[atl_mask], bias[atl_mask]) if atl_mask.sum() > 10 else (np.nan, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, mask, name, r, p in [
        (axes[0], med_mask, "Mediterranean", r_med, p_med),
        (axes[1], atl_mask, "Atlantic", r_atl, p_atl),
    ]:
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        hb = ax.hexbin(
            cos_align[mask],
            bias[mask],
            gridsize=60,
            cmap="viridis",
            mincnt=1,
            edgecolors="none",
        )
        ax.set_xlabel(r"$\cos(\theta - \varphi)$")
        ax.set_ylabel("Bias [m]")
        ax.set_title(f"{name}\nn={mask.sum():,}, Pearson r={r:.3f}" + (f" (p={p:.2e})" if np.isfinite(p) else ""))
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)
        plt.colorbar(hb, ax=ax, label="Count")

    fig.suptitle("Wind-Wave Alignment vs Bias by Region (Gibraltar lon=-5.5°)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_correlation_with_features(
    cos_align: np.ndarray,
    bias: np.ndarray,
    features_dict: dict[str, np.ndarray],
    output_path: Path,
    title_suffix: str = "",
):
    """Bar chart and heatmap of cos(θ−φ) correlation with all features + bias."""
    # Build combined dict: cos_alignment + bias + all features
    all_vars = {"bias": bias, **features_dict}
    # Common valid mask
    valid = np.isfinite(cos_align)
    for arr in all_vars.values():
        valid &= np.isfinite(arr)

    cos_valid = cos_align[valid]
    n_valid = valid.sum()
    if n_valid < 10:
        print(f"  ⚠ Skipping correlation plot: only {n_valid} valid samples")
        return

    # Compute correlations
    corr_pearson = {}
    corr_spearman = {}
    for name, arr in all_vars.items():
        arr_valid = arr[valid]
        r_p, _ = pearsonr(cos_valid, arr_valid)
        r_s, _ = spearmanr(cos_valid, arr_valid)
        corr_pearson[name] = r_p
        corr_spearman[name] = r_s

    # Sort by absolute Pearson correlation
    names = sorted(corr_pearson.keys(), key=lambda x: abs(corr_pearson[x]), reverse=True)
    pearsons = [corr_pearson[n] for n in names]
    spearmans = [corr_spearman[n] for n in names]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Bar chart - Pearson (bias=red, corrected_*=green, rest=steelblue)
    def _color(n: str) -> str:
        if n == "bias":
            return "#C62828"
        if n.startswith("corrected_"):
            return "#2E7D32"
        return "steelblue"

    colors = [_color(n) for n in names]
    x = range(len(names))
    bars = axes[0].bar(x, pearsons, color=colors, alpha=0.8, edgecolor="black")
    axes[0].axhline(0, color="gray", linestyle="-", linewidth=1)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title(f"cos(θ−φ) vs Features (Pearson){title_suffix}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")
    for _i, (bar, v) in enumerate(zip(bars, pearsons, strict=False)):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02 if v >= 0 else v - 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    # Bar chart - Spearman
    bars = axes[1].bar(x, spearmans, color=colors, alpha=0.8, edgecolor="black")
    axes[1].axhline(0, color="gray", linestyle="-", linewidth=1)
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title(f"cos(θ−φ) vs Features (Spearman){title_suffix}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")
    for _i, (bar, v) in enumerate(zip(bars, spearmans, strict=False)):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02 if v >= 0 else v - 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle(
        f"Correlation of cos(θ−φ) with Features (n={n_valid:,})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def plot_cos_distribution(
    cos_align: np.ndarray,
    region: np.ndarray,
    output_path: Path,
):
    """Histogram of cos(θ−φ) overall and by region."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].hist(cos_align, bins=80, color="steelblue", alpha=0.8, edgecolor="black")
    axes[0].set_xlabel(r"$\cos(\theta - \varphi)$")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].grid(True, alpha=0.3)

    med_mask = region == 1
    atl_mask = region == 0

    if med_mask.sum() > 0:
        axes[1].hist(
            cos_align[med_mask],
            bins=80,
            color="darkgreen",
            alpha=0.7,
            label=f"Med (n={med_mask.sum():,})",
        )
    axes[1].set_xlabel(r"$\cos(\theta - \varphi)$")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Mediterranean")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[1].grid(True, alpha=0.3)

    if atl_mask.sum() > 0:
        axes[2].hist(
            cos_align[atl_mask],
            bins=80,
            color="darkblue",
            alpha=0.7,
            label=f"Atlantic (n={atl_mask.sum():,})",
        )
    axes[2].set_xlabel(r"$\cos(\theta - \varphi)$")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Atlantic")
    axes[2].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(r"Distribution of $\cos(\theta - \varphi)$", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Wind-Wave Alignment (cos(θ−φ)) Correlation Analysis"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="s3://medwav-dev-data/parquet/hourly/year=2018",
        help="S3 or local directory with parquet files",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2018,
        help="Year to analyze (used for file pattern if input-dir has year=YYYY)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max number of parquet files to load (for quick tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/wind_wave_alignment",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--subsample-rows",
        type=float,
        default=None,
        help="Subsample fraction of rows per file (0.01–1.0). Use e.g. 0.01 for 365 files to avoid OOM.",
    )
    parser.add_argument(
        "--min-vhm0",
        type=float,
        default=0.01,
        help="Minimum VHM0 [m] to keep (excludes land/near-zero). Default 0.01.",
    )
    args = parser.parse_args()

    # If input-dir ends with year=YYYY, use that; else append year
    input_dir = args.input_dir
    if "year=" not in input_dir:
        input_dir = f"{input_dir.rstrip('/')}/year={args.year}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wind-Wave Alignment Correlation Analysis")
    print("=" * 60)

    cos_align, bias, lat, lon, region, features_dict = load_wind_wave_data(
        input_dir,
        args.year,
        args.max_files,
        load_all_features=True,
        subsample_rows=args.subsample_rows,
        min_vhm0=args.min_vhm0,
    )

    if len(cos_align) == 0:
        print("No valid data. Exiting.")
        return

    # Summary stats
    r_p, p_p = pearsonr(cos_align, bias)
    r_s, p_s = spearmanr(cos_align, bias)
    print(f"\nOverall correlation (n={len(cos_align):,}):")
    print(f"  Pearson:  r={r_p:.4f}, p={p_p:.2e}")
    print(f"  Spearman: ρ={r_s:.4f}, p={p_s:.2e}")

    med_n = (region == 1).sum()
    atl_n = (region == 0).sum()
    print(f"\nRegional split: Mediterranean n={med_n:,}, Atlantic n={atl_n:,}")

    # Plots
    plot_hexbin_scatter(
        cos_align, bias,
        output_dir / f"cos_wind_wave_vs_bias_hexbin_{args.year}.png",
        title_suffix=f" (year={args.year})",
    )
    plot_bias_by_regime(
        cos_align, bias,
        output_dir / f"bias_by_regime_{args.year}.png",
        title_suffix=f" (year={args.year})",
    )
    plot_regional_comparison(
        cos_align, bias, region,
        output_dir / f"cos_wind_wave_regional_{args.year}.png",
    )
    plot_cos_distribution(
        cos_align, region,
        output_dir / f"cos_wind_wave_distribution_{args.year}.png",
    )

    if features_dict:
        plot_correlation_with_features(
            cos_align, bias, features_dict,
            output_dir / f"cos_wind_wave_correlation_features_{args.year}.png",
            title_suffix=f" (year={args.year})",
        )
    else:
        print("  ⚠ No feature columns found in parquet; skipping correlation-with-features plot")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
