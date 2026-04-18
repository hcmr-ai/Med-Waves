"""
Bias Stationarity Diagnostic
=============================
Answers: How much of the reanalysis bias (corrected - raw) is stationary
(same spatial pattern every year) vs non-stationary (year-specific)?

Outputs:
  1. Per-year mean bias maps + grand mean
  2. Variance decomposition: stationary vs year-specific fraction at each pixel
  3. Pairwise spatial correlations between annual bias maps
  4. Conditional analysis by wave-height bin
  5. Seasonal decomposition
  6. Summary statistics printed to console

Usage:
  poetry run python scripts/bias_stationarity_diagnostic.py \
    --data_path /mnt/blobstorage/preprocessed_extended_subsampled_step_5/
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_year_month(filename):
    name = Path(filename).stem
    marker = "WAVEAN"
    idx = name.find(marker)
    if idx != -1 and len(name) >= idx + 12:
        return int(name[idx+6:idx+10]), int(name[idx+10:idx+12])
    return None, None


def load_bias_fields(file_path, vhm0_idx, corrected_vhm0_idx, vtm02_idx=None, corrected_vtm02_idx=None):
    """Load one .pt file, return bias fields for all 24 hours.

    Returns:
        vhm0_bias: (24, H, W) - NaN on land
        raw_vhm0:  (24, H, W) - for binning
        vtm02_bias: (24, H, W) or None
    """
    data = torch.load(file_path, map_location="cpu")
    tensor = data["tensor"]  # (24, H, W, C)

    raw_vhm0 = tensor[..., vhm0_idx].numpy()            # (24, H, W)
    corrected_vhm0 = tensor[..., corrected_vhm0_idx].numpy()
    vhm0_bias = corrected_vhm0 - raw_vhm0                # (24, H, W)

    vtm02_bias = None
    if vtm02_idx is not None and corrected_vtm02_idx is not None:
        raw_vtm02 = tensor[..., vtm02_idx].numpy()
        corrected_vtm02 = tensor[..., corrected_vtm02_idx].numpy()
        vtm02_bias = corrected_vtm02 - raw_vtm02

    return vhm0_bias, raw_vhm0, vtm02_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--file_pattern", default="WAVEAN*.pt")
    parser.add_argument("--output_dir", default="/mnt/blobstorage/diagnostics/bias_stationarity")
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["all", "atlantic", "mediterranean"],
        help="Region filter (Gibraltar + Bay of Biscay logic)",
    )
    parser.add_argument("--max_files_per_year", type=int, default=None,
                        help="Limit files per year for quick runs (None=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.region
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files and group by year
    all_files = sorted(glob.glob(f"{args.data_path}/{args.file_pattern}"))
    print(f"Found {len(all_files)} total files")

    files_by_year = defaultdict(list)
    for f in all_files:
        year, month = parse_year_month(f)
        if year:
            files_by_year[year].append(f)

    years = sorted(files_by_year.keys())
    print(f"Years: {years}")
    for y in years:
        print(f"  {y}: {len(files_by_year[y])} files")

    # Get feature indices from first file
    data0 = torch.load(all_files[0], map_location="cpu")
    feature_cols = data0["feature_cols"]
    tensor_shape = data0["tensor"].shape
    print(f"\nTensor shape per file: {tensor_shape}  (24hours, H, W, C)")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    vhm0_idx = feature_cols.index("VHM0")
    corrected_vhm0_idx = feature_cols.index("corrected_VHM0")
    vtm02_idx = feature_cols.index("VTM02") if "VTM02" in feature_cols else None
    corrected_vtm02_idx = feature_cols.index("corrected_VTM02") if "corrected_VTM02" in feature_cols else None
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")

    H, W = tensor_shape[1], tensor_shape[2]

    # Region mask from first file coordinates (same logic as dataset/eval pipelines)
    if args.region == "all":
        region_mask = np.ones((H, W), dtype=bool)
    else:
        t0 = data0["tensor"][0]  # (H, W, C)
        lat = t0[..., lat_idx].numpy()
        lon = t0[..., lon_idx].numpy()

        GIBRALTAR_LON = -5.5
        BISCAY_LAT = 43.0
        BISCAY_LON = 0.0
        biscay = (lat > BISCAY_LAT) & (lon < BISCAY_LON)

        if args.region == "atlantic":
            region_mask = (lon < GIBRALTAR_LON) | biscay
        else:  # mediterranean
            region_mask = (lon >= GIBRALTAR_LON) & ~biscay

        region_mask = region_mask & ~np.isnan(lat) & ~np.isnan(lon)

    print(f"Region filter: {args.region}")
    print(
        f"Region pixels kept: {int(region_mask.sum()):,}/{region_mask.size:,} "
        f"({100.0 * region_mask.mean():.1f}%)"
    )

    # ================================================================
    # PASS 1: Accumulate per-year running statistics
    # We accumulate: sum, sum_of_squares, count per pixel per year
    # This avoids loading all data into memory
    # ================================================================
    print("\n=== Pass 1: Accumulating per-year bias statistics ===")

    # For VHM0 bias
    year_sum = {}       # year -> (H, W) running sum of bias
    year_sq_sum = {}    # year -> (H, W) running sum of bias^2
    year_count = {}     # year -> (H, W) count of valid (non-NaN) values
    year_raw_sum = {}   # year -> (H, W) running sum of raw VHM0

    # Per-bin accumulators: {year: {bin_name: {"sum": ..., "count": ...}}}
    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float("inf")]
    bin_names = [
        "0-1m", "1-2m", "2-3m", "3-4m", "4-5m", "5-6m", "6-7m",
        "7-8m", "8-9m", "9-10m", "10-11m", "11-12m", "12m+",
    ]
    bin_labels = [
        "0.0-1.0m", "1.0-2.0m", "2.0-3.0m", "3.0-4.0m", "4.0-5.0m",
        "5.0-6.0m", "6.0-7.0m", "7.0-8.0m", "8.0-9.0m", "9.0-10.0m",
        "10.0-11.0m", "11.0-12.0m", "12.0m+",
    ]
    year_bin_sum = {}
    year_bin_sq_sum = {}
    year_bin_abs_sum = {}
    year_bin_count = {}

    # Per-season accumulators: {year: {season: {"sum": ..., "count": ...}}}
    def month_to_season(m):
        if m in (12, 1, 2): return "winter"
        if m in (3, 4, 5): return "spring"
        if m in (6, 7, 8): return "summer"
        return "autumn"

    year_season_sum = {}
    year_season_count = {}

    # Relative bias accumulators: (corrected - raw) / (raw + eps), per bin per year
    REL_EPS = 0.05  # 5 cm floor to avoid division by near-zero in calm seas
    year_relbin_sum = {}
    year_relbin_count = {}

    # Per-pixel absolute bias accumulator (for spatial MAE maps)
    year_abs_sum = {}

    # Global VHM0 histograms for distribution plots (Plot 10)
    VHM0_HIST_BINS = np.linspace(0, 15, 301)   # 300 bins, 0.05 m resolution
    _n_hist = len(VHM0_HIST_BINS) - 1
    vhm0_hist_raw       = np.zeros(_n_hist, dtype=np.float64)
    vhm0_hist_corrected = np.zeros(_n_hist, dtype=np.float64)

    for year in years:
        year_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_sq_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_count[year] = np.zeros((H, W), dtype=np.float64)
        year_raw_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_bin_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_bin_sq_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_bin_abs_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_bin_count[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_season_sum[year] = {s: np.zeros((H, W), dtype=np.float64) for s in ["winter", "spring", "summer", "autumn"]}
        year_season_count[year] = {s: np.zeros((H, W), dtype=np.float64) for s in ["winter", "spring", "summer", "autumn"]}
        year_relbin_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_relbin_count[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_abs_sum[year] = np.zeros((H, W), dtype=np.float64)

    for year in years:
        file_list = files_by_year[year]
        if args.max_files_per_year:
            file_list = file_list[:args.max_files_per_year]

        for fi, fpath in enumerate(file_list):
            _, month = parse_year_month(fpath)
            season = month_to_season(month) if month else "unknown"

            vhm0_bias, raw_vhm0, _ = load_bias_fields(
                fpath, vhm0_idx, corrected_vhm0_idx, vtm02_idx, corrected_vtm02_idx
            )
            # vhm0_bias: (24, H, W), raw_vhm0: (24, H, W)

            for hour in range(24):
                bias_hw = vhm0_bias[hour]   # (H, W), NaN on land
                raw_hw = raw_vhm0[hour]     # (H, W)
                valid = ~np.isnan(bias_hw) & region_mask

                # Overall per-year
                bias_filled = np.where(valid, bias_hw, 0.0)
                year_sum[year] += bias_filled
                year_sq_sum[year] += np.where(valid, bias_hw**2, 0.0)
                year_abs_sum[year] += np.where(valid, np.abs(bias_hw), 0.0)
                year_count[year] += valid.astype(np.float64)
                year_raw_sum[year] += np.where(valid, raw_hw, 0.0)

                # Per-bin (absolute and relative)
                rel_bias_hw = np.where(valid, bias_hw / (np.abs(raw_hw) + REL_EPS), 0.0)
                for bi in range(len(bin_names)):
                    lo, hi = bin_edges[bi], bin_edges[bi+1]
                    in_bin = valid & (raw_hw >= lo) & (raw_hw < hi)
                    year_bin_sum[year][bin_names[bi]] += np.where(in_bin, bias_hw, 0.0)
                    year_bin_sq_sum[year][bin_names[bi]] += np.where(in_bin, bias_hw**2, 0.0)
                    year_bin_abs_sum[year][bin_names[bi]] += np.where(in_bin, np.abs(bias_hw), 0.0)
                    year_bin_count[year][bin_names[bi]] += in_bin.astype(np.float64)
                    year_relbin_sum[year][bin_names[bi]] += np.where(in_bin, rel_bias_hw, 0.0)
                    year_relbin_count[year][bin_names[bi]] += in_bin.astype(np.float64)

                # Per-season
                if season in year_season_sum[year]:
                    year_season_sum[year][season] += bias_filled
                    year_season_count[year][season] += valid.astype(np.float64)

                # VHM0 distribution histograms
                raw_valid = raw_hw[valid].ravel()
                corrected_valid = (raw_hw + bias_hw)[valid].ravel()
                vhm0_hist_raw       += np.histogram(raw_valid,       bins=VHM0_HIST_BINS)[0]
                vhm0_hist_corrected += np.histogram(corrected_valid, bins=VHM0_HIST_BINS)[0]

            if (fi + 1) % 10 == 0 or fi == len(file_list) - 1:
                print(f"  {year}: {fi+1}/{len(file_list)} files processed")

    # ================================================================
    # ANALYSIS 1: Per-year mean bias maps
    # ================================================================
    print("\n=== Analysis 1: Per-year mean bias maps ===")

    year_mean_bias = {}
    for year in years:
        with np.errstate(divide="ignore", invalid="ignore"):
            year_mean_bias[year] = np.where(
                year_count[year] > 0,
                year_sum[year] / year_count[year],
                np.nan
            )
        sea_mask = year_count[year] > 0
        valid_vals = year_mean_bias[year][sea_mask]
        print(f"  {year}: mean bias = {np.nanmean(valid_vals):.4f} m, "
              f"std = {np.nanstd(valid_vals):.4f} m, "
              f"median = {np.nanmedian(valid_vals):.4f} m, "
              f"|bias| mean = {np.nanmean(np.abs(valid_vals)):.4f} m")

    # Grand mean (stationary component)
    total_sum = sum(year_sum[y] for y in years)
    total_count = sum(year_count[y] for y in years)
    with np.errstate(divide="ignore", invalid="ignore"):
        grand_mean_bias = np.where(total_count > 0, total_sum / total_count, np.nan)

    sea_mask = total_count > 0
    print(f"\n  GRAND MEAN (stationary component):")
    print(f"    mean = {np.nanmean(grand_mean_bias[sea_mask]):.4f} m, "
          f"std = {np.nanstd(grand_mean_bias[sea_mask]):.4f} m")

    # ================================================================
    # ANALYSIS 2: Variance decomposition
    # ================================================================
    print("\n=== Analysis 2: Variance decomposition (stationary vs year-specific) ===")

    # Between-year variance at each pixel:
    # Var_between = (1/N_years) * sum_y (mean_y - grand_mean)^2
    n_years_with_data = np.zeros((H, W))
    between_year_var = np.zeros((H, W))
    for year in years:
        has_data = year_count[year] > 0
        n_years_with_data += has_data.astype(float)
        between_year_var += np.where(
            has_data,
            (year_mean_bias[year] - grand_mean_bias)**2,
            0.0
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        between_year_var = np.where(n_years_with_data > 1,
                                     between_year_var / n_years_with_data, np.nan)

    # Total variance at each pixel (across all timesteps, all years)
    with np.errstate(divide="ignore", invalid="ignore"):
        total_mean = np.where(total_count > 0, total_sum / total_count, 0)
        total_var = np.where(
            total_count > 1,
            (sum(year_sq_sum[y] for y in years) / total_count) - total_mean**2,
            np.nan
        )

    # Stationary fraction = 1 - (between_year_var / total_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary_fraction = np.where(
            (total_var > 1e-10) & (~np.isnan(between_year_var)),
            1.0 - between_year_var / total_var,
            np.nan
        )

    valid_sf = stationary_fraction[sea_mask & ~np.isnan(stationary_fraction)]
    print(f"  Stationary fraction (per pixel, then domain-averaged):")
    print(f"    Mean:   {np.nanmean(valid_sf):.4f}  (1.0 = fully stationary)")
    print(f"    Median: {np.nanmedian(valid_sf):.4f}")
    print(f"    Std:    {np.nanstd(valid_sf):.4f}")
    print(f"    10th percentile: {np.nanpercentile(valid_sf, 10):.4f}")
    print(f"    90th percentile: {np.nanpercentile(valid_sf, 90):.4f}")

    print(f"\n  Between-year std (domain-averaged): {np.sqrt(np.nanmean(between_year_var[sea_mask])):.4f} m")
    print(f"  Total std (domain-averaged):        {np.sqrt(np.nanmean(total_var[sea_mask])):.4f} m")

    # ================================================================
    # ANALYSIS 3: Pairwise spatial correlation of annual bias maps
    # ================================================================
    print("\n=== Analysis 3: Pairwise correlation of annual mean bias maps ===")

    corr_matrix = np.full((len(years), len(years)), np.nan)
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            mask = sea_mask & ~np.isnan(year_mean_bias[y1]) & ~np.isnan(year_mean_bias[y2])
            v1 = year_mean_bias[y1][mask]
            v2 = year_mean_bias[y2][mask]
            if len(v1) > 10:
                corr_matrix[i, j] = np.corrcoef(v1, v2)[0, 1]

    print("  Correlation matrix:")
    header = "       " + "  ".join(f"{y}" for y in years)
    print(header)
    for i, y1 in enumerate(years):
        row = f"  {y1}  " + "  ".join(f"{corr_matrix[i,j]:.3f}" for j in range(len(years)))
        print(row)

    # How well does each year correlate with the grand mean?
    print("\n  Correlation of each year's mean bias with grand mean:")
    for year in years:
        mask_y = sea_mask & ~np.isnan(year_mean_bias[year]) & ~np.isnan(grand_mean_bias)
        v1 = year_mean_bias[year][mask_y]
        v2 = grand_mean_bias[mask_y]
        r = np.corrcoef(v1, v2)[0, 1]
        print(f"    {year}: r = {r:.4f}")

    # ================================================================
    # ANALYSIS 4: Per-bin stationarity
    # ================================================================
    print("\n=== Analysis 4: Per-bin mean bias by year ===")
    print(f"  {'Bin':<18s}", end="")
    for y in years:
        print(f"  {y:>8}", end="")
    print(f"  {'std_across_years':>16}")

    for bname in bin_names:
        vals = []
        print(f"  {bname:<18s}", end="")
        for y in years:
            with np.errstate(divide="ignore", invalid="ignore"):
                bcount = year_bin_count[y][bname]
                bmean = np.where(bcount > 0, year_bin_sum[y][bname] / bcount, np.nan)
            domain_mean = np.nanmean(bmean[sea_mask & (bcount > 0)])
            vals.append(domain_mean)
            print(f"  {domain_mean:>8.4f}", end="")
        print(f"  {np.nanstd(vals):>16.4f}")

    # ================================================================
    # ANALYSIS 5: Seasonal decomposition
    # ================================================================
    print("\n=== Analysis 5: Seasonal mean bias by year ===")
    for season in ["winter", "spring", "summer", "autumn"]:
        print(f"\n  {season.upper()}:")
        vals = []
        for y in years:
            cnt = year_season_count[y][season]
            with np.errstate(divide="ignore", invalid="ignore"):
                smean = np.where(cnt > 0, year_season_sum[y][season] / cnt, np.nan)
            domain_mean = np.nanmean(smean[sea_mask & (cnt > 0)])
            vals.append(domain_mean)
            print(f"    {y}: mean bias = {domain_mean:.4f} m")
        print(f"    std across years: {np.nanstd(vals):.4f} m")

    # ================================================================
    # ANALYSIS 6: How much would a static correction capture?
    # ================================================================
    print("\n=== Analysis 6: Static correction baseline (grand mean applied to each year) ===")
    for year in years:
        mask_y = sea_mask & ~np.isnan(year_mean_bias[year])
        raw_bias_rms = np.sqrt(np.nanmean(year_mean_bias[year][mask_y]**2))
        residual_after_static = year_mean_bias[year] - grand_mean_bias
        residual_rms = np.sqrt(np.nanmean(residual_after_static[mask_y]**2))
        reduction_pct = (1 - residual_rms / raw_bias_rms) * 100 if raw_bias_rms > 0 else 0
        print(f"  {year}: raw bias RMS = {raw_bias_rms:.4f} m -> "
              f"after static correction RMS = {residual_rms:.4f} m "
              f"({reduction_pct:.1f}% reduction)")

    # ================================================================
    # PLOTS
    # ================================================================
    print(f"\n=== Saving plots to {output_dir} ===")

    # Plot 1: Per-year mean bias maps
    n_cols = min(len(years), 3)
    n_rows = (len(years) + n_cols - 1) // n_cols + 1  # +1 for grand mean
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten()

    vmin = np.nanpercentile(grand_mean_bias[sea_mask], 2)
    vmax = np.nanpercentile(grand_mean_bias[sea_mask], 98)
    vabs = max(abs(vmin), abs(vmax))

    for i, year in enumerate(years):
        im = axes[i].imshow(year_mean_bias[year], cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                           aspect="auto", origin="upper")
        axes[i].set_title(f"{year} mean bias (m)")
        plt.colorbar(im, ax=axes[i], shrink=0.7)

    im = axes[len(years)].imshow(grand_mean_bias, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                                  aspect="auto", origin="upper")
    axes[len(years)].set_title("GRAND MEAN (stationary)")
    plt.colorbar(im, ax=axes[len(years)], shrink=0.7)

    for j in range(len(years)+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "01_per_year_mean_bias_maps.png", dpi=150)
    plt.close()

    # Plot 2: Year anomalies (year_mean - grand_mean)
    fig, axes = plt.subplots(n_rows-1, n_cols, figsize=(6*n_cols, 5*(n_rows-1)))
    axes = np.atleast_2d(axes).flatten()
    for i, year in enumerate(years):
        anomaly = year_mean_bias[year] - grand_mean_bias
        amax = np.nanpercentile(np.abs(anomaly[sea_mask]), 98)
        im = axes[i].imshow(anomaly, cmap="RdBu_r", vmin=-amax, vmax=amax,
                           aspect="auto", origin="upper")
        axes[i].set_title(f"{year} anomaly (year - grand mean)")
        plt.colorbar(im, ax=axes[i], shrink=0.7)
    for j in range(len(years), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "02_year_anomaly_maps.png", dpi=150)
    plt.close()

    # Plot 3: Stationary fraction map
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(stationary_fraction, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", origin="upper")
    ax.set_title("Stationary fraction per pixel (1.0 = same every year, 0.0 = all year-specific)")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "03_stationary_fraction_map.png", dpi=150)
    plt.close()

    # Plot 4: Between-year std map
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    bstd = np.sqrt(between_year_var)
    im = ax.imshow(bstd, cmap="hot_r", vmin=0,
                   vmax=np.nanpercentile(bstd[sea_mask], 98),
                   aspect="auto", origin="upper")
    ax.set_title("Between-year std of bias (m) — where the bias changes most across years")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "04_between_year_std_map.png", dpi=150)
    plt.close()

    # Plot 5: Correlation matrix heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    for i in range(len(years)):
        for j in range(len(years)):
            ax.text(j, i, f"{corr_matrix[i,j]:.3f}", ha="center", va="center", fontsize=10)
    ax.set_title("Pairwise correlation of annual mean bias maps")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "05_correlation_matrix.png", dpi=150)
    plt.close()

    # Plot 6: Per-bin bias by year
    ncols = min(5, len(bin_names))
    nrows = (len(bin_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for bi, bname in enumerate(bin_names):
        vals = []
        for y in years:
            cnt = year_bin_count[y][bname]
            with np.errstate(divide="ignore", invalid="ignore"):
                bmean = np.where(cnt > 0, year_bin_sum[y][bname] / cnt, np.nan)
            vals.append(np.nanmean(bmean[sea_mask & (cnt > 0)]))
        axes_flat[bi].bar([str(y) for y in years], vals)
        axes_flat[bi].set_title(bname)
        axes_flat[bi].set_ylabel("Mean bias (m)")
        axes_flat[bi].axhline(y=0, color="k", linewidth=0.5)
        axes_flat[bi].tick_params(axis="x", rotation=45)
    for bi in range(len(bin_names), len(axes_flat)):
        axes_flat[bi].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "06_per_bin_bias_by_year.png", dpi=150)
    plt.close()

    # ================================================================
    # ANALYSIS 7: Relative bias (corrected - raw) / (|raw| + eps)
    # ================================================================
    print("\n=== Analysis 7: Relative bias per bin — absolute vs relative stability ===")
    print(f"  (relative bias = (corrected - raw) / (|raw| + {REL_EPS}))\n")

    print(f"  {'Bin':<18s}", end="")
    for y in years:
        print(f"  {y:>8}", end="")
    print(f"  {'std':>8}  {'CV%':>6}")

    abs_stds = {}
    rel_stds = {}
    for bname in bin_names:
        abs_vals = []
        rel_vals = []
        for y in years:
            with np.errstate(divide="ignore", invalid="ignore"):
                acnt = year_bin_count[y][bname]
                amean = np.where(acnt > 0, year_bin_sum[y][bname] / acnt, np.nan)
                rcnt = year_relbin_count[y][bname]
                rmean = np.where(rcnt > 0, year_relbin_sum[y][bname] / rcnt, np.nan)
            abs_vals.append(np.nanmean(amean[sea_mask & (acnt > 0)]))
            rel_vals.append(np.nanmean(rmean[sea_mask & (rcnt > 0)]))
        abs_stds[bname] = np.nanstd(abs_vals)
        rel_stds[bname] = np.nanstd(rel_vals)

        # Print relative bias row
        abs_mean_all = np.nanmean(abs_vals)
        rel_mean_all = np.nanmean(rel_vals)
        print(f"  {bname:<18s}", end="")
        for rv in rel_vals:
            print(f"  {rv:>8.4f}", end="")
        cv_pct = (rel_stds[bname] / abs(rel_mean_all) * 100) if abs(rel_mean_all) > 1e-6 else float("inf")
        print(f"  {rel_stds[bname]:>8.4f}  {cv_pct:>5.1f}%")

    print(f"\n  Stability comparison (std across years):")
    print(f"  {'Bin':<18s}  {'abs_std(m)':>10}  {'rel_std':>10}  {'more_stable':>12}")
    for bname in bin_names:
        winner = "RELATIVE" if rel_stds[bname] < abs_stds[bname] else "ABSOLUTE"
        print(f"  {bname:<18s}  {abs_stds[bname]:>10.4f}  {rel_stds[bname]:>10.4f}  {winner:>12}")

    # Plot 7: Absolute vs relative bias stability per bin (grid layout)
    ncols7 = min(5, len(bin_names))
    nrows7 = (len(bin_names) + ncols7 - 1) // ncols7
    fig, axes_abs = plt.subplots(nrows7, ncols7, figsize=(5 * ncols7, 4 * nrows7))
    fig2, axes_rel = plt.subplots(nrows7, ncols7, figsize=(5 * ncols7, 4 * nrows7))
    axes_abs_flat = axes_abs.flatten() if hasattr(axes_abs, "flatten") else [axes_abs]
    axes_rel_flat = axes_rel.flatten() if hasattr(axes_rel, "flatten") else [axes_rel]
    for bi, bname in enumerate(bin_names):
        abs_vals = []
        rel_vals = []
        for y in years:
            with np.errstate(divide="ignore", invalid="ignore"):
                acnt = year_bin_count[y][bname]
                amean = np.where(acnt > 0, year_bin_sum[y][bname] / acnt, np.nan)
                rcnt = year_relbin_count[y][bname]
                rmean = np.where(rcnt > 0, year_relbin_sum[y][bname] / rcnt, np.nan)
            abs_vals.append(np.nanmean(amean[sea_mask & (acnt > 0)]))
            rel_vals.append(np.nanmean(rmean[sea_mask & (rcnt > 0)]))

        x_labels = [str(y) for y in years]
        axes_abs_flat[bi].bar(x_labels, abs_vals, color="steelblue")
        axes_abs_flat[bi].set_title(f"{bname} (absolute, m)")
        axes_abs_flat[bi].axhline(y=0, color="k", linewidth=0.5)
        axes_abs_flat[bi].tick_params(axis="x", rotation=45)

        axes_rel_flat[bi].bar(x_labels, rel_vals, color="coral")
        axes_rel_flat[bi].set_title(f"{bname} (relative)")
        axes_rel_flat[bi].axhline(y=0, color="k", linewidth=0.5)
        axes_rel_flat[bi].tick_params(axis="x", rotation=45)

    for bi in range(len(bin_names), len(axes_abs_flat)):
        axes_abs_flat[bi].set_visible(False)
        axes_rel_flat[bi].set_visible(False)
    fig.suptitle("Absolute bias per bin across years", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "07a_absolute_bias_per_bin.png", dpi=150)
    plt.close(fig)
    fig2.suptitle("Relative bias per bin across years\n"
                  "(more uniform bars = more stable = better prediction target)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(output_dir / "07b_relative_bias_per_bin.png", dpi=150)
    plt.close(fig2)

    # ================================================================
    # ANALYSIS 8 / PLOT 8: Sea-bin performance (matches evaluate_bunet style)
    # Metrics pooled across all years; binned by raw VHM0 value.
    #
    # Two "models" compared against true wave (corrected_VHM0):
    #   Reference       : raw ERA5 VHM0 used as-is
    #   Static correction: raw VHM0 + per-bin mean bias (grand-mean per sea state)
    # ================================================================
    print("\n=== Analysis 8: Sea-bin performance (raw ERA5 vs static correction vs true wave) ===")

    # Pool across years
    pool_bin_sum = {b: sum(year_bin_sum[y][b] for y in years) for b in bin_names}
    pool_bin_sq_sum = {b: sum(year_bin_sq_sum[y][b] for y in years) for b in bin_names}
    pool_bin_abs_sum = {b: sum(year_bin_abs_sum[y][b] for y in years) for b in bin_names}
    pool_bin_count = {b: sum(year_bin_count[y][b] for y in years) for b in bin_names}

    # Per-bin mean bias (the static correction offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        pool_bin_mean = {
            b: np.where(pool_bin_count[b] > 0,
                        pool_bin_sum[b] / pool_bin_count[b], 0.0)
            for b in bin_names
        }

    # --- Pass 2: accumulate |bias - bin_mean| for static-correction MAE
    #             and static-corrected VHM0 histogram for distribution plot ----------
    print("  Pass 2: computing static-correction residuals for MAE …")
    pool_bin_abs_residual_sum = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
    vhm0_hist_static = np.zeros(_n_hist, dtype=np.float64)

    for year in years:
        file_list = files_by_year[year]
        if args.max_files_per_year:
            file_list = file_list[:args.max_files_per_year]
        for fpath in file_list:
            vhm0_bias, raw_vhm0, _ = load_bias_fields(
                fpath, vhm0_idx, corrected_vhm0_idx, vtm02_idx, corrected_vtm02_idx
            )
            for hour in range(24):
                bias_hw = vhm0_bias[hour]
                raw_hw  = raw_vhm0[hour]
                valid = ~np.isnan(bias_hw) & region_mask
                for bi, bname in enumerate(bin_names):
                    lo, hi = bin_edges[bi], bin_edges[bi + 1]
                    in_bin = valid & (raw_hw >= lo) & (raw_hw < hi)
                    residual = bias_hw - pool_bin_mean[bname]   # error after static correction
                    pool_bin_abs_residual_sum[bname] += np.where(in_bin, np.abs(residual), 0.0)
                    # static prediction = raw + per-bin mean; accumulate its histogram
                    static_pred = (raw_hw + pool_bin_mean[bname])[in_bin].ravel()
                    vhm0_hist_static += np.histogram(static_pred, bins=VHM0_HIST_BINS)[0]

    sb_rmse, sb_static_rmse, sb_mae, sb_static_mae, sb_mean_bias, sb_count, sb_pct = [], [], [], [], [], [], []
    sb_labels_plot = []
    total_sea_count = sum(
        float(pool_bin_count[b][sea_mask].sum()) for b in bin_names
    )

    for bi, bname in enumerate(bin_names):
        cnt_map = pool_bin_count[bname]
        n = float(cnt_map[sea_mask].sum())
        if n == 0:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            rmse_map = np.where(
                cnt_map > 0,
                np.sqrt(pool_bin_sq_sum[bname] / cnt_map),
                np.nan,
            )
            mae_map = np.where(
                cnt_map > 0,
                pool_bin_abs_sum[bname] / cnt_map,
                np.nan,
            )
            mean_map = np.where(
                cnt_map > 0,
                pool_bin_sum[bname] / cnt_map,
                np.nan,
            )
            # Static per-bin correction RMSE = sqrt(Var(bias within bin))
            # = sqrt(mean(bias²) - mean_bias²)
            var_map = np.where(
                cnt_map > 0,
                pool_bin_sq_sum[bname] / cnt_map - (pool_bin_sum[bname] / cnt_map) ** 2,
                np.nan,
            )
            static_rmse_map = np.where(
                ~np.isnan(var_map),
                np.sqrt(np.maximum(var_map, 0.0)),
                np.nan,
            )

        static_mae_map = np.where(
            cnt_map > 0,
            pool_bin_abs_residual_sum[bname] / cnt_map,
            np.nan,
        )

        rmse_val        = float(np.nanmean(rmse_map[sea_mask & (cnt_map > 0)]))
        static_rmse_val = float(np.nanmean(static_rmse_map[sea_mask & (cnt_map > 0)]))
        mae_val         = float(np.nanmean(mae_map[sea_mask & (cnt_map > 0)]))
        static_mae_val  = float(np.nanmean(static_mae_map[sea_mask & (cnt_map > 0)]))
        mean_val        = float(np.nanmean(mean_map[sea_mask & (cnt_map > 0)]))

        sb_rmse.append(rmse_val)
        sb_static_rmse.append(static_rmse_val)
        sb_mae.append(mae_val)
        sb_static_mae.append(static_mae_val)
        sb_mean_bias.append(mean_val)
        sb_count.append(int(n))
        sb_pct.append(100.0 * n / total_sea_count if total_sea_count > 0 else 0.0)
        sb_labels_plot.append(bin_labels[bi])

        print(
            f"  {bin_labels[bi]:<14s}: "
            f"RMSE raw={rmse_val:.4f}m  static={static_rmse_val:.4f}m  |  "
            f"MAE raw={mae_val:.4f}m  static={static_mae_val:.4f}m  |  "
            f"n={int(n):,}  ({sb_pct[-1]:.1f}%)"
        )

    # Plot 8: 2×2 sea-bin performance figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Sea-Bin Performance: Raw ERA5 vs Static Per-Bin Correction vs True Wave\n"
        "(bias = corrected_VHM0 − raw_VHM0, pooled across all years)",
        fontsize=15,
        fontweight="bold",
    )

    x = np.arange(len(sb_labels_plot))
    width = 0.35

    # Panel [0,0]: RMSE grouped bars — Raw ERA5 vs Static per-bin correction
    bars_raw = axes[0, 0].bar(
        x - width / 2, sb_rmse, width,
        label="Raw ERA5 (RMSE vs true wave)", color="steelblue", alpha=0.85,
    )
    bars_static = axes[0, 0].bar(
        x + width / 2, sb_static_rmse, width,
        label="Static per-bin correction (residual RMSE)", color="darkorange", alpha=0.85,
    )
    axes[0, 0].set_title("RMSE by Sea State\n(Raw ERA5 vs static per-bin correction)", fontweight="bold")
    axes[0, 0].set_ylabel("RMSE (m)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sb_labels_plot, rotation=45, ha="right")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=9)
    y_top_rmse = max(max(sb_rmse), max(sb_static_rmse))
    for i, (v_raw, v_st) in enumerate(zip(sb_rmse, sb_static_rmse)):
        if v_raw > 0:
            axes[0, 0].text(
                i - width / 2, v_raw + y_top_rmse * 0.01, f"{v_raw:.3f}",
                ha="center", va="bottom", fontsize=7,
            )
        if v_st > 0:
            axes[0, 0].text(
                i + width / 2, v_st + y_top_rmse * 0.01, f"{v_st:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    single_width = 0.5  # full-width bars for single-series panels

    # Panel [0,1]: MAE grouped bars — Raw ERA5 vs Static correction
    axes[0, 1].bar(
        x - width / 2, sb_mae, width,
        label="Raw ERA5", color="steelblue", alpha=0.85,
    )
    axes[0, 1].bar(
        x + width / 2, sb_static_mae, width,
        label="Static per-bin correction", color="darkorange", alpha=0.85,
    )
    axes[0, 1].set_title("MAE by Sea State\n(Raw ERA5 vs static per-bin correction)", fontweight="bold")
    axes[0, 1].set_ylabel("MAE (m)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sb_labels_plot, rotation=45, ha="right")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].legend(fontsize=9)
    y_top_mae = max(max(sb_mae), max(sb_static_mae))
    for i, (v_raw, v_st) in enumerate(zip(sb_mae, sb_static_mae)):
        if v_raw > 0:
            axes[0, 1].text(
                i - width / 2, v_raw + y_top_mae * 0.01, f"{v_raw:.3f}",
                ha="center", va="bottom", fontsize=7,
            )
        if v_st > 0:
            axes[0, 1].text(
                i + width / 2, v_st + y_top_mae * 0.01, f"{v_st:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    # Panel [1,0]: Mean bias per sea bin (signed — shows under/over-correction)
    colors_bias = ["green" if v >= 0 else "red" for v in sb_mean_bias]
    axes[1, 0].bar(x, sb_mean_bias, single_width, color=colors_bias, alpha=0.7)
    axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Mean Bias by Sea State\n(+= over-correction, −= under-correction)", fontweight="bold")
    axes[1, 0].set_ylabel("Mean Bias (m)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sb_labels_plot, rotation=45, ha="right")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    bias_range = max(sb_mean_bias) - min(sb_mean_bias)
    for i, v in enumerate(sb_mean_bias):
        axes[1, 0].text(
            i,
            v + bias_range * 0.02 if v >= 0 else v - bias_range * 0.02,
            f"{v:+.3f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
        )

    # Panel [1,1]: Sample distribution per sea bin
    axes[1, 1].bar(x, sb_pct, single_width, color="gold", alpha=0.7)
    axes[1, 1].set_title("Sample Distribution by Sea State", fontweight="bold")
    axes[1, 1].set_ylabel("Percentage of Sea Samples (%)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(sb_labels_plot, rotation=45, ha="right")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for i, (v, cnt) in enumerate(zip(sb_pct, sb_count)):
        axes[1, 1].text(
            i,
            v + max(sb_pct) * 0.02,
            f"{v:.1f}%\n({cnt:,})",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "08_sea_bin_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_dir / '08_sea_bin_performance.png'}")

    # ================================================================
    # PLOT 9: Spatial RMSE / MAE / improvement maps (evaluate_bunet style)
    #
    # "Reference"  = raw VHM0 used as-is  → error = bias itself
    # "Corrector"  = static grand-mean correction applied per pixel
    #                → residual = bias − grand_mean_bias
    # Maps produced (one file each, matching evaluate_bunet naming):
    #   09a_rmse_reference.png         sqrt(mean(bias²))      per pixel
    #   09b_rmse_static_corrector.png  sqrt(Var(bias))        per pixel
    #   09c_rmse_improvement.png       ref_rmse − corr_rmse   per pixel
    #   09d_rmse_improvement_binary.png  same, blue/red binary cmap
    #   09e_mae_reference.png          mean(|bias|)           per pixel
    #   09f_mean_bias.png              grand_mean_bias        per pixel
    # ================================================================
    print("\n=== Plot 9: Spatial RMSE / MAE / improvement maps ===")

    # --- lat/lon grids (from first file, hour 0) --------------------------
    t0_hw = data0["tensor"][0]          # (H, W, C)
    lat_grid = t0_hw[..., lat_idx].numpy()   # (H, W)
    lon_grid = t0_hw[..., lon_idx].numpy()   # (H, W)

    # --- pool pixel-level accumulators across all years -------------------
    total_sq_bias  = sum(year_sq_sum[y]  for y in years)   # sum of bias²
    total_abs_bias = sum(year_abs_sum[y] for y in years)   # sum of |bias|
    # total_count and total_sum already computed above

    with np.errstate(divide="ignore", invalid="ignore"):
        rmse_ref = np.where(total_count > 0,
                            np.sqrt(total_sq_bias / total_count), np.nan)
        mae_ref  = np.where(total_count > 0,
                            total_abs_bias / total_count, np.nan)
        # Static corrector residual RMSE = sqrt(mean((bias - grand_mean)²))
        #   = sqrt(mean(bias²) - grand_mean²)  [when count is large]
        #   equivalently = sqrt(total_var) which is already computed
        rmse_corr = np.where(~np.isnan(total_var) & (total_count > 0),
                             np.sqrt(np.maximum(total_var, 0.0)), np.nan)

    improvement = rmse_ref - rmse_corr   # >0 means correction helps

    # --- helper: one cartopy pcolormesh figure ----------------------------
    def _save_geo_map(data, save_path, title, cmap, vmin, vmax, cbar_label, norm=None):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                     linewidth=0.5, alpha=0.5)
        im = ax.pcolormesh(
            lon_grid, lat_grid, data,
            cmap=cmap,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        valid_lons = lon_grid[~np.isnan(lon_grid)]
        valid_lats = lat_grid[~np.isnan(lat_grid)]
        ax.set_extent([valid_lons.min(), valid_lons.max(),
                       valid_lats.min(), valid_lats.max()],
                      crs=ccrs.PlateCarree())
        plt.colorbar(im, ax=ax, orientation="vertical",
                     label=cbar_label, pad=0.05, shrink=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")

    cmap_jet = plt.get_cmap("jet").copy()
    cmap_jet.set_bad("white")

    vmax_rmse = np.nanpercentile(rmse_ref[sea_mask], 98)
    vmax_mae  = np.nanpercentile(mae_ref[sea_mask],  98)

    # 09a: Reference RMSE
    _save_geo_map(
        rmse_ref,
        output_dir / "09a_rmse_reference.png",
        title="Reference RMSE — sqrt(mean(bias²))  [raw VHM0 as predictor]",
        cmap=cmap_jet, vmin=0, vmax=vmax_rmse, cbar_label="RMSE (m)",
    )

    # 09b: Static-corrector residual RMSE
    vmax_corr = np.nanpercentile(rmse_corr[sea_mask & ~np.isnan(rmse_corr)], 98)
    _save_geo_map(
        rmse_corr,
        output_dir / "09b_rmse_static_corrector.png",
        title="Static-Corrector Residual RMSE — sqrt(Var(bias))  [after removing grand mean]",
        cmap=cmap_jet, vmin=0, vmax=max(vmax_rmse, vmax_corr), cbar_label="RMSE (m)",
    )

    # 09c: RMSE improvement — symmetric diverging colormap
    imp_abs_max = np.nanpercentile(np.abs(improvement[sea_mask & ~np.isnan(improvement)]), 98)
    _save_geo_map(
        improvement,
        output_dir / "09c_rmse_improvement.png",
        title="RMSE Improvement (Reference − Static Corrector)\n+red = correction helps, −blue = correction hurts",
        cmap="RdBu", vmin=-imp_abs_max, vmax=imp_abs_max, cbar_label="ΔRMSE (m)",
    )

    # 09d: RMSE improvement — binary blue/red (exact match to evaluate_bunet)
    cmap_binary = mcolors.LinearSegmentedColormap.from_list(
        "improvement_binary", ["#0000FF", "#FF0000"], N=256
    )
    cmap_binary.set_bad("white")
    norm_binary = mcolors.BoundaryNorm(
        boundaries=[-imp_abs_max, 0, imp_abs_max], ncolors=256, clip=True
    )
    _save_geo_map(
        improvement,
        output_dir / "09d_rmse_improvement_binary.png",
        title="RMSE Improvement binary (red = static correction better, blue = worse)",
        cmap=cmap_binary, vmin=-imp_abs_max, vmax=imp_abs_max,
        cbar_label="ΔRMSE (m)", norm=norm_binary,
    )

    # 09e: MAE reference
    _save_geo_map(
        mae_ref,
        output_dir / "09e_mae_reference.png",
        title="Reference MAE — mean(|bias|)  [raw VHM0 as predictor]",
        cmap=cmap_jet, vmin=0, vmax=vmax_mae, cbar_label="MAE (m)",
    )

    # 09f: Grand mean bias (stationary component)
    bias_abs_max = np.nanpercentile(np.abs(grand_mean_bias[sea_mask]), 98)
    _save_geo_map(
        grand_mean_bias,
        output_dir / "09f_mean_bias.png",
        title="Grand Mean Bias — mean(corrected − raw)  [stationary component]",
        cmap="RdBu_r", vmin=-bias_abs_max, vmax=bias_abs_max, cbar_label="Bias (m)",
    )

    # ================================================================
    # PLOT 10: VHM0 distributions — raw ERA5 / static correction / true wave
    # Matches evaluate_bunet.plot_vhm0_distributions() style:
    #   green  = corrected_VHM0 (true wave / reference)
    #   orange = static per-bin correction (treated as "model")
    #   red    = raw ERA5 VHM0 (uncorrected)
    # Produces three files mirroring the three sub-plots in evaluate_bunet:
    #   10a  all three
    #   10b  static correction vs true wave  (model vs reference)
    #   10c  true wave vs raw ERA5           (reference vs uncorrected)
    # Also produces per-sea-state variants for extreme bins (≥ 11 m).
    # ================================================================
    print("\n=== Plot 10: VHM0 distributions ===")

    from scipy import stats as _scipy_stats

    _bin_centers = 0.5 * (VHM0_HIST_BINS[:-1] + VHM0_HIST_BINS[1:])
    _x_grid = np.linspace(0, 15, 300)

    def _hist_kde(hist_counts, bw_scale=0.5):
        """Build a KDE from histogram counts using bin centres as weighted points."""
        w = hist_counts.astype(np.float64)
        total = w.sum()
        if total == 0:
            return np.zeros_like(_x_grid)
        w_norm = w / total
        std_est = np.sqrt(np.sum(w_norm * (_bin_centers - np.sum(w_norm * _bin_centers)) ** 2))
        bw = bw_scale * std_est * total ** (-1 / 5) if std_est > 0 else 0.1
        bw = max(bw, 1e-3)
        kde = _scipy_stats.gaussian_kde(_bin_centers, weights=w + 1e-12, bw_method=bw)
        return kde(_x_grid)

    def _vhm0_range_hist(hist_full, lo, hi):
        """Slice a histogram array to a VHM0 sub-range [lo, hi)."""
        idx_lo = int(np.searchsorted(VHM0_HIST_BINS, lo, side="left"))
        idx_hi = int(np.searchsorted(VHM0_HIST_BINS, hi, side="right")) - 1
        sliced = hist_full.copy()
        sliced[:idx_lo] = 0
        sliced[idx_hi:] = 0
        return sliced

    def _save_dist_plot(kde_vals_list, labels, colors, title, fname, vhm0_range=None):
        range_tag = f" (VHM0 {vhm0_range[0]}–{vhm0_range[1]} m)" if vhm0_range else ""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for kde_vals, label, color in zip(kde_vals_list, labels, colors):
            ax.plot(_x_grid, kde_vals, label=label, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(_x_grid, kde_vals, alpha=0.15, color=color)
        ax.set_xlabel("VHM0 (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title(title + range_tag, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)
        plt.tight_layout()
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {output_dir / fname}")

    # --- full-distribution KDEs ---
    kde_raw       = _hist_kde(vhm0_hist_raw)
    kde_corrected = _hist_kde(vhm0_hist_corrected)
    kde_static    = _hist_kde(vhm0_hist_static)

    _save_dist_plot(
        [kde_corrected, kde_static, kde_raw],
        ["True wave (corrected_VHM0)", "Static per-bin correction", "Raw ERA5"],
        ["green", "darkorange", "red"],
        "VHM0 Distributions — True Wave vs Static Correction vs Raw ERA5",
        "10a_vhm0_distributions.png",
    )
    _save_dist_plot(
        [kde_corrected, kde_static],
        ["True wave (corrected_VHM0)", "Static per-bin correction"],
        ["green", "darkorange"],
        "VHM0 Distribution — Static Correction vs True Wave",
        "10b_vhm0_static_vs_true.png",
    )
    _save_dist_plot(
        [kde_corrected, kde_raw],
        ["True wave (corrected_VHM0)", "Raw ERA5"],
        ["green", "red"],
        "VHM0 Distribution — True Wave vs Raw ERA5",
        "10c_vhm0_true_vs_raw.png",
    )

    # --- per-range variants for extreme sea states (matching evaluate_bunet usage) ---
    for lo, hi in [(11, 12), (12, 13)]:
        h_raw  = _vhm0_range_hist(vhm0_hist_raw,       lo, hi)
        h_corr = _vhm0_range_hist(vhm0_hist_corrected, lo, hi)
        h_stat = _vhm0_range_hist(vhm0_hist_static,    lo, hi)
        if h_raw.sum() == 0:
            print(f"  No samples in {lo}–{hi} m range, skipping.")
            continue
        suffix = f"_{lo}-{hi}m"
        _save_dist_plot(
            [_hist_kde(h_corr), _hist_kde(h_stat), _hist_kde(h_raw)],
            ["True wave (corrected_VHM0)", "Static per-bin correction", "Raw ERA5"],
            ["green", "darkorange", "red"],
            f"VHM0 Distributions ({lo}–{hi} m)",
            f"10a_vhm0_distributions{suffix}.png",
            vhm0_range=(lo, hi),
        )
        _save_dist_plot(
            [_hist_kde(h_corr), _hist_kde(h_stat)],
            ["True wave (corrected_VHM0)", "Static per-bin correction"],
            ["green", "darkorange"],
            f"VHM0 — Static Correction vs True Wave",
            f"10b_vhm0_static_vs_true{suffix}.png",
            vhm0_range=(lo, hi),
        )

    print(f"\nDone. Check the output directory for plots.")

    # ================================================================
    # SUMMARY VERDICT
    # ================================================================
    print("\n" + "="*70)
    print("SUMMARY VERDICT")
    print("="*70)

    mean_sf = np.nanmean(valid_sf)
    off_diag_corrs = corr_matrix[np.triu_indices(len(years), k=1)]
    mean_corr = np.nanmean(off_diag_corrs)
    # Specifically check 2023 vs training years
    test_year_idx = years.index(2023) if 2023 in years else None
    if test_year_idx is not None:
        train_year_idxs = [years.index(y) for y in [2018, 2019, 2020, 2021] if y in years]
        test_corrs = [corr_matrix[test_year_idx, ti] for ti in train_year_idxs]
        mean_test_corr = np.nanmean(test_corrs)
    else:
        mean_test_corr = np.nan

    print(f"\n  Stationary fraction (domain-averaged): {mean_sf:.3f}")
    print(f"  Mean pairwise correlation of annual maps: {mean_corr:.3f}")
    if not np.isnan(mean_test_corr):
        print(f"  Mean correlation of 2023 with training years: {mean_test_corr:.3f}")

    if mean_sf > 0.8 and mean_corr > 0.9:
        print("\n  → VERDICT: Bias is HIGHLY STATIONARY.")
        print("    A simple model with strong regularization should work well.")
        print("    The static grand-mean correction captures most of the signal.")
    elif mean_sf > 0.6 and mean_corr > 0.7:
        print("\n  → VERDICT: Bias is MODERATELY STATIONARY with some year-specific variation.")
        print("    A regularized model should capture the bulk, but you may need")
        print("    conditioning on large-scale climate state for the residual.")
    else:
        print("\n  → VERDICT: Bias has SIGNIFICANT year-to-year variation.")
        print("    Fixed models will struggle. Consider:")
        print("    - Conditioning on synoptic-scale indices (NAO, etc.)")
        print("    - Online/adaptive correction")
        print("    - Ensemble approaches")


if __name__ == "__main__":
    main()