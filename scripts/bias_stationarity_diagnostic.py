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
    year_bin_sum = {}
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

    for year in years:
        year_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_sq_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_count[year] = np.zeros((H, W), dtype=np.float64)
        year_raw_sum[year] = np.zeros((H, W), dtype=np.float64)
        year_bin_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_bin_count[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_season_sum[year] = {s: np.zeros((H, W), dtype=np.float64) for s in ["winter", "spring", "summer", "autumn"]}
        year_season_count[year] = {s: np.zeros((H, W), dtype=np.float64) for s in ["winter", "spring", "summer", "autumn"]}
        year_relbin_sum[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}
        year_relbin_count[year] = {b: np.zeros((H, W), dtype=np.float64) for b in bin_names}

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
                year_count[year] += valid.astype(np.float64)
                year_raw_sum[year] += np.where(valid, raw_hw, 0.0)

                # Per-bin (absolute and relative)
                rel_bias_hw = np.where(valid, bias_hw / (np.abs(raw_hw) + REL_EPS), 0.0)
                for bi in range(len(bin_names)):
                    lo, hi = bin_edges[bi], bin_edges[bi+1]
                    in_bin = valid & (raw_hw >= lo) & (raw_hw < hi)
                    year_bin_sum[year][bin_names[bi]] += np.where(in_bin, bias_hw, 0.0)
                    year_bin_count[year][bin_names[bi]] += in_bin.astype(np.float64)
                    year_relbin_sum[year][bin_names[bi]] += np.where(in_bin, rel_bias_hw, 0.0)
                    year_relbin_count[year][bin_names[bi]] += in_bin.astype(np.float64)

                # Per-season
                if season in year_season_sum[year]:
                    year_season_sum[year][season] += bias_filled
                    year_season_count[year][season] += valid.astype(np.float64)

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