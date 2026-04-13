"""
Static Bias Map Baseline
=========================
1. Computes the per-pixel mean bias from training years (the "static map")
2. Saves the map as a reusable .npy artifact
3. Evaluates on every hourly timestep for each year:
   - "No correction": raw VHM0 vs corrected_VHM0
   - "Static map":    (VHM0 + static_bias) vs corrected_VHM0
4. Reports RMSE, MAE, bias, per wave-height bin, per year

Any ML model must consistently beat the "Static map" row to justify its existence.

Usage:
  PYTHONUNBUFFERED=1 poetry run python scripts/static_baseline_evaluation.py \
    --data_path /mnt/blobstorage/preprocessed_extended_subsampled_step_5/ \
    --train_years 2018 2019 2020 2021 \
    --eval_years 2017 2018 2019 2020 2021 2022 2023 \
    --region mediterranean
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path for src imports (same pattern as evaluation pipeline scripts)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluation_plots import (
    plot_rmse_maps as plot_rmse_maps_fn,
)
from src.evaluation.evaluation_plots import (
    plot_sea_bin_metrics as plot_sea_bin_metrics_fn,
)
from src.evaluation.metrics import compute_sea_bin_metrics_from_accumulators


def parse_year_month(filename):
    name = Path(filename).stem
    marker = "WAVEAN"
    idx = name.find(marker)
    if idx != -1 and len(name) >= idx + 12:
        return int(name[idx + 6 : idx + 10]), int(name[idx + 10 : idx + 12])
    return None, None


def get_vhm0_sea_bins():
    """Match corrected_VHM0 sea bins used in evaluate_bunet.py."""
    return [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--file_pattern", default="WAVEAN*.pt")
    parser.add_argument("--train_years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    parser.add_argument("--eval_years", nargs="+", type=int, default=[2017, 2018, 2019, 2020, 2021, 2022, 2023])
    parser.add_argument("--output_dir", default="/mnt/blobstorage/diagnostics/static_baseline")
    parser.add_argument("--max_files_per_year", type=int, default=None)
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["all", "atlantic", "mediterranean"],
        help="Region filter to evaluate (Gibraltar + Bay of Biscay logic)",
    )
    parser.add_argument("--bias_map", type=str, default=None,
                        help="Path to precomputed static_bias_map.npy (skips Step 1)")
    parser.add_argument(
        "--sea_bin_source",
        type=str,
        default="true",
        choices=["true", "raw"],
        help="Source used for sea-bin assignment: 'true' (corrected_VHM0) or 'raw' (VHM0).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.region
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(glob.glob(f"{args.data_path}/{args.file_pattern}"))
    print(f"Found {len(all_files)} total files")

    files_by_year = defaultdict(list)
    for f in all_files:
        year, _ = parse_year_month(f)
        if year:
            files_by_year[year].append(f)

    years_available = sorted(files_by_year.keys())
    print(f"Years available: {years_available}")
    print(f"Training years: {args.train_years}")
    print(f"Eval years: {args.eval_years}")

    # Get feature indices
    data0 = torch.load(all_files[0], map_location="cpu")
    feature_cols = data0["feature_cols"]
    H, W = data0["tensor"].shape[1], data0["tensor"].shape[2]
    print(f"Grid: {H} x {W}")

    vhm0_idx = feature_cols.index("VHM0")
    corrected_vhm0_idx = feature_cols.index("corrected_VHM0")
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    lat_grid = data0["tensor"][0, ..., lat_idx].numpy()
    lon_grid = data0["tensor"][0, ..., lon_idx].numpy()
    del data0

    # Build static region mask from coordinates in the first file
    # Match CachedWaveDataset logic:
    # atlantic: lon < -5.5 OR (lat > 43 AND lon < 0)
    # mediterranean: lon >= -5.5 AND NOT (lat > 43 AND lon < 0)
    if args.region == "all":
        region_mask = np.ones((H, W), dtype=bool)
    else:
        data_coords = torch.load(all_files[0], map_location="cpu")
        t0 = data_coords["tensor"][0]  # (H, W, C)
        lat = t0[..., lat_idx].numpy()
        lon = t0[..., lon_idx].numpy()
        del data_coords

        GIBRALTAR_LON = -5.5
        BISCAY_LAT = 43.0
        BISCAY_LON = 0.0
        biscay = (lat > BISCAY_LAT) & (lon < BISCAY_LON)

        if args.region == "atlantic":
            region_mask = (lon < GIBRALTAR_LON) | biscay
        else:  # mediterranean
            region_mask = (lon >= GIBRALTAR_LON) & ~biscay

        # Keep only finite coordinate pixels
        region_mask = region_mask & ~np.isnan(lat) & ~np.isnan(lon)

    print(f"Region filter: {args.region}")
    print(f"Region pixels kept: {int(region_mask.sum()):,}/{region_mask.size:,} ({100.0 * region_mask.mean():.1f}%)")
    print(f"Sea-bin source: {args.sea_bin_source}")

    # ==================================================================
    # STEP 1: Compute or load static bias map
    # ==================================================================
    if args.bias_map is not None:
        print(f"\n{'='*70}")
        print(f"STEP 1: Loading precomputed bias map from {args.bias_map}")
        print(f"{'='*70}")
        static_bias_map = np.load(args.bias_map)
        assert static_bias_map.shape == (H, W), \
            f"Bias map shape {static_bias_map.shape} != grid {(H, W)}"
        sea_mask = ~np.isnan(static_bias_map) & region_mask
    else:
        print(f"\n{'='*70}")
        print("STEP 1: Computing static bias map from training years")
        print(f"{'='*70}")

        bias_sum = np.zeros((H, W), dtype=np.float64)
        bias_count = np.zeros((H, W), dtype=np.float64)

        for year in args.train_years:
            if year not in files_by_year:
                print(f"  WARNING: year {year} not found in data, skipping")
                continue
            file_list = files_by_year[year]
            if args.max_files_per_year:
                file_list = file_list[: args.max_files_per_year]

            for fi, fpath in enumerate(file_list):
                data = torch.load(fpath, map_location="cpu")
                tensor = data["tensor"]  # (24, H, W, C)
                raw = tensor[..., vhm0_idx].numpy()
                corrected = tensor[..., corrected_vhm0_idx].numpy()
                bias = corrected - raw  # (24, H, W)

                for hour in range(24):
                    valid = ~np.isnan(bias[hour])
                    bias_sum += np.where(valid, bias[hour], 0.0)
                    bias_count += valid.astype(np.float64)

                if (fi + 1) % 20 == 0 or fi == len(file_list) - 1:
                    print(f"  {year}: {fi+1}/{len(file_list)} files", flush=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            static_bias_map = np.where(bias_count > 0, bias_sum / bias_count, np.nan)

        sea_mask = (bias_count > 0) & region_mask

        map_path = output_dir / f"static_bias_map_{args.region}.npy"
        np.save(map_path, static_bias_map)
        print(f"  Saved static bias map to {map_path}")

    # Always keep a region-masked map for plotting and region-specific reuse
    static_bias_map_region = np.where(sea_mask, static_bias_map, np.nan)
    masked_map_path = output_dir / f"static_bias_map_{args.region}.npy"
    np.save(masked_map_path, static_bias_map_region)
    print(f"  Saved region-masked static bias map to {masked_map_path}")

    print(f"\n  Static map stats (sea pixels only):")
    print(f"    Mean:   {np.nanmean(static_bias_map[sea_mask]):.4f} m")
    print(f"    Std:    {np.nanstd(static_bias_map[sea_mask]):.4f} m")
    print(f"    Median: {np.nanmedian(static_bias_map[sea_mask]):.4f} m")
    print(f"    Min:    {np.nanmin(static_bias_map[sea_mask]):.4f} m")
    print(f"    Max:    {np.nanmax(static_bias_map[sea_mask]):.4f} m")

    # ==================================================================
    # STEP 2: Evaluate on each year — hourly timestep level
    # ==================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Evaluating static baseline on every hourly timestep")
    print(f"{'='*70}")

    sea_bins = get_vhm0_sea_bins()
    bin_names = [b["name"] for b in sea_bins]
    bin_labels = {b["name"]: b["label"] for b in sea_bins}

    results = {}
    sea_bin_accumulators = {
        bin_config["name"]: {
            "count": np.int64(0),
            "sum_mae": np.float64(0.0),
            "sum_mse": np.float64(0.0),
            "sum_bias": np.float64(0.0),
            "sum_baseline_mae": np.float64(0.0),
            "sum_baseline_mse": np.float64(0.0),
            "sum_baseline_bias": np.float64(0.0),
            "count_model_better": np.int64(0),
            "count_model_worse": np.int64(0),
        }
        for bin_config in sea_bins
    }
    spatial_error_sq_model = np.zeros((H, W), dtype=np.float64)
    spatial_error_abs_model = np.zeros((H, W), dtype=np.float64)
    spatial_error_sq_baseline = np.zeros((H, W), dtype=np.float64)
    spatial_error_abs_baseline = np.zeros((H, W), dtype=np.float64)
    spatial_count = np.zeros((H, W), dtype=np.float64)

    for year in args.eval_years:
        if year not in files_by_year:
            print(f"  WARNING: year {year} not found, skipping")
            continue

        file_list = files_by_year[year]
        if args.max_files_per_year:
            file_list = file_list[: args.max_files_per_year]

        # Accumulators for "no correction" and "static correction"
        # Overall
        nc_sum_sq_err = np.float64(0.0)  # no correction: squared error
        nc_sum_abs_err = np.float64(0.0)
        nc_sum_err = np.float64(0.0)
        sc_sum_sq_err = np.float64(0.0)  # static correction: squared error
        sc_sum_abs_err = np.float64(0.0)
        sc_sum_err = np.float64(0.0)
        total_pixels = np.int64(0)

        # Per-bin
        nc_bin_sq = {b: np.float64(0.0) for b in bin_names}
        nc_bin_abs = {b: np.float64(0.0) for b in bin_names}
        nc_bin_err = {b: np.float64(0.0) for b in bin_names}
        sc_bin_sq = {b: np.float64(0.0) for b in bin_names}
        sc_bin_abs = {b: np.float64(0.0) for b in bin_names}
        sc_bin_err = {b: np.float64(0.0) for b in bin_names}
        bin_count = {b: np.int64(0) for b in bin_names}

        for fi, fpath in enumerate(file_list):
            data = torch.load(fpath, map_location="cpu")
            tensor = data["tensor"]
            raw = tensor[..., vhm0_idx].numpy()        # (24,H,W)
            corrected = tensor[..., corrected_vhm0_idx].numpy()

            for hour in range(24):
                raw_h = raw[hour]
                cor_h = corrected[hour]
                valid = ~np.isnan(cor_h) & ~np.isnan(raw_h) & sea_mask

                if not valid.any():
                    continue

                raw_v = raw_h[valid]
                cor_v = cor_h[valid]
                static_v = static_bias_map[valid]

                # "No correction" error: using raw VHM0 as prediction
                nc_err = raw_v - cor_v  # error = prediction - truth
                nc_sum_sq_err += np.sum(nc_err ** 2)
                nc_sum_abs_err += np.sum(np.abs(nc_err))
                nc_sum_err += np.sum(nc_err)

                # "Static correction" error: using VHM0 + static_map
                sc_pred = raw_v + static_v
                sc_err = sc_pred - cor_v
                sc_sum_sq_err += np.sum(sc_err ** 2)
                sc_sum_abs_err += np.sum(np.abs(sc_err))
                sc_sum_err += np.sum(sc_err)

                total_pixels += valid.sum()

                # Per-bin
                bin_values = cor_v if args.sea_bin_source == "true" else raw_v
                for bin_config in sea_bins:
                    lo = bin_config["min"]
                    hi = bin_config["max"]
                    bname = bin_config["name"]
                    in_bin = (bin_values >= lo) & (bin_values < hi)
                    if not in_bin.any():
                        continue
                    nc_e = nc_err[in_bin]
                    sc_e = sc_err[in_bin]
                    nc_bin_sq[bname] += np.sum(nc_e ** 2)
                    nc_bin_abs[bname] += np.sum(np.abs(nc_e))
                    nc_bin_err[bname] += np.sum(nc_e)
                    sc_bin_sq[bname] += np.sum(sc_e ** 2)
                    sc_bin_abs[bname] += np.sum(np.abs(sc_e))
                    sc_bin_err[bname] += np.sum(sc_e)
                    bin_count[bname] += in_bin.sum()

                    sea_bin_accumulators[bname]["count"] += in_bin.sum()
                    sea_bin_accumulators[bname]["sum_mae"] += np.sum(np.abs(sc_e))
                    sea_bin_accumulators[bname]["sum_mse"] += np.sum(sc_e ** 2)
                    sea_bin_accumulators[bname]["sum_bias"] += np.sum(sc_e)
                    sea_bin_accumulators[bname]["sum_baseline_mae"] += np.sum(
                        np.abs(nc_e)
                    )
                    sea_bin_accumulators[bname]["sum_baseline_mse"] += np.sum(nc_e ** 2)
                    sea_bin_accumulators[bname]["sum_baseline_bias"] += np.sum(nc_e)
                    sea_bin_accumulators[bname]["count_model_better"] += np.sum(
                        np.abs(sc_e) < np.abs(nc_e)
                    )
                    sea_bin_accumulators[bname]["count_model_worse"] += np.sum(
                        np.abs(sc_e) > np.abs(nc_e)
                    )

                valid_count = valid.astype(np.float64)
                spatial_count += valid_count
                spatial_model_sq_h = np.zeros((H, W), dtype=np.float64)
                spatial_model_abs_h = np.zeros((H, W), dtype=np.float64)
                spatial_baseline_sq_h = np.zeros((H, W), dtype=np.float64)
                spatial_baseline_abs_h = np.zeros((H, W), dtype=np.float64)
                spatial_model_sq_h[valid] = sc_err ** 2
                spatial_model_abs_h[valid] = np.abs(sc_err)
                spatial_baseline_sq_h[valid] = nc_err ** 2
                spatial_baseline_abs_h[valid] = np.abs(nc_err)
                spatial_error_sq_model += spatial_model_sq_h
                spatial_error_abs_model += spatial_model_abs_h
                spatial_error_sq_baseline += spatial_baseline_sq_h
                spatial_error_abs_baseline += spatial_baseline_abs_h

            if (fi + 1) % 20 == 0 or fi == len(file_list) - 1:
                print(f"  {year}: {fi+1}/{len(file_list)} files", flush=True)

        n = float(total_pixels)
        is_train = year in args.train_years
        label = "TRAIN" if is_train else "TEST"

        results[year] = {
            "label": label,
            "n_pixels": total_pixels,
            "sea_bin_source": args.sea_bin_source,
            "nc_rmse": np.sqrt(nc_sum_sq_err / n),
            "nc_mae": nc_sum_abs_err / n,
            "nc_bias": nc_sum_err / n,
            "sc_rmse": np.sqrt(sc_sum_sq_err / n),
            "sc_mae": sc_sum_abs_err / n,
            "sc_bias": sc_sum_err / n,
            "rmse_improvement_pct": (1 - np.sqrt(sc_sum_sq_err / n) / np.sqrt(nc_sum_sq_err / n)) * 100,
            "mae_improvement_pct": (1 - (sc_sum_abs_err / n) / (nc_sum_abs_err / n)) * 100,
            "bins": {},
        }

        for bname in bin_names:
            bn = float(bin_count[bname]) if bin_count[bname] > 0 else 1.0
            results[year]["bins"][bname] = {
                "label": bin_labels[bname],
                "count": int(bin_count[bname]),
                "nc_rmse": np.sqrt(nc_bin_sq[bname] / bn),
                "nc_mae": nc_bin_abs[bname] / bn,
                "nc_bias": nc_bin_err[bname] / bn,
                "sc_rmse": np.sqrt(sc_bin_sq[bname] / bn),
                "sc_mae": sc_bin_abs[bname] / bn,
                "sc_bias": sc_bin_err[bname] / bn,
                "rmse_improvement_pct": (1 - (np.sqrt(sc_bin_sq[bname] / bn) / np.sqrt(nc_bin_sq[bname] / bn))) * 100
                if nc_bin_sq[bname] > 0
                else 0.0,
                "mae_improvement_pct": (1 - ((sc_bin_abs[bname] / bn) / (nc_bin_abs[bname] / bn))) * 100
                if nc_bin_abs[bname] > 0
                else 0.0,
            }

    # ==================================================================
    # STEP 3: Print results
    # ==================================================================
    print(f"\n{'='*70}")
    print("RESULTS: Static Baseline vs No Correction (per hourly timestep)")
    print(f"{'='*70}")

    print(f"\n  Overall metrics:")
    print(f"  {'Year':<6} {'Split':<6} {'NC RMSE':>9} {'SC RMSE':>9} {'Improv%':>8} "
          f"{'NC MAE':>8} {'SC MAE':>8} {'Improv%':>8} "
          f"{'NC Bias':>8} {'SC Bias':>8}")
    print(f"  {'-'*90}")

    for year in sorted(results.keys()):
        r = results[year]
        rmse_imp = (1 - r["sc_rmse"] / r["nc_rmse"]) * 100
        mae_imp = (1 - r["sc_mae"] / r["nc_mae"]) * 100
        print(
            f"  {year:<6} {r['label']:<6} "
            f"{r['nc_rmse']:>9.4f} {r['sc_rmse']:>9.4f} {rmse_imp:>7.1f}% "
            f"{r['nc_mae']:>8.4f} {r['sc_mae']:>8.4f} {mae_imp:>7.1f}% "
            f"{r['nc_bias']:>8.4f} {r['sc_bias']:>8.4f}"
        )

    # Per-bin breakdown
    for bname in bin_names:
        print(f"\n  {bname}:")
        print(f"  {'Year':<6} {'Split':<6} {'Count':>10} "
              f"{'NC RMSE':>9} {'SC RMSE':>9} {'Improv%':>8} "
              f"{'NC MAE':>8} {'SC MAE':>8} {'Improv%':>8} "
              f"{'SC Bias':>8}")
        print(f"  {'-'*100}")
        for year in sorted(results.keys()):
            r = results[year]
            b = r["bins"][bname]
            if b["count"] == 0:
                continue
            rmse_imp = (1 - b["sc_rmse"] / b["nc_rmse"]) * 100
            mae_imp = (1 - b["sc_mae"] / b["nc_mae"]) * 100
            print(
                f"  {year:<6} {r['label']:<6} {b['count']:>10,} "
                f"{b['nc_rmse']:>9.4f} {b['sc_rmse']:>9.4f} {rmse_imp:>7.1f}% "
                f"{b['nc_mae']:>8.4f} {b['sc_mae']:>8.4f} {mae_imp:>7.1f}% "
                f"{b['sc_bias']:>8.4f}"
            )

    # ==================================================================
    # STEP 4: Save results table and plot
    # ==================================================================

    # Save as text
    import json
    results_serializable = {}
    for y, r in results.items():
        results_serializable[str(y)] = {
            k: (int(v) if isinstance(v, np.integer) else
                float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in r.items()
        }
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)

    sea_bin_metrics = compute_sea_bin_metrics_from_accumulators(
        sea_bins=sea_bins,
        sea_bin_accumulators=sea_bin_accumulators,
    )
    with open(output_dir / "sea_bin_metrics_overall.json", "w") as f:
        json.dump(sea_bin_metrics, f, indent=2)

    # Plot: RMSE comparison bar chart
    eval_years = sorted(results.keys())
    nc_rmses = [results[y]["nc_rmse"] for y in eval_years]
    sc_rmses = [results[y]["sc_rmse"] for y in eval_years]

    x = np.arange(len(eval_years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, nc_rmses, width, label="No correction", color="salmon")
    bars2 = ax.bar(x + width / 2, sc_rmses, width, label="Static bias map", color="steelblue")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Baseline: No Correction vs Static Bias Map (hourly timestep evaluation)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n{'(train)' if y in args.train_years else '(test)'}" for y in eval_years])
    ax.legend()
    ax.axhline(y=0, color="k", linewidth=0.3)

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_rmse_comparison.png", dpi=150)
    plt.close()

    # Plot: Per-bin RMSE for static correction across years
    n_bins = len(bin_names)
    ncols = 4
    nrows = (n_bins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten()
    for bi, bname in enumerate(bin_names):
        nc_vals = [results[y]["bins"][bname]["nc_rmse"] for y in eval_years]
        sc_vals = [results[y]["bins"][bname]["sc_rmse"] for y in eval_years]
        x = np.arange(len(eval_years))
        axes_flat[bi].bar(x - width / 2, nc_vals, width, label="No correction", color="salmon")
        axes_flat[bi].bar(x + width / 2, sc_vals, width, label="Static map", color="steelblue")
        axes_flat[bi].set_title(bname)
        axes_flat[bi].set_xticks(x)
        axes_flat[bi].set_xticklabels([str(y) for y in eval_years], rotation=45)
        axes_flat[bi].set_ylabel("RMSE (m)")
        if bi == 0:
            axes_flat[bi].legend(fontsize=8)
    for j in range(n_bins, len(axes_flat)):
        axes_flat[j].axis("off")
    plt.suptitle("Per-bin RMSE: No Correction vs Static Bias Map", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_per_bin_rmse.png", dpi=150)
    plt.close()

    # Plot: Per-bin RMSE improvement (%) heatmap across years
    improvement_matrix = np.zeros((len(eval_years), len(bin_names)), dtype=float)
    for yi, y in enumerate(eval_years):
        for bi, bname in enumerate(bin_names):
            improvement_matrix[yi, bi] = results[y]["bins"][bname]["rmse_improvement_pct"]

    fig, ax = plt.subplots(figsize=(1.05 * len(bin_names) + 4, 0.6 * len(eval_years) + 3))
    vmax_abs = max(1.0, np.nanpercentile(np.abs(improvement_matrix), 98))
    im = ax.imshow(
        improvement_matrix,
        cmap="RdYlGn",
        vmin=-vmax_abs,
        vmax=vmax_abs,
        aspect="auto",
        origin="upper",
    )
    ax.set_title("Per-bin RMSE Improvement (%) - Static Map vs No Correction")
    ax.set_xlabel("Wave-height bin")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(len(bin_names)))
    ax.set_xticklabels(bin_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(eval_years)))
    ax.set_yticklabels([str(y) for y in eval_years])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Improvement (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_per_bin_rmse_improvement.png", dpi=150)
    plt.close()

    # Plot: Per-year improvement bars across bins (one panel per year)
    n_years = len(eval_years)
    ncols = 3
    nrows = (n_years + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for yi, year in enumerate(eval_years):
        ax = axes[yi]
        vals = [
            results[year]["bins"][bname]["rmse_improvement_pct"] for bname in bin_names
        ]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        x = np.arange(len(bin_names))
        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(
            f"{year} ({'train' if year in args.train_years else 'test'})", fontsize=11
        )
        ax.set_xticks(x)
        ax.set_xticklabels(bin_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE improvement (%)")
        ax.grid(True, axis="y", alpha=0.3)

        # Add percentage labels on top of bars
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.6 if v >= 0 else -0.6),
                f"{v:.1f}%",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )

    for j in range(n_years, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Per-bin RMSE improvement by year", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_per_bin_rmse_improvement_bars.png", dpi=150)
    plt.close()

    # Export the same sea-bin performance figure as evaluate_bunet.py
    plot_sea_bin_metrics_fn(
        sea_bin_metrics=sea_bin_metrics,
        sea_bins=sea_bins,
        target_column="corrected_VHM0",
        unit="m",
        output_dir=output_dir,
    )

    # Export the same RMSE/MAE spatial maps as evaluate_bunet.py
    spatial_errors_model = [
        {
            "error_sq": spatial_error_sq_model,
            "error_sq_mae": spatial_error_abs_model,
            "count": spatial_count,
        }
    ]
    spatial_errors_baseline = [
        {
            "error_sq": spatial_error_sq_baseline,
            "error_sq_mae": spatial_error_abs_baseline,
            "count": spatial_count,
        }
    ]
    plot_rmse_maps_fn(
        spatial_errors_model=spatial_errors_model,
        spatial_errors_baseline=spatial_errors_baseline,
        test_files=all_files,
        subsample_step=1,
        geo_bounds=None,
        unit="m",
        output_dir=output_dir,
        dataset_coords=(lat_grid, lon_grid),
    )

    # Plot: Static bias map
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    vabs = np.nanpercentile(np.abs(static_bias_map_region[sea_mask]), 98)
    im = ax.imshow(static_bias_map_region, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="auto", origin="lower")
    ax.set_title(f"Static bias map [{args.region}] (mean of training years {args.train_years})")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Bias (m)")
    plt.tight_layout()
    plt.savefig(output_dir / "static_bias_map.png", dpi=150)
    plt.close()

    print(f"\n  Plots and results saved to {output_dir}")
    print(f"\n  KEY NUMBERS for your ML model to beat:")
    for year in sorted(results.keys()):
        r = results[year]
        print(f"    {year} ({r['label']}): Static map RMSE = {r['sc_rmse']:.4f} m, MAE = {r['sc_mae']:.4f} m")


if __name__ == "__main__":
    main()
