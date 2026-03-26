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
    --eval_years 2017 2018 2019 2020 2021 2022 2023
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
        return int(name[idx + 6 : idx + 10]), int(name[idx + 10 : idx + 12])
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--file_pattern", default="WAVEAN*.pt")
    parser.add_argument("--train_years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    parser.add_argument("--eval_years", nargs="+", type=int, default=[2017, 2018, 2019, 2020, 2021, 2022, 2023])
    parser.add_argument("--output_dir", default="/mnt/blobstorage/diagnostics/static_baseline")
    parser.add_argument("--max_files_per_year", type=int, default=None)
    parser.add_argument("--bias_map", type=str, default=None,
                        help="Path to precomputed static_bias_map.npy (skips Step 1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
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
    del data0

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
        sea_mask = ~np.isnan(static_bias_map)
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

        sea_mask = bias_count > 0

        map_path = output_dir / "static_bias_map.npy"
        np.save(map_path, static_bias_map)
        print(f"  Saved static bias map to {map_path}")

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

    bin_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, float("inf")]
    bin_names = [
        "0-1m", "1-2m", "2-3m", "3-4m", "4-5m", "5-6m", "6-7m",
        "7-8m", "8-9m", "9-10m", "10-11m", "11-12m", "12m+",
    ]

    results = {}

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
                for bi in range(len(bin_names)):
                    lo, hi = bin_edges[bi], bin_edges[bi + 1]
                    in_bin = (raw_v >= lo) & (raw_v < hi)
                    if not in_bin.any():
                        continue
                    bname = bin_names[bi]
                    nc_e = nc_err[in_bin]
                    sc_e = sc_err[in_bin]
                    nc_bin_sq[bname] += np.sum(nc_e ** 2)
                    nc_bin_abs[bname] += np.sum(np.abs(nc_e))
                    nc_bin_err[bname] += np.sum(nc_e)
                    sc_bin_sq[bname] += np.sum(sc_e ** 2)
                    sc_bin_abs[bname] += np.sum(np.abs(sc_e))
                    sc_bin_err[bname] += np.sum(sc_e)
                    bin_count[bname] += in_bin.sum()

            if (fi + 1) % 20 == 0 or fi == len(file_list) - 1:
                print(f"  {year}: {fi+1}/{len(file_list)} files", flush=True)

        n = float(total_pixels)
        is_train = year in args.train_years
        label = "TRAIN" if is_train else "TEST"

        results[year] = {
            "label": label,
            "n_pixels": total_pixels,
            "nc_rmse": np.sqrt(nc_sum_sq_err / n),
            "nc_mae": nc_sum_abs_err / n,
            "nc_bias": nc_sum_err / n,
            "sc_rmse": np.sqrt(sc_sum_sq_err / n),
            "sc_mae": sc_sum_abs_err / n,
            "sc_bias": sc_sum_err / n,
            "bins": {},
        }

        for bname in bin_names:
            bn = float(bin_count[bname]) if bin_count[bname] > 0 else 1.0
            results[year]["bins"][bname] = {
                "count": int(bin_count[bname]),
                "nc_rmse": np.sqrt(nc_bin_sq[bname] / bn),
                "nc_mae": nc_bin_abs[bname] / bn,
                "nc_bias": nc_bin_err[bname] / bn,
                "sc_rmse": np.sqrt(sc_bin_sq[bname] / bn),
                "sc_mae": sc_bin_abs[bname] / bn,
                "sc_bias": sc_bin_err[bname] / bn,
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

    # Plot: Static bias map
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    vabs = np.nanpercentile(np.abs(static_bias_map[sea_mask]), 98)
    im = ax.imshow(static_bias_map, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="auto", origin="upper")
    ax.set_title(f"Static bias map (mean of training years {args.train_years})")
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
