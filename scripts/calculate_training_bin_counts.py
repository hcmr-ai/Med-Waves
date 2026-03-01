#!/usr/bin/env python3
"""
Calculate per-bin pixel counts in the training set.

Loads all training .pt files (split by year from config), applies region
filtering, and counts how many valid pixels fall into each 1m wave-height
bin for the target column (corrected_VHM0).

Usage:
    python scripts/calculate_training_bin_counts.py
    python scripts/calculate_training_bin_counts.py --config src/configs/config_dnn.yaml
    python scripts/calculate_training_bin_counts.py --region mediterranean
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commons.helpers import get_file_list, split_files_by_year

SEA_BINS = [
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

GIBRALTAR_LON = -5.5


def _build_region_mask(tensor, feature_cols, region_filter):
    """Build a spatial boolean mask for the requested region.

    Returns a (H, W) bool tensor that is True for pixels in the region.
    """
    lat_idx = feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    lon_data = tensor[0, :, :, lon_idx]

    if region_filter == "mediterranean":
        return lon_data >= GIBRALTAR_LON
    elif region_filter == "atlantic":
        return lon_data < GIBRALTAR_LON
    else:
        raise ValueError(f"Unknown region_filter: {region_filter}")


def _parse_date_from_filename(file_path):
    """Extract date from filename like WAVEAN20180101... -> (2018, 1, 1)."""
    name = Path(file_path).stem
    m = re.search(r"WAVEAN(\d{4})(\d{2})(\d{2})", name)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None, None, None


def count_bins_for_file(args):
    """Process a single .pt file and return per-bin counts.

    Each .pt file has shape (24, H, W, C) with 24 hourly timesteps.
    We count every valid (non-NaN) pixel across all 24 hours.
    Also returns per-hour counts for each bin so callers can report
    which hours of which day had specific wave heights.
    """
    file_path, target_col, region_filter = args

    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        tensor = data["tensor"]  # (24, H, W, C)
        feature_cols = data["feature_cols"]

        target_idx = feature_cols.index(target_col)

        if region_filter:
            region_mask = _build_region_mask(tensor, feature_cols, region_filter)
        else:
            region_mask = None

        target_vals = tensor[:, :, :, target_idx]  # (24, H, W)
        valid_mask = ~torch.isnan(target_vals)

        if region_mask is not None:
            valid_mask = valid_mask & region_mask.unsqueeze(0)

        values = target_vals[valid_mask].numpy()

        bin_counts = {}
        for b in SEA_BINS:
            mask = (values >= b["min"]) & (values < b["max"])
            bin_counts[b["name"]] = int(mask.sum())

        above_max = int((values >= SEA_BINS[-1]["max"]).sum())
        bin_counts["above_15"] = above_max

        n_hours = tensor.shape[0]
        hourly_bin_counts = {}
        for hour in range(n_hours):
            hour_vals = target_vals[hour]  # (H, W)
            hour_valid = ~torch.isnan(hour_vals)
            if region_mask is not None:
                hour_valid = hour_valid & region_mask
            h_values = hour_vals[hour_valid].numpy()

            h_counts = {}
            for b in SEA_BINS:
                cnt = int(((h_values >= b["min"]) & (h_values < b["max"])).sum())
                if cnt > 0:
                    h_counts[b["name"]] = cnt
            above = int((h_values >= SEA_BINS[-1]["max"]).sum())
            if above > 0:
                h_counts["above_15"] = above
            if h_counts:
                hourly_bin_counts[hour] = h_counts

        year, month, day = _parse_date_from_filename(file_path)

        return {
            "file": str(file_path),
            "date": {"year": year, "month": month, "day": day},
            "total_valid_pixels": int(len(values)),
            "bins": bin_counts,
            "hourly_bins": hourly_bin_counts,
        }

    except Exception as e:
        return {"file": str(file_path), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-bin pixel counts in the training set"
    )
    parser.add_argument(
        "--config",
        default="src/configs/config_dnn.yaml",
        help="Path to DNN config YAML",
    )
    parser.add_argument(
        "--region",
        default=None,
        help='Override region filter (e.g. "mediterranean", "atlantic", or "none")',
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Override target column name (default: from config)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which data split to analyse (default: train)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for JSON output (default: auto-generated)",
    )
    parser.add_argument(
        "--show-dates",
        default=None,
        help='Show dates/months where a specific bin appeared (e.g. "extreme_12_13")',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    target_columns = data_cfg.get("target_columns", {"vhm0": "corrected_VHM0"})
    target_col = args.target_column or list(target_columns.values())[0]

    if args.region == "none":
        region_filter = None
    else:
        region_filter = args.region or data_cfg.get("region_filter", None)

    files = get_file_list(
        data_cfg["data_path"], data_cfg["file_pattern"], data_cfg.get("max_files")
    )
    print(f"Found {len(files)} total files in {data_cfg['data_path']}")

    train_files, val_files, test_files = split_files_by_year(
        files,
        train_year=data_cfg.get("train_year", 2021),
        val_year=data_cfg.get("val_year", 2022),
        test_year=data_cfg.get("test_year", 2023),
        val_months=data_cfg.get("val_months", []),
        test_months=data_cfg.get("test_months", []),
    )

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "all": files,
    }
    selected_files = split_map[args.split]
    print(f"Split '{args.split}': {len(selected_files)} files")
    print(f"Target column: {target_col}")
    print(f"Region filter: {region_filter or 'none'}")
    print()

    worker_args = [
        (fp, target_col, region_filter) for fp in selected_files
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(count_bins_for_file, a): a[0] for a in worker_args}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            results.append(future.result())

    totals = defaultdict(int)
    total_pixels = 0
    errors = []

    for r in results:
        if "error" in r:
            errors.append(r)
            continue
        total_pixels += r["total_valid_pixels"]
        for bin_name, count in r["bins"].items():
            totals[bin_name] += count

    print()
    print("=" * 70)
    print(f"  TRAINING SET BIN COUNTS  ({args.split} split)")
    print(f"  Target: {target_col}  |  Region: {region_filter or 'all'}")
    print("=" * 70)
    print(f"  Files processed : {len(results) - len(errors)}")
    if errors:
        print(f"  Files with errors: {len(errors)}")
    print(f"  Total valid pixels: {total_pixels:,}")
    print()
    print(f"  {'Bin':<20} {'Range':>12} {'Count':>14} {'%':>8}")
    print("  " + "-" * 56)

    for b in SEA_BINS:
        name = b["name"]
        count = totals.get(name, 0)
        pct = 100.0 * count / total_pixels if total_pixels else 0.0
        print(f"  {name:<20} {b['label']:>12} {count:>14,} {pct:>7.3f}%")

    above = totals.get("above_15", 0)
    pct_above = 100.0 * above / total_pixels if total_pixels else 0.0
    print(f"  {'above_15':<20} {'15.0m+':>12} {above:>14,} {pct_above:>7.3f}%")
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<20} {'':>12} {total_pixels:>14,} {'100.000%':>8}")
    print()

    cumulative = 0
    print(f"  {'Bin':<20} {'Cumulative %':>14}")
    print("  " + "-" * 36)
    for b in SEA_BINS:
        cumulative += totals.get(b["name"], 0)
        cum_pct = 100.0 * cumulative / total_pixels if total_pixels else 0.0
        print(f"  {b['name']:<20} {cum_pct:>13.3f}%")
    cumulative += above
    print(f"  {'above_15':<20} {100.0:>13.3f}%")
    print()

    # --- Per-date breakdown for a specific bin ---
    target_bin = args.show_dates
    date_details = {}
    if target_bin:
        for r in results:
            if "error" in r:
                continue
            file_count = r["bins"].get(target_bin, 0)
            if file_count == 0:
                continue
            d = r["date"]
            year, month, day = d["year"], d["month"], d["day"]
            date_key = f"{year}-{month:02d}-{day:02d}" if year else Path(r["file"]).stem

            hours_with_bin = []
            for hour, hcounts in sorted(r["hourly_bins"].items()):
                if target_bin in hcounts:
                    hours_with_bin.append((hour, hcounts[target_bin]))

            date_details[date_key] = {
                "total_pixels": file_count,
                "hours": hours_with_bin,
                "year": year,
                "month": month,
                "day": day,
            }

        print("=" * 70)
        print(f"  DATES WITH BIN '{target_bin}' PIXELS")
        print("=" * 70)

        if not date_details:
            print(f"  No files contained pixels in bin '{target_bin}'.")
        else:
            sorted_dates = sorted(date_details.keys())
            print(f"  {len(sorted_dates)} days found\n")

            # Per-month summary
            month_agg = defaultdict(lambda: {"days": 0, "pixels": 0})
            for dk in sorted_dates:
                info = date_details[dk]
                mk = f"{info['year']}-{info['month']:02d}" if info["year"] else "unknown"
                month_agg[mk]["days"] += 1
                month_agg[mk]["pixels"] += info["total_pixels"]

            print(f"  {'Month':<12} {'Days':>6} {'Pixels':>14}")
            print("  " + "-" * 34)
            for mk in sorted(month_agg.keys()):
                ma = month_agg[mk]
                print(f"  {mk:<12} {ma['days']:>6} {ma['pixels']:>14,}")
            print()

            # Per-day detail
            print(f"  {'Date':<14} {'Pixels':>10} {'Hours (UTC)':>40}")
            print("  " + "-" * 66)
            for dk in sorted_dates:
                info = date_details[dk]
                hour_strs = [f"{h:02d}h({c:,}px)" for h, c in info["hours"]]
                hours_summary = ", ".join(hour_strs) if hour_strs else "—"
                print(f"  {dk:<14} {info['total_pixels']:>10,} {hours_summary:>40}")
            print()

    # --- Save JSON ---
    output_path = args.output or f"training_bin_counts_{args.split}_{region_filter or 'all'}.json"
    output_data = {
        "config": args.config,
        "split": args.split,
        "target_column": target_col,
        "region_filter": region_filter,
        "total_valid_pixels": total_pixels,
        "bin_counts": dict(totals),
        "bins_definition": SEA_BINS,
        "errors": errors,
    }
    if target_bin and date_details:
        output_data["date_details_for_bin"] = {
            "bin": target_bin,
            "dates": {
                dk: {
                    "total_pixels": info["total_pixels"],
                    "hours": [{"hour": h, "pixels": c} for h, c in info["hours"]],
                }
                for dk, info in sorted(date_details.items())
            },
        }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}")

    if errors:
        print(f"\nFiles with errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e['file']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
