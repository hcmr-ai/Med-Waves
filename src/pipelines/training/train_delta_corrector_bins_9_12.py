#!/usr/bin/env python3
"""
Train DeltaCorrector on training set, filtered to wave height bins 9-12m.

Fits mean or median bias between VHM0 (uncorrected) and corrected_VHM0
for pixels in bins 9-12m only. Saves the fitted model for use during evaluation.

Usage:
    python scripts/train_delta_corrector_bins_9_12.py
    python scripts/train_delta_corrector_bins_9_12.py --config src/configs/config_dnn.yaml
    python scripts/train_delta_corrector_bins_9_12.py --method median_bias --output correctors/delta_bins_9_12.joblib
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifiers.delta_corrector import DeltaCorrector
from src.commons.helpers import get_file_list, split_files_by_year

GIBRALTAR_LON = -5.5
BIN_MIN, BIN_MAX = 9.0, 12.0  # Bins 9-12m


def _build_region_mask(tensor, feature_cols, region_filter):
    """Build (H, W) boolean mask for the requested region."""
    feature_cols.index("latitude")
    lon_idx = feature_cols.index("longitude")
    # Handle both (H,W,C) and (24,H,W,C)
    if tensor.ndim == 4:
        lon_data = tensor[0, :, :, lon_idx]
    else:
        lon_data = tensor[:, :, lon_idx]

    if region_filter == "mediterranean":
        return lon_data >= GIBRALTAR_LON
    elif region_filter == "atlantic":
        return lon_data < GIBRALTAR_LON
    raise ValueError(f"Unknown region_filter: {region_filter}")


def extract_bins_9_12_from_file(args):
    """Extract (VHM0, corrected_VHM0) for pixels in bins 9-12m from a single .pt file."""
    file_path, region_filter = args

    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        tensor = data["tensor"]
        feature_cols = data["feature_cols"]

        vhm0_idx = feature_cols.index("VHM0")
        corrected_idx = feature_cols.index("corrected_VHM0")

        if tensor.ndim == 4:
            # Daily: (24, H, W, C)
            target_vals = tensor[:, :, :, corrected_idx]
            vhm0_vals = tensor[:, :, :, vhm0_idx]
        else:
            # Hourly: (H, W, C)
            target_vals = tensor[:, :, corrected_idx].unsqueeze(0)
            vhm0_vals = tensor[:, :, vhm0_idx].unsqueeze(0)

        valid_mask = ~torch.isnan(target_vals) & ~torch.isnan(vhm0_vals)
        bin_mask = (target_vals >= BIN_MIN) & (target_vals < BIN_MAX)
        mask = valid_mask & bin_mask

        if region_filter:
            region_mask = _build_region_mask(tensor, feature_cols, region_filter)
            if tensor.ndim == 4:
                region_mask = region_mask.unsqueeze(0).expand_as(mask)
            mask = mask & region_mask

        n = mask.sum().item()
        if n == 0:
            return None

        vhm0_flat = vhm0_vals[mask].numpy()
        corrected_flat = target_vals[mask].numpy()

        return {"VHM0": vhm0_flat, "corrected_VHM0": corrected_flat}

    except Exception as e:
        return {"error": str(e), "file": str(file_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Train DeltaCorrector on training set (bins 9-12m)"
    )
    parser.add_argument(
        "--config",
        default="src/configs/config_dnn.yaml",
        help="Path to DNN config YAML",
    )
    parser.add_argument(
        "--region",
        default=None,
        help='Region filter: "mediterranean", "atlantic", or "none"',
    )
    parser.add_argument(
        "--method",
        default="mean_bias",
        choices=["mean_bias", "median_bias"],
        help="Bias calculation method",
    )
    parser.add_argument(
        "--output",
        default="correctors/delta_corrector_bins_9_12.joblib",
        help="Output path for fitted model",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max files to process (for testing)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    data_path = data_cfg["data_path"]
    file_pattern = data_cfg.get("file_pattern", "WAVEAN*.pt")
    train_year = data_cfg.get("train_year", [2018, 2019, 2020, 2021])
    val_year = data_cfg.get("val_year", [2022])
    val_months = data_cfg.get("val_months", list(range(1, 13)))
    test_year = data_cfg.get("test_year", [2023])
    test_months = data_cfg.get("test_months", list(range(1, 13)))

    region_filter = args.region
    if region_filter and region_filter.lower() == "none":
        region_filter = None
    elif region_filter is None and "region_filter" in data_cfg:
        region_filter = data_cfg["region_filter"]

    files = get_file_list(data_path, file_pattern, max_files=args.max_files)
    train_files, _, _ = split_files_by_year(
        files,
        train_year=train_year,
        val_year=val_year,
        test_year=test_year,
        val_months=val_months,
        test_months=test_months,
    )

    if not train_files:
        print("No training files found. Exiting.")
        sys.exit(1)

    print(f"Training DeltaCorrector (bins {BIN_MIN}-{BIN_MAX}m)")
    print(f"  Config: {args.config}")
    print(f"  Train files: {len(train_files)}")
    print(f"  Region: {region_filter or 'all'}")
    print(f"  Method: {args.method}")
    print()

    worker_args = [(fp, region_filter) for fp in train_files]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_bins_9_12_from_file, a): a for a in worker_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
            results.append(future.result())

    vhm0_list, corrected_list = [], []
    errors = 0
    for r in results:
        if r is None:
            continue
        if "error" in r:
            errors += 1
            continue
        vhm0_list.append(r["VHM0"])
        corrected_list.append(r["corrected_VHM0"])

    if errors:
        print(f"  Files with errors: {errors}")

    if not vhm0_list:
        print("No pixels in bins 9-12m found. Exiting.")
        sys.exit(1)

    vhm0_all = np.concatenate(vhm0_list)
    corrected_all = np.concatenate(corrected_list)

    df = pl.DataFrame({"VHM0": vhm0_all, "corrected_VHM0": corrected_all})
    print(f"  Total pixels in bins 9-12m: {len(df):,}")
    print()

    corrector = DeltaCorrector(method=args.method)
    corrector.fit(df, variables=["VHM0"], corrected_suffix="corrected_")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrector.save_model(str(output_path))
    print(f"\nSaved model to {output_path}")


if __name__ == "__main__":
    main()
