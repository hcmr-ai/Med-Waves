"""
Debug / test script for TimestepPatchWaveDataset.

Tests the valid sampling combinations:
  1. random     + plain DataLoader   (natural bin distribution)
  2. stratified + plain DataLoader   (round-robin bins via idx % n_bins)
  3. stratified + BalancedBinBatchSampler (sampler controls bins)
  4. exhaustive + plain DataLoader   (all non-overlapping tiles, shuffled)

Prints tensor stats, bin distributions, and produces georeferenced maps.

Usage:
    poetry run python scripts/debug_patch_dataset.py
    poetry run python scripts/debug_patch_dataset.py --region atlantic --batch-size 8
    poetry run python scripts/debug_patch_dataset.py --num-files 5 --steps 8
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.commons.datasets.samplers import BalancedBinBatchSampler
from src.commons.datasets.time_step_patch_dataset import (
    PatchSamplingConfig,
    TimestepPatchWaveDataset,
)

EXCLUDED_COLUMNS = [
    "time", "latitude", "longitude", "timestamp",
    "corrected_VTM02", "WDIR", "VMDR",
]

OUT_DIR = "debug_output"


# ------------------------------------------------------------------ helpers
def print_sample_stats(tag, X, y, mask, vhm0, patch_bin, i0, j0, ph, pw):
    y_valid = y[mask]
    print(f"\n{'='*60}")
    print(f"{tag}")
    print(f"{'='*60}")
    print(f"  Patch anchor : ({i0}, {j0})  size: {ph}x{pw}  bin: {patch_bin}")
    print(f"  X            : {X.shape}  range [{X.min():.4f}, {X.max():.4f}]")
    if y_valid.numel() > 0:
        print(f"  y (sea)      : {y.shape}  range [{y_valid.min():.4f}, {y_valid.max():.4f}]")
    else:
        print(f"  y (sea)      : {y.shape}  NO valid sea pixels")
    print(f"  mask         : {mask.shape}  sea {mask.sum().item()}/{mask.numel()} ({mask.float().mean()*100:.1f}%)")
    print(f"  NaN in X     : {torch.isnan(X).any().item()}")
    print(f"  NaN in y sea : {torch.isnan(y_valid).any().item() if y_valid.numel() > 0 else 'N/A'}")


def print_batch_stats(tag, X_b, y_b, mask_b, vhm0_b, bins_b, coords_b):
    yb_valid = y_b[mask_b]
    print(f"\n{'='*60}")
    print(f"{tag}")
    print(f"{'='*60}")
    print(f"  X_batch      : {X_b.shape}   dtype={X_b.dtype}")
    print(f"  y_batch      : {y_b.shape}   dtype={y_b.dtype}")
    print(f"  mask_batch   : {mask_b.shape}   dtype={mask_b.dtype}")
    print(f"  vhm0_batch   : {vhm0_b.shape}   dtype={vhm0_b.dtype}")
    print(f"  bins         : {bins_b.tolist()}")
    print(f"  coords       : i0={coords_b[0].tolist()}, j0={coords_b[1].tolist()}")
    print(f"  X range      : [{X_b.min():.4f}, {X_b.max():.4f}]")
    if yb_valid.numel() > 0:
        print(f"  y range (sea): [{yb_valid.min():.4f}, {yb_valid.max():.4f}]")
    else:
        print(f"  y range (sea): NO valid sea pixels")
    print(f"  sea pixels   : {mask_b.sum().item()} / {mask_b.numel()} ({mask_b.float().mean()*100:.1f}%)")
    print(f"  NaN in X     : {torch.isnan(X_b).any().item()}")
    print(f"  NaN in y sea : {torch.isnan(yb_valid).any().item() if yb_valid.numel() > 0 else 'N/A'}")


def plot_batch(X_b, y_b, mask_b, vhm0_b, bins_b, coords_b,
               lat_grid, lon_grid, ph, pw, title, out_path):
    proj = ccrs.PlateCarree()
    n_show = min(X_b.shape[0], 4)
    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 10),
                             subplot_kw={"projection": proj})
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_show):
        i0 = coords_b[0][col].item()
        j0 = coords_b[1][col].item()
        lat_p = lat_grid[i0:i0 + ph, j0:j0 + pw]
        lon_p = lon_grid[i0:i0 + ph, j0:j0 + pw]
        mask_s = mask_b[col].squeeze().numpy().astype(bool)

        # Row 0: VHM0
        ax = axes[0, col]
        v = vhm0_b[col].squeeze().numpy().copy()
        v[~mask_s] = np.nan
        im = ax.pcolormesh(lon_p, lat_p, v, cmap="viridis",
                           transform=proj, shading="auto")
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none")
        plt.colorbar(im, ax=ax, shrink=0.7, label="m")
        ax.set_title(f"VHM0  bin={bins_b[col].item()}\n({i0},{j0})", fontsize=10)

        # Row 1: Target bias
        ax = axes[1, col]
        t = y_b[col].squeeze().numpy().copy()
        t[~mask_s] = np.nan
        vabs = max(abs(np.nanmin(t)), abs(np.nanmax(t)), 1e-6)
        im = ax.pcolormesh(lon_p, lat_p, t, cmap="RdBu_r",
                           vmin=-vabs, vmax=vabs,
                           transform=proj, shading="auto")
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none")
        plt.colorbar(im, ax=ax, shrink=0.7, label="m")
        ax.set_title("Bias (corr \u2212 raw)", fontsize=10)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved \u2192 {out_path}")


# ------------------------------------------------------------------ main
def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)

    pt_files = sorted(glob.glob(f"{args.data_dir}/WAVEAN*.pt"))[: args.num_files]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {args.data_dir}")
    print(f"Using {len(pt_files)} file(s): {[os.path.basename(f) for f in pt_files]}")

    patch_cfg = PatchSamplingConfig(
        patch_size=(args.patch_h, args.patch_w),
        min_valid_fraction=args.min_sea,
    )

    # ================================================================
    #  Test the valid combinations:
    #    1. random     + plain DataLoader  (natural bin distribution)
    #    2. stratified + plain DataLoader  (round-robin bin via idx % n_bins)
    #    3. stratified + BalancedBinBatchSampler (sampler controls bins)
    #    4. exhaustive + plain DataLoader  (all tiles, shuffled)
    # ================================================================
    tests = [
        ("random",     "plain"),
        ("stratified", "plain"),
        ("stratified", "sampler"),
        ("exhaustive", "plain"),
    ]

    for mode, loader_type in tests:
        label = f"{mode}_{loader_type}"
        print(f"\n{'#'*60}")
        print(f"#  {mode.upper()} + {loader_type.upper()}")
        print(f"{'#'*60}")

        ds = TimestepPatchWaveDataset(
            file_paths=pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED_COLUMNS,
            region_filter=args.region,
            return_coords=True,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode=mode,
            patch_cfg=patch_cfg,
            seed=42,
        )
        ph, pw = ds.patch_cfg.patch_size
        lat_grid, lon_grid = ds.get_coordinates()
        n_bins = len(ds.patch_cfg.bin_edges_m) + 1

        print(f"\n  Dataset length: {len(ds)} samples")
        if mode == "exhaustive" and ds.tile_grid is not None:
            print(f"  Tile grid    : {len(ds.tile_grid)} tiles/frame")

        # ---------- single sample ----------
        X, y, mask, vhm0, patch_bin, (i0, j0) = ds[0]
        print_sample_stats(
            f"SINGLE SAMPLE  mode={mode}", X, y, mask, vhm0, patch_bin, i0, j0, ph, pw,
        )

        # ---------- build loader ----------
        if loader_type == "sampler":
            sampler = BalancedBinBatchSampler(
                dataset_len=len(ds),
                n_bins=n_bins,
                batch_size=args.batch_size,
                bins_per_batch=None,
                steps_per_epoch=args.steps,
                seed=123,
            )
            loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
        else:
            shuffle = (mode == "exhaustive")
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

        batch = next(iter(loader))
        X_b, y_b, mask_b, vhm0_b, bins_b, coords_b = batch

        print_batch_stats(
            f"BATCH  {label}  bs={args.batch_size}" + (f"  n_bins={n_bins}" if loader_type == "sampler" else ""),
            X_b, y_b, mask_b, vhm0_b, bins_b, coords_b,
        )
        plot_batch(
            X_b, y_b, mask_b, vhm0_b, bins_b, coords_b,
            lat_grid, lon_grid, ph, pw,
            title=f"{label}  |  bs={args.batch_size}  |  patch={ph}\u00d7{pw}",
            out_path=os.path.join(OUT_DIR, f"batch_{label}.png"),
        )

        # ---------- bin distribution ----------
        max_batches = args.steps
        if mode == "exhaustive":
            max_batches = min(args.steps, len(loader))
        print(f"\n  Bin distribution (first {max_batches} batches):")
        all_bins = []
        for i, b in enumerate(loader):
            all_bins.extend(b[4].tolist())
            if i + 1 >= max_batches:
                break
        bin_counts = Counter(sorted(all_bins))
        print(f"  {bin_counts}")
        total = sum(bin_counts.values())
        for bin_id in sorted(bin_counts):
            print(f"    bin {bin_id}: {bin_counts[bin_id]:>4}  ({bin_counts[bin_id]/total*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"All outputs saved to {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug TimestepPatchWaveDataset")
    parser.add_argument("--data-dir", default="/opt/dlami/nvme/preprocessed_subsampled_step_5/")
    parser.add_argument("--region", default="mediterranean", choices=["mediterranean", "atlantic", None])
    parser.add_argument("--num-files", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4, help="Batches to test for bin distribution")
    parser.add_argument("--patch-h", type=int, default=64)
    parser.add_argument("--patch-w", type=int, default=160)
    parser.add_argument("--min-sea", type=float, default=0.2)
    args = parser.parse_args()
    run(args)
