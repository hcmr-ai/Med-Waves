"""
Debug / test script for CachedWaveDataset (full-frame training dataset).

Tests:
  1. Full-frame (no patching, no subsampling) — current training setup
  2. Full-frame + region filtering
  3. Full-frame + random patch crop (optional)

Prints tensor stats, produces georeferenced maps.

Usage:
    poetry run python scripts/debug_cached_dataset.py
    poetry run python scripts/debug_cached_dataset.py --region atlantic --batch-size 4
    poetry run python scripts/debug_cached_dataset.py --patch-h 64 --patch-w 160
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
from src.commons.datasets.cache_wave_dataset import CachedWaveDataset

EXCLUDED_COLUMNS = [
    "time", "latitude", "longitude", "timestamp",
    "corrected_VTM02", "WDIR", "VMDR",
]

OUT_DIR = "debug_output"


# ------------------------------------------------------------------ helpers
def print_sample_stats(tag, X, y, mask, vhm0):
    y_valid = y[mask]
    print(f"\n{'='*60}")
    print(f"{tag}")
    print(f"{'='*60}")
    print(f"  X            : {X.shape}  dtype={X.dtype}")
    print(f"  y            : {y.shape}  dtype={y.dtype}")
    print(f"  mask         : {mask.shape}  dtype={mask.dtype}")
    print(f"  vhm0         : {vhm0.shape}")
    print(f"  X range      : [{X.min():.4f}, {X.max():.4f}]")
    if y_valid.numel() > 0:
        print(f"  y range (sea): [{y_valid.min():.4f}, {y_valid.max():.4f}]")
    else:
        print(f"  y range (sea): NO valid sea pixels")
    print(f"  sea pixels   : {mask.sum().item()} / {mask.numel()} ({mask.float().mean()*100:.1f}%)")
    print(f"  NaN in X     : {torch.isnan(X).any().item()}")
    print(f"  NaN in y sea : {torch.isnan(y_valid).any().item() if y_valid.numel() > 0 else 'N/A'}")


def print_batch_stats(tag, X_b, y_b, mask_b, vhm0_b):
    yb_valid = y_b[mask_b]
    print(f"\n{'='*60}")
    print(f"{tag}")
    print(f"{'='*60}")
    print(f"  X_batch      : {X_b.shape}   dtype={X_b.dtype}")
    print(f"  y_batch      : {y_b.shape}   dtype={y_b.dtype}")
    print(f"  mask_batch   : {mask_b.shape}   dtype={mask_b.dtype}")
    print(f"  vhm0_batch   : {vhm0_b.shape}   dtype={vhm0_b.dtype}")
    print(f"  X range      : [{X_b.min():.4f}, {X_b.max():.4f}]")
    if yb_valid.numel() > 0:
        print(f"  y range (sea): [{yb_valid.min():.4f}, {yb_valid.max():.4f}]")
    else:
        print(f"  y range (sea): NO valid sea pixels")
    print(f"  sea pixels   : {mask_b.sum().item()} / {mask_b.numel()} ({mask_b.float().mean()*100:.1f}%)")
    print(f"  NaN in X     : {torch.isnan(X_b).any().item()}")
    print(f"  NaN in y sea : {torch.isnan(yb_valid).any().item() if yb_valid.numel() > 0 else 'N/A'}")


def plot_full_frame(y_b, mask_b, vhm0_b, lat_grid, lon_grid, title, out_path):
    """Plot full-frame maps for up to 4 samples in a batch."""
    proj = ccrs.PlateCarree()
    n_show = min(y_b.shape[0], 4)
    fig, axes = plt.subplots(2, n_show, figsize=(6 * n_show, 10),
                             subplot_kw={"projection": proj})
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_show):
        mask_s = mask_b[col].squeeze().numpy().astype(bool)

        # Row 0: VHM0
        ax = axes[0, col]
        v = vhm0_b[col].squeeze().numpy().copy()
        v[~mask_s] = np.nan
        im = ax.pcolormesh(lon_grid, lat_grid, v, cmap="viridis",
                           transform=proj, shading="auto")
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="none")
        plt.colorbar(im, ax=ax, shrink=0.7, label="m")
        ax.set_title(f"VHM0 (sample {col})", fontsize=10)

        # Row 1: Target bias
        ax = axes[1, col]
        t = y_b[col].squeeze().numpy().copy()
        t[~mask_s] = np.nan
        vabs = max(abs(np.nanmin(t)), abs(np.nanmax(t)), 1e-6)
        im = ax.pcolormesh(lon_grid, lat_grid, t, cmap="RdBu_r",
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

    patch_size = None
    if args.patch_h and args.patch_w:
        patch_size = (args.patch_h, args.patch_w)

    # ================================================================
    #  Test configurations:
    #    1. No region filter (full grid)
    #    2. With region filter
    #    3. With region filter + patch crop (if --patch-h/--patch-w given)
    # ================================================================
    tests = [
        {"label": "full_no_filter", "region": None,         "patch": None},
        {"label": f"full_{args.region}", "region": args.region, "patch": None},
    ]
    if patch_size is not None:
        tests.append({
            "label": f"patch_{args.patch_h}x{args.patch_w}_{args.region}",
            "region": args.region,
            "patch": patch_size,
        })

    for cfg in tests:
        label = cfg["label"]
        print(f"\n{'#'*60}")
        print(f"#  {label.upper()}")
        print(f"{'#'*60}")

        ds = CachedWaveDataset(
            file_paths=pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED_COLUMNS,
            predict_bias=True,
            patch_size=cfg["patch"],
            region_filter=cfg["region"],
            use_cache=True,
        )

        lat_grid, lon_grid = ds.get_coordinates()

        print(f"\n  Dataset length : {len(ds)} samples")
        print(f"  Region filter  : {cfg['region']}")
        print(f"  Patch size     : {cfg['patch']}")

        # ---------- single sample ----------
        sample = ds[0]
        X, y, mask, vhm0 = sample
        print_sample_stats(f"SINGLE SAMPLE  {label}", X, y, mask, vhm0)

        # ---------- batch ----------
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        batch = next(iter(loader))
        X_b, y_b, mask_b, vhm0_b = batch

        print_batch_stats(
            f"BATCH  {label}  bs={args.batch_size}",
            X_b, y_b, mask_b, vhm0_b,
        )

        # ---------- plot ----------
        plot_full_frame(
            y_b, mask_b, vhm0_b,
            lat_grid, lon_grid,
            title=f"{label}  |  bs={args.batch_size}  |  shape={X_b.shape[-2]}x{X_b.shape[-1]}",
            out_path=os.path.join(OUT_DIR, f"cached_{label}.png"),
        )

        # ---------- VHM0 bin distribution across a few samples ----------
        print(f"\n  VHM0 bin distribution (first {min(args.steps * args.batch_size, len(ds))} samples):")
        all_bins = []
        bin_edges = [2.0, 4.0]
        for i, b in enumerate(loader):
            vhm0_batch = b[3]
            mask_batch = b[2]
            for s in range(vhm0_batch.shape[0]):
                v_sea = vhm0_batch[s][mask_batch[s]]
                if v_sea.numel() > 0:
                    p90 = v_sea.quantile(0.90).item()
                    bin_id = 0
                    for edge in bin_edges:
                        if p90 >= edge:
                            bin_id += 1
                    all_bins.append(bin_id)
            if i + 1 >= args.steps:
                break
        bin_counts = Counter(sorted(all_bins))
        total = sum(bin_counts.values())
        print(f"  {bin_counts}")
        for bin_id in sorted(bin_counts):
            print(f"    bin {bin_id} (p90 {'<' if bin_id == 0 else '>='}{bin_edges[min(bin_id, len(bin_edges))-1] if bin_id > 0 else bin_edges[0]}m): "
                  f"{bin_counts[bin_id]:>4}  ({bin_counts[bin_id]/total*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"All outputs saved to {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug CachedWaveDataset")
    parser.add_argument("--data-dir", default="/opt/dlami/nvme/preprocessed_subsampled_step_5/")
    parser.add_argument("--region", default="mediterranean", choices=["mediterranean", "atlantic"])
    parser.add_argument("--num-files", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4, help="Batches to scan for bin distribution")
    parser.add_argument("--patch-h", type=int, default=None, help="Patch height (omit for full-frame)")
    parser.add_argument("--patch-w", type=int, default=None, help="Patch width (omit for full-frame)")
    args = parser.parse_args()
    run(args)
