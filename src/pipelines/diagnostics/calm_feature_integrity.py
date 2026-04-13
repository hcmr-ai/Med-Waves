#!/usr/bin/env python3
"""
Calm-regime feature integrity diagnostics.

Outputs:
  - calm_feature_missingness_by_year.csv
  - calm_feature_quantiles_by_year.csv
  - calm_feature_target_corr_by_year.csv
  - calm_missingness_heatmap_<bin>.png
  - calm_median_by_year_heatmap_<bin>.png
  - calm_corr_delta_heatmap_<bin>.png

This script is intentionally approximate for quantiles (reservoir sampling),
but exact for missingness and correlation moments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.commons.helpers import DNNConfig, get_file_list


GIBRALTAR_LON = -5.5
BISCAY_LAT = 43.0
BISCAY_LON = 0.0


class Reservoir:
    def __init__(self, max_size: int, rng: np.random.Generator):
        self.max_size = int(max_size)
        self.rng = rng
        self.n_seen = 0
        self.buf = np.empty(self.max_size, dtype=np.float32)
        self.size = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        vals = values.reshape(-1).astype(np.float32, copy=False)
        for v in vals:
            self.n_seen += 1
            if self.size < self.max_size:
                self.buf[self.size] = v
                self.size += 1
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.max_size:
                    self.buf[j] = v

    def quantiles(self, probs: List[float]) -> List[float]:
        if self.size == 0:
            return [math.nan] * len(probs)
        arr = self.buf[: self.size]
        return [float(np.quantile(arr, p)) for p in probs]


def parse_year(path: str) -> int | None:
    name = Path(path).name
    m = re.search(r"WAVEAN(20\d{2})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(20\d{2})", name)
    return int(m.group(1)) if m else None


def apply_region_mask(lat: np.ndarray, lon: np.ndarray, region_filter: str | None) -> np.ndarray:
    valid_geo = np.isfinite(lat) & np.isfinite(lon)
    if region_filter is None:
        return valid_geo
    biscay = (lat > BISCAY_LAT) & (lon < BISCAY_LON)
    if region_filter == "atlantic":
        region = (lon < GIBRALTAR_LON) | biscay
    elif region_filter == "mediterranean":
        region = (lon >= GIBRALTAR_LON) & (~biscay)
    else:
        raise ValueError(f"Unknown region_filter={region_filter}")
    return valid_geo & region


def sanitize_bin_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}_{hi:.1f}".replace(".", "p")


def heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    cbar_label: str,
    save_path: Path,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    if matrix.size == 0:
        return
    fig_h = max(6, 0.25 * len(row_labels))
    fig_w = max(8, 1.8 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = matrix.copy().astype(np.float64)
    if center_zero:
        vmax = np.nanpercentile(np.abs(data[np.isfinite(data)]), 98) if np.any(np.isfinite(data)) else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_progress(
    progress_path: Path | None,
    phase: str,
    year: int | None = None,
    file_idx: int | None = None,
    files_in_year: int | None = None,
    years_done: int | None = None,
    total_years: int | None = None,
    note: str | None = None,
) -> None:
    if progress_path is None:
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "year": year,
        "file_idx": file_idx,
        "files_in_year": files_in_year,
        "years_done": years_done,
        "total_years": total_years,
        "note": note,
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calm-regime feature integrity diagnostics")
    parser.add_argument("--config", type=str, default="src/configs/config_dnn.yaml")
    parser.add_argument("--output-dir", type=str, default="evaluation_results/data_audit/calm_feature_integrity")
    parser.add_argument("--bin-source", type=str, choices=["raw", "true"], default="raw")
    parser.add_argument(
        "--calm-bins",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.5],
        help="Bin edges for calm regime. Example: 0.0 0.1 0.2 0.5",
    )
    parser.add_argument("--max-files-per-year", type=int, default=None)
    parser.add_argument("--reservoir-size", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress-json",
        type=str,
        default=None,
        help="Optional path to write current progress as JSON.",
    )
    parser.add_argument(
        "--heartbeat-every-files",
        type=int,
        default=20,
        help="Print/write progress every N files within each year.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_json) if args.progress_json else None

    config = DNNConfig(args.config).config
    data_cfg = config["data"]
    data_path = data_cfg["data_path"]
    pattern = data_cfg["file_pattern"]
    region_filter = data_cfg.get("region_filter", None)
    excluded = set(data_cfg.get("excluded_columns", []))
    target_cols = list(data_cfg.get("target_columns", {"vhm0": "corrected_VHM0"}).values())
    primary_target = target_cols[0]
    raw_col = primary_target.replace("corrected_", "")

    files = get_file_list(data_path, pattern, max_files=data_cfg.get("max_files", None))
    year_to_files: Dict[int, List[str]] = defaultdict(list)
    for f in files:
        y = parse_year(f)
        if y is not None:
            year_to_files[y].append(f)

    years = sorted(year_to_files.keys())
    if not years:
        raise RuntimeError("No files with parsable years found.")
    write_progress(
        progress_path,
        phase="initialized",
        years_done=0,
        total_years=len(years),
        note=f"Found {len(files)} files across {len(years)} years",
    )

    calm_edges = args.calm_bins
    if len(calm_edges) < 2:
        raise ValueError("Need at least two calm bin edges.")
    calm_bins: List[Tuple[float, float]] = [(calm_edges[i], calm_edges[i + 1]) for i in range(len(calm_edges) - 1)]

    rng = np.random.default_rng(args.seed)

    # Stats containers keyed by (year, bin_label, feature)
    miss_stats = defaultdict(lambda: {"count_bin": 0, "count_nan": 0, "count_finite": 0})
    moment_stats = defaultdict(lambda: {"sum": 0.0, "sum_sq": 0.0})
    corr_stats = defaultdict(
        lambda: {"n": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0}
    )
    reservoirs = {}
    feature_names: List[str] = []

    for y_idx, year in enumerate(years):
        year_files = sorted(year_to_files[year])
        if args.max_files_per_year is not None:
            year_files = year_files[: args.max_files_per_year]
        print(f"[audit] year={year}: files={len(year_files)}")
        write_progress(
            progress_path,
            phase="processing_year",
            year=year,
            file_idx=0,
            files_in_year=len(year_files),
            years_done=y_idx,
            total_years=len(years),
            note="Starting year",
        )

        for f_idx, fp in enumerate(year_files, start=1):
            blob = torch.load(fp, map_location="cpu", weights_only=False)
            tensor = blob["tensor"]
            feature_cols = blob["feature_cols"]

            if not feature_names:
                feature_names = [c for c in feature_cols if c not in excluded and c not in target_cols]
                if raw_col not in feature_cols or primary_target not in feature_cols:
                    raise RuntimeError(f"Required columns missing in {fp}: {raw_col}, {primary_target}")
                print(f"[audit] using {len(feature_names)} input features")

            # Promote hourly tensor (H,W,C) to (1,H,W,C)
            if tensor.ndim == 3:
                arr = tensor.unsqueeze(0).numpy()
            elif tensor.ndim == 4:
                arr = tensor.numpy()
            else:
                continue

            idx_raw = feature_cols.index(raw_col)
            idx_corr = feature_cols.index(primary_target)
            idx_lat = feature_cols.index("latitude")
            idx_lon = feature_cols.index("longitude")
            feat_indices = [feature_cols.index(f) for f in feature_names]

            for t in range(arr.shape[0]):
                frame = arr[t]  # (H, W, C)
                raw = frame[..., idx_raw]
                corr = frame[..., idx_corr]
                lat = frame[..., idx_lat]
                lon = frame[..., idx_lon]

                region_mask = apply_region_mask(lat, lon, region_filter)
                base_valid = region_mask & np.isfinite(raw) & np.isfinite(corr)
                if not np.any(base_valid):
                    continue

                delta = corr - raw
                gate_values = raw if args.bin_source == "raw" else corr

                for lo, hi in calm_bins:
                    bin_label = f"{lo:.1f}-{hi:.1f}"
                    m_bin = base_valid & (gate_values >= lo) & (gate_values < hi)
                    n_bin = int(np.sum(m_bin))
                    if n_bin == 0:
                        continue

                    for feat_name, fi in zip(feature_names, feat_indices, strict=False):
                        key = (year, bin_label, feat_name)
                        feat = frame[..., fi]
                        finite_feat = np.isfinite(feat)
                        finite_xy = m_bin & finite_feat & np.isfinite(delta)

                        miss_stats[key]["count_bin"] += n_bin
                        miss_stats[key]["count_nan"] += int(np.sum(m_bin & (~finite_feat)))
                        miss_stats[key]["count_finite"] += int(np.sum(finite_xy))

                        if np.any(finite_xy):
                            x = feat[finite_xy].astype(np.float64, copy=False)
                            y = delta[finite_xy].astype(np.float64, copy=False)

                            moment_stats[key]["sum"] += float(np.sum(x))
                            moment_stats[key]["sum_sq"] += float(np.sum(x * x))

                            c = corr_stats[key]
                            c["n"] += int(x.size)
                            c["sum_x"] += float(np.sum(x))
                            c["sum_y"] += float(np.sum(y))
                            c["sum_x2"] += float(np.sum(x * x))
                            c["sum_y2"] += float(np.sum(y * y))
                            c["sum_xy"] += float(np.sum(x * y))

                            if key not in reservoirs:
                                reservoirs[key] = Reservoir(args.reservoir_size, rng)
                            reservoirs[key].update(x)

            if (
                f_idx == 1
                or f_idx == len(year_files)
                or (args.heartbeat_every_files > 0 and f_idx % args.heartbeat_every_files == 0)
            ):
                print(f"[audit] year={year} progress: {f_idx}/{len(year_files)} files")
                write_progress(
                    progress_path,
                    phase="processing_year",
                    year=year,
                    file_idx=f_idx,
                    files_in_year=len(year_files),
                    years_done=y_idx,
                    total_years=len(years),
                    note="In progress",
                )

    # Build CSV rows
    missing_rows = []
    quant_rows = []
    corr_rows = []
    quant_probs = [0.01, 0.5, 0.99]

    all_keys = sorted(miss_stats.keys(), key=lambda x: (x[0], x[1], x[2]))
    for key in all_keys:
        year, bin_label, feat = key
        m = miss_stats[key]
        n_bin = m["count_bin"]
        n_nan = m["count_nan"]
        n_f = m["count_finite"]
        nan_pct = 100.0 * n_nan / max(1, n_bin)

        missing_rows.append(
            {
                "year": year,
                "bin_label": bin_label,
                "feature": feat,
                "count_bin": n_bin,
                "count_nan": n_nan,
                "nan_pct": nan_pct,
                "count_finite_xy": n_f,
            }
        )

        r = reservoirs.get(key, None)
        q01, q50, q99 = (r.quantiles(quant_probs) if r is not None else [math.nan, math.nan, math.nan])
        ms = moment_stats[key]
        mean = ms["sum"] / max(1, n_f)
        var = ms["sum_sq"] / max(1, n_f) - mean * mean
        std = math.sqrt(max(0.0, var))
        quant_rows.append(
            {
                "year": year,
                "bin_label": bin_label,
                "feature": feat,
                "count_finite_xy": n_f,
                "mean": mean,
                "std": std,
                "q01": q01,
                "q50": q50,
                "q99": q99,
            }
        )

        c = corr_stats[key]
        n = c["n"]
        if n > 1:
            num = n * c["sum_xy"] - c["sum_x"] * c["sum_y"]
            denx = n * c["sum_x2"] - c["sum_x"] * c["sum_x"]
            deny = n * c["sum_y2"] - c["sum_y"] * c["sum_y"]
            den = math.sqrt(max(0.0, denx) * max(0.0, deny))
            corr = float(num / den) if den > 0 else math.nan
        else:
            corr = math.nan
        corr_rows.append(
            {
                "year": year,
                "bin_label": bin_label,
                "feature": feat,
                "count_finite_xy": n,
                "corr_feature_to_delta": corr,
            }
        )

    def write_csv(path: Path, rows: List[Dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "calm_feature_missingness_by_year.csv", missing_rows)
    write_csv(out_dir / "calm_feature_quantiles_by_year.csv", quant_rows)
    write_csv(out_dir / "calm_feature_target_corr_by_year.csv", corr_rows)
    write_progress(
        progress_path,
        phase="writing_plots",
        years_done=len(years),
        total_years=len(years),
        note="CSVs written, generating plots",
    )

    # Heatmaps by bin: rows=features, cols=years
    years_s = [str(y) for y in years]
    bins_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in calm_bins]

    miss_lookup = {(r["year"], r["bin_label"], r["feature"]): r["nan_pct"] for r in missing_rows}
    q50_lookup = {(r["year"], r["bin_label"], r["feature"]): r["q50"] for r in quant_rows}
    corr_lookup = {(r["year"], r["bin_label"], r["feature"]): r["corr_feature_to_delta"] for r in corr_rows}

    for b in bins_labels:
        mtx_miss = np.full((len(feature_names), len(years)), np.nan, dtype=np.float64)
        mtx_q50 = np.full((len(feature_names), len(years)), np.nan, dtype=np.float64)
        mtx_corr = np.full((len(feature_names), len(years)), np.nan, dtype=np.float64)
        for i, feat in enumerate(feature_names):
            for j, year in enumerate(years):
                mtx_miss[i, j] = miss_lookup.get((year, b, feat), np.nan)
                mtx_q50[i, j] = q50_lookup.get((year, b, feat), np.nan)
                mtx_corr[i, j] = corr_lookup.get((year, b, feat), np.nan)

        bslug = sanitize_bin_label(*tuple(map(float, b.split("-"))))
        heatmap(
            mtx_miss,
            feature_names,
            years_s,
            title=f"Missingness (%) in bin {b} ({args.bin_source})",
            cbar_label="NaN %",
            save_path=out_dir / f"calm_missingness_heatmap_{bslug}.png",
            cmap="magma",
            center_zero=False,
        )
        heatmap(
            mtx_q50,
            feature_names,
            years_s,
            title=f"Feature median (q50) in bin {b} ({args.bin_source})",
            cbar_label="Median value",
            save_path=out_dir / f"calm_median_by_year_heatmap_{bslug}.png",
            cmap="viridis",
            center_zero=False,
        )
        heatmap(
            mtx_corr,
            feature_names,
            years_s,
            title=f"Corr(feature, corrected-raw) in bin {b} ({args.bin_source})",
            cbar_label="Pearson r",
            save_path=out_dir / f"calm_corr_delta_heatmap_{bslug}.png",
            cmap="coolwarm",
            center_zero=True,
        )

    write_progress(
        progress_path,
        phase="done",
        years_done=len(years),
        total_years=len(years),
        note=f"Outputs in {out_dir}",
    )
    print(f"[done] wrote diagnostics to: {out_dir}")


if __name__ == "__main__":
    main()

