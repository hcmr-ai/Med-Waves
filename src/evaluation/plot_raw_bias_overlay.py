#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def parse_year_month(name: str):
    m = re.search(r"WAVEAN(\d{4})(\d{2})", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def region_mask(lat, lon, region):
    if region is None or region == "all":
        return np.ones_like(lat, dtype=bool)

    gibraltar_lon = -5.5
    biscay = (lat > 43.0) & (lon < 0.0)

    if region == "atlantic":
        return (lon < gibraltar_lon) | biscay
    if region == "mediterranean":
        return (lon >= gibraltar_lon) & (~biscay)
    if region == "aegean":
        return (
            (lat >= 35.0) & (lat <= 42.0) &
            (lon >= 23.0) & (lon <= 28.0)
        )
    raise ValueError(f"Unknown region: {region}")


def summarize(x):
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def extract_abs_bias_from_file(path: Path, raw_col: str, corrected_col: str, region: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    tensor = data["tensor"]        # (H,W,C) or (T,H,W,C)
    cols = data["feature_cols"]

    i_raw = cols.index(raw_col)
    i_cor = cols.index(corrected_col)
    i_lat = cols.index("latitude")
    i_lon = cols.index("longitude")

    def one_frame(frame):
        raw = frame[..., i_raw].numpy()
        cor = frame[..., i_cor].numpy()
        lat = frame[..., i_lat].numpy()
        lon = frame[..., i_lon].numpy()

        valid = np.isfinite(raw) & np.isfinite(cor) & np.isfinite(lat) & np.isfinite(lon)
        valid &= region_mask(lat, lon, region)
        if not np.any(valid):
            return np.array([], dtype=np.float32)
        return np.abs(raw - cor)[valid].astype(np.float32)

    if tensor.ndim == 3:
        return one_frame(tensor)

    parts = []
    for t in range(tensor.shape[0]):
        arr = one_frame(tensor[t])
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def collect_year_bias(data_path: Path, file_pattern: str, year: int, months, raw_col, corrected_col, region):
    files = sorted(data_path.rglob(file_pattern))
    selected = []
    for p in files:
        y, m = parse_year_month(p.name)
        if y != year:
            continue
        if months and m not in months:
            continue
        selected.append(p)

    values = []
    for p in selected:
        arr = extract_abs_bias_from_file(p, raw_col, corrected_col, region)
        if arr.size:
            values.append(arr)

    if values:
        return np.concatenate(values, axis=0), selected
    return np.array([], dtype=np.float32), selected


def write_txt_metadata(path: Path, meta: dict):
    lines = []
    lines.append("Raw Bias Overlay Metadata")
    lines.append("=========================")
    lines.append(f"data_path: {meta['data_path']}")
    lines.append(f"file_pattern: {meta['file_pattern']}")
    lines.append(f"region: {meta['region']}")
    lines.append(f"raw_column: {meta['raw_column']}")
    lines.append(f"corrected_column: {meta['corrected_column']}")
    lines.append(f"denoise_abs_threshold: {meta['denoise_abs_threshold']}")
    lines.append(f"years: {meta['years']}")
    lines.append("")

    for y in meta["years"]:
        ys = meta["per_year"][str(y)]
        lines.append(f"Year {y}")
        lines.append(f"  files_used: {ys['files_used']}")
        lines.append(f"  count: {ys['stats']['count']}")
        lines.append(f"  mean: {ys['stats']['mean']}")
        lines.append(f"  std: {ys['stats']['std']}")
        lines.append(f"  median: {ys['stats']['median']}")
        lines.append(f"  p90: {ys['stats']['p90']}")
        lines.append(f"  p95: {ys['stats']['p95']}")
        lines.append(f"  min: {ys['stats']['min']}")
        lines.append(f"  max: {ys['stats']['max']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description="Overlay |raw - corrected| distributions for two years from raw dataset files."
    )
    p.add_argument("--data-path", required=True, help="Root folder with WAVEAN*.pt files")
    p.add_argument("--file-pattern", default="WAVEAN*.pt")
    p.add_argument("--year-a", type=int, default=2017)
    p.add_argument("--year-b", type=int, default=2023)
    p.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    p.add_argument("--region", choices=["all", "atlantic", "mediterranean", "aegean"], default="atlantic")
    p.add_argument("--raw-column", default="VTM02")
    p.add_argument("--corrected-column", default="corrected_VTM02")
    p.add_argument("--denoise-abs-threshold", type=float, default=None)
    p.add_argument("--out-plot", required=True)
    p.add_argument("--out-metadata-json", required=True)
    p.add_argument("--out-metadata-txt", required=True)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--xmin", type=float, default=None)
    args = p.parse_args()

    data_path = Path(args.data_path)
    months = [int(x) for x in args.months.split(",") if x.strip()]
    years = [args.year_a, args.year_b]

    year_vals = {}
    year_files = {}
    for y in years:
        vals, files = collect_year_bias(
            data_path=data_path,
            file_pattern=args.file_pattern,
            year=y,
            months=months,
            raw_col=args.raw_column,
            corrected_col=args.corrected_column,
            region=args.region,
        )
        if args.denoise_abs_threshold is not None:
            vals = vals[vals > args.denoise_abs_threshold]
        year_vals[y] = vals
        year_files[y] = files

    # Plot
    plt.figure(figsize=(10, 6))
    for y in years:
        vals = year_vals[y]
        if vals.size > 0:
            sns.kdeplot(vals, linewidth=2, label=f"{y} (n={vals.size:,})")
        else:
            print(f"Warning: year {y} has no valid samples after filtering.")

    title = f"{args.region.title()} |{args.raw_column} - {args.corrected_column}| distribution"
    if args.denoise_abs_threshold is not None:
        title += f" (>{args.denoise_abs_threshold})"
    plt.title(title)
    plt.xlabel("Absolute bias")
    plt.ylabel("Density")
    if args.xmin is not None or args.xmax is not None:
        plt.xlim(left=args.xmin, right=args.xmax)

    plt.legend()
    plt.tight_layout()

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=300)
    plt.close()

    # Metadata
    meta = {
        "data_path": str(data_path),
        "file_pattern": args.file_pattern,
        "region": args.region,
        "raw_column": args.raw_column,
        "corrected_column": args.corrected_column,
        "denoise_abs_threshold": args.denoise_abs_threshold,
        "years": years,
        "months": months,
        "out_plot": str(out_plot),
        "per_year": {
            str(y): {
                "files_used": len(year_files[y]),
                "example_files": [str(p) for p in year_files[y][:5]],
                "stats": summarize(year_vals[y]),
            }
            for y in years
        },
    }

    out_json = Path(args.out_metadata_json)
    out_txt = Path(args.out_metadata_txt)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_txt_metadata(out_txt, meta)

    print(f"Saved plot: {out_plot}")
    print(f"Saved metadata json: {out_json}")
    print(f"Saved metadata txt: {out_txt}")


if __name__ == "__main__":
    main()
