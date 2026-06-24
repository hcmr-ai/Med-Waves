"""Mediterranean-only: per grid point, plots/<csv_stem>/<lat_lon>/ (value + abs-error PNGs, map, CSVs) + map_overview.png."""

from __future__ import annotations

import sys
import seaborn as sns

import matplotlib
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches

matplotlib.use("Agg")



from load_csv import DATA_PATH, load_data
from plot_point_evolution import (
    lat_lon_dirname,
    timeseries_for_point,
    abs_error_means,
    with_point_key,
    weekly_mean_timeseries,
    fifteen_day_mean_timeseries,
    monthly_mean_timeseries,
    timeseries_table_for_export,
    write_absdiff_csv,
    save_15day_timeseries_csvs,
    plot_evolution,
    plot_abs_errors,
)
def plot_point_on_map(
    save_path: Path,
    plat: float,
    plon: float,
    *,
    padding_deg: float = 5.0,
    figsize: tuple[float, float] = (7.0, 6.5),
) -> None:
    """Mark the grid point on a small regional map (Cartopy coastlines if installed)."""
    pad = padding_deg
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([plon - pad, plon + pad, plat - pad, plat + pad], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="0.78", alpha=0.85)
        ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="0.55", linewidth=0.25)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45)
        gl.top_labels = False
        gl.right_labels = False
        ax.scatter(
            [plon],
            [plat],
            transform=ccrs.PlateCarree(),
            c="crimson",
            s=110,
            zorder=10,
            edgecolors="black",
            linewidths=0.9,
        )
        ax.set_title(f"Location ({plat:.5f}°, {plon:.5f}°)")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception as exc:
        print(
            f"Warning: Cartopy map rendering failed in plot_point_on_map "
            f"({type(exc).__name__}: {exc}). Falling back to plain matplotlib."
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter([plon], [plat], c="crimson", s=100, zorder=10, edgecolors="black", linewidths=0.9)
    ax.set_xlim(plon - pad, plon + pad)
    ax.set_ylim(plat - pad, plat + pad)
    ax.set_aspect(1.0 / max(abs(math.cos(math.radians(plat))), 0.2))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Location ({plat:.5f}°, {plon:.5f}°)\n(install cartopy for coastlines)")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_points_overview_map(
    save_path: Path,
    points: list[tuple[float, float]],
    *,
    padding_deg: float = 1.5,
    figsize: tuple[float, float] = (9.0, 7.0),
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """All grid points on one map (e.g. Mediterranean batch)."""
    if not points:
        return
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    pad = padding_deg
    extent = [
        min(lons) - pad,
        max(lons) + pad,
        min(lats) - pad,
        max(lats) + pad,
    ]
    if title is None:
        title = f"Overview — {len(points)} grid point(s)"
    if labels is not None and len(labels) != len(points):
        raise ValueError("labels must be the same length as points")

    def _annotate(ax, transform) -> None:
        if not labels:
            return
        for lo, la, lab in zip(lons, lats, labels, strict=False):
            ax.annotate(
                lab,
                xy=(lo, la),
                transform=transform,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                ha="left",
                va="bottom",
                clip_on=True,
            )

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#f2f6fa", alpha=0.85)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="0.55", linewidth=0.25)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        ax.coastlines(resolution="10m", linewidth=0.7, color="black", zorder=2)

        gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45)
        gl.top_labels = False
        gl.right_labels = False
        ax.scatter(
            lons,
            lats,
            s=22,
            color="#d73027",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.25,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        _annotate(ax, ccrs.PlateCarree())
        ax.set_title(title)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception as exc:
        print(
            f"Warning: Cartopy map rendering failed in plot_points_overview_map "
            f"({type(exc).__name__}: {exc}). Falling back to plain matplotlib."
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(lons, lats, c="crimson", s=55, zorder=10, edgecolors="black", linewidths=0.5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    mid_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(abs(math.cos(math.radians(mid_lat))), 0.2))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    _annotate(ax, ax.transData)
    ax.set_title(f"{title}\n(install cartopy for coastlines)")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_points_overview_map_enh(
    save_path: Path,
    points: list[tuple[float, float]],
    points_split,
    *,
    padding_deg: float = 1.5,
    figsize: tuple[float, float] = (9.0, 7.0),
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Overview map with separate styling for best, worst, degraded, and improved points."""
    if not points:
        return

    best = points_split["best"][0]
    worst = points_split["worst"][0]
    degraded = points_split["degraded"]
    improved = points_split["improved"]

    def _as_set(x):
        return {(float(a), float(b)) for a, b in x}

    best_set = _as_set([best])
    worst_set = _as_set([worst])
    degraded_set = _as_set(degraded)
    improved_set = _as_set(improved)

    # avoid double plotting best/worst inside improved/degraded
    improved_set = improved_set - best_set - worst_set
    degraded_set = degraded_set - best_set - worst_set

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    pad = padding_deg
    extent = [
        min(lons) - pad,
        max(lons) + pad,
        min(lats) - pad,
        max(lats) + pad,
    ]

    if title is None:
        # title = f"Overview — {len(points)} grid points"
        pass
    def split_xy(point_set):
        if not point_set:
            return [], []
        la = [p[0] for p in point_set]
        lo = [p[1] for p in point_set]
        return lo, la

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN, facecolor="#f2f6fa", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
        ax.coastlines(resolution="10m", linewidth=0.7, color="black", zorder=2)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.35,
            color="gray",
            alpha=0.35,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 9}
        gl.ylabel_style = {"size": 9}

        transform = ccrs.PlateCarree()

        # improved
        lo, la = split_xy(improved_set)
        ax.scatter(
            lo, la,
            s=18,
            color="#2ca25f",
            alpha=0.65,
            edgecolor="black",
            transform=transform,
            linewidth=0.25,
            zorder=4,
            label=f"Improved ({len(improved_set)})",
        )

        # degraded
        lo, la = split_xy(degraded_set)
        ax.scatter(
            lo, la,
            s=18,
            color="#d73027",
            alpha=0.65,
            edgecolor="black",
            linewidth=0.25,
            transform=transform,
            zorder=5,
            label=f"Degraded ({len(degraded_set)})",
        )

        # best
        lo, la = split_xy(best_set)
        ax.scatter(
            lo, la,
            s=130,
            marker="*",
            color="#1a9850",
            edgecolor="black",
            linewidth=0.7,
            transform=transform,
            zorder=7,
            label="Best",
        )

        # worst
        lo, la = split_xy(worst_set)
        ax.scatter(
            lo, la,
            s=70,
            marker="X",
            color="#b2182b",
            edgecolor="black",
            linewidth=0.7,
            transform=transform,
            zorder=7,
            label="Worst",
        )

        ax.set_title(title, fontsize=13, fontweight="bold")

        leg = ax.legend(
            loc="lower left",
            fontsize=9,
            frameon=True,
        )
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_alpha(0.95)

        # highlight regions:
        # --- Region (Aegean) ---
        lon_min, lon_max = 23, 28
        lat_min, lat_max = 35, 42

        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=0.7,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.text(
            lon_min + 0.3,
            lat_max - 0.5,
            "Region 1",
            transform=ccrs.PlateCarree(),
            fontsize=5,
            fontweight="bold"
        )
        ax.add_patch(rect)

        # --- Region (Cyprus) ---
        lon_min, lon_max = 31, 40
        lat_min, lat_max = 31, 37

        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=0.7,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.text(
            lon_min + 0.3,
            lat_max - 0.5,
            "Region 2",
            transform=ccrs.PlateCarree(),
            fontsize=5,
            fontweight="bold"
        )

        ax.add_patch(rect)

        # --- Region (North Italy) ---
        lon_min, lon_max = 12, 18
        lat_min, lat_max = 42, 46

        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=0.7,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.text(
            lon_min + 0.3,
            lat_max - 0.5,
            "Region 3",
            transform=ccrs.PlateCarree(),
            fontsize=5,
            fontweight="bold"
        )

        ax.add_patch(rect)

        # central mediterannean good region
        lon_min, lon_max = 15, 20
        lat_min, lat_max = 32, 40
        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=0.7,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.text(
            lon_min + 0.3,
            lat_max - 0.5,
            "Region 4",
            transform=ccrs.PlateCarree(),
            fontsize=5,
            fontweight="bold"
        )
        ax.add_patch(rect)


        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf",format="pdf", dpi=300, bbox_inches="tight")

        plt.close(fig)
        return

    except Exception as exc:
        print(
            f"Warning: Cartopy map rendering failed in plot_points_overview_map_enh "
            f"({type(exc).__name__}: {exc}). Falling back to plain matplotlib."
        )

    # fallback without cartopy
    fig, ax = plt.subplots(figsize=figsize)

    lo, la = split_xy(improved_set)
    ax.scatter(lo, la, s=18, color="#2ca25f", alpha=0.65, edgecolor="black",
               label=f"Improved ({len(improved_set)})")

    lo, la = split_xy(degraded_set)
    ax.scatter(lo, la, s=24, color="#d73027", alpha=0.8, edgecolor="black",
               linewidth=0.25, label=f"Degraded ({len(degraded_set)})")

    lo, la = split_xy(best_set)
    ax.scatter(lo, la, s=130, marker="*", color="#1a9850", edgecolor="black",
               linewidth=0.7, label="Best")

    lo, la = split_xy(worst_set)
    ax.scatter(lo, la, s=90, marker="X", color="#b2182b", edgecolor="black",
               linewidth=0.7, label="Worst")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    mid_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(abs(math.cos(math.radians(mid_lat))), 0.2))

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"{title}\n(install cartopy for coastlines)", fontsize=13, fontweight="bold")
    ax.grid(True, linewidth=0.35, alpha=0.35)

    leg = ax.legend(loc="lower left", fontsize=9, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
def plot_points_overview_map_enh3(
    save_path: Path,
    points: list[tuple[float, float]],
    points_split,
    *,
    padding_deg: float = 1.5,
    figsize: tuple[float, float] = (9.0, 7.0),
    labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """All grid points on one map (e.g. Mediterranean batch)."""
    if not points:
        return
    points_split["best"][0]
    points_split["worst"][0]
    points_split["degraded"]
    points_split["improved"]
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    pad = padding_deg
    extent = [
        min(lons) - pad,
        max(lons) + pad,
        min(lats) - pad,
        max(lats) + pad,
    ]
    if title is None:
        pass
        # title = f"Overview — {len(points)} grid point(s)"
    if labels is not None and len(labels) != len(points):
        raise ValueError("labels must be the same length as points")

    def _annotate(ax, transform) -> None:
        if not labels:
            return
        for lo, la, lab in zip(lons, lats, labels, strict=False):
            ax.annotate(
                lab,
                xy=(lo, la),
                transform=transform,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6,
                ha="left",
                va="bottom",
                clip_on=True,
            )

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#f2f6fa", alpha=0.85)
        ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="0.55", linewidth=0.25)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        ax.coastlines(resolution="10m", linewidth=0.7, color="black", zorder=2)

        gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45)
        gl.top_labels = False
        gl.right_labels = False
        ax.scatter(
            lons,
            lats,
            s=22,
            color="#d73027",
            alpha=0.75,
            edgecolor="black",
            linewidth=0.25,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        _annotate(ax, ccrs.PlateCarree())
        ax.set_title(title)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    except Exception as exc:
        print(
            f"Warning: Cartopy map rendering failed in plot_points_overview_map_enh3 "
            f"({type(exc).__name__}: {exc}). Falling back to plain matplotlib."
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(lons, lats, c="crimson", s=55, zorder=10, edgecolors="black", linewidths=0.5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    mid_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(abs(math.cos(math.radians(mid_lat))), 0.2))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    _annotate(ax, ax.transData)
    ax.set_title(f"{title}\n(install cartopy for coastlines)")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if "region" not in df.columns:
        return df.copy()
    if region == "all":
        return df.copy()
    return df[df["region"].str.lower() == region].copy()


def top_n_points(df, n: int):
    counts = df.groupby(["_plat", "_plon"], as_index=False).size()
    return counts.sort_values("size", ascending=False).head(n)

def plot_scatter(x,y, save_path: Path ):
    if len(x) == 0 or len(y) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(x, y)
    # Diagonal (y = x)
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    # Labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Uncorrected")

    # Optional: improve layout
    fig.tight_layout()

    # Save
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_degraded_pdf(d: pd.DataFrame, save_path: Path) -> None:
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))  # Αυξήσαμε ελαφρώς το figsize για να χωρέσουν τα μεγαλύτερα γράμματα
    import seaborn as sns

    sns.kdeplot(d["ref_unc"], ax=ax, color="red", linewidth=2.5, bw_adjust=0.7)
    sns.kdeplot(d["ref_cor"], ax=ax, label="blue",  linewidth=2.2, bw_adjust=0.7)
    plt.tight_layout()
    plt.savefig(save_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-points",
        type=int,
        default=300,
        metavar="N",
        help="how many grid points to plot (default 25; use 20 or 30 as you prefer)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV path (default: med_simple23 next to load_csv.py)",
    )
    p.add_argument(
        "--region",
        type=str,
        default="mediterranean",
        choices=["mediterranean", "atlantic", "aegean", "all"],
        help="Region filter for input CSV rows (default: mediterranean).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for generated plot files.",
    )
    args = p.parse_args()
    n = args.n_points
    if n < 1 or n > 400:
        print("Use --n-points between 1 and 400.", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv) if args.csv is not None else DATA_PATH
    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "plots_300" / csv_path.stem
    )
    # out_dir = Path(__file__).resolve().parent / "plots_300_mlp" / csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(csv_path)
    df = with_point_key(filter_region(df_raw, args.region))
    if df.empty:
        raise SystemExit(f"No rows with region '{args.region}'.")

    top = top_n_points(df, n)
    written = 0
    overview_points: list[tuple[float, float]] = []
    overview_points_2 = {"best": "", "worst":"", "degraded": [], "improved": []}
    means = {"ref_unc": [], "ref_cor": [], "point":[], "rel_imp": []}
    all_ts = []

    for _, row in tqdm(top.iterrows()):
        plat, plon = float(row["_plat"]), float(row["_plon"])
        ts = timeseries_for_point(df, plat, plon)

        if ts.empty:
            continue
        point_tag = lat_lon_dirname(plat, plon)
        point_dir = out_dir / point_tag
        point_dir.mkdir(parents=True, exist_ok=True)

        # Restore per-point artifacts (timeseries CSVs + plots + location map).
        ts_w = weekly_mean_timeseries(ts)
        ts_15 = fifteen_day_mean_timeseries(ts)
        ts_m = monthly_mean_timeseries(ts)
        timeseries_table_for_export(ts, plat, plon).to_csv(point_dir / "timeseries_native.csv", index=False)
        write_absdiff_csv(point_dir / "timeseries_native_absdiff.csv", ts)
        timeseries_table_for_export(ts_w, plat, plon).to_csv(point_dir / "timeseries_7d.csv", index=False)
        write_absdiff_csv(point_dir / "timeseries_7d_absdiff.csv", ts_w)
        save_15day_timeseries_csvs(point_dir, ts_15, plat, plon)
        timeseries_table_for_export(ts_m, plat, plon).to_csv(point_dir / "timeseries_monthly.csv", index=False)
        write_absdiff_csv(point_dir / "timeseries_monthly_absdiff.csv", ts_m)
        plot_evolution(ts, plat, plon, point_dir / "plot_native.png", title=f"Native timestep — grid ({plat}, {plon})")
        plot_evolution(
            ts_w,
            plat,
            plon,
            point_dir / "plot_7d.png",
            title=f"7-day mean ({len(ts_w)} values) — grid ({plat}, {plon})",
        )
        plot_evolution(
            ts_15,
            plat,
            plon,
            point_dir / "plot_15d.png",
            title=f"15-day mean ({len(ts_15)} values) — grid ({plat}, {plon})",
        )
        plot_evolution(
            ts_m,
            plat,
            plon,
            point_dir / "plot_monthly.png",
            title=f"Monthly mean ({len(ts_m)} values) — grid ({plat}, {plon})",
        )
        plot_abs_errors(ts, plat, plon, point_dir / "plot_abs_errors_native.png", resolution_label="Native timestep")
        plot_abs_errors(ts_w, plat, plon, point_dir / "plot_abs_errors_7d.png", resolution_label="7-day mean")
        plot_abs_errors(ts_15, plat, plon, point_dir / "plot_abs_errors_15d.png", resolution_label="15-day mean")
        plot_abs_errors(ts_m, plat, plon, point_dir / "plot_abs_errors_monthly.png", resolution_label="Monthly mean")
        plot_point_on_map(point_dir / "map_location.png", plat, plon)

        x = abs_error_means(ts)
        means["ref_unc"].append(x[0])
        means["ref_cor"].append(x[1])
        means["point"].append(point_tag)
        means["rel_imp"].append((x[0]-x[1])/x[0])

        # hourly-level dataframe
        ts = ts.copy()
        ts["lat"] = plat
        ts["lon"] = plon
        ts["timestamp"] = pd.to_datetime(ts["timestamp"])

        ts["hour"] = ts["timestamp"].dt.hour
        ts["hour_str"] = ts["timestamp"].dt.strftime("%H:00")
        ts["month"] = ts["timestamp"].dt.month
        ts["month_name"] = ts["timestamp"].dt.month_name()

        ts["err_unc"] = (ts["uncorrected"] - ts["reference"]) ** 2
        ts["err_cor"] = (ts["corrected"] - ts["reference"]) ** 2

        all_ts.append(ts)

        overview_points.append((plat, plon))
        written += 1

    d = pd.DataFrame(means)
    d_time = pd.concat(all_ts, ignore_index=True)  # hourly metrics
    d["ref_unc"] = pd.to_numeric(d["ref_unc"], errors='coerce')
    d["ref_cor"] = pd.to_numeric(d["ref_cor"], errors='coerce')
    d["diff"] = d["ref_unc"] - d["ref_cor"]
    d["improvement"] = ((d["ref_unc"] - d["ref_cor"]) / d["ref_unc"]) * 100
    d.iloc[d["improvement"].argmax()]
    d.iloc[d["improvement"].argmin()]

    overview_points_2["best"] = [overview_points[d["improvement"].argmax()]]
    overview_points_2["worst"] = [overview_points[d["improvement"].argmin()]]
    overview_points_2["degraded"] = [overview_points[i] for i in list(d[d["improvement"] < 0].index)]
    overview_points_2["improved"] = [overview_points[i] for i in list(d[d["improvement"] >=0].index)]

    deg = d[d["improvement"] < 0]
    imp = d[d["improvement"] >= 0]
    # lon_min, lon_max = 12, 18
    # lat_min, lat_max = 42, 46
    #
    # d_time = d_time[
    #     (d_time["lon"] >= lon_min) & (d_time["lon"] <= lon_max) &
    #     (d_time["lat"] >= lat_min) & (d_time["lat"] <= lat_max)
    #     ]
    # print(len(d_time.groupby(["lat", "lon"]).size()))
    g = d_time.groupby(["hour", "month"]).agg(
        mse_unc=("err_unc", "mean"),
        mse_cor=("err_cor", "mean")
    ).reset_index()

    g["rmse_unc"] = np.sqrt(g["mse_unc"])
    g["rmse_cor"] = np.sqrt(g["mse_cor"])
    g["rmse_impr"] = 100 * (g["rmse_unc"] - g["rmse_cor"]) / g["rmse_unc"]

    pivot = g.pivot(index="hour", columns="month", values="rmse_impr")
    # plt.figure(figsize=(9, 5))
    vmax = np.nanpercentile(np.abs(pivot.values), 98)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.heatmap(
        pivot,
        ax=ax,
        vmin=-vmax,
        vmax=vmax,
        # linewidths=0.05,
        # linecolor="white",
        cmap="RdBu",
        center=0,
        cbar_kws={
            "label": "RMSE improvement (%)",
            "shrink": 1,
            "pad": 0.02
        }
    )

    plt.xlabel("Month")
    plt.ylabel("Hour of day")
    # plt.title("Temporal RMSE improvement")

    plt.tight_layout()

    plt.savefig(out_dir / "heatmap_rmse_improvement_cyprus.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "heatmap_rmse_improvement_cyprus.pdf", dpi=300, bbox_inches="tight")  # for paper

    plt.close()
    B = deg[["ref_unc", "ref_cor"]]
    C = imp[["ref_unc", "ref_cor"]]

    plot_scatter(B["ref_cor"], B["ref_unc"], out_dir / "degraded_scatter.png")
    plot_scatter(C["ref_cor"], C["ref_unc"], out_dir / "improved_scatter.png")

    plot_degraded_pdf(B, out_dir / "degraded_pdf.png")
    plot_degraded_pdf(C, out_dir / "improved_pdf.png")


    if overview_points:
        plot_points_overview_map_enh(out_dir / "map_overview.png", overview_points, overview_points_2)

    print(
        f"Wrote per-point and summary plots for {written} sampled point(s) under {out_dir}."
    )


if __name__ == "__main__":
    main()
