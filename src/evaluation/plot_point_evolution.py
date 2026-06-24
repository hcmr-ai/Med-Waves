"""Plot reference, uncorrected, and corrected over time for one grid point."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_csv import load_data

POINT_DECIMALS = 5

COL_ABS_REF_UNC = "abs(reference-uncorrected)"
COL_ABS_REF_CORR = "abs(reference-corrected)"


def with_point_key(df: pd.DataFrame, decimals: int = POINT_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    out["_plat"] = out["grid_lat"].round(decimals)
    out["_plon"] = out["grid_lon"].round(decimals)
    return out


def timeseries_for_point(
    df: pd.DataFrame, plat: float, plon: float
) -> pd.DataFrame:
    """One row per timestamp (mean if duplicates exist for that time)."""
    d = df[(df["_plat"] == round(plat, POINT_DECIMALS)) & (df["_plon"] == round(plon, POINT_DECIMALS))]
    cols = ["reference", "uncorrected", "corrected"]
    return d.groupby("timestamp", as_index=False)[cols].mean().sort_values("timestamp")


def weekly_mean_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Mean in each 7-day window (~52 values per year)."""
    idx = pd.DatetimeIndex(ts["timestamp"])
    g = ts.set_index(idx).sort_index()
    out = g[["reference", "uncorrected", "corrected"]].resample("7D", label="left").mean()
    out = out.reset_index()
    out.columns = ["timestamp", "reference", "uncorrected", "corrected"]
    return out


def fifteen_day_mean_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Mean in each 15-day window (~24 values per year)."""
    idx = pd.DatetimeIndex(ts["timestamp"])
    g = ts.set_index(idx).sort_index()
    out = g[["reference", "uncorrected", "corrected"]].resample("15D", label="left").mean()
    out = out.reset_index()
    out.columns = ["timestamp", "reference", "uncorrected", "corrected"]
    return out


def monthly_mean_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month mean (12 values per full year)."""
    idx = pd.DatetimeIndex(ts["timestamp"])
    g = ts.set_index(idx).sort_index()
    out = g[["reference", "uncorrected", "corrected"]].resample("MS").mean()
    out = out.reset_index()
    out.columns = ["timestamp", "reference", "uncorrected", "corrected"]
    return out


def absdiff_only_table(ts: pd.DataFrame) -> pd.DataFrame:
    """Only timestamp and absolute errors vs reference (the 'diff abs' columns)."""
    ref = ts["reference"]
    return pd.DataFrame(
        {
            "timestamp": ts["timestamp"],
            COL_ABS_REF_UNC: (ref - ts["uncorrected"]).abs(),
            COL_ABS_REF_CORR: (ref - ts["corrected"]).abs(),
        }
    )


def write_absdiff_csv(path: Path, ts: pd.DataFrame) -> None:
    absdiff_only_table(ts).to_csv(path, index=False)


def save_15day_timeseries_csvs(point_dir: Path, ts_15: pd.DataFrame, plat: float, plon: float) -> None:
    """15-day mean: full wide table (two names) + abs-diff-only CSVs (two names)."""
    df = timeseries_table_for_export(ts_15, plat, plon)
    df.to_csv(point_dir / "timeseries_15d.csv", index=False)
    df.to_csv(point_dir / "timeseries_15_days.csv", index=False)
    ad = absdiff_only_table(ts_15)
    ad.to_csv(point_dir / "timeseries_15d_absdiff.csv", index=False)
    ad.to_csv(point_dir / "timeseries_15_days_absdiff.csv", index=False)


def timeseries_table_for_export(ts: pd.DataFrame, plat: float, plon: float) -> pd.DataFrame:
    """Wide table like the spreadsheet example: timestamp, series, simple: tag column, abs errors."""
    tag = lat_lon_dirname(plat, plon)
    simple_col = f"simple: {tag}"
    ref = ts["reference"]
    unc = ts["uncorrected"]
    corr = ts["corrected"]
    out = pd.DataFrame(
        {
            "timestamp": ts["timestamp"],
            "reference": ref,
            "uncorrected": unc,
            "corrected": corr,
        }
    )
    out[simple_col] = ""
    out[COL_ABS_REF_UNC] = (ref - unc).abs()
    out[COL_ABS_REF_CORR] = (ref - corr).abs()
    return out


def abs_error_means(ts: pd.DataFrame) -> tuple[float, float]:
    ref = ts["reference"]
    a1 = (ref - ts["uncorrected"]).abs().mean(skipna=True)
    a2 = (ref - ts["corrected"]).abs().mean(skipna=True)
    return float(a1), float(a2)


def _fmt_mean_line(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.11f}"


def plot_abs_errors(
    ts: pd.DataFrame,
    plat: float,
    plon: float,
    save_path: Path,
    *,
    resolution_label: str,
) -> None:
    """|reference − uncorrected| and |reference − corrected| vs time (+ mean lines in legend style via text box)."""
    ref = ts["reference"]
    y1 = (ref - ts["uncorrected"]).abs()
    y2 = (ref - ts["corrected"]).abs()
    m1, m2 = abs_error_means(ts)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts["timestamp"], y1, label=COL_ABS_REF_UNC)
    ax.plot(ts["timestamp"], y2, label=COL_ABS_REF_CORR)
    ax.legend(loc="upper right")
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute error")
    ax.set_title(f"{resolution_label} — {lat_lon_dirname(plat, plon)}")
    ax.text(
        0.02,
        0.98,
        f"mean {COL_ABS_REF_UNC} = {_fmt_mean_line(m1)}\nmean {COL_ABS_REF_CORR} = {_fmt_mean_line(m2)}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    # return {"ref_unc": _fmt_mean_line(m1), "ref_cor": _fmt_mean_line(m2), "rel_imp": (_fmt_mean_line(m1) - _fmt_mean_line(m2))/_fmt_mean_line(m1)}

def lat_lon_dirname(plat: float, plon: float, decimals: int = POINT_DECIMALS) -> str:
    """Filesystem-safe folder name from grid coordinates."""

    def fmt(x: float) -> str:
        return f"{x:.{decimals}f}".replace("-", "m").replace(".", "p")

    return f"lat{fmt(plat)}_lon{fmt(plon)}"


def decode_lat_lon_dirname(name: str) -> tuple[float, float]:
    """Decode a folder-style tag like lat42p47917_lon17p50000 to (latitude, longitude)."""
    name = Path(name.strip()).name
    sep = "_lon"
    if not name.startswith("lat") or sep not in name:
        raise ValueError(f"Expected 'lat{{...}}_lon{{...}}', got {name!r}")
    i = name.index(sep)
    lat_body, lon_body = name[3:i], name[i + len(sep) :]
    if not lat_body or not lon_body:
        raise ValueError(f"Incomplete name {name!r}")

    def dec(s: str) -> float:
        return float(s.replace("p", ".").replace("m", "-"))

    return dec(lat_body), dec(lon_body)


def output_root_from_arg(path: Path) -> Path:
    """If user passed a *.png path, use its parent as the root directory."""
    if path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg"}:
        return path.parent
    return path


def pick_default_point(df: pd.DataFrame) -> tuple[float, float]:
    """Point with the most timestamps (after rounding to grid)."""
    counts = df.groupby(["_plat", "_plon"], as_index=False).size()
    row = counts.sort_values("size", ascending=False).iloc[0]
    return float(row["_plat"]), float(row["_plon"])


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
        ax.add_feature(cfeature.OCEAN, facecolor="0.78", alpha=0.85)
        ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="0.55", linewidth=0.25)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.55)
        gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45)
        gl.top_labels = False
        gl.right_labels = False
        ax.scatter(
            lons,
            lats,
            transform=ccrs.PlateCarree(),
            c="crimson",
            s=55,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
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


def plot_evolution(
    ts: pd.DataFrame,
    plat: float,
    plon: float,
    save_path: Path | None,
    *,
    title: str | None = None,
) -> None:
    if title is None:
        title = f"Temporal evolution at grid ({plat}, {plon})"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts["timestamp"], ts["reference"], label="reference")
    ax.plot(ts["timestamp"], ts["uncorrected"], label="uncorrected")
    ax.plot(ts["timestamp"], ts["corrected"], label="corrected")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lat",
        type=float,
        default=None,
        help="grid latitude (uses same rounding as grid_lon in the CSV)",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=None,
        help="grid longitude",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "output root directory: creates <lat_lon>/ with value plots, abs-error plots, "
            "map, and timeseries_*.csv (export layout matches spreadsheet example; "
            "if you pass name.png, its parent folder is the root)"
        ),
    )
    args = p.parse_args()

    df = with_point_key(load_data())
    if (args.lat is None) ^ (args.lon is None):
        p.error("Pass both --lat and --lon, or neither to use the busiest grid point.")
    if args.lat is None:
        plat, plon = pick_default_point(df)
    else:
        plat, plon = round(args.lat, POINT_DECIMALS), round(args.lon, POINT_DECIMALS)

    ts = timeseries_for_point(df, plat, plon)
    if ts.empty:
        raise SystemExit(
            f"No rows for grid point ({plat}, {plon}). "
            "Try --lat/--lon from the CSV (grid_lat, grid_lon columns)."
        )

    if args.output is None:
        plot_evolution(ts, plat, plon, None)
    else:
        root = output_root_from_arg(args.output)
        point_dir = root / lat_lon_dirname(plat, plon)
        point_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"Wrote plots, maps, and CSVs under {point_dir}")


if __name__ == "__main__":
    main()
