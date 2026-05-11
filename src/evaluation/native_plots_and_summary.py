"""Mediterranean-only: per grid point, plots/<csv_stem>/<lat_lon>/ (value + abs-error PNGs, map, CSVs) + map_overview.png."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from load_csv import DATA_PATH, load_data
from plot_point_evolution import (
    _fmt_mean_line,
    lat_lon_dirname,
    timeseries_for_point,
    with_point_key,
    abs_error_means
)

COL_ABS_REF_UNC = "abs(reference-uncorrected)"
COL_ABS_REF_CORR = "abs(reference-corrected)"

def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if "region" not in df.columns:
        return df.copy()
    if region == "all":
        return df.copy()
    return df[df["region"].str.lower() == region].copy()


def top_n_points(df, n: int):
    counts = df.groupby(["_plat", "_plon"], as_index=False).size()
    return counts.sort_values("size", ascending=False).head(n)

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
    ax.plot(ts["timestamp"], y1, label=COL_ABS_REF_UNC, color="#4285f4")
    ax.plot(ts["timestamp"], y2, label=COL_ABS_REF_CORR, color="#eb9999")
    ax.legend(loc="upper right")
    ax.set_xlabel("Time",  fontweight="bold", fontsize=20)
    ax.set_ylabel("Absolute error",  fontweight="bold", fontsize=20)
    ax.set_title(f"{resolution_label} — {lat_lon_dirname(plat, plon)}")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.savefig(f"{save_path}.pdf", format="pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)
    rel_imp = 100* ((ts["uncorrected"]-ts["corrected"])/ts["uncorrected"])
    return {"ref_unc": _fmt_mean_line(m1),"ref_cor": _fmt_mean_line(m2), "rel_imp": rel_imp}

def plot_best_worst(ts_best, ts_worst):

    def prepare(df):
        df = df.copy().sort_values("timestamp")
        df["err_unc"] = (df["reference"] - df["uncorrected"]).abs()
        df["err_cor"] = (df["reference"] - df["corrected"]).abs()
        df = df.set_index("timestamp")

        roll_unc = df["err_unc"].rolling("7D").median()
        roll_cor = df["err_cor"].rolling("7D").median()

        improved_pct = 100 * (df["err_cor"] < df["err_unc"]).mean()

        return df, roll_unc, roll_cor, improved_pct

    df_b, ru_b, rc_b, imp_b = prepare(ts_best)
    df_w, ru_w, rc_w, imp_w = prepare(ts_worst)

    # ---- Shared figure ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # =========================
    # BEST
    # =========================
    ax = axes[0]

    ax.plot(df_b.index, df_b["err_unc"],
            color="#4285f4", alpha=0.05, linewidth=0.5)
    ax.plot(df_b.index, df_b["err_cor"],
            color="#eb9999", alpha=0.05, linewidth=0.5)

    ax.plot(ru_b.index, ru_b,
            color="#4285f4", linewidth=2.2, label="Uncorrected")
    ax.plot(rc_b.index, rc_b,
            color="#eb9999", linewidth=2.2, label="Corrected")

    ax.set_title("Best-case location",  fontweight="bold", fontsize=20)
    ax.set_ylabel("Absolute error",  fontweight="bold", fontsize=20)

    ax.text(0.02, 0.90,
            f"Improved: {imp_b:.1f}%",
            transform=ax.transAxes,
            fontsize=10)

    ax.legend(loc="upper right")

    # =========================
    # WORST
    # =========================
    ax = axes[1]

    ax.plot(df_w.index, df_w["err_unc"],
            color="#4285f4", alpha=0.05, linewidth=0.5)
    ax.plot(df_w.index, df_w["err_cor"],
            color="#eb9999", alpha=0.05, linewidth=0.5)

    ax.plot(ru_w.index, ru_w,
            color="#4285f4", linewidth=2.2, label="Uncorrected")
    ax.plot(rc_w.index, rc_w,
            color="#eb9999", linewidth=2.2, label="Corrected")

    ax.set_title("Worst-case location",  fontweight="bold", fontsize=20)
    ax.set_ylabel("Absolute error",  fontweight="bold", fontsize=20)
    ax.set_xlabel("Time",  fontweight="bold", fontsize=20)

    ax.text(0.02, 0.90,
            f"Improved: {imp_w:.1f}%",
            transform=ax.transAxes,
            fontsize=10)

    ax.legend(loc="upper right")

    # ---- Formatting ----
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
    plt.savefig("x.png")

def plot_abs_error_trend(ts, save_path:Path, title="", ):

    df = ts.copy().sort_values("timestamp")

    # ---- Compute errors ----
    df["err_unc"] = (df["reference"] - df["uncorrected"]).abs()
    df["err_cor"] = (df["reference"] - df["corrected"]).abs()

    # ---- Rolling trend (robust) ----
    df = df.set_index("timestamp")
    roll_unc = df["err_unc"].rolling("7D").median()
    roll_cor = df["err_cor"].rolling("7D").median()

    # ---- Compute mean improvement (%) ----
    eps = 1e-8
    rel_imp = (df["err_unc"] - df["err_cor"]) / (df["err_unc"] + eps) * 100
    rel_imp.mean()

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 4))

    # RAW (very faint)
    ax.plot(df.index, df["err_unc"],
            color="#4285f4", alpha=0.3, linewidth=0.5)
    ax.plot(df.index, df["err_cor"],
            color="#eb9999", alpha=0.5, linewidth=0.5)

    # TRENDS (main signal)
    ax.plot(roll_unc.index, roll_unc,
            color="#4285f4", linewidth=2.2, label="Uncorrected")
    ax.plot(roll_cor.index, roll_cor,
            color="#eb9999", linewidth=2.2, label="TransUNet")

    # OPTIONAL: highlight improvement
    ax.fill_between(
        roll_unc.index,
        roll_unc,
        roll_cor,
        where=(roll_unc > roll_cor),
        color="green",
        alpha=0.15
    )

    # ---- Labels ----
    ax.set_xlabel("Time",  fontweight="bold", fontsize=14 )
    ax.set_ylabel("Absolute error",   fontweight="bold", fontsize=14)
    # ax.set_title(title)

    # ---- Clean legend ----
    ax.legend(loc="upper right")

    # ---- Formatting ----
    fig.autofmt_xdate()
    plt.tight_layout()

    # ---- Save ----
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        fig.savefig(f"{save_path}.pdf", format="pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-points",
        type=int,
        default=300,
        metavar="N",
        help="how many grid points to analyze (default: 300)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV path (default: DATA_PATH from load_csv.py)",
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
        else Path(__file__).resolve().parent / "plots_300_native" / csv_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(csv_path)

    df = with_point_key(filter_region(df_raw, args.region))
    if df.empty:
        raise SystemExit(f"No rows with region '{args.region}'.")

    top = top_n_points(df, n)
    written = 0
    overview_points: list[tuple[float, float]] = []
    means = {"ref_unc": [], "ref_cor": [], "point":[], "rel_imp": []}
    for _, row in top.iterrows():
        plat, plon = float(row["_plat"]), float(row["_plon"])
        ts = timeseries_for_point(df, plat, plon)
        if ts.empty:
            continue
        point_dir = out_dir / lat_lon_dirname(plat, plon)
        point_dir.mkdir(parents=True, exist_ok=True)
        x= plot_abs_errors(ts, plat, plon, point_dir / "plot_abs_errors_native.png", resolution_label="Native timestep")
        plot_abs_error_trend(ts,point_dir /"plot_abs_errors_native_trend.png", "")

        means["ref_unc"].append(x["ref_unc"])
        means["ref_cor"].append(x["ref_cor"])
        means["point"].append(point_dir)
        means["rel_imp"].append(x["rel_imp"])

        overview_points.append((plat, plon))
        written += 1

    d = pd.DataFrame(means)
    d["ref_unc"] = pd.to_numeric(d["ref_unc"], errors='coerce')
    d["ref_cor"] = pd.to_numeric(d["ref_cor"], errors='coerce')
    d["diff"] = d["ref_unc"] - d["ref_cor"]
    d["improvement"] = ((d["ref_unc"] - d["ref_cor"]) / d["ref_unc"]) * 100

    A = d["diff"].describe()
    B = d[d["diff"] < 0]["diff"].describe()
    C = (len(d[d["diff"] < 0]) / len(d)) * 100

    D = d[d["diff"] >= 0]["diff"].describe()
    print(A)
    print(B)
    print(C)
    print(D)
    print("---------------------")

    A = d["improvement"].describe()
    B = d[d["improvement"] < 0]["diff"].describe()
    C = (len(d[d["improvement"] < 0]) / len(d)) * 100

    D = d[d["improvement"] >= 0]["improvement"].describe()
    print(A)
    print(B)
    print(C)
    print(D)
    print("---------------------")
    max_improvement = d.iloc[d["improvement"].argmax()]
    min_improvement = d.iloc[d["improvement"].argmin()]
    print(f"Best: {max_improvement['point']}")
    print(f"Worst: {min_improvement['point']}")

    # plot_best_worst(max_improvement["rel_imp"], min_improvement["rel_imp"])
    # plot_imp(ts, max_improvement["rel_imp"], min_improvement["rel_imp"], max_improvement["point"], min_improvement["point"])
    # PosixPath(
    #     '/Users/elisavetpalogiannidi/Documents/Work/HCMR/codes/evaluation/plots/med_simple23/lat38p52083_lon12p08333')
    # PosixPath(
    #     '/Users/elisavetpalogiannidi/Documents/Work/HCMR/codes/evaluation/plots/med_simple23/lat43p10417_lon10p00000')
    print(
        f"Wrote {written} point folder(s) under {out_dir} "
        f"(each: 4 value + 4 abs-error plots + map + 10 CSVs incl. absdiff); map_overview.png in run folder"
    )


if __name__ == "__main__":
    main()
