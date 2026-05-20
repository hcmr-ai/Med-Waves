#!/usr/bin/env python3
"""
Global evaluation — όλες οι ρυθμίσεις παρακάτω (dictionaries).
Τρέξε: python evaluation/global_evaluation.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot_maps import plot_dist, plot_scatter_per_wave_bin, rmse  # noqa: E402

# =============================================================================
# ΡΥΘΜΙΣΕΙΣ
# =============================================================================

CFG = {
    "model": "transunet",
    "data_dir": ROOT / "data",
    "out_dir": ROOT / "evaluation_results",
    "input_npz": None,
    "threshold_cm": 5.0,
    "eps": 1e-12,
}

def get_regimes() -> dict[str, str]:
    return {
        "overall": "όλα τα δείγματα",
        "noise": f"|y_unco − y_true| < {CFG['threshold_cm']} cm",
        "clean": f"|y_unco − y_true| ≥ {CFG['threshold_cm']} cm",
    }

# Όρια για «μικρό» vs «μεγάλο» λάθος (failures)
FAILURE_TIERS = {
    "near_unc_ratio": 1.10,       # err_pred / err_unc < 1.1 → σχεδόν ίδιο με uncorrected
    "large_ratio": 2.00,          # >2× χειρότερο squared error
    "small_excess_m2": 0.001,     # μικρή επιπλέον ζημία (m²)
    "large_excess_m2": 0.01,      # σημαντική επιπλέον ζημία (m²)
}

PLOTS = {
    "map_extent": [-6, 36, 30, 46],
    "dpi": 300,
    "top_n_locations": 25,
    "hs_bin_width_m": 0.25,
    "colors": {"better": "#2ca25f", "worse": "#d73027", "neutral": "#4575b4"},
}

OUTPUT_FILES = {
    "metrics_csv": "metrics_summary.csv",
    "report_txt": "statistics_report.txt",
    "glossary_txt": "metrics_glossary.txt",
    "failures_json": "failure_summaries.json",
}

# Επεξήγηση κάθε μετρικής (εμφανίζεται στο report)
METRIC_GLOSSARY = {
    "n_samples": (
        "Πλήθος δειγμάτων στο regime.",
        "N",
    ),
    "coverage_pct": (
        "Ποσοστό των συνολικών δειγμάτων που ανήκουν σε αυτό το regime.",
        "100 × N_regime / N_total",
    ),
    "rmse_uncorrected_m": (
        "RMSE του μη διορθωμένου μοντέλου (vhm0) έναντι αναφοράς, σε μέτρα.",
        "RMSE_unc = sqrt( mean( (y_unco − y_true)² ) )",
    ),
    "rmse_corrected_m": (
        "RMSE του διορθωμένου μοντέλου (y_pred) έναντι αναφοράς, σε μέτρα.",
        "RMSE_pred = sqrt( mean( (y_pred − y_true)² ) )",
    ),
    "rmse_improvement_pct": (
        "Ποσοστιαία μείωση RMSE μετά τη διόρθωση. Θετικό = καλύτερο corrected.",
        "100 × (RMSE_unc − RMSE_pred) / RMSE_unc",
    ),
    "mae_uncorrected_m": (
        "Μέσο απόλυτο σφάλμα uncorrected, σε μέτρα.",
        "mean( |y_unco − y_true| )",
    ),
    "mae_corrected_m": (
        "Μέσο απόλυτο σφάλμα corrected, σε μέτρα.",
        "mean( |y_pred − y_true| )",
    ),
    "bias_uncorrected_m": (
        "Συστηματική μεροχή uncorrected. Θετικό = υπερεκτίμηση κύματος.",
        "mean( y_unco − y_true )",
    ),
    "bias_corrected_m": (
        "Συστηματική μεροχή corrected. Θετικό = υπερεκτίμηση κύματος.",
        "mean( y_pred − y_true )",
    ),
    "pct_improved": (
        "Ποσοστό δειγμάτων όπου το corrected είναι καλύτερο (μικρότερο squared error).",
        "100 × mean( err_pred < err_unc )  [ισοδύναμα delta > 0]",
    ),
    "pct_worse": (
        "Ποσοστό δειγμάτων όπου το corrected είναι χειρότερο (Fail).",
        "100 × mean( err_pred > err_unc )",
    ),
    "mean_rel_impr_pct": (
        "Μέση σχετική βελτίωση ανά δείγμα (%). NaN/ακραία αν err_unc≈0.",
        "100 × mean( (err_unc − err_pred) / err_unc )",
    ),
    "median_rel_impr_pct": (
        "Διάμεσος της σχετικής βελτίωσης (%). Πιο ανθεκτικό από τον μέσο.",
        "median( 100 × (err_unc − err_pred) / err_unc )",
    ),
    "mean_hs_m": (
        "Μέσο ύψος κύματος (Hs) αναφοράς στο regime.",
        "mean( y_true )",
    ),
    "median_hs_m": (
        "Διάμεσο Hs αναφοράς στο regime.",
        "median( y_true )",
    ),
    # --- failure / severity ---
    "failure_rate_pct": (
        "Ποσοστό δειγμάτων όπου το corrected χάνει (ίδιο με pct_worse στο ίδιο regime).",
        "100 × N_fail / N_regime",
    ),
    "mean_excess_err_m2": (
        "Μέση ΕΠΙΠΛΕΟΝ τετραγωνική ζημία στα failures: πόσο χειρότερο είναι το corrected.",
        "mean( err_pred − err_unc )  [μόνο όπου err_pred > err_unc]",
    ),
    "median_excess_err_m2": (
        "Διάμεσος επιπλέον squared error στα failures.",
        "median( err_pred − err_unc )",
    ),
    "mean_err_ratio_fail": (
        "Μέσος λόγος err_pred/err_unc στα failures. 1.0 = ίδιο· 2.0 = διπλάσιο sq. error.",
        "mean( err_pred / err_unc )",
    ),
    "median_err_ratio_fail": (
        "Διάμεσος λόγος err_pred/err_unc στα failures.",
        "median( err_pred / err_unc )",
    ),
    "mean_abs_err_increase_m": (
        "Μέση αύξηση RMSE-like σφάλματος σε μέτρα στα failures: sqrt(err_pred)−sqrt(err_unc).",
        "mean( sqrt(err_pred) − sqrt(err_unc) )",
    ),
    "mean_abs_pred_minus_unc_m": (
        "Πόσο αλλάζει η πρόβλεψη σε μέτρα στα failures (όχι vs truth).",
        "mean( |y_pred − y_unco| )",
    ),
    "pct_fail_near_unc": (
        "Failures «κοντά» στο uncorrected: ratio<1.1 ή excess<0.001 m². Μικρή ζημία.",
        "100 × mean( ratio<1.1 OR excess<small )",
    ),
    "pct_fail_large": (
        "Failures «μεγάλα»: ratio>2 ή excess>0.01 m². Σημαντική επιδείνωση.",
        "100 × mean( ratio>2 OR excess>large )",
    ),
    "overprediction_pct": (
        "Στα failures, πόσο συχνά το y_pred > y_true (υπερεκτίμηση).",
        "100 × mean( y_pred > y_true )",
    ),
    "underprediction_pct": (
        "Στα failures, πόσο συχνά το y_pred < y_true.",
        "100 × mean( y_pred < y_true )",
    ),
    "mean_d_error_m": (
        "Μέση διαφορά απόλυτων σφαλμάτων: |corrected−truth| − |unc−truth|. "
        "Θετικό = το corrected είναι κατά μέσο όρο χειρότερο σε |error|.",
        "mean( |y_pred−y_true| − |y_unco−y_true| )",
    ),
    "median_d_error_m": (
        "Διάμεσος D_error (m). Αρνητικό = βελτίωση απόλυτου σφάλματος.",
        "median( |y_pred−y_true| − |y_unco−y_true| )",
    ),
}

# =============================================================================


def _threshold_m() -> float:
    return CFG["threshold_cm"] / 100.0


def load_predictions() -> pd.DataFrame:
    if CFG.get("input_npz"):
        path = Path(CFG["input_npz"])
    else:
        path = CFG["data_dir"] / f"{CFG['model']}_coords.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    d = np.load(path)
    return pd.DataFrame({
        "lat": d["lat"], "lon": d["lon"],
        "y_true": d["y_true"], "y_pred": d["y_pred"], "y_unco": d["vhm0"],
    })


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    eps = CFG["eps"]
    out = copy.deepcopy(df)
    out["err_unc"] = (out["y_unco"] - out["y_true"]) ** 2
    out["err_pred"] = (out["y_pred"] - out["y_true"]) ** 2
    out["abs_unc"] = np.abs(out["y_unco"] - out["y_true"])
    out["res_unc"] = out["y_unco"] - out["y_true"]
    out["res_pred"] = out["y_pred"] - out["y_true"]
    out["delta"] = out["err_unc"] - out["err_pred"]
    out["worse"] = out["err_pred"] > out["err_unc"]
    out["improved"] = out["delta"] > 0
    out["excess_err"] = out["err_pred"] - out["err_unc"]
    out["err_ratio"] = np.where(out["err_unc"] > eps, out["err_pred"] / out["err_unc"], np.nan)
    out["abs_err_increase_m"] = np.sqrt(out["err_pred"]) - np.sqrt(out["err_unc"])
    out["abs_pred_minus_unc"] = np.abs(out["y_pred"] - out["y_unco"])
    out["rel_impr_pct"] = np.where(
        out["err_unc"] > eps, (out["delta"] / out["err_unc"]) * 100, np.nan
    )
    # D_error (m): αύξηση απόλυτου σφάλματος μετά τη διόρθωση
    out["d_error_m"] = np.abs(out["res_pred"]) - np.abs(out["res_unc"])
    return out


def filter_regime(df: pd.DataFrame, name: str) -> pd.DataFrame:
    th = _threshold_m()
    if name == "overall":
        return df
    if name == "noise":
        return df[df["abs_unc"] < th]
    if name == "clean":
        return df[df["abs_unc"] >= th]
    raise KeyError(name)


def metrics(df: pd.DataFrame, name: str, n_total: int) -> dict:
    if len(df) == 0:
        return {"regime": name, "n_samples": 0}
    rel = df["rel_impr_pct"].dropna()
    ru, rp = rmse(df["y_unco"], df["y_true"]), rmse(df["y_pred"], df["y_true"])
    return {
        "regime": name,
        "n_samples": len(df),
        "coverage_pct": 100 * len(df) / n_total,
        "rmse_uncorrected_m": float(ru),
        "rmse_corrected_m": float(rp),
        "rmse_improvement_pct": float(100 * (ru - rp) / ru),
        "mae_uncorrected_m": float(np.mean(np.abs(df["res_unc"]))),
        "mae_corrected_m": float(np.mean(np.abs(df["res_pred"]))),
        "bias_uncorrected_m": float(df["res_unc"].mean()),
        "bias_corrected_m": float(df["res_pred"].mean()),
        "pct_improved": float(100 * df["improved"].mean()),
        "pct_worse": float(100 * df["worse"].mean()),
        "mean_rel_impr_pct": float(rel.mean()) if len(rel) else np.nan,
        "median_rel_impr_pct": float(rel.median()) if len(rel) else np.nan,
        "mean_hs_m": float(df["y_true"].mean()),
        "median_hs_m": float(df["y_true"].median()),
    }


def failure_severity(fail: pd.DataFrame) -> dict:
    """Πόσο «χαλάει» το μοντέλο όταν κάνει λάθος: κοντά vs μεγάλη ζημία."""
    if len(fail) == 0:
        return {}
    t = FAILURE_TIERS
    ratio = fail["err_ratio"].dropna()
    excess = fail["excess_err"]
    near = (ratio < t["near_unc_ratio"]) | (excess < t["small_excess_m2"])
    large = (ratio > t["large_ratio"]) | (excess > t["large_excess_m2"])
    r = fail["res_pred"]
    return {
        "mean_excess_err_m2": float(excess.mean()),
        "median_excess_err_m2": float(excess.median()),
        "p90_excess_err_m2": float(np.percentile(excess, 90)),
        "mean_err_ratio_fail": float(ratio.mean()),
        "median_err_ratio_fail": float(ratio.median()),
        "p90_err_ratio_fail": float(np.percentile(ratio, 90)),
        "mean_abs_err_increase_m": float(fail["abs_err_increase_m"].mean()),
        "median_abs_err_increase_m": float(fail["abs_err_increase_m"].median()),
        "mean_abs_pred_minus_unc_m": float(fail["abs_pred_minus_unc"].mean()),
        "pct_fail_near_unc": float(100 * near.mean()),
        "pct_fail_large": float(100 * large.mean()),
        "overprediction_pct": float(100 * (r > 0).mean()),
        "underprediction_pct": float(100 * (r < 0).mean()),
        "mean_hs_failures_m": float(fail["y_true"].mean()),
        "n_failures": len(fail),
    }


def failure_stats(df: pd.DataFrame, fail: pd.DataFrame) -> dict:
    n = len(df)
    if len(fail) == 0:
        return {"n_total": n, "n_failures": 0, "failure_rate_pct": 0.0}
    out = {
        "n_total": n,
        "failure_rate_pct": 100 * len(fail) / n,
        "n_locations": int(fail[["lat", "lon"]].drop_duplicates().shape[0]),
    }
    out.update(failure_severity(fail))
    out["regime"] = None  # filled by caller
    return out


def analyze_wave_bins(df: pd.DataFrame, name: str, out: Path) -> pd.DataFrame:
    """Ανάλυση ανά Hs bin: failure rate, RMSE, severity."""
    w = PLOTS["hs_bin_width_m"]
    max_hs = max(4.0, df["y_true"].max() + w)
    bins = np.arange(0, max_hs + w, w)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    df = df.copy()
    df["hs_bin"] = pd.cut(df["y_true"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for label, g in df.groupby("hs_bin", observed=True):
        if len(g) == 0:
            continue
        fail = g[g["worse"]]
        row = {
            "regime": name,
            "hs_bin_m": str(label),
            "n_samples": len(g),
            "n_failures": len(fail),
            "failure_rate_pct": 100 * len(fail) / len(g),
            "pct_improved": 100 * g["improved"].mean(),
            "rmse_uncorrected_m": rmse(g["y_unco"], g["y_true"]),
            "rmse_corrected_m": rmse(g["y_pred"], g["y_true"]),
            "rmse_improvement_pct": 100 * (rmse(g["y_unco"], g["y_true"]) - rmse(g["y_pred"], g["y_true"]))
            / rmse(g["y_unco"], g["y_true"]),
            "mean_hs_m": float(g["y_true"].mean()),
        }
        if len(fail) > 0:
            row.update({
                "mean_excess_err_m2": float(fail["excess_err"].mean()),
                "median_err_ratio_fail": float(fail["err_ratio"].median()),
                "pct_fail_near_unc": float(100 * (
                    (fail["err_ratio"] < FAILURE_TIERS["near_unc_ratio"])
                    | (fail["excess_err"] < FAILURE_TIERS["small_excess_m2"])
                ).mean()),
                "pct_fail_large": float(100 * (
                    (fail["err_ratio"] > FAILURE_TIERS["large_ratio"])
                    | (fail["excess_err"] > FAILURE_TIERS["large_excess_m2"])
                ).mean()),
            })
        rows.append(row)

    table = pd.DataFrame(rows)
    sub = out / name
    sub.mkdir(parents=True, exist_ok=True)
    table.to_csv(sub / "metrics_by_wave_bin.csv", index=False)

    if len(table) == 0:
        return table

    x = np.arange(len(table))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(x, table["failure_rate_pct"], color=PLOTS["colors"]["worse"], alpha=0.85)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(table["hs_bin_m"], rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel("%")
    axes[0, 0].set_title(f"{name}: failure rate ανά Hs bin")

    axes[0, 1].bar(x, table["rmse_improvement_pct"], color=PLOTS["colors"]["neutral"], alpha=0.85)
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(table["hs_bin_m"], rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("%")
    axes[0, 1].set_title(f"{name}: RMSE improvement % ανά bin")

    if "mean_excess_err_m2" in table.columns:
        axes[1, 0].bar(x, table["mean_excess_err_m2"].fillna(0), color="#fc8d59", alpha=0.85)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(table["hs_bin_m"], rotation=45, ha="right", fontsize=8)
        axes[1, 0].set_ylabel("m²")
        axes[1, 0].set_title(f"{name}: mean excess error (failures) ανά bin")

    if "pct_fail_near_unc" in table.columns:
        w_b = np.arange(len(table))
        axes[1, 1].bar(w_b - 0.2, table["pct_fail_near_unc"].fillna(0), width=0.4,
                        label="Κοντά unc (<1.1×)", color="#fee08b")
        axes[1, 1].bar(w_b + 0.2, table["pct_fail_large"].fillna(0), width=0.4,
                        label="Μεγάλο λάθος (>2×)", color="#d73027")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(table["hs_bin_m"], rotation=45, ha="right", fontsize=8)
        axes[1, 1].set_ylabel("% των failures")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_title(f"{name}: μικρό vs μεγάλο λάθος ανά bin")

    fig.suptitle(f"Wave-bin analysis — {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(sub / "metrics_by_wave_bin.pdf", dpi=PLOTS["dpi"])
    plt.close(fig)
    return table


def save_map(loc_df, col, title, path, cmap="YlOrRd", vmin=None, vmax=None):
    lats, lons = np.sort(loc_df["lat"].unique()), np.sort(loc_df["lon"].unique())
    Z = loc_df.pivot(index="lat", columns="lon", values=col).reindex(index=lats, columns=lons)
    Lon, Lat = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    vmax = vmax if vmax is not None else np.nanpercentile(Z.values, 99)
    vmin = vmin if vmin is not None else 0
    im = ax.pcolormesh(Lon, Lat, Z.values, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="auto", transform=ccrs.PlateCarree())
    ax.set_extent(PLOTS["map_extent"])
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#fff")
    ax.coastlines(resolution="10m", linewidth=0.6)
    plt.colorbar(im, ax=ax, shrink=0.9).set_label(col)
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=PLOTS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_residual_hist(df: pd.DataFrame, name: str, sub: Path) -> None:
    """Κατανομή residual (y_pred − y_true) για OK vs Fail."""
    ok, fail = df[~df["worse"]], df[df["worse"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(ok) > 0:
        sns.kdeplot(ok["res_pred"], ax=ax, color=PLOTS["colors"]["better"],
                    label=f"OK (n={len(ok):,})", lw=2)
    if len(fail) > 0:
        sns.kdeplot(fail["res_pred"], ax=ax, color=PLOTS["colors"]["worse"],
                    label=f"Fail (n={len(fail):,})", lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Residual: y_pred − y_true [m]")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}: residual distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sub / "residual_hist.pdf", dpi=PLOTS["dpi"])
    plt.close(fig)


def plot_d_error_hist(df: pd.DataFrame, name: str, sub: Path) -> None:
    """
    D_error (m) = |corrected − truth| − |uncorrected − truth|
                = |y_pred − y_true| − |y_unco − y_true|
    D_error < 0 → μικρότερο απόλυτο σφάλμα μετά τη διόρθωση (καλύτερο).
    D_error > 0 → μεγαλύτερο απόλυτο σφάλμα (χειρότερο).
    """
    ok, fail = df[~df["worse"]], df[df["worse"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(ok) > 0:
        sns.kdeplot(ok["d_error_m"], ax=ax, color=PLOTS["colors"]["better"],
                    label=f"OK (n={len(ok):,})", lw=2)
    if len(fail) > 0:
        sns.kdeplot(fail["d_error_m"], ax=ax, color=PLOTS["colors"]["worse"],
                    label=f"Fail (n={len(fail):,})", lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("D_error = |y_pred−y_true| − |y_unco−y_true| [m]")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}: D_error distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sub / "d_error_hist.pdf", dpi=PLOTS["dpi"])
    plt.close(fig)


def plot_failure_severity(fail: pd.DataFrame, name: str, sub: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sns.histplot(fail["excess_err"], bins=80, kde=True, ax=axes[0], color=PLOTS["colors"]["worse"])
    axes[0].axvline(FAILURE_TIERS["small_excess_m2"], ls="--", c="green", label="small")
    axes[0].axvline(FAILURE_TIERS["large_excess_m2"], ls="--", c="red", label="large")
    axes[0].set_xlabel("excess = err_pred − err_unc [m²]")
    axes[0].set_title("Επιπλέον squared error στα failures")
    axes[0].legend(fontsize=8)

    ratio = fail["err_ratio"].dropna()
    sns.histplot(ratio, bins=80, kde=True, ax=axes[1], color=PLOTS["colors"]["neutral"])
    axes[1].axvline(1.0, ls="-", c="k", lw=1)
    axes[1].axvline(FAILURE_TIERS["near_unc_ratio"], ls="--", c="green")
    axes[1].axvline(FAILURE_TIERS["large_ratio"], ls="--", c="red")
    axes[1].set_xlabel("err_pred / err_unc")
    axes[1].set_title("Λόγος σφάλματος (1=ίδιο με unc)")

    sns.histplot(fail["abs_err_increase_m"], bins=80, kde=True, ax=axes[2], color="#fc8d59")
    axes[2].axvline(0, ls="--", c="k")
    axes[2].set_xlabel("sqrt(err_pred) − sqrt(err_unc) [m]")
    axes[2].set_title("Αύξηση σφάλματος σε μέτρα")

    fig.suptitle(f"{name}: πόσο χαλάει όταν κάνει λάθος", fontweight="bold")
    fig.tight_layout()
    fig.savefig(sub / "failure_severity.pdf", dpi=PLOTS["dpi"])
    plt.close(fig)


def analyze_failures(df: pd.DataFrame, name: str, out: Path) -> dict:
    sub = out / f"{name}_failure_analysis"
    sub.mkdir(parents=True, exist_ok=True)
    fail = df[df["worse"]]
    summ = failure_stats(df, fail)
    summ["regime"] = name
    summ["mean_d_error_m"] = float(df["d_error_m"].mean())
    summ["median_d_error_m"] = float(df["d_error_m"].median())
    if len(fail) > 0:
        summ["mean_d_error_failures_m"] = float(fail["d_error_m"].mean())

    plot_residual_hist(df, name, sub)
    plot_d_error_hist(df, name, sub)

    if len(fail) == 0:
        (sub / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        return summ

    fail.to_csv(sub / "failure_samples.csv", index=False)
    plot_failure_severity(fail, name, sub)
    (sub / "summary.json").write_text(json.dumps(summ, indent=2, default=str))

    loc = fail.groupby(["lat", "lon"], as_index=False).agg(
        n_failures=("worse", "count"),
        mean_hs=("y_true", "mean"),
        mean_excess=("excess_err", "mean"),
        mean_ratio=("err_ratio", "mean"),
    )
    loc.to_csv(sub / "failure_locations.csv", index=False)

    rate = df.groupby(["lat", "lon"], as_index=False).agg(n=("worse", "count"), f=("worse", "sum"))
    rate["failure_rate_pct"] = 100 * rate["f"] / rate["n"]
    rate.to_csv(sub / "failure_rate_by_location.csv", index=False)
    save_map(rate, "failure_rate_pct", f"{name}: failure rate %", sub / "failure_rate_map.pdf", vmax=100)
    save_map(loc, "n_failures", f"{name}: failure count", sub / "failure_count_map.pdf")

    ok = df[~df["worse"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(ok["y_true"], ax=axes[0], color=PLOTS["colors"]["better"], label="OK", lw=2)
    sns.kdeplot(fail["y_true"], ax=axes[0], color=PLOTS["colors"]["worse"], label="Fail", lw=2)
    axes[0].set_xlabel("Hs [m]"); axes[0].legend(); axes[0].set_title("Hs: OK vs Fail")
    sns.kdeplot(ok["abs_unc"], ax=axes[1], color=PLOTS["colors"]["better"], label="OK", lw=2)
    sns.kdeplot(fail["abs_unc"], ax=axes[1], color=PLOTS["colors"]["worse"], label="Fail", lw=2)
    axes[1].set_xlabel("|y_unco − y_true| [m]"); axes[1].legend()
    fig.savefig(sub / "fail_vs_ok.pdf", dpi=PLOTS["dpi"])
    plt.close(fig)
    return summ


def write_glossary(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ΛΕΞΙΚΟ ΜΕΤΡΙΚΩΝ — Global Evaluation\n")
        f.write("=" * 70 + "\n\n")
        for key, (desc, formula) in METRIC_GLOSSARY.items():
            f.write(f"{key}\n")
            f.write(f"  Τι είναι: {desc}\n")
            f.write(f"  Τύπος:    {formula}\n\n")
        f.write("\nΟρισμοί failures:\n")
        f.write("  OK:   err_pred <= err_unc  (το corrected δεν χειροτερεύει)\n")
        f.write("  Fail: err_pred > err_unc   (το corrected χάνει)\n\n")
        f.write("Όρια severity (FAILURE_TIERS):\n")
        for k, v in FAILURE_TIERS.items():
            f.write(f"  {k}: {v}\n")


def write_report(
    path: Path,
    all_metrics: list[dict],
    failures: dict[str, dict],
    wave_tables: dict[str, pd.DataFrame],
) -> None:
    write_glossary(path.parent / OUTPUT_FILES["glossary_txt"])

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Model: {CFG['model']}\n")
        f.write(f"Noise/clean threshold: {CFG['threshold_cm']} cm\n")
        f.write(f"Λεξικό μετρικών: {OUTPUT_FILES['glossary_txt']}\n")
        f.write("=" * 70 + "\n\n")

        f.write("ΠΩΣ ΔΙΑΒΆΖΕΙΣ ΤΑ FAILURES\n")
        f.write("-" * 70 + "\n")
        f.write("• pct_fail_near_unc: το λάθος είναι ΜΙΚΡΟ — σχεδόν όπως το uncorrected.\n")
        f.write("• pct_fail_large: το λάθος είναι ΜΕΓΑΛΟ — σημαντικά χειρότερο από uncorrected.\n")
        f.write("• mean_err_ratio_fail ≈ 1.05 → λίγο χειρότερο· ≈ 2 → διπλάσιο sq. error.\n")
        f.write("• mean_abs_err_increase_m → πόσα cm επιπλέον σφάλμα vs truth στα failures.\n\n")

        for m in all_metrics:
            regime = m.get("regime", "?")
            f.write(
                f"\n{'=' * 70}\nREGIME: {regime.upper()} — {get_regimes().get(regime, '')}\n{'=' * 70}\n"
            )
            for k, v in m.items():
                if k == "regime":
                    continue
                gloss = METRIC_GLOSSARY.get(k, ("", ""))[0]
                extra = f"  # {gloss}" if gloss else ""
                f.write(f"  {k}: {v}{extra}\n")

            if regime in failures:
                f.write(f"\n  --- Failure severity ({regime}) ---\n")
                for k, v in failures[regime].items():
                    if k in ("regime", "n_total"):
                        continue
                    gloss = METRIC_GLOSSARY.get(k, ("", ""))[0]
                    extra = f"  # {gloss}" if gloss else ""
                    f.write(f"  {k}: {v}{extra}\n")

            if regime in wave_tables and len(wave_tables[regime]) > 0:
                f.write(f"\n  --- Ανά Hs bin (δείτε {regime}/metrics_by_wave_bin.csv) ---\n")
                wt = wave_tables[regime]
                for _, row in wt.iterrows():
                    line = (
                        f"    {row['hs_bin_m']} m: n={row['n_samples']}, "
                        f"fail={row['failure_rate_pct']:.1f}%, "
                        f"RMSE impr={row['rmse_improvement_pct']:.1f}%"
                    )
                    if "mean_excess_err_m2" in row and pd.notna(row.get("mean_excess_err_m2")):
                        line += f", excess={row['mean_excess_err_m2']:.4f} m²"
                    if "pct_fail_near_unc" in row and pd.notna(row.get("pct_fail_near_unc")):
                        line += f", near_unc={row['pct_fail_near_unc']:.0f}%"
                    if "pct_fail_large" in row and pd.notna(row.get("pct_fail_large")):
                        line += f", large={row['pct_fail_large']:.0f}%"
                    f.write(line + "\n")


def evaluate_regime(df_all: pd.DataFrame, name: str, out: Path, n_total: int) -> dict:
    df = filter_regime(df_all, name)
    sub = out / name
    sub.mkdir(parents=True, exist_ok=True)
    m = metrics(df, name, n_total)
    if len(df) == 0:
        return m
    plot_dist(df, str(sub / "distributions.pdf"))
    plot_scatter_per_wave_bin(df, str(sub / "scatter_per_bin.png"), name)
    plot_residual_hist(df, name, sub)
    plot_d_error_hist(df, name, sub)
    m["mean_d_error_m"] = float(df["d_error_m"].mean())
    m["median_d_error_m"] = float(df["d_error_m"].median())
    return m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global evaluation for plot_samples NPZ outputs."
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        default=None,
        help=(
            "Path to input NPZ with keys: y_true, y_pred, vhm0, lat, lon. "
            "If omitted, uses CFG['data_dir']/f\"{CFG['model']}_coords.npz\"."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for global evaluation artifacts. "
            "If omitted, uses CFG['out_dir']/CFG['model']."
        ),
    )
    parser.add_argument(
        "--threshold-cm",
        type=float,
        default=None,
        help="Noise/clean split threshold in cm (default from CFG).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    if args.input_npz:
        CFG["input_npz"] = args.input_npz
        try:
            CFG["model"] = Path(args.input_npz).stem
        except Exception:
            pass
    if args.threshold_cm is not None:
        CFG["threshold_cm"] = float(args.threshold_cm)
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = CFG["out_dir"] / CFG["model"]
    out.mkdir(parents=True, exist_ok=True)

    print(f"Model: {CFG['model']}  |  threshold: {CFG['threshold_cm']} cm")
    print(f"Output: {out}\n")

    df = add_errors(load_predictions())
    n_total = len(df)
    all_metrics, failures, wave_tables = [], {}, {}

    for name in get_regimes():
        dreg = filter_regime(df, name)
        print(f"  {name}: {len(dreg):,} samples")
        all_metrics.append(evaluate_regime(df, name, out, n_total))
        wave_tables[name] = analyze_wave_bins(dreg, name, out)
        if name in ("noise", "clean"):
            failures[name] = analyze_failures(dreg, name, out)

    pd.DataFrame(all_metrics).to_csv(out / OUTPUT_FILES["metrics_csv"], index=False)
    (out / OUTPUT_FILES["failures_json"]).write_text(
        json.dumps(failures, indent=2, default=str)
    )
    write_report(out / OUTPUT_FILES["report_txt"], all_metrics, failures, wave_tables)
    print(f"\nDone → {out}")
    print(f"  Report: {OUTPUT_FILES['report_txt']}")
    print(f"  Glossary: {OUTPUT_FILES['glossary_txt']}")


if __name__ == "__main__":
    run()