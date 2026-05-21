import pandas as pd
import os
import copy
import argparse
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature


matplotlib.use("Agg")



def get_stats(x):
    return np.mean(x), np.std(x), np.median(x), np.percentile(x, 95)

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def plot_map(data_type):
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # ---- symmetric color scale ----
    if metric == "impr":
        vmax = 35.146782
    else:
        vmax = np.nanpercentile(np.abs(Z.values), 99)  # more robust than max
    # vmax = np.nanpercentile(np.abs(Z.values), 99)  # more robust than max

    im = ax.pcolormesh(
        Lon, Lat, Z.values,
        cmap="RdBu",  # cleaner than RdYlBu
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # ---- map styling ----
    ax.set_extent(map_extent)

    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)

    ax.coastlines(resolution="10m", linewidth=0.6, color="black", zorder=2)
    ax.contour(
        Lon, Lat, Z.values,
        levels=8,
        colors="black",
        linewidths=0.3,
        alpha=0.3
    )
    # subtle gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # ---- colorbar ----
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)

    cbar.set_label(
        "Relative RMSE improvement (%)",
        fontsize=11,
        fontweight="bold",
    )

    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/rmse_{metric}_{data_type}_map.pdf", dpi=300, bbox_inches="tight")
    plt.close()

def get_lat_lon_Z(dframe):
    # Version-safe RMSE by location (avoid groupby.apply(...).reset_index(name=...),
    # which behaves differently across pandas versions).
    rmse_unc = (
        dframe.assign(se_unc=(dframe["y_unco"] - dframe["y_true"]) ** 2)
        .groupby(["lat", "lon"], as_index=False)["se_unc"]
        .mean()
        .rename(columns={"se_unc": "mse_unc"})
    )
    rmse_unc["rmse_unc"] = np.sqrt(rmse_unc["mse_unc"])
    rmse_unc = rmse_unc.drop(columns=["mse_unc"])

    rmse_cor = (
        dframe.assign(se_cor=(dframe["y_pred"] - dframe["y_true"]) ** 2)
        .groupby(["lat", "lon"], as_index=False)["se_cor"]
        .mean()
        .rename(columns={"se_cor": "mse_cor"})
    )
    rmse_cor["rmse_cor"] = np.sqrt(rmse_cor["mse_cor"])
    rmse_cor = rmse_cor.drop(columns=["mse_cor"])

    # merge
    d = rmse_unc.merge(rmse_cor, on=["lat", "lon"])

    # difference (what your plot shows)
    d["rmse_diff"] = d["rmse_unc"] - d["rmse_cor"]
    d["rmse_impr"] = 100 * ((d["rmse_unc"] - d["rmse_cor"]) / d["rmse_unc"])

    # unique grid
    lats_unique = np.sort(d["lat"].unique())
    lons_unique = np.sort(d["lon"].unique())

    # pivot to grid
    Z = d.pivot(index="lat", columns="lon", values=f"rmse_{metric}")

    # align axes
    Z = Z.reindex(index=lats_unique, columns=lons_unique)

    Lon, Lat = np.meshgrid(lons_unique, lats_unique)

    f.write("\nMean Relative Per point statistics per location:\n")
    f.write(f'{d[d[f"rmse_{metric}"] > 0][f"rmse_{metric}"].describe()}\n')
    improved_locations = len(d[d[f"rmse_{metric}"] > 0][f"rmse_{metric}"])
    f.write(f"\nImproved locations (locations_only): {improved_locations}\n"
            f"Improvement: {100 * (improved_locations / len(d))}\n")

    return Lat, Lon, Z

def filter_area(data_full, lat_min, lon_min, lat_max, lon_max):
    df_filtered = data_full[
        (data_full["lon"] >= lon_min) & (data_full["lon"] <= lon_max) &
        (data_full["lat"] >= lat_min) & (data_full["lat"] <= lat_max)
        ]
    return df_filtered

def evaluate(dataset, denoise, subset="all"):
    df = copy.deepcopy(dataset)

    # ---- Per-sample squared error improvement ----
    df["err_unc"] = (df["y_unco"] - df["y_true"]) ** 2
    df["err_pred"] = (df["y_pred"] - df["y_true"]) ** 2

    df["delta"] = df["err_unc"] - df["err_pred"]  # >0 means improvement

    # Optional subset analysis
    subset = (subset or "all").lower()
    if subset == "degraded":
        df = df[df["delta"] < 0].copy()
    elif subset == "improved":
        df = df[df["delta"] > 0].copy()

    if len(df) == 0:
        f.write(f"\nSubset '{subset}' has no samples. Skipping metrics.\n")
        return None, None, None, df, None

    df_filtered_out = None
    if denoise:
        abs_th = denoise
        sq_th = abs_th ** 2
        mask = df["err_unc"] > sq_th
        f.write(f"\nDenoising: \nabs err > {abs_th:.2f} m  ->  sq err > {sq_th:.4g}  ->  coverage {100 * np.mean(mask):.2f}%\n")

        df_filtered_out = df[~mask].copy()
        df = df[mask]

        if len(df) == 0:
            f.write(
                f"\nSubset '{subset}' has no samples after denoising threshold {abs_th:.2f}m.\n"
            )
            return None, None, None, df, df_filtered_out

    lat, lon, z = get_lat_lon_Z(df)
    rmse_y_unco = rmse(df["y_unco"], df["y_true"])
    rmse_y_pred = rmse(df["y_pred"], df["y_true"])
    global_rmse = 100 * ((rmse_y_unco - rmse_y_pred) / rmse_y_unco)
    f.write(f"Global RMSE (samples level): {global_rmse}\n\n")

    improved_samples = np.sum(df["delta"] > 0)
    improvement = 100 * (improved_samples / len(df))
    f.write(f"Per point squared improvement stats:\n{df["delta"].describe()}\n\n"
            f"(y_unc - y_true)^2 - (y_pred-y_true)^2 Globally: doesn't group by per location to count the improvement of the location\n\n"
            f"Improved samples (including hours): {improved_samples}\n"
            f"Improvement (samples percentage): {improvement}\n"
            f"Mean Improvement (mean_delta): {df['delta'].mean()},s={df['delta'].std()}\n")

    #----- Per sample relative improvement ----- #
    eps = 1e-8
    improvement_pct = (df["delta"] / (df["err_unc"] + eps)) * 100
    bad_mask = improvement_pct < 0
    bad_improvements = improvement_pct[bad_mask]

    # ---- Stats ----
    mean_improvement = np.mean(improvement_pct)
    median_improvement = np.median(improvement_pct)
    std_improvement = np.std(improvement_pct)
    count_improved = np.sum(improvement_pct > 0)
    percentage_improved = 100 * count_improved / len(improvement_pct)

    def _write_case_stats(case_name, mask=None):
        if mask is None:
            sub = df
            sub_imp = improvement_pct
        else:
            sub = df[mask]
            sub_imp = improvement_pct[mask]
        if len(sub) == 0:
            f.write(f"\n=== {case_name} Error Stats ===\nNo samples.\n")
            return

        sub_abs_unc = np.abs(sub["y_unco"] - sub["y_true"])
        sub_abs_pred = np.abs(sub["y_pred"] - sub["y_true"])
        sub_sq_unc = (sub["y_unco"] - sub["y_true"]) ** 2
        sub_sq_pred = (sub["y_pred"] - sub["y_true"]) ** 2

        f.write(f"\n=== {case_name} Error Stats ===\n")
        f.write(f"Count: {len(sub)} ({100 * len(sub) / len(df):.2f}% of current section)\n")
        f.write("Absolute error (uncorrected) describe:\n")
        f.write(f"{sub_abs_unc.describe()}\n")
        f.write("Absolute error (corrected) describe:\n")
        f.write(f"{sub_abs_pred.describe()}\n")
        f.write("Squared error (uncorrected) describe:\n")
        f.write(f"{sub_sq_unc.describe()}\n")
        f.write("Squared error (corrected) describe:\n")
        f.write(f"{sub_sq_pred.describe()}\n")
        f.write("Relative improvement (%) describe:\n")
        f.write(f"{sub_imp.describe()}\n")

    f.write("\n=== Per-sample Relative Improvement (%) ===\n")
    f.write(f"Mean: {mean_improvement:.2f}%\n")
    f.write(f"Std: {std_improvement:.2f}%\n")
    f.write(f"Median: {median_improvement:.2f}%\n")
    f.write(f"Improved samples: {percentage_improved:.2f}%\n\n")

    f.write("=== Degraded samples (model worse) ===\n")
    f.write(f"Count: {len(bad_improvements)} ({100 * len(bad_improvements) / len(improvement_pct):.2f}%)\n")
    f.write(f"Mean: {np.mean(bad_improvements):.2f}%\n")
    f.write(f"Std: {np.std(bad_improvements):.2f}%\n")
    f.write(f"Median: {np.median(bad_improvements):.2f}%\n")

    y_bad = df["y_true"][bad_mask]

    f.write("\n=== Wave height (bad samples) ===\n")
    f.write(f"Mean Hs: {np.mean(y_bad):.2f}\n")
    f.write(f"Std Hs: {np.std(y_bad):.2f}\n")
    f.write(f"Median Hs: {np.median(y_bad):.2f}\n")
    f.write(f"P90 Hs: {np.percentile(y_bad, 90):.2f}\n")

    # Detailed error summaries to verify absolute error magnitudes.
    _write_case_stats("All samples")
    _write_case_stats("Improved samples", improvement_pct > 0)
    _write_case_stats("Degraded samples", improvement_pct < 0)

    if df_filtered_out is not None and len(df_filtered_out) > 0:
        f.write("\n=== Filtered-out subset (low-noise points) ===\n")
        f.write(f"Count: {len(df_filtered_out)}\n")
        f.write(f"Percentage of original samples: {100 * len(df_filtered_out) / len(dataset):.2f}%\n")

        rmse_out_unc = rmse(df_filtered_out["y_unco"], df_filtered_out["y_true"])
        rmse_out_pred = rmse(df_filtered_out["y_pred"], df_filtered_out["y_true"])
        mae_out_unc = np.mean(np.abs(df_filtered_out["y_unco"] - df_filtered_out["y_true"]))
        mae_out_pred = np.mean(np.abs(df_filtered_out["y_pred"] - df_filtered_out["y_true"]))

        rmse_out_impr = (
            100 * (rmse_out_unc - rmse_out_pred) / rmse_out_unc if rmse_out_unc > 0 else 0.0
        )
        mae_out_impr = (
            100 * (mae_out_unc - mae_out_pred) / mae_out_unc if mae_out_unc > 0 else 0.0
        )

        f.write(f"RMSE uncorrected: {rmse_out_unc:.6f}\n")
        f.write(f"RMSE corrected:   {rmse_out_pred:.6f}\n")
        f.write(f"RMSE improvement: {rmse_out_impr:.4f}%\n")
        f.write(f"MAE uncorrected:  {mae_out_unc:.6f}\n")
        f.write(f"MAE corrected:    {mae_out_pred:.6f}\n")
        f.write(f"MAE improvement:  {mae_out_impr:.4f}%\n")

        eps_out = 1e-8
        err_unc_out = (df_filtered_out["y_unco"] - df_filtered_out["y_true"]) ** 2
        err_pred_out = (df_filtered_out["y_pred"] - df_filtered_out["y_true"]) ** 2
        imp_pct_out = ((err_unc_out - err_pred_out) / (err_unc_out + eps_out)) * 100

        f.write("\nFiltered-out denominator diagnostics (err_unc):\n")
        f.write(f"{err_unc_out.describe()}\n")
        for thr in [1e-8, 1e-6, 1e-4, 1e-3]:
            cnt_thr = int((err_unc_out < thr).sum())
            pct_thr = 100.0 * cnt_thr / len(err_unc_out)
            f.write(f"err_unc < {thr:.0e}: {cnt_thr} ({pct_thr:.2f}%)\n")

        f.write("\nFiltered-out relative improvement (%) diagnostics:\n")
        f.write(f"{imp_pct_out.describe()}\n")
        q = imp_pct_out.quantile([0.01, 0.05, 0.95, 0.99])
        f.write(
            "p01/p05/p95/p99: "
            f"{q.loc[0.01]:.2f}, {q.loc[0.05]:.2f}, {q.loc[0.95]:.2f}, {q.loc[0.99]:.2f}\n"
        )
        imp_pct_out_clipped = imp_pct_out.clip(-500, 500)
        f.write(
            "clipped[-500,500] mean/std: "
            f"{imp_pct_out_clipped.mean():.2f} / {imp_pct_out_clipped.std():.2f}\n"
        )

        delta_out = (
            (df_filtered_out["y_unco"] - df_filtered_out["y_true"]) ** 2
            - (df_filtered_out["y_pred"] - df_filtered_out["y_true"]) ** 2
        )
        improved_out = np.sum(delta_out > 0)
        f.write(
            f"Improved samples in filtered-out subset: {improved_out} "
            f"({100 * improved_out / len(df_filtered_out):.2f}%)\n"
        )

    return lat, lon, z, df, df_filtered_out

def plot_dist(dfd, path):
    ref_color = "#4285f4"
    trans_color = "#eb9999"
    uncor_color = '#484848'#"#a4c2f4"

    fig, ax = plt.subplots(figsize=(10, 7))  # Αυξήσαμε ελαφρώς το figsize για να χωρέσουν τα μεγαλύτερα γράμματα
    label_ref = "Reference"
    label_tr = "TransUNet"
    label_unc = "Uncorrected"

    sns.kdeplot(dfd["y_true"], ax=ax, label=label_ref, color=ref_color, linewidth=2.5, bw_adjust=0.7)
    sns.kdeplot(dfd["y_pred"], ax=ax, label=label_tr, color=trans_color, linewidth=1.5, bw_adjust=0.7, linestyle="--")
    sns.kdeplot(dfd["y_unco"], ax=ax, label=label_unc, color=uncor_color, linewidth=1.5, bw_adjust=0.7, alpha=0.5)
    ax.set_xlim(0, 6)
    ax.legend(
        loc="upper right",
        # bbox_to_anchor=(0.5, 1.25),  # Τοποθέτηση πάνω από το γράφημα
        ncol=1,  # Χωρισμός σε 2 στήλες για να απλωθεί σε πλάτος
        columnspacing=1.0,  # Αυξάνει την απόσταση μεταξύ των στηλών
        fontsize=10,
        frameon=True
    )

    plt.tight_layout()

    # Αποθήκευση σε PDF
    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")

def plot_scatter(dfs, save_path):
    err_pred = dfs["err_pred"]
    err_unc = dfs["err_unc"]
    fig, ax = plt.subplots(figsize=(8, 8))

    # Boolean masks
    better = err_pred < err_unc  # improvement → below diagonal
    worse = err_pred >= err_unc  # worse → above diagonal

    # Scatter
    ax.scatter(err_unc[better], err_pred[better], label="Improved", alpha=0.5)
    ax.scatter(err_unc[worse], err_pred[worse], label="Worse", alpha=0.5)

    # Diagonal
    min_val = min(err_unc.min(), err_pred.min())
    max_val = max(err_unc.max(), err_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    # Labels
    ax.set_xlabel("err_unc")
    ax.set_ylabel("err_pred")
    ax.set_title("Error comparison (per point)")
    ax.legend()

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(fig)


def plot_scatter_filtered_out(df_out, save_path):
    """Scatter of filtered-out (low-noise) points.

    x: |corrected − ground truth|, y: |uncorrected − ground truth|, with 1:1 line.
    """
    abs_err_cor = np.abs(df_out["y_pred"] - df_out["y_true"])
    abs_err_unc = np.abs(df_out["y_unco"] - df_out["y_true"])

    fig, ax = plt.subplots(figsize=(7, 7))

    better = abs_err_cor < abs_err_unc
    worse  = abs_err_cor >= abs_err_unc

    ax.scatter(
        abs_err_cor[better], abs_err_unc[better],
        s=6, alpha=0.4, color="#2ca25f", label=f"Improved ({100*better.mean():.1f}%)",
        rasterized=True,
    )
    ax.scatter(
        abs_err_cor[worse], abs_err_unc[worse],
        s=6, alpha=0.4, color="#d73027", label=f"Degraded ({100*worse.mean():.1f}%)",
        rasterized=True,
    )

    lim = max(abs_err_cor.max(), abs_err_unc.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="1:1")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("|corrected − ground truth|  (m)", fontsize=12)
    ax.set_ylabel("|uncorrected − ground truth|  (m)", fontsize=12)
    ax.set_title(f"Filtered-out points  (n={len(df_out):,})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_scatter_per_wave_bin(dfs, save_path, label):
    bin_width = 1.0
    err_pred = dfs["err_pred"].values
    err_unc = dfs["err_unc"].values

    wave = dfs["y_true"].values  # or y_unco depending on your definition

    max_wave = wave.max()
    # define bins
    bins = np.arange(0, max_wave + bin_width, bin_width)
    n_bins = len(bins) - 1

    ncols = 3
    nrows = int(np.ceil(n_bins / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_bins):
        ax = axes[i]

        bmin, bmax = bins[i], bins[i + 1]

        mask = (wave >= bmin) & (wave < bmax)

        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        e_unc = err_unc[mask]
        e_pred = err_pred[mask]

        better = e_pred < e_unc
        worse = e_pred >= e_unc

        labelb = f"{100*(better.sum()/len(e_pred)):.2f}% Improved"
        labelw = f"{100*(worse.sum()/len(e_pred)):.2f}% Degraded"

        # scatter
        ax.scatter(e_unc[better], e_pred[better], label=labelb,
                   color="#2ca25f", alpha=0.5, s=10)

        ax.scatter(e_unc[worse], e_pred[worse],
                   color="#d73027", alpha=0.5, s=10, label=labelw)

        ax.legend(
            loc="upper left",
            # bbox_to_anchor=(0.5, 1.25),  # Τοποθέτηση πάνω από το γράφημα
            ncol=1,  # Χωρισμός σε 2 στήλες για να απλωθεί σε πλάτος
            columnspacing=1.0,  # Αυξάνει την απόσταση μεταξύ των στηλών
            fontsize=10,
            frameon=True
        )
        # diagonal
        min_val = min(e_unc.min(), e_pred.min())
        max_val = max(e_unc.max(), e_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

        ax.set_title(f"{bmin:.0f}–{bmax:.0f} m\n(n={mask.sum()})")

        ax.set_aspect('equal', adjustable='box')

        if i == 0:
            ax.set_ylabel("err_pred")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("err_unc")

    # remove empty axes
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(f"Error comparison per wave regime in {label} ", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot RMSE maps and diagnostics from evaluator NPZ output."
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        default="data/transunet_coords.npz",
        help="Path to input NPZ with y_true, y_pred, vhm0, lat, lon.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/transunet",
        help="Directory for generated figures and statistics.txt.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="impr",
        choices=["diff", "impr"],
        help="Map metric: diff (rmse_unc-rmse_cor) or impr (relative improvement %).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="auto",
        choices=["auto", "mediterranean", "atlantic", "aegean"],
        help=(
            "Region mode for map extent and subregion reports. "
            "'mediterranean' enables Aegean/Cyprus/North Italy/Central Med blocks; "
            "'atlantic' skips them."
        ),
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "degraded", "improved"],
        help=(
            "Sample subset for diagnostics: all points, only degraded points "
            "(model worse than uncorrected), or only improved points."
        ),
    )
    args = parser.parse_args()

    data = np.load(args.input_npz)
    metric = args.metric
    out_dir = args.output_dir
    model = os.path.splitext(os.path.basename(args.input_npz))[0]
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/statistics.txt", "w") as f:
        f.write(f"Model: {model}\n"
                f"Metric: {metric}\n"
                f"Subset: {args.subset}\n"
                f"==================\n\n")
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_unco = data["vhm0"]
        y_lat = data['lat']
        y_lon = data['lon']

        df = pd.DataFrame({
            "lat": y_lat,
            "lon": y_lon,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_unco": y_unco
        })

        region = args.region
        if region == "auto":
            lon_min_data = float(df["lon"].min())
            region = "atlantic" if lon_min_data < -6.0 else "mediterranean"

        if region == "mediterranean":
            map_extent = [-6, 36, 30, 46]
        elif region == "aegean":
            map_extent = [23, 28, 35, 42]
        else:
            pad_deg = 0.25
            map_extent = [
                float(df["lon"].min()) - pad_deg,
                float(df["lon"].max()) + pad_deg,
                float(df["lat"].min()) - pad_deg,
                float(df["lat"].max()) + pad_deg,
            ]

        f.write(f"Region: {region}\n")


        # df has only sea points, columns: lat, lon
        lats = np.sort(df["lat"].unique())
        lons = np.sort(df["lon"].unique())

        dlat = np.median(np.diff(lats))
        dlon = np.median(np.diff(lons))

        fig = plt.figure(figsize=(9, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent([lons.min() - 1, lons.max() + 1, lats.min() - 1, lats.max() + 1])

        # background: land white
        ax.add_feature(cfeature.OCEAN, facecolor="#eaf2f8", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.4, zorder=2)
        ax.coastlines(resolution="10m", linewidth=0.5, color="black", zorder=3)

        # pick one random sea point
        rand_row = df[["lat", "lon"]].drop_duplicates().sample(1).iloc[0]
        rand_lat = rand_row["lat"]
        rand_lon = rand_row["lon"]

        # draw only sea grid cells
        for _, row in df[["lat", "lon"]].drop_duplicates().iterrows():
            lat = row["lat"]
            lon = row["lon"]

            # check if this is the random one
            is_random = (lat == rand_lat) and (lon == rand_lon)

            rect = patches.Rectangle(
                (lon - dlon / 2, lat - dlat / 2),
                dlon,
                dlat,
                facecolor="#d73027" if is_random else "none",
                edgecolor="#d73027" if is_random else "black",
                linewidth=1.2 if is_random else 0.08,
                alpha=0.6 if is_random else 0.4,
                transform=ccrs.PlateCarree(),
                zorder=5 if is_random else 1
            )
            ax.add_patch(rect)

        plt.tight_layout()
        plt.savefig("sea_grid_cells_5798.pdf", dpi=300, bbox_inches="tight")
        plt.show()


        # Evaluate all
        f.write("\n====================================\n")
        f.write("======= Full Map Evaluation ========\n")
        f.write("====================================\n")
        subset_tag = args.subset
        if subset_tag == "all":
            subset_tag = "all"

        Lat, Lon, Z, dfd, _ = evaluate(df, None, subset=args.subset)
        if Lat is not None:
            plot_map(subset_tag)
            plot_dist(dfd, f"{out_dir}/{subset_tag}_distr.pdf")
            plot_scatter_per_wave_bin(
                dfd,
                f"{out_dir}/{subset_tag}_scatter_per_bin.png",
                subset_tag,
            )
        else:
            f.write(f"Skipping full-map plots for subset '{args.subset}' (no data).\n")

        Lat, Lon, Z, dfd, dfd_filtered_out = evaluate(df, 0.05, subset=args.subset)
        # Lat, Lon, Z, dfd = evaluate(df, 0.10)
        # Lat, Lon, Z, dfd = evaluate(df, 0.15)
        # Lat, Lon, Z, dfd = evaluate(df, 0.20)
        # Lat, Lon, Z, dfd = evaluate(df, 0.30)
        # Lat, Lon, Z, dfd = evaluate(df, 0.40)
        # Lat, Lon, Z, dfd = evaluate(df, 0.50)
        # Lat, Lon, Z, dfd = evaluate(df, 1)
        # Lat, Lon, Z, dfd = evaluate(df, 2)
        # Lat, Lon, Z, dfd = evaluate(df, 5)
        if Lat is not None:
            plot_map(f"{subset_tag}_denoised_0.05")
            plot_dist(dfd, f"{out_dir}/{subset_tag}_denoised_0.05_distr.pdf")
            plot_scatter_per_wave_bin(
                dfd,
                f"{out_dir}/{subset_tag}_denoised_0.05_scatter_per_bin.png",
                f"{subset_tag}_denoised_0.05",
            )
        else:
            f.write(f"Skipping denoised plots for subset '{args.subset}' (no data).\n")
        if dfd_filtered_out is not None and len(dfd_filtered_out) > 0:
            plot_dist(dfd_filtered_out, f"{out_dir}/{subset_tag}_filtered_out_0.05_distr.pdf")
            plot_scatter_per_wave_bin(
                dfd_filtered_out,
                f"{out_dir}/{subset_tag}_filtered_out_0.05_scatter_per_bin.png",
                f"{subset_tag}_filtered_out_0.05",
            )
            f.write("\n====================================\n")
            f.write("=== Filtered-out subset map eval ===\n")
            f.write("====================================\n")
            Lat, Lon, Z = get_lat_lon_Z(dfd_filtered_out)
            plot_map(f"{subset_tag}_filtered_out_0.05")

        # if region == "mediterranean":
        #     f.write("\n====================================\n")
        #     f.write("======= Aegean Evaluation ==========\n")
        #     f.write("====================================\n")
        #     # Aegean Sea bounds (approx)
        #     lon_min, lon_max = 23, 28
        #     lat_min, lat_max = 35, 42

        #     df_aegean = filter_area(df, lat_min, lon_min, lat_max, lon_max)
        #     Lat, Lon, Z, dfd = evaluate(df_aegean, None)
        #     plot_map("aegean_all")
        #     plot_dist(dfd, f"{out_dir}/aegean_all_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/aegean_all_scatter_per_bin.png", "aegean_all")

        #     Lat, Lon, Z, dfd = evaluate(df_aegean, 0.05)
        #     plot_map("aegean_denoised_0.05")
        #     plot_dist(dfd, f"{out_dir}/aegean_denoised_0.05_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/aegean_denoised_0.05_scatter_per_bin.png", "aegean_denoised_0.05")

        #     f.write("\n====================================\n")
        #     f.write("======= Cyprus Evaluation ==========\n")
        #     f.write("====================================\n")
        #     # Cyprus Sea bounds (approx)
        #     lon_min, lon_max = 31, 40
        #     lat_min, lat_max = 31, 37

        #     df_cyprus = filter_area(df, lat_min, lon_min, lat_max, lon_max)
        #     Lat, Lon, Z, dfd = evaluate(df_cyprus, None)
        #     plot_map("cyprus_all")
        #     plot_dist(dfd, f"{out_dir}/cyprus_all_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/cyprus_all_scatter_per_bin.png", "cyprus_all")

        #     Lat, Lon, Z, dfd = evaluate(df_cyprus, 0.05)
        #     plot_map("cyprus_denoised")
        #     plot_dist(dfd, f"{out_dir}/cyprus_denoised_0.05_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/cyprus_denoised_0.05_scatter_per_bin.png", "cyprus_denoised_0.05")

        #     f.write("\n====================================\n")
        #     f.write("======= North Italy Evaluation =====\n")
        #     f.write("====================================\n")
        #     # North Italy sea bounds approx)
        #     lon_min, lon_max = 12, 18
        #     lat_min, lat_max = 42, 46

        #     df_nitaly = filter_area(df, lat_min, lon_min, lat_max, lon_max)
        #     Lat, Lon, Z, dfd = evaluate(df_nitaly, None)
        #     plot_map("north_italy_all")
        #     plot_dist(dfd, f"{out_dir}/north_italy_all_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/north_italy_all_scatter_per_bin.png", "north_italy_all")

        #     Lat, Lon, Z, dfd = evaluate(df_nitaly, 0.05)
        #     plot_map("north_italy_denoised_0.05")
        #     plot_dist(dfd, f"{out_dir}/north_italy_denoised_0.05_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/north_italy_denoised_0.05_scatter_per_bin.png", "north_italy_denoised_0.05")

        #     f.write("\n====================================\n")
        #     f.write("======= Central Med Evaluation =====\n")
        #     f.write("====================================\n")
        #     # central mediterannean good region
        #     lon_min, lon_max = 15, 20
        #     lat_min, lat_max = 32, 40

        #     df_central = filter_area(df, lat_min, lon_min, lat_max, lon_max)
        #     Lat, Lon, Z, dfd = evaluate(df_central, None)
        #     plot_map("central_all")
        #     plot_dist(dfd, f"{out_dir}/central_all_distr.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/central_all_scatter_per_bin.png", "central_all")

        #     Lat, Lon, Z, dfd = evaluate(df_central, 0.05)
        #     plot_map("central_denoised_0.05")
        #     plot_dist(dfd, f"{out_dir}/central_denoised_0.05.pdf")
        #     plot_scatter_per_wave_bin(dfd, f"{out_dir}/central_denoised_0.05_scatter_per_bin.png", "canteal_denoised_0.05")
        # else:
        #     f.write("\nSubregion evaluations skipped (region != mediterranean).\n")
