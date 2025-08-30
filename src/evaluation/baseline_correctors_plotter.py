from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from comet_ml import ExistingExperiment


class PredictionPlotter:
    def __init__(self, prediction_dir: str, comet_exp: ExistingExperiment):
        self.prediction_dir = Path(prediction_dir)
        self.comet = comet_exp
        # Don't load data immediately - use lazy loading
        self._df = None

    @property
    def df(self):
        """Lazy loading of the DataFrame - only loads when first accessed"""
        if self._df is None:
            self._df = self._load_all_predictions()
        return self._df

    def _load_all_predictions(self) -> pl.DataFrame:
        """Optimized data loading with streaming and memory-efficient operations"""
        print(f"📂 Loading daily predictions from {self.prediction_dir}...")
        files = sorted(self.prediction_dir.glob("WAVEAN*.parquet"))

        # Use lazy evaluation to avoid loading all files into memory at once
        lazy_dfs = []
        for file in files:
            lazy_df = pl.scan_parquet(file).with_columns([
                pl.col("timestamp").dt.date().alias("day")
            ])
            lazy_dfs.append(lazy_df)

        # Concatenate lazily and collect only when needed
        return pl.concat(lazy_dfs, how="vertical").collect(engine="streaming")

    def compute_metrics(self, resolution: float = 0.25):
        """Memory-optimized metrics computation with streaming and reduced copies"""
        print("📊 Computing metrics per day, month, and spatial bins...")

        # Use lazy evaluation for the initial data processing
        files = sorted(self.prediction_dir.glob("WAVEAN*.parquet"))
        if not files:
            raise FileNotFoundError(f"No WAVEAN*.parquet files found in {self.prediction_dir}")

        # Create lazy DataFrames for each file
        lazy_dfs = []
        for file in files:
            lazy_df = pl.scan_parquet(file).with_columns([
                pl.col("timestamp").dt.date().alias("day")
            ]).filter(
                pl.all_horizontal([
                    pl.col("VHM0").is_not_null(),
                    pl.col("VTM02").is_not_null(),
                    pl.col("predicted_VHM0").is_not_null(),
                    pl.col("predicted_VTM02").is_not_null(),
                    pl.col("latitude").is_not_null(),
                    pl.col("longitude").is_not_null()
                ])
            )
            lazy_dfs.append(lazy_df)

        # Concatenate all lazy DataFrames
        lazy_df = pl.concat(lazy_dfs, how="vertical")

        def metric_block(pred: str, target: str):
            base = pred.replace("predicted_", "")
            return (
                ((pl.col(pred) - pl.col(target)) ** 2).mean().sqrt().alias(f"rmse_{base}"),
                (pl.col(pred) - pl.col(target)).abs().mean().alias(f"mae_{base}"),
                (pl.col(pred) - pl.col(target)).mean().alias(f"diff_{base}"),
                (pl.col(pred).mean() - pl.col(target).mean()).alias(f"bias_{base}"),
                ((pl.col(pred) - pl.col(pred).mean()) ** 2).mean().alias(f"var_{base}"),
                pl.corr(pred, target).alias(f"corr_{base}"),
            )

        # Process metrics by day with streaming
        print("Computing daily metrics...")
        metrics_day = lazy_df.group_by("day").agg([
            *metric_block("predicted_VHM0", "corrected_VHM0"),
            *metric_block("predicted_VTM02", "corrected_VTM02")
        ]).collect(engine="streaming")
        metrics_day.write_parquet(self.prediction_dir / "metrics_by_day.parquet")

        # Process metrics by month with streaming
        print("Computing monthly metrics...")
        metrics_month = lazy_df.with_columns([
            pl.col("day").dt.strftime("%Y-%m").alias("month")
        ]).group_by("month").agg([
            *metric_block("predicted_VHM0", "corrected_VHM0"),
            *metric_block("predicted_VTM02", "corrected_VTM02")
        ]).collect(engine="streaming")
        metrics_month.write_parquet(self.prediction_dir / "metrics_by_month.parquet")

        # Process spatial metrics with streaming
        print("Computing spatial metrics...")
        metrics_spatial = lazy_df.with_columns([
            pl.col("day").dt.strftime("%Y-%m").alias("month"),
            pl.col("day").dt.month().alias("month_num"),
            (pl.col("latitude") / resolution).round(0) * resolution,
            (pl.col("longitude") / resolution).round(0) * resolution
        ]).with_columns([
            pl.when(pl.col("month_num").is_in([12, 1, 2])).then(pl.lit("winter"))
            .when(pl.col("month_num").is_in([3, 4, 5])).then(pl.lit("spring"))
            .when(pl.col("month_num").is_in([6, 7, 8])).then(pl.lit("summer"))
            .when(pl.col("month_num").is_in([9, 10, 11])).then(pl.lit("autumn"))
            .alias("season")
        ]).drop("month_num").rename({
            "latitude": "lat_bin",
            "longitude": "lon_bin"
        }).group_by(["month", "season", "lat_bin", "lon_bin"]).agg([
            *metric_block("predicted_VHM0", "corrected_VHM0"),
            *metric_block("predicted_VTM02", "corrected_VTM02")
        ]).collect(engine="streaming")
        metrics_spatial.write_parquet(self.prediction_dir / "metrics_spatial_by_month.parquet")

        return metrics_day, metrics_month, metrics_spatial

    def _log_metric_trends_lines(
        self,
        metric_table: pl.DataFrame,
        group_col: str,
        metrics: list[str] | None = None
    ):
        if metrics is None:
            metrics = ["rmse", "mae", "bias"]
        import matplotlib.pyplot as plt

        group_values = metric_table[group_col].to_list()

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 4))

            for col in metric_table.columns:
                if col.startswith(metric + "_") and col != metric:
                    label = col.replace(f"{metric}_", "")
                    ax.plot(group_values, metric_table[col].to_numpy(), label=label, marker='o')

            ax.set_title(f"{metric.upper()} per {group_col}")
            ax.set_xlabel(group_col.capitalize())
            ax.set_ylabel(metric.upper())
            ax.grid(True)
            ax.legend()

            self.comet.log_figure(
                figure_name=f"metric_trend__{metric}__per_{group_col}.png",
                figure=fig
            )
            plt.close(fig)

    def _log_metric_trends(
        self,
        metric_table: pl.DataFrame,
        group_col: str,
        metrics: list[str] | None = None
    ):
        if metrics is None:
            metrics = ["rmse", "mae", "bias", "diff", "var"]
        print(f"🗺️ Plotting metrics grouped by {group_col}...")
        import matplotlib.pyplot as plt
        import numpy as np

        # Sort the metric table by the group column to ensure chronological order
        if group_col in ["day", "month"]:
            # For date-based columns, sort chronologically
            if group_col == "day":
                sorted_table = metric_table.sort("day")
            else:  # month
                sorted_table = metric_table.sort("month")
        else:
            # For non-date columns, sort alphabetically
            sorted_table = metric_table.sort(group_col)

        group_values = sorted_table[group_col].to_list()
        x = np.arange(len(group_values))  # the label locations

        for metric in metrics:
            # Identify variables (e.g., "vhm0", "vtm02")
            metric_cols = [col for col in sorted_table.columns if col.startswith(metric + "_")]
            variables = [col.replace(f"{metric}_", "") for col in metric_cols]

            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.8 / len(variables)  # bar width

            for i, (col, var) in enumerate(zip(metric_cols, variables, strict=False)):
                values = sorted_table[col].to_numpy()
                offset = x + (i - len(variables) / 2) * width + width / 2
                ax.bar(offset, values, width=width, label=var.upper())

            ax.set_title(f"{metric.upper()} per {group_col}")
            ax.set_xlabel(group_col.capitalize())
            ax.set_ylabel(metric.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(group_values, rotation=90)
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            ax.legend()

            self.comet.log_figure(
                figure_name=f"metric_bar__{metric}__per_{group_col}.png",
                figure=fig
            )
            plt.close(fig)


    def _log_distributions(self, group_col: str, group_fmt: str):
        """Memory-optimized distribution logging with streaming"""
        print(f"📊 Logging histograms grouped by {group_col}...")

        # Use lazy evaluation for distribution analysis
        files = sorted(self.prediction_dir.glob("WAVEAN*.parquet"))
        if not files:
            raise FileNotFoundError(f"No WAVEAN*.parquet files found in {self.prediction_dir}")

        # Create lazy DataFrames for each file
        lazy_dfs = []
        for file in files:
            lazy_df = pl.scan_parquet(file).with_columns([
                pl.col("timestamp").dt.date().alias("day"),
                pl.col("timestamp").dt.strftime(group_fmt).alias(group_col)
            ]).filter(
                pl.all_horizontal([
                    pl.col("VHM0").is_not_null(),
                    pl.col("VTM02").is_not_null(),
                    pl.col("predicted_VHM0").is_not_null(),
                    pl.col("predicted_VTM02").is_not_null()
                ])
            )
            lazy_dfs.append(lazy_df)

        # Concatenate all lazy DataFrames
        lazy_df = pl.concat(lazy_dfs, how="vertical")

        # Get unique groups first
        unique_groups = lazy_df.select(group_col).unique().collect(engine="streaming")[group_col].to_list()

        for grp in unique_groups:
            # Process each group separately to avoid loading all data
            group_df = lazy_df.filter(pl.col(group_col) == grp).collect(engine="streaming")

            for col in ["VHM0", "VTM02"]:
                self._plot_histogram_group(
                    group_df, col, f"{col}_{grp}_{group_col}"
                )

    def _plot_histogram_group(self, df: pl.DataFrame, col: str, label: str):
        bins = np.linspace(
            float(df[col].min()),
            float(df[col].max()),
            101
        )

        def compute_hist(data: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
            bin_indices = np.digitize(data, bins) - 1
            bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
            counts = np.bincount(bin_indices, minlength=len(bins) - 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            return bin_centers, counts

        fig, ax = plt.subplots(figsize=(8, 4))
        for kind, c, lbl in zip(["input", "predicted", "target"],
                                ["blue", "green", "orange"],
                                [col, f"predicted_{col}", f"corrected_{col}"], strict=False):
            if lbl not in df.columns:
                continue
            x, y = compute_hist(df[lbl].to_numpy(), lbl)
            ax.bar(x, y, width=(x[1] - x[0]), alpha=0.3, edgecolor='black', color=c, label=kind)

        ax.set_title(f"Distribution of {col} - {label}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)
        self.comet.log_figure(figure_name=f"hist__{label}.png", figure=fig)
        plt.close(fig)

    def _log_maps(
        self,
        grouped_metrics: pl.DataFrame,
        time_col: str,
        metrics: list[str] | None = None,
        variables: list[str] | None = None,
        cmap: str = "viridis"
    ):
        if metrics is None:
            metrics = ["rmse", "mae", "bias"]
        if variables is None:
            variables = ["VHM0", "VTM02"]
        import matplotlib.pyplot as plt

        def month_to_season(month: int) -> str:
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            elif month in [9, 10, 11]:
                return "Autumn"
            return "Unknown"

        print(f"🗺️ Plotting metric maps grouped by '{time_col}'...")

        unique_times = grouped_metrics.select(time_col).unique().to_series().to_list()

        # Optional: enforce season ordering
        if time_col == "season":
            season_order = ["winter", "spring", "summer", "autumn"]
            unique_times = [s for s in season_order if s in unique_times]

        for grp in unique_times:
            gdf = grouped_metrics.filter(pl.col(time_col) == grp)

            grp_str = grp  # Can adjust formatting if needed

            for var in variables:
                for metric in metrics:
                    col = f"{metric.lower()}_{var}"

                    if col not in gdf.columns:
                        continue

                    fig = plt.figure(figsize=(10, 6))
                    ax = plt.axes(projection=ccrs.PlateCarree())
                    ax.coastlines()
                    ax.add_feature(cfeature.BORDERS, linestyle=":")

                    sc = ax.scatter(
                        gdf["lon_bin"], gdf["lat_bin"], c=gdf[col],
                        cmap=cmap, s=10, alpha=0.8, transform=ccrs.PlateCarree(), rasterized=True
                    )
                    ax.set_title(f"{metric.upper()} Map - {var} - {grp_str}")
                    plt.colorbar(sc, ax=ax, label=metric.upper())
                    plt.tight_layout()
                    self.comet.log_figure(
                        figure_name=f"map__{time_col}__{grp_str}__{metric.lower()}__{var}.png",
                        figure=fig
                    )
                    plt.close(fig)

    def run_all(self):
        """Memory-optimized main execution with streaming operations"""
        print("🚀 Starting memory-optimized analysis...")

        # Compute metrics with streaming
        metrics_day, metrics_month, metrics_spatial = self.compute_metrics()
        # metrics_day = pl.read_parquet(self.prediction_dir / "metrics_by_day.parquet")
        # metrics_month = pl.read_parquet(self.prediction_dir / "metrics_by_month.parquet")
        # metrics_spatial = pl.read_parquet(self.prediction_dir / "metrics_spatial_by_month.parquet")
        self._log_metric_trends(metrics_month, group_col="month")
        # self._log_metric_trends(metrics_day, group_col="day")
        # self._log_distributions(group_col="month", group_fmt="%Y-%m")
        # self._log_maps(metrics_spatial, time_col="month")
        self._log_maps(metrics_spatial, time_col="season")

        # Clear memory after processing
        del metrics_day, metrics_month, metrics_spatial
        if hasattr(self, '_df') and self._df is not None:
            del self._df
            self._df = None


def main(run_id: int, corrector: str):
    # Handle different directory structures
    if corrector in ["DeltaCorrector", "EDCDFCorrector", "DiffCorrector"]:
        run_to_experiment_map = {
            "DeltaCorrector": "d74ddb4c81b94c7db381b5c901ebe0af",
            "EDCDFCorrector": "8c7f2b19ade34477b884f430d2d90dcb",
            "DiffCorrector": "d39937d34aeb4544800345356e3106b8",
        }
        experiment = ExistingExperiment(
            api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
            previous_experiment=run_to_experiment_map[corrector]
        )
        subfolder = "run_delta_v1" if corrector == "DeltaCorrector" else "run_edcdf_v1" if corrector == "EDCDFCorrector" else "run_diff_v1"
        prediction_dir = f"/data/tsolis/AI_project/output/experiments/{corrector}/{subfolder}/individual_predictions"
    else:
        run_to_experiment_map = {
            0: "33030a94a54a402794061658e6d8f8d2",
            1: "0196eeb0034e48fe8cf3dcb8a762e9c8",
            2: "25c44ad4a83e4b64a42228d9ed3d959f",
            3: "4adf9763101140c584f3ae78b38794ba",
            4: "f2f2a8da31584694bcc0ac54d1ff3044"
        }
        experiment = ExistingExperiment(
            api_key="y2tkTNGtg7kP3HX9mfdy8JHaM",
            previous_experiment=run_to_experiment_map[run_id]
        )
        # For random_regressor, use the numeric run directory
        prediction_dir = f"/data/tsolis/AI_project/output/experiments/{corrector}/{run_id}"

    plotter = PredictionPlotter(
        prediction_dir=prediction_dir,
        comet_exp=experiment
    )

    plotter.run_all()

if __name__ == "__main__":
    for i in range(1):
        main(run_id=i, corrector="DiffCorrector")
