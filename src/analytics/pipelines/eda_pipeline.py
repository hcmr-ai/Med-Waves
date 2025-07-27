from pathlib import Path

import polars as pl

from src.analytics.plots.eda_plots import (
    plot_distribution,
    plot_hourly_by_weekday,
    plot_monthly_seasonality,
    plot_outliers,
    plot_smoothed_differenced,
    plot_time_series,
    plot_time_series_decomposition,
    plot_weekly_month_seasonality,
    plot_weekly_seasonality,
)
from src.analytics.utils.eda_helpers import scale_series


class EDAPipeline:
    def __init__(
        self,
        year: str,
        feature_col: str,
        main_parquet_data_path: str,
        main_output_path: str
    ):
        self.year = year
        self.feature_col = feature_col
        self.main_output_path = Path(main_output_path)
        self.main_output_path.mkdir(parents=True, exist_ok=True)
        self.agg_file = Path(f"{main_parquet_data_path}/WAVEAN{year}.parquet")

    def load_data(self):
        if self.agg_file.exists():
            print(f"Loading aggregated data: {self.agg_file}")
            ts_df = pl.read_parquet(self.agg_file)
            ts_df = self._add_time_columns(ts_df)
            self.ts_pd = (
                ts_df.select([pl.col("time"), pl.col("month"), pl.col("weekday"), pl.col("hour"), pl.col(f"{self.feature_col}_mean")])
                .to_pandas()
                .set_index("time")
            )
            print("Data loaded successfully.")
        else:
            raise FileNotFoundError(f"No data file found: {self.agg_file}")

    def _add_time_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add 'month', 'weekday', 'hour' columns for time-based grouping."""
        df = df.with_columns([
            pl.col("time").dt.month().alias("month"),
            pl.col("time").dt.weekday().alias("weekday"),
            pl.col("time").dt.hour().alias("hour"),
        ])

        return df

    def plot_all(self):
        # 1. Distribution
        plot_distribution(
            self.ts_pd[f"{self.feature_col}_mean"],
            self.feature_col,
            show=False,
            save_path=self.main_output_path / "distribution.png"
        )
        # 2. Actual time series
        plot_time_series(
            self.ts_pd[f"{self.feature_col}_mean"],
            show=False,
            save_path=self.main_output_path / "actual_time_series.png"
        )
        # 3. Outliers
        series = self.ts_pd[f"{self.feature_col}_mean"]
        outlier_mask = ((series - series.mean()).abs() / series.std()) > 3
        plot_outliers(
            self.ts_pd[f"{self.feature_col}_mean"],
            outlier_mask,
            show=False,
            save_path=self.main_output_path / "filled_outliers.png"
        )
        # 4. Smoothed & differenced
        smoothed = self.ts_pd[f"{self.feature_col}_mean"].rolling(7, center=True, min_periods=1).mean()
        differenced = self.ts_pd[f"{self.feature_col}_mean"].diff().fillna(0)
        scaled_series = scale_series(self.ts_pd[f"{self.feature_col}_mean"])
        plot_smoothed_differenced(
            self.ts_pd[f"{self.feature_col}_mean"],
            smoothed,
            differenced,
            scaled_series,
            show=False,
            save_path=self.main_output_path / "smoothed_differenced.png"
        )
        monthly = (
            self.ts_pd
            .groupby("month")[f"{self.feature_col}_mean"]
            .mean()
            .reset_index(name="monthly_mean")
            .sort_values("month")
        )
        plot_monthly_seasonality(
            monthly,
            self.feature_col,
            save_path=self.main_output_path / "monthly_seasonality.png"
        )
        del monthly

        weekly = (
            self.ts_pd
            .groupby("weekday")[f"{self.feature_col}_mean"]
            .mean()
            .reset_index(name="weekday_mean")
            .sort_values("weekday")
        )
        plot_weekly_seasonality(
            weekly,
            self.feature_col,
            save_path=self.main_output_path / "weekly_seasonality.png"
        )
        del weekly

        plot_weekly_month_seasonality(
            self.ts_pd,
            self.feature_col,
            save_path=self.main_output_path / "weekly_monthly_seasonality.png"
        )

        plot_hourly_by_weekday(
            self.ts_pd,
            f"{self.feature_col}_mean",
            save_path=self.main_output_path / "hourly_by_weekday_seasonality.png"
        )
        plot_time_series_decomposition(
            self.ts_pd[f"{self.feature_col}_mean"],
            save_path=self.main_output_path / "time_series_decomposition_add.png"
        )
        try:
            plot_time_series_decomposition(
                self.ts_pd[f"{self.feature_col}_mean"],
                model="multivariate",
                save_path=self.main_output_path / "time_series_decomposition_multi.png"
            )
        except ValueError:
            pass

    def run(self):
        self.load_data()
        self.plot_all()

if __name__ == "__main__":
    years = ["2022", "2023"]
    # feature_cols = [ "WSPD", "VHM0", "VTM02", "WDIR", "VMDR" ]
    feature_cols = ['WSPD', 'VHM0', 'VTM02', 'corrected_VHM0', 'corrected_VTM02', 'U10', 'V10',
    'wave_dir_sin', 'wave_dir_cos', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
    'sin_month', 'cos_month', 'lat_norm', 'lon_norm']
    data_origin = "augmented_with_labels"

    for year in years:
        for feature_col in feature_cols:
            main_parquet_data_path = f"/data/tsolis/AI_project/parquet/{data_origin}/hourly_mean/"
            main_output_path = f"outputs/eda/{data_origin}/{year}/{feature_col}"

            pipeline = EDAPipeline(
                year=year,
                feature_col=feature_col,
                main_parquet_data_path=main_parquet_data_path,
                main_output_path=main_output_path,
            )
            pipeline.run()
