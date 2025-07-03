import polars as pl
import os
from pathlib import Path
from typing import List

from src.analytics.utils.eda_helpers import (
    flag_missing, flag_outliers_zscore, smooth_rolling, difference_series, scale_series
)
from src.analytics.plots.eda_plots import (
    plot_time_series, plot_outliers, plot_smoothed_differenced, plot_distribution
)

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
            self.ts_pd = (
                ts_df.select([pl.col("time"), pl.col(f"{self.feature_col}_mean")])
                .to_pandas()
                .set_index("time")[f"{self.feature_col}_mean"]
            )
            print("Data loaded successfully.")
        else:
            raise FileNotFoundError(f"No data file found: {self.agg_file}")

    def plot_all(self):
        # 1. Distribution
        plot_distribution(
            self.ts_pd,
            self.feature_col,
            show=False,
            save_path=self.main_output_path / "distribution.png"
        )
        # 2. Actual time series
        plot_time_series(
            self.ts_pd,
            show=False,
            save_path=self.main_output_path / "actual_time_series.png"
        )
        # 3. Outliers
        outlier_mask = ((self.ts_pd - self.ts_pd.mean()).abs() / self.ts_pd.std()) > 3
        plot_outliers(
            self.ts_pd,
            outlier_mask,
            show=False,
            save_path=self.main_output_path / "filled_outliers.png"
        )
        # 4. Smoothed & differenced
        smoothed = self.ts_pd.rolling(7, center=True, min_periods=1).mean()
        differenced = self.ts_pd.diff().fillna(0)
        scaled_series = scale_series(self.ts_pd)
        plot_smoothed_differenced(
            self.ts_pd,
            smoothed,
            differenced,
            scaled_series,
            show=False,
            save_path=self.main_output_path / "smoothed_differenced.png"
        )

    def run(self):
        self.load_data()
        self.plot_all()

if __name__ == "__main__":
    years = ["2021", "2022", "2023"]
    feature_cols = [ "WSPD", "VHM0", "VTM02", "WDIR", "VMDR" ]
    data_origin = "without_reduced"

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
