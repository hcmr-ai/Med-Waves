from pathlib import Path

import polars as pl


class YearlySpatialAggregator:
    def __init__(
        self,
        parquet_files_path: str,
        parquet_files_output_path: str,
        year: str,
        dry_run: bool = False,
        exclude_cols: set | None = None
    ):
        self.year = year
        if exclude_cols is None:
            self.exclude_cols = {"time", "latitude", "longitude"}
        self.parquet_files_path = Path(parquet_files_path)
        self.parquet_files_output_path = Path(parquet_files_output_path)
        self.dry_run = dry_run

        self.agg_dfs = []

    def monthly_files(self):
        all_files = sorted(self.parquet_files_path.glob(f"WAVEAN{self.year}*.parquet"))
        if self.dry_run:
            print("Number of total daily files: ", len(all_files))
        months = {str(m).zfill(2): [] for m in range(1, 13)}
        for f in all_files:
            fname = f.name.split(".parquet")[0]
            month = fname[-4:-2]
            if month in months:
                months[month].append(f)
        return [months[m] for m in sorted(months.keys())]

    def aggregate_month(self, files):
        if not files:
            return None
        print(f"Aggregating month with {len(files)} files")
        df = pl.scan_parquet([str(f) for f in files])
        ts_schema = df.collect_schema()
        feature_cols = [col for col in ts_schema if col not in self.exclude_cols]
        agg_exprs = []
        for f in feature_cols:
            agg_exprs += [
                pl.col(f).mean().alias(f"{f}_mean"),
                pl.col(f).std().alias(f"{f}_std"),
                (pl.col(f).is_null().sum() / pl.len()).alias(f"{f}_pct_missing"),
            ]
        df = df.with_columns([
            pl.col("time").dt.year().alias("year"),
            pl.col("time").dt.month().alias("month"),
        ])
        agg_df = (
            df.group_by(["latitude", "longitude", "year", "month"])
            .agg(agg_exprs)
            .collect()
        )
        return agg_df

    def aggregate_year(self):
        self.agg_dfs = []
        if self.dry_run:
            monthly_files = self.monthly_files()
            print("Number of total monthly-files", len(monthly_files))
            print({i: len(files) for i, files in enumerate(monthly_files)})
        else:
            for files in self.monthly_files():
                agg = self.aggregate_month(files)
                if agg is not None:
                    self.agg_dfs.append(agg)
            if not self.agg_dfs:
                print("No monthly data aggregated!")
                return None

            print("Concatenating monthly data for annual aggregation...")
            year_df = pl.concat(self.agg_dfs)
            save_path = self.parquet_files_output_path / f"spatial_stats_{self.year}.parquet"
            year_df.write_parquet(save_path)
            print(f"Saved yearly spatial stats to {save_path}")
            return year_df


if __name__ == "__main__":
    year = "2021"
    data_origin = "without_reduced"

    parquet_files_path = f"/data/tsolis/AI_project/parquet/{data_origin}/hourly"
    parquet_files_output_path = f"/data/tsolis/AI_project/parquet/{data_origin}/monthly_spatial_stats"

    aggregator = YearlySpatialAggregator(
        parquet_files_path=parquet_files_path,
        parquet_files_output_path=parquet_files_output_path,
        year=year,
        dry_run=False,
    )
    year_df = aggregator.aggregate_year()
