import polars as pl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path

from src.analytics.plots.spatial_plots import plot_missing_spatial_heatmap, plot_spatial_feature_heatmap

SEASON_MAP = {
    12: "WINTER", 1: "WINTER", 2: "WINTER",
    3: "AUTUMN", 4: "AUTUMN", 5: "AUTUMN",
    6: "SUMMER", 7: "SUMMER", 8: "SUMMER",
    9: "SPRING", 10: "SPRING", 11: "SPRING",
}

def add_season_columns(df):
    """Add 'season' and 'season_year' columns for climatological analysis."""

    df = df.with_columns([
        pl.col("month").map_elements(lambda m: SEASON_MAP.get(m), return_dtype=pl.String).alias("season"),
        pl.col("year").alias("season_year"),  # Calendar year, no shifting!
    ])
    return df

class SpatialEDAPipeline:
    def __init__(
        self,
        parquet_files_path: str,
        exclude_cols: set = {"time", "latitude", "longitude"}
    ):
        self.parquet_files = parquet_files_path

    def load_data(self):
        print(f"Loading data: {self.parquet_files}")
        self.df = pl.read_parquet(self.parquet_files)
        print(f"Loaded Data Successfully.")

    def plot_missing_heatmap(self, df, label=""):
        """Plot percentage of missing values for each grid cell."""
        print("Computing missing data statistics...")
        
        miss_pd = df.to_pandas().pivot(index="latitude", columns="longitude", values=f"{self.feature_col}_pct_missing")
        out_path = self.output_dir / f"missing_heatmap{('_' + label) if label else ''}.png"
        plot_missing_spatial_heatmap(miss_pd, out_path)
        
        print(f"Saved missing heatmap to {out_path}")

    def plot_mean_std_heatmaps(self, df, label=""):
        """Plot mean and std feature as spatial scatter heatmaps."""
        # Plot mean
        stats = df.to_pandas()
        out_mean = self.output_dir / f"{self.feature_col.lower()}_mean_map{('_' + label) if label else ''}.png"
        plot_spatial_feature_heatmap(
            df=stats,
            feature_col=f"{self.feature_col}_mean",
            output_dir=out_mean,
            stat_name="mean",
            cmap="viridis",
        )
        print(f"Saved mean map to {out_mean}")
        
        # Plot std
        out_std = self.output_dir / f"{self.feature_col.lower()}_std_map{('_' + label) if label else ''}.png"
        plot_spatial_feature_heatmap(
            df=stats,
            feature_col=f"{self.feature_col}_std",
            output_dir=out_std,
            stat_name="std",
            cmap="viridis",
        )
        print(f"Saved std map to {out_std}")

    def aggregate_annual(self):
        """Aggregate all months in a year per grid cell (mean, std, missing%)."""
        agg = (
            self.df.group_by(["latitude", "longitude"])
            .agg([
                pl.col(f"{self.feature_col}_mean").mean().alias(f"{self.feature_col}_mean"),
                pl.col(f"{self.feature_col}_std").mean().alias(f"{self.feature_col}_std"),
                pl.col(f"{self.feature_col}_pct_missing").mean().alias(f"{self.feature_col}_pct_missing"),
            ])
        )
        return agg
        
    def aggregate_seasonal(self):
        """Return a Polars DataFrame of seasonal averages per grid point."""
        df = add_season_columns(self.df)
        # For each (lat, lon, season), mean over all years (seasonal climatology)
        agg = (
            df.group_by(["latitude", "longitude", "season"])
            .agg([
                pl.col(f"{self.feature_col}_mean").mean(),
                pl.col(f"{self.feature_col}_std").mean(),
                pl.col(f"{self.feature_col}_pct_missing").mean(),
            ])
        )
        return agg

    def plot_seasonal_means(self, agg):
        """Plot seasonal mean maps for each season."""
        seasons = agg["season"].unique().to_list()
        for season in seasons:
            df_s = agg.filter(pl.col("season") == season)
            label = f"{season}_mean"
            self.plot_mean_std_heatmaps(df=df_s, label=label)
            self.plot_missing_heatmap(df=df_s, label=label)

    def run(self, feature_col: str, output_dir: str):
        self.feature_col = feature_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Aggregate per grid for the annual plots
        annual_agg = self.aggregate_annual()
        self.plot_missing_heatmap(df=annual_agg)
        self.plot_mean_std_heatmaps(df=annual_agg)
        del annual_agg
        # --- Seasonal
        seasonal_agg = self.aggregate_seasonal()
        self.plot_seasonal_means(seasonal_agg)
        del seasonal_agg


if __name__ == "__main__":
    years = ["2021"]
    feature_cols = [ "WSPD", "VHM0", "VTM02", "WDIR", "VMDR" ]
    data_origin = "without_reduced"

    for year in years:
        parquet_files_path = f"/data/tsolis/AI_project/parquet/{data_origin}/monthly_spatial_stats/spatial_stats_{year}.parquet"

        pipeline = SpatialEDAPipeline(
            parquet_files_path=parquet_files_path,
        )
        pipeline.load_data()

        for feature_col in feature_cols:
            output_dir = f"outputs/eda/spatial_heatmaps/{data_origin}/{year}/{feature_col}"
            pipeline.run(feature_col, output_dir)
