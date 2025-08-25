import sys
from pathlib import Path
import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.random_classifier_plotter import PredictionPlotter


def main():
    # Configuration
    base_data_dir = "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly"  # Use existing data
    output_dir = "/data/tsolis/AI_project/output/experiments/DiffCorrector/run_diff_v1"
    
    # Use the same date patterns as in training
    patterns = ["WAVEAN2023"]
    print(f"📅 Using date patterns: {patterns}")
    
    print("🚀 Starting Diff Corrector Evaluation")
    print("=" * 50)
    print(f"📂 Data directory: {base_data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    class DummyExperiment:
        def log_figure(self, figure_name, figure):
            import matplotlib.pyplot as plt
            figure.savefig(output_path / figure_name, dpi=300, bbox_inches='tight')
            print(f"📊 Saved figure: {figure_name}")
    
    dummy_experiment = DummyExperiment()
    
    class FilteredPredictionPlotter(PredictionPlotter):
        def __init__(self, prediction_dir, comet_exp, patterns):
            super().__init__(prediction_dir, comet_exp)
            self.patterns = patterns
        
        def _load_all_predictions(self) -> pl.DataFrame:
            """Load predictions with date pattern filtering"""
            print(f"📂 Loading daily predictions from {self.prediction_dir} with patterns: {self.patterns}")
            
            all_files = []
            for pattern in self.patterns:
                pattern_files = sorted(self.prediction_dir.glob(f"*{pattern}*.parquet"))
                all_files.extend(pattern_files)
            
            all_files = sorted(set(all_files))  # Remove duplicates
            print(f"📁 Found {len(all_files)} files matching patterns")
            
            lazy_dfs = []
            for file in all_files:
                lazy_df = pl.scan_parquet(file).with_columns([
                    pl.col("timestamp").dt.date().alias("day")
                ])
                lazy_dfs.append(lazy_df)
            
            return pl.concat(lazy_dfs, how="vertical").collect(engine="streaming")
        
        def compute_metrics(self, resolution: float = 0.25):
            """Memory-optimized metrics computation with streaming and reduced copies"""
            print("📊 Computing metrics per day, month, and spatial bins...")
            
            # Filter files based on patterns
            all_files = []
            for pattern in self.patterns:
                pattern_files = sorted(self.prediction_dir.glob(f"*{pattern}*.parquet"))
                all_files.extend(pattern_files)
            
            all_files = sorted(set(all_files))  # Remove duplicates
            if not all_files:
                raise FileNotFoundError(f"No files matching patterns {self.patterns} found in {self.prediction_dir}")
            
            print(f"📁 Processing {len(all_files)} files matching patterns")
            
            lazy_dfs = []
            for file in all_files:
                lazy_df = pl.scan_parquet(file).with_columns([
                    pl.col("timestamp").dt.date().alias("day")
                ]).filter(
                    pl.all_horizontal([
                        pl.col("VHM0").is_not_null(),
                        pl.col("VTM02").is_not_null(),
                        pl.col("corrected_VHM0").is_not_null(),
                        pl.col("corrected_VTM02").is_not_null(),
                        pl.col("latitude").is_not_null(),
                        pl.col("longitude").is_not_null()
                    ])
                )
                lazy_dfs.append(lazy_df)
            
            lazy_df = pl.concat(lazy_dfs, how="vertical")

            def metric_block(uncorrected: str, corrected: str):
                base = uncorrected  # Use the variable name directly
                return (
                    ((pl.col(uncorrected) - pl.col(corrected)) ** 2).mean().sqrt().alias(f"rmse_{base}"),
                    (pl.col(uncorrected) - pl.col(corrected)).abs().mean().alias(f"mae_{base}"),
                    (pl.col(uncorrected) - pl.col(corrected)).mean().alias(f"diff_{base}"),
                    (pl.col(uncorrected).mean() - pl.col(corrected).mean()).alias(f"bias_{base}"),
                    ((pl.col(uncorrected) - pl.col(uncorrected).mean()) ** 2).mean().alias(f"var_{base}"),
                    pl.corr(uncorrected, corrected).alias(f"corr_{base}"),
                )

            print("Computing daily metrics...")
            metrics_day = lazy_df.group_by("day").agg([
                *metric_block("VHM0", "corrected_VHM0"),
                *metric_block("VTM02", "corrected_VTM02")
            ]).collect(engine="streaming")

            print("Computing monthly metrics...")
            metrics_month = lazy_df.with_columns([
                pl.col("day").dt.strftime("%Y-%m").alias("month")
            ]).group_by("month").agg([
                *metric_block("VHM0", "corrected_VHM0"),
                *metric_block("VTM02", "corrected_VTM02")
            ]).collect(engine="streaming")

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
                *metric_block("VHM0", "corrected_VHM0"),
                *metric_block("VTM02", "corrected_VTM02")
            ]).collect(engine="streaming")

            return metrics_day, metrics_month, metrics_spatial
    
    plotter = FilteredPredictionPlotter(base_data_dir, dummy_experiment, patterns)
    
    try:
        print("📊 Computing Diff corrector metrics (uncorrected vs corrected)...")
        metrics_day, metrics_month, metrics_spatial = plotter.compute_metrics(resolution=0.25)
        
        predictions_dir = output_path / "individual_predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        metrics_day.write_parquet(predictions_dir / "metrics_by_day.parquet")
        metrics_month.write_parquet(predictions_dir / "metrics_by_month.parquet")
        metrics_spatial.write_parquet(predictions_dir / "metrics_spatial_by_month.parquet")
        
        print(f"✅ Saved metrics files:")
        print(f"   - metrics_by_day.parquet ({len(metrics_day)} days)")
        print(f"   - metrics_by_month.parquet ({len(metrics_month)} months)")
        print(f"   - metrics_spatial_by_month.parquet ({len(metrics_spatial)} spatial bins)")
        
        print("\n📈 Diff Corrector Summary Statistics:")
        print("=" * 50)
        
        for var in ["VHM0", "VTM02"]:
            print(f"\n{var}:")
            rmse_avg = metrics_day[f"rmse_{var}"].mean()
            mae_avg = metrics_day[f"mae_{var}"].mean()
            bias_avg = metrics_day[f"bias_{var}"].mean()
            corr_avg = metrics_day[f"corr_{var}"].mean()
            
            print(f"  RMSE: {rmse_avg:.4f}")
            print(f"  MAE:  {mae_avg:.4f}")
            print(f"  Bias: {bias_avg:.4f}")
            print(f"  Corr: {corr_avg:.4f}")
        
        print(f"\n✅ Diff corrector evaluation complete!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
