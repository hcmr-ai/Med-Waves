from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


class CorrectorComparisonPlotter:
    def __init__(self, base_dir: str = "/data/tsolis/AI_project/output/experiments"):
        self.base_dir = Path(base_dir)
        self.correctors = ["DeltaCorrector", "random_regressor", "DiffCorrector", "EDCDFCorrector"]
        self.metrics = ["rmse", "mae", "diff", "bias", "corr"]
        self.variables = ["VHM0", "VTM02"]

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_all_metrics(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """Load metrics from all correctors and runs."""
        all_metrics = {}

        for corrector in self.correctors:
            if corrector == "random_regressor":
                # For random_regressor, look for numeric run directories directly under corrector
                corrector_dir = self.base_dir / corrector
                if not corrector_dir.exists():
                    print(f"⚠️ Directory not found: {corrector_dir}")
                    continue

                all_metrics[corrector] = {}

                # Find all numeric run directories
                run_dirs = []
                for d in corrector_dir.iterdir():
                    if d.is_dir() and d.name.isdigit():
                        run_dirs.append(d)

                for run_dir in run_dirs:
                    run_id = run_dir.name

                    # Load metrics files directly from run directory
                    metrics_files = {
                        "daily": run_dir / "metrics_by_day.parquet",
                        "monthly": run_dir / "metrics_by_month.parquet",
                        "spatial": run_dir / "metrics_spatial_by_month.parquet"
                    }

                    run_metrics = {}
                    for period, file_path in metrics_files.items():
                        if file_path.exists():
                            try:
                                df = pl.read_parquet(file_path)
                                run_metrics[period] = df
                                print(f"✅ Loaded {period} metrics for {corrector} run {run_id}")
                            except Exception as e:
                                print(f"❌ Error loading {file_path}: {e}")
                        else:
                            print(f"⚠️ File not found: {file_path}")

                    if run_metrics:
                        all_metrics[corrector][run_id] = run_metrics

            else:
                # For DeltaCorrector, EDCDFCorrector, and DiffCorrector
                if corrector == "DeltaCorrector":
                    subfolder = "run_delta_v1"
                elif corrector == "EDCDFCorrector":
                    subfolder = "run_edcdf_v1"
                elif corrector == "DiffCorrector":
                    subfolder = "run_diff_v1"
                else:
                    continue

                corrector_dir = self.base_dir / corrector / subfolder
                if not corrector_dir.exists():
                    print(f"⚠️ Directory not found: {corrector_dir}")
                    continue

                all_metrics[corrector] = {}

                # For these correctors, metrics are in individual_predictions directory
                metrics_dir = corrector_dir / "individual_predictions"
                if not metrics_dir.exists():
                    print(f"⚠️ Metrics directory not found: {metrics_dir}")
                    continue

                # Load metrics files directly from individual_predictions
                metrics_files = {
                    "daily": metrics_dir / "metrics_by_day.parquet",
                    "monthly": metrics_dir / "metrics_by_month.parquet",
                    "spatial": metrics_dir / "metrics_spatial_by_month.parquet"
                }

                run_metrics = {}
                for period, file_path in metrics_files.items():
                    if file_path.exists():
                        try:
                            df = pl.read_parquet(file_path)
                            run_metrics[period] = df
                            print(f"✅ Loaded {period} metrics for {corrector}")
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
                    else:
                        print(f"⚠️ File not found: {file_path}")

                if run_metrics:
                    # Use subfolder name as run_id for these correctors
                    all_metrics[corrector][subfolder] = run_metrics

        return all_metrics

    def aggregate_metrics_across_runs(self, all_metrics: Dict[str, Dict[str, pl.DataFrame]]) -> Dict[str, pl.DataFrame]:
        """Aggregate metrics across all runs for each corrector and time period."""
        aggregated = {}

        for corrector, runs in all_metrics.items():
            for run_id, periods in runs.items():
                for period, df in periods.items():
                    key = f"{corrector}_{period}"

                    if key not in aggregated:
                        aggregated[key] = []

                    # Add corrector and run_id columns for identification
                    df_with_meta = df.with_columns([
                        pl.lit(corrector).alias("corrector"),
                        pl.lit(run_id).alias("run_id")
                    ])
                    aggregated[key].append(df_with_meta)

        # Concatenate a§ll DataFrames for each key
        for key in list(aggregated.keys()):
            if aggregated[key]:
                aggregated[key] = pl.concat(aggregated[key], how="vertical")
            else:
                del aggregated[key]

        return aggregated

    def _compute_weekly_metrics(self, daily_metrics: pl.DataFrame) -> pl.DataFrame:
        """Compute weekly metrics from daily metrics."""
        if "day" not in daily_metrics.columns:
            return daily_metrics

        # Convert day to datetime if it's not already
        df = daily_metrics.with_columns([
            pl.col("day").cast(pl.Date)
        ])

        # Add week column
        df = df.with_columns([
            pl.col("day").dt.strftime("%Y-%W").alias("week")
        ])

        # Get metric columns (excluding day, week, corrector, run_id)
        metric_cols = [col for col in df.columns if col not in ["day", "week", "corrector", "run_id"]]

        # Group by week and compute mean for each metric
        weekly_metrics = df.group_by("week").agg([
            pl.col(col).mean().alias(col) for col in metric_cols
        ]).sort("week")

        return weekly_metrics

    def create_comparison_plots(self, aggregated_metrics: Dict[str, pl.DataFrame], output_dir: str = "comparison_plots"):
        """Create comparison line plots for all metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Process each time period
        for period in ["daily", "weekly", "monthly"]:
            # Create period-specific subfolder
            period_path = output_path / period
            period_path.mkdir(exist_ok=True)

            if period == "weekly":
                # Compute weekly metrics on the fly from daily data
                daily_metrics = {k: v for k, v in aggregated_metrics.items() if "daily" in k}
                if not daily_metrics:
                    print("⚠️ No daily metrics found to compute weekly metrics")
                    continue

                period_metrics = {}
                for corrector_name, daily_df in daily_metrics.items():
                    corrector = corrector_name.split("_")[0]
                    weekly_df = self._compute_weekly_metrics(daily_df)
                    period_metrics[f"{corrector}_weekly"] = weekly_df

                print(f"📊 Computed weekly metrics for {len(period_metrics)} correctors")
            else:
                period_metrics = {k: v for k, v in aggregated_metrics.items() if period in k}

            if not period_metrics:
                print(f"⚠️ No {period} metrics found")
                continue

            # Create plots for each metric and variable combination
            for metric in self.metrics:
                for variable in self.variables:
                    self._create_single_comparison_plot(
                        period_metrics, metric, variable, period, period_path
                    )

    def create_random_regressor_runs_plots(self, all_metrics: Dict[str, Dict[str, pl.DataFrame]], output_dir: str):
        """Create dedicated plots showing individual runs of random_regressor."""
        output_path = Path(output_dir) / "random_delta_sampling_runs"
        output_path.mkdir(exist_ok=True)

        print("📊 Creating random_regressor individual runs plots...")

        # Get random_regressor data
        if "random_regressor" not in all_metrics:
            print("⚠️ No random_regressor data found")
            return

        random_runs = all_metrics["random_regressor"]

        # Process each time period
        for period in ["daily", "weekly", "monthly"]:
            # Create period-specific subfolder
            period_path = output_path / period
            period_path.mkdir(exist_ok=True)

            if period == "weekly":
                # Compute weekly metrics on the fly from daily data
                period_runs = {}
                for run_id, periods_data in random_runs.items():
                    if "daily" in periods_data:
                        daily_df = periods_data["daily"]
                        weekly_df = self._compute_weekly_metrics(daily_df)
                        period_runs[run_id] = weekly_df

                if not period_runs:
                    print("⚠️ No daily data found to compute weekly metrics for random_regressor runs")
                    continue

                print(f"📊 Computed weekly metrics for {len(period_runs)} random_regressor runs")
            else:
                period_runs = {}
                # Collect all runs for this period
                for run_id, periods_data in random_runs.items():
                    if period in periods_data:
                        period_runs[run_id] = periods_data[period]

            if not period_runs:
                print(f"⚠️ No {period} data found for random_regressor runs")
                continue

            # Create plots for each metric and variable combination
            for metric in self.metrics:
                for variable in self.variables:
                    self._create_random_regressor_runs_plot(
                        period_runs, metric, variable, period, period_path
                    )

    def _create_random_regressor_runs_plot(
        self,
        period_runs: Dict[str, pl.DataFrame],
        metric: str,
        variable: str,
        period: str,
        output_path: Path
    ):
        """Create a plot showing individual random_regressor runs."""

        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Colors for different runs
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Get time column
        if period == "daily":
            time_col = "day"
        elif period == "weekly":
            time_col = "week"
        else:  # monthly
            time_col = "month"

        # Process each run in numerical order
        sorted_runs = sorted(period_runs.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        print(f"📊 Processing {len(sorted_runs)} runs: {[run_id for run_id, _ in sorted_runs]}")

        # Debug: Check what data each run contains
        for run_id, df in sorted_runs:
            print(f"   Run {run_id}: DataFrame shape {df.shape}, columns: {list(df.columns)}")
            if 'day' in df.columns:
                print(f"     Date range: {df['day'].min()} to {df['day'].max()}")

        for i, (run_id, df) in enumerate(sorted_runs):
            color = colors[i % len(colors)]
            print(f"   Plotting run {run_id} with color {color}")

            # Get the metric column
            metric_col = f"{metric}_{variable}"
            if metric_col not in df.columns:
                print(f"⚠️ Column {metric_col} not found in run {run_id}")
                continue

            if time_col not in df.columns:
                print(f"⚠️ Time column {time_col} not found in run {run_id}")
                continue

            # Sort by time
            sorted_df = df.select([time_col, metric_col]).sort(time_col)

            # Convert time to proper format for plotting
            if period == "daily":
                x_values = [pd.to_datetime(d) for d in sorted_df[time_col].to_list()]
            elif period == "weekly":
                # Convert week format (YYYY-WW) to datetime
                x_values = []
                for week_str in sorted_df[time_col].to_list():
                    try:
                        if '-' in week_str and len(week_str.split('-')) == 2:
                            year, week = week_str.split('-')
                            # Create datetime from year and week number (ISO week format)
                            x_values.append(pd.to_datetime(f"{year}-W{week.zfill(2)}-1", format="%Y-W%W-%w"))
                        else:
                            # Fallback: try to parse as regular date
                            x_values.append(pd.to_datetime(week_str))
                    except Exception as e:
                        print(f"⚠️ Warning: Could not parse week string '{week_str}': {e}")
                        # Use original string as fallback
                        x_values.append(week_str)
            else:  # monthly
                x_values = sorted_df[time_col].to_list()

            y_values = sorted_df[metric_col].to_numpy()

            # Plot the line for this run
            ax.plot(x_values, y_values,
                   label=f"Run {run_id}",
                   color=color,
                   linewidth=2,
                   marker='o',
                   markersize=4,
                   alpha=0.8)

            # Debug: Print some statistics for this run
            print(f"   Run {run_id}: mean={y_values.mean():.4f}, std={y_values.std():.4f}, min={y_values.min():.4f}, max={y_values.max():.4f}")

            # Check if this run is identical to the previous one
            if i > 0:
                prev_y_values = sorted_runs[i-1][1].select([time_col, metric_col]).sort(time_col)[metric_col].to_numpy()
                if len(y_values) == len(prev_y_values):
                    diff = np.abs(y_values - prev_y_values).max()
                    print(f"     Max difference from previous run: {diff:.8f}")
                    if diff < 1e-6:
                        print(f"     ⚠️ WARNING: Run {run_id} is essentially identical to previous run!")

        # Customize the plot
        ax.set_title(f"Random Regressor Individual Runs - {metric.upper()} {variable} ({period.capitalize()})",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.legend(fontsize=10, title="Run ID")
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        if period == "monthly":
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout and save
        plt.tight_layout()
        filename = f"random_regressor_runs_{metric}_{variable}_{period}.png"
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved: {filename}")

    def _create_single_comparison_plot(
        self,
        period_metrics: Dict[str, pl.DataFrame],
        metric: str,
        variable: str,
        period: str,
        output_path: Path
    ):
        """Create a single comparison plot for one metric-variable combination."""

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Colors for different correctors
        colors = {
            "DeltaCorrector": "#1f77b4",
            "EDCDFCorrector": "#ff7f0e",
            "random_regressor": "#2ca02c",
            "DiffCorrector": "#d62728"  # Red for baseline
        }

        # Process each corrector
        for corrector_name, df in period_metrics.items():
            corrector = corrector_name.split("_")[0]
            color = colors.get(corrector, "#9467bd")  # Purple as default

            # Get the metric column
            metric_col = f"{metric}_{variable}"
            if metric_col not in df.columns:
                print(f"⚠️ Column {metric_col} not found in {corrector_name}")
                continue

            # Get time column
            if period == "daily":
                time_col = "day"
            elif period == "weekly":
                time_col = "week"
            else:  # monthly
                time_col = "month"
            if time_col not in df.columns:
                print(f"⚠️ Time column {time_col} not found in {corrector_name}")
                continue

            # Group by time and compute mean across runs
            if "run_id" in df.columns:
                # Multiple runs - compute mean
                grouped = df.group_by(time_col).agg([
                    pl.col(metric_col).mean().alias(metric_col),
                    pl.col(metric_col).std().alias(f"{metric_col}_std")
                ]).sort(time_col)
            else:
                # Single run
                grouped = df.select([time_col, metric_col]).sort(time_col)

            # Convert time to proper format for plotting
            if period == "daily":
                x_values = [pd.to_datetime(d) for d in grouped[time_col].to_list()]
            elif period == "weekly":
                # Convert week format (YYYY-WW) to datetime
                x_values = []
                for week_str in grouped[time_col].to_list():
                    try:
                        if '-' in week_str and len(week_str.split('-')) == 2:
                            year, week = week_str.split('-')
                            # Create datetime from year and week number (ISO week format)
                            x_values.append(pd.to_datetime(f"{year}-W{week.zfill(2)}-1", format="%Y-W%W-%w"))
                        else:
                            # Fallback: try to parse as regular date
                            x_values.append(pd.to_datetime(week_str))
                    except Exception as e:
                        print(f"⚠️ Warning: Could not parse week string '{week_str}': {e}")
                        # Use original string as fallback
                        x_values.append(week_str)
            else:  # monthly
                x_values = grouped[time_col].to_list()

            y_values = grouped[metric_col].to_numpy()

            # Plot the line
            ax.plot(x_values, y_values,
                   label=corrector,
                   color=color,
                   linewidth=2,
                   marker='o',
                   markersize=4)

            # Add error bars if we have multiple runs
            if "run_id" in df.columns and f"{metric_col}_std" in grouped.columns:
                std_values = grouped[f"{metric_col}_std"].to_numpy()
                ax.fill_between(x_values,
                               y_values - std_values,
                               y_values + std_values,
                               alpha=0.2,
                               color=color)

        # Customize the plot
        ax.set_title(f"{metric.upper()} Comparison - {variable} ({period.capitalize()})",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        if period == "monthly":
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout and save
        plt.tight_layout()
        filename = f"{metric}_{variable}_{period}_comparison.png"
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved: {filename}")

    def create_summary_table(self, aggregated_metrics: Dict[str, pl.DataFrame], output_dir: str = "comparison_plots"):
        """Create a summary table with overall performance metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        summary_data = []

        for corrector_name, df in aggregated_metrics.items():
            corrector = corrector_name.split("_")[0]
            period = corrector_name.split("_")[1]

            for metric in self.metrics:
                for variable in self.variables:
                    metric_col = f"{metric}_{variable}"

                    if metric_col in df.columns:
                        values = df[metric_col].to_numpy()

                        summary_data.append({
                            "Corrector": corrector,
                            "Period": period,
                            "Metric": metric.upper(),
                            "Variable": variable,
                            "Mean": np.mean(values),
                            "Std": np.std(values),
                            "Min": np.min(values),
                            "Max": np.max(values),
                            "Count": len(values)
                        })

        if summary_data:
            summary_df = pl.DataFrame(summary_data)

            # Save as CSV
            summary_df.write_csv(output_path / "performance_summary.csv")

            # Create a formatted table for display
            print("\n" + "="*80)
            print("PERFORMANCE SUMMARY")
            print("="*80)

            # Group by metric and variable for better readability
            for metric in self.metrics:
                for variable in self.variables:
                    subset = summary_df.filter(
                        (pl.col("Metric") == metric.upper()) &
                        (pl.col("Variable") == variable)
                    )

                    if len(subset) > 0:
                        print(f"\n{metric.upper()} - {variable}:")
                        print("-" * 50)

                        for row in subset.iter_rows(named=True):
                            print(f"{row['Corrector']:15} | {row['Period']:8} | "
                                  f"Mean: {row['Mean']:8.4f} ± {row['Std']:6.4f} "
                                  f"[{row['Min']:6.4f}, {row['Max']:6.4f}]")

        return summary_df if summary_data else None

    def run_comparison_analysis(self, output_dir: str = "comparison_plots"):
        """Run the complete comparison analysis."""
        print("🚀 Starting Corrector Comparison Analysis")
        print("=" * 50)

        # Load all metrics
        print("📂 Loading metrics from all correctors...")
        all_metrics = self.load_all_metrics()

        if not all_metrics:
            print("❌ No metrics found for any corrector")
            return

        # Aggregate metrics across runs
        print("📊 Aggregating metrics across runs...")
        aggregated_metrics = self.aggregate_metrics_across_runs(all_metrics)

        if not aggregated_metrics:
            print("❌ No aggregated metrics available")
            return

        # Create comparison plots
        print("📈 Creating comparison plots...")
        self.create_comparison_plots(aggregated_metrics, output_dir)

        # Create random_regressor individual runs plots
        print("📊 Creating random_regressor individual runs plots...")
        self.create_random_regressor_runs_plots(all_metrics, output_dir)

        # Create summary table
        print("📋 Creating summary table...")
        self.create_summary_table(aggregated_metrics, output_dir)

        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


def main():
    """Main function to run the comparison analysis."""
    plotter = CorrectorComparisonPlotter()
    plotter.run_comparison_analysis()


if __name__ == "__main__":
    main()
