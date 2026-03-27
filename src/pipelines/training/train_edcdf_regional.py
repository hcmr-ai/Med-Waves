"""
Train and evaluate EDCDF corrector per region (Atlantic / Mediterranean).

Loads parquet files from S3 (or local), filters by longitude-based region boundary,
fits a proper (non-incremental) EDCDF per region, evaluates on held-out test years,
and saves predictions, metrics, and models.

Usage:
    poetry run python -m src.pipelines.training.train_edcdf_regional \
        --train-years 2018 2019 2020 2021 \
        --test-years 2022 2023 \
        --regions atlantic mediterranean \
        --variables VHM0 VTM02

    # Single region, local output only:
    poetry run python -m src.pipelines.training.train_edcdf_regional \
        --train-years 2018 2019 2020 2021 \
        --test-years 2022 \
        --regions mediterranean \
        --no-s3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import fsspec
import joblib
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from tqdm import tqdm

from src.classifiers.edcdf_corrector import EDCDFCorrector

S3_BUCKET = "medwav-dev-data"
S3_PARQUET_BASE = "s3://medwav-dev-data/parquet/hourly_extra_features"
LOCAL_OUTPUT_BASE = "data/edcdf_regional"

GIBRALTAR_LON = -5.5

REGIONS = {
    "atlantic": lambda df: df.filter(pl.col("longitude") < GIBRALTAR_LON),
    "mediterranean": lambda df: df.filter(pl.col("longitude") >= GIBRALTAR_LON),
    "all": lambda df: df,
}


def subsample_spatial(df: pl.DataFrame, step: int) -> pl.DataFrame:
    """Keep every `step`-th unique latitude and longitude (regular grid thinning)."""
    if step <= 1:
        return df
    lats = sorted(df["latitude"].unique().to_list())
    lons = sorted(df["longitude"].unique().to_list())
    keep_lats = set(lats[::step])
    keep_lons = set(lons[::step])
    before = len(df)
    df = df.filter(
        pl.col("latitude").is_in(keep_lats) & pl.col("longitude").is_in(keep_lons)
    )
    print(f"  Spatial subsample (step={step}): {before:,} -> {len(df):,} rows "
          f"({len(keep_lats)} lats × {len(keep_lons)} lons)")
    return df


def list_parquet_files(years: List[int], base_path: str = S3_PARQUET_BASE) -> List[str]:
    """List parquet file paths for given years from S3 or local."""
    is_s3 = base_path.startswith("s3://")
    all_files = []

    for year in years:
        if is_s3:
            fs = fsspec.filesystem("s3")
            pattern = f"{base_path.replace('s3://', '')}/year={year}/WAVEAN{year}*.parquet"
            year_files = sorted(f"s3://{f}" for f in fs.glob(pattern))
        else:
            year_dir = Path(base_path) / f"year={year}"
            year_files = sorted(str(p) for p in year_dir.glob(f"WAVEAN{year}*.parquet"))

        all_files.extend(year_files)
        print(f"  Year {year}: {len(year_files)} files")

    return all_files


def _build_spatial_mask(
    first_file: str, columns: List[str], spatial_step: int,
) -> Tuple[set, set]:
    """Read one file to discover the unique lat/lon grid, return the keep sets."""
    df = pl.read_parquet(first_file, columns=["latitude", "longitude"])
    lats = sorted(df["latitude"].unique().to_list())
    lons = sorted(df["longitude"].unique().to_list())
    return set(lats[::spatial_step]), set(lons[::spatial_step])


def load_parquet_files(
    file_paths: List[str],
    columns: List[str],
    batch_size: int = 30,
    spatial_step: int = 1,
) -> pl.DataFrame:
    """Load parquet files, applying spatial subsampling per-batch to limit memory."""
    keep_lats, keep_lons = None, None
    if spatial_step > 1:
        keep_lats, keep_lons = _build_spatial_mask(file_paths[0], columns, spatial_step)
        print(f"  Spatial subsample (step={spatial_step}): keeping {len(keep_lats)} lats × {len(keep_lons)} lons")

    all_dfs = []
    for i in tqdm(range(0, len(file_paths), batch_size), desc="Loading parquet batches"):
        batch = file_paths[i : i + batch_size]
        batch_dfs = []
        for fp in batch:
            try:
                df = pl.read_parquet(fp, columns=columns)
                batch_dfs.append(df)
            except Exception as e:
                print(f"  Warning: skipping {fp}: {e}")
                continue
        if batch_dfs:
            chunk = pl.concat(batch_dfs)
            if keep_lats is not None:
                chunk = chunk.filter(
                    pl.col("latitude").is_in(keep_lats)
                    & pl.col("longitude").is_in(keep_lons)
                )
            all_dfs.append(chunk)

    if not all_dfs:
        raise RuntimeError("No parquet files loaded successfully")

    combined = pl.concat(all_dfs)
    print(f"  Loaded {len(combined):,} rows from {len(file_paths)} files")
    return combined


def fit_edcdf(
    df: pl.DataFrame,
    variables: List[str],
    corrected_suffix: str = "corrected_",
) -> EDCDFCorrector:
    """Fit EDCDF on the full DataFrame (non-incremental, exact CDF)."""
    df_clean = df.drop_nulls(
        subset=[v for v in variables] + [f"{corrected_suffix}{v}" for v in variables]
    )
    print(f"  Fitting on {len(df_clean):,} valid rows (dropped {len(df) - len(df_clean):,} nulls)")

    corrector = EDCDFCorrector()
    corrector.fit(df_clean, variables=variables, corrected_suffix=corrected_suffix)

    for var in variables:
        model_vals = df_clean[var].to_numpy().astype(float)
        obs_vals = df_clean[f"{corrected_suffix}{var}"].to_numpy().astype(float)
        finite_mask = np.isfinite(model_vals) & np.isfinite(obs_vals)
        n_finite = int(finite_mask.sum())
        mv, ov = model_vals[finite_mask], obs_vals[finite_mask]
        print(
            f"    {var}: model range [{mv.min():.3f}, {mv.max():.3f}], "
            f"obs range [{ov.min():.3f}, {ov.max():.3f}], "
            f"n_finite={n_finite:,} / {len(model_vals):,}"
        )

    return corrector


def evaluate_predictions(
    df: pl.DataFrame,
    variables: List[str],
    corrected_suffix: str = "corrected_",
) -> Dict[str, Dict[str, float]]:
    """Compute evaluation metrics: RMSE, MAE, bias, correlation, R2."""
    metrics = {}
    for var in variables:
        pred_col = f"predicted_{var}"
        target_col = f"{corrected_suffix}{var}"

        valid = df.drop_nulls(subset=[pred_col, target_col])
        y_pred = valid[pred_col].to_numpy().astype(float)
        y_true = valid[target_col].to_numpy().astype(float)
        y_raw = valid[var].to_numpy().astype(float)

        finite = np.isfinite(y_pred) & np.isfinite(y_true) & np.isfinite(y_raw)
        y_pred, y_true, y_raw = y_pred[finite], y_true[finite], y_raw[finite]

        if len(y_pred) == 0:
            metrics[var] = {"error": "no valid data"}
            continue

        residuals = y_pred - y_true
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        bias = np.mean(residuals)
        corr, _ = pearsonr(y_true, y_pred)

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Baseline metrics (uncorrected model vs observed)
        baseline_residuals = y_raw - y_true
        baseline_rmse = np.sqrt(np.mean(baseline_residuals**2))
        baseline_mae = np.mean(np.abs(baseline_residuals))
        baseline_bias = np.mean(baseline_residuals)

        rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse * 100) if baseline_rmse > 0 else 0.0
        mae_improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0.0

        metrics[var] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "bias": float(bias),
            "correlation": float(corr),
            "r2": float(r2),
            "baseline_rmse": float(baseline_rmse),
            "baseline_mae": float(baseline_mae),
            "baseline_bias": float(baseline_bias),
            "rmse_improvement_pct": float(rmse_improvement),
            "mae_improvement_pct": float(mae_improvement),
            "n_samples": len(y_pred),
        }

    return metrics


def compute_detailed_metrics(
    df: pl.DataFrame,
    variables: List[str],
    corrected_suffix: str = "corrected_",
    resolution: float = 0.25,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compute daily, monthly, and spatial metrics from prediction DataFrame."""
    nan_filter_cols = (
        variables
        + [f"{corrected_suffix}{v}" for v in variables]
        + [f"predicted_{v}" for v in variables]
    )
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in nan_filter_cols
    ]).drop_nulls(subset=nan_filter_cols)
    lazy = df.lazy().with_columns(pl.col("timestamp").dt.date().alias("day"))

    def metric_block(source: str, target: str, prefix: str):
        return [
            ((pl.col(source) - pl.col(target)) ** 2).mean().sqrt().alias(f"rmse_{prefix}"),
            (pl.col(source) - pl.col(target)).abs().mean().alias(f"mae_{prefix}"),
            (pl.col(source) - pl.col(target)).mean().alias(f"bias_{prefix}"),
            pl.corr(source, target).alias(f"corr_{prefix}"),
        ]

    agg_exprs = []
    for var in variables:
        # Baseline: uncorrected model vs ground truth
        agg_exprs.extend(metric_block(var, f"{corrected_suffix}{var}", f"baseline_{var}"))
        # EDCDF prediction vs ground truth
        agg_exprs.extend(metric_block(f"predicted_{var}", f"{corrected_suffix}{var}", f"edcdf_{var}"))

    metrics_day = lazy.group_by("day").agg(agg_exprs).sort("day").collect()

    metrics_month = (
        lazy.with_columns(pl.col("day").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(agg_exprs)
        .sort("month")
        .collect()
    )

    metrics_spatial = (
        lazy.with_columns([
            pl.col("day").dt.strftime("%Y-%m").alias("month"),
            pl.col("day").dt.month().alias("month_num"),
            (pl.col("latitude") / resolution).round(0) * resolution,
            (pl.col("longitude") / resolution).round(0) * resolution,
        ])
        .with_columns(
            pl.when(pl.col("month_num").is_in([12, 1, 2]))
            .then(pl.lit("winter"))
            .when(pl.col("month_num").is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("month_num").is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .when(pl.col("month_num").is_in([9, 10, 11]))
            .then(pl.lit("autumn"))
            .alias("season")
        )
        .drop("month_num")
        .rename({"latitude": "lat_bin", "longitude": "lon_bin"})
        .group_by(["month", "season", "lat_bin", "lon_bin"])
        .agg(agg_exprs)
        .collect()
    )

    return metrics_day, metrics_month, metrics_spatial


def run_region(
    region: str,
    train_files: List[str],
    test_files: List[str],
    variables: List[str],
    corrected_suffix: str,
    output_base: str,
    upload_s3: bool,
    train_years: List[int],
    test_years: List[int],
    spatial_step: int = 1,
) -> Dict:
    """Train and evaluate EDCDF for a single region."""
    print(f"\n{'=' * 80}")
    print(f"  REGION: {region.upper()}")
    print(f"{'=' * 80}")

    region_filter = REGIONS[region]
    needed_cols = (
        ["timestamp", "latitude", "longitude"]
        + variables
        + [f"{corrected_suffix}{v}" for v in variables]
    )

    # --- Load & filter training data ---
    print(f"\nLoading training data ({len(train_files)} files)...")
    train_df = load_parquet_files(train_files, columns=needed_cols, spatial_step=spatial_step)
    train_df = region_filter(train_df)
    print(f"  After region filter: {len(train_df):,} rows")

    # --- Fit EDCDF ---
    print(f"\nFitting EDCDF ({region})...")
    corrector = fit_edcdf(train_df, variables, corrected_suffix)
    del train_df

    # --- Load & filter test data ---
    print(f"\nLoading test data ({len(test_files)} files)...")
    test_df = load_parquet_files(test_files, columns=needed_cols, spatial_step=spatial_step)
    test_df = region_filter(test_df)
    print(f"  After region filter: {len(test_df):,} rows")

    # --- Predict ---
    print("\nGenerating predictions...")
    test_df = corrector.predict(test_df)

    # --- Evaluate ---
    print("\nEvaluating...")
    overall_metrics = evaluate_predictions(test_df, variables, corrected_suffix)

    print(f"\n  Overall metrics ({region}):")
    for var, m in overall_metrics.items():
        if "error" in m:
            print(f"    {var}: {m['error']}")
            continue
        print(
            f"    {var}: RMSE={m['rmse']:.4f} (baseline={m['baseline_rmse']:.4f}, "
            f"improvement={m['rmse_improvement_pct']:.1f}%), "
            f"MAE={m['mae']:.4f}, bias={m['bias']:.4f}, corr={m['correlation']:.4f}"
        )

    # --- Detailed metrics ---
    print("\nComputing detailed metrics (daily/monthly/spatial)...")
    metrics_day, metrics_month, metrics_spatial = compute_detailed_metrics(
        test_df, variables, corrected_suffix
    )

    # --- Save outputs ---
    train_years_str = "-".join(str(y) for y in sorted(train_years))
    test_years_str = "-".join(str(y) for y in sorted(test_years))
    run_id = f"edcdf_{region}_train_{train_years_str}_test_{test_years_str}"

    output_dir = Path(output_base) / region / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / f"{run_id}.joblib"
    joblib.dump(corrector, model_path)
    print(f"\n  Saved model: {model_path}")

    metrics_day.write_parquet(predictions_dir / "metrics_by_day.parquet")
    metrics_month.write_parquet(predictions_dir / "metrics_by_month.parquet")
    metrics_spatial.write_parquet(predictions_dir / "metrics_spatial_by_month.parquet")
    print(f"  Saved metrics: {predictions_dir}")

    # Save per-file predictions
    print("  Saving per-file predictions...")
    if "timestamp" in test_df.columns:
        dates = test_df.with_columns(
            pl.col("timestamp").dt.date().alias("_date")
        ).get_column("_date").unique().sort()

        for date_val in tqdm(dates, desc="Saving daily predictions"):
            day_df = test_df.filter(pl.col("timestamp").dt.date() == date_val)
            fname = f"WAVEAN{date_val.strftime('%Y%m%d')}.parquet"
            day_df.write_parquet(predictions_dir / fname)

    metadata = {
        "run_id": run_id,
        "region": region,
        "train_years": sorted(train_years),
        "test_years": sorted(test_years),
        "variables": variables,
        "corrected_suffix": corrected_suffix,
        "gibraltar_lon": GIBRALTAR_LON,
        "spatial_step": spatial_step,
        "overall_metrics": overall_metrics,
        "train_files_count": len(train_files),
        "test_files_count": len(test_files),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata: {metadata_path}")

    if upload_s3:
        import boto3

        s3 = boto3.client("s3")
        s3_prefix = f"edcdf_regional/{region}/{run_id}"

        s3.upload_file(str(model_path), S3_BUCKET, f"{s3_prefix}/models/{model_path.name}")
        s3.upload_file(str(metadata_path), S3_BUCKET, f"{s3_prefix}/metadata.json")
        print(f"  Uploaded to s3://{S3_BUCKET}/{s3_prefix}/")

    del test_df
    return {"region": region, "run_id": run_id, "metrics": overall_metrics}


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate EDCDF per region")
    parser.add_argument(
        "--train-years", type=int, nargs="+", required=True,
        help="Training years, e.g. --train-years 2018 2019 2020 2021",
    )
    parser.add_argument(
        "--test-years", type=int, nargs="+", required=True,
        help="Test years, e.g. --test-years 2022 2023",
    )
    parser.add_argument(
        "--regions", nargs="+", default=["atlantic", "mediterranean"],
        choices=["atlantic", "mediterranean", "all"],
        help="Regions to train on",
    )
    parser.add_argument(
        "--variables", nargs="+", default=["VHM0", "VTM02"],
        help="Variables to correct",
    )
    parser.add_argument("--corrected-suffix", default="corrected_")
    parser.add_argument(
        "--parquet-base", default=S3_PARQUET_BASE,
        help="Base path for parquet files (S3 or local)",
    )
    parser.add_argument(
        "--output-base", default=LOCAL_OUTPUT_BASE,
        help="Local output directory",
    )
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument(
        "--spatial-step", type=int, default=1,
        help="Keep every Nth lat/lon for spatial thinning (default: 1 = no thinning, 5 matches .pt subsampling)",
    )
    args = parser.parse_args()

    print("EDCDF Regional Training & Evaluation")
    print("=" * 80)
    print(f"  Train years:  {args.train_years}")
    print(f"  Test years:   {args.test_years}")
    print(f"  Regions:      {args.regions}")
    print(f"  Variables:    {args.variables}")
    print(f"  Data source:  {args.parquet_base}")
    print(f"  Spatial step: {args.spatial_step}")
    print()

    print("Listing training files...")
    train_files = list_parquet_files(args.train_years, args.parquet_base)
    print(f"  Total training files: {len(train_files)}")

    print("\nListing test files...")
    test_files = list_parquet_files(args.test_years, args.parquet_base)
    print(f"  Total test files: {len(test_files)}")

    results = []
    for region in args.regions:
        result = run_region(
            region=region,
            train_files=train_files,
            test_files=test_files,
            variables=args.variables,
            corrected_suffix=args.corrected_suffix,
            output_base=args.output_base,
            upload_s3=not args.no_s3,
            train_years=args.train_years,
            test_years=args.test_years,
            spatial_step=args.spatial_step,
        )
        results.append(result)

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for r in results:
        print(f"\n  {r['region'].upper()} ({r['run_id']}):")
        for var, m in r["metrics"].items():
            if "error" in m:
                print(f"    {var}: {m['error']}")
                continue
            print(
                f"    {var}: RMSE={m['rmse']:.4f} (baseline={m['baseline_rmse']:.4f}, "
                f"{m['rmse_improvement_pct']:+.1f}%), "
                f"MAE={m['mae']:.4f}, R2={m['r2']:.4f}"
            )


if __name__ == "__main__":
    main()
