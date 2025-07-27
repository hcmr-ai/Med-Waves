from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def sample_file(file_path: Path, feature_cols: list[str], frac: float = 0.01) -> pl.DataFrame:
    df = pl.read_parquet(file_path).select(feature_cols)
    n_rows = int(len(df) * frac)
    if n_rows == 0:
        return pl.DataFrame(schema={col: df[col].dtype for col in feature_cols})  # Return empty frame
    return df.sample(n=n_rows)

def collect_samples(parquet_dir: Path, feature_cols: list[str], glob_pattern: str, frac: float = 0.01) -> pl.DataFrame:
    all_samples = []
    files = sorted(parquet_dir.glob(glob_pattern))

    for file in files:
        print(f"📦 Sampling from: {file.name}")
        sample = sample_file(file, feature_cols, frac)
        all_samples.append(sample)

    return pl.concat(all_samples)

def compute_descriptive_stats(df: pl.DataFrame, save_path: Path):
    stats = df.describe()
    stats_path = save_path / "sampled_descriptive_stats.csv"
    stats.write_csv(stats_path)
    print(f"📊 Descriptive statistics saved to: {stats_path}")

def compute_and_plot_correlation(df: pl.DataFrame, save_path: Path, method: str = "pearson"):
    df_pd = df.to_pandas()
    corr = df_pd.corr(method=method)

    csv_path = save_path / f"sampled_correlation_{method}.csv"
    png_path = save_path / f"sampled_correlation_{method}.png"

    corr.to_csv(csv_path)
    print(f"📈 Correlation matrix saved to: {csv_path}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"{method.capitalize()} Correlation Matrix (Sampled)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"🖼️ Correlation heatmap saved to: {png_path}")

def main():
    year = "2023"
    frac = 0.01  # 1% sample from each file
    feature_cols = [
        'WSPD', 'VHM0', 'VTM02', 'corrected_VHM0', 'corrected_VTM02',
        'U10', 'V10', 'wave_dir_sin', 'wave_dir_cos',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'sin_month', 'cos_month', 'lat_norm', 'lon_norm'
    ]

    parquet_dir = Path("/data/tsolis/AI_project/parquet/augmented_with_labels/hourly")
    output_dir = Path(f"/data/tsolis/AI_project/output/eda/augmented_with_labels/{year}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_df = collect_samples(parquet_dir, feature_cols, f"WAVEAN{year}*.parquet", frac=frac)

    compute_descriptive_stats(sampled_df, output_dir)
    compute_and_plot_correlation(sampled_df, output_dir, method="pearson")
    # compute_and_plot_correlation(sampled_df, output_dir, method="spearman")

if __name__ == "__main__":
    main()
