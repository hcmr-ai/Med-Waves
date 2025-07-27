from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def load_data(parquet_path: str) -> pl.DataFrame:
    print(f"📥 Loading data from {parquet_path}")
    return pl.read_parquet(parquet_path)

def load_data_lazy(parquet_glob_path: str) -> pl.LazyFrame:
    print(f"📥 Scanning data lazily from {parquet_glob_path}")
    return pl.scan_parquet(parquet_glob_path)

def compute_descriptive_stats(df: pl.DataFrame, feature_cols: list[str], save_path: Path):
    stats = df.select(feature_cols).describe()
    stats.write_csv(save_path / "descriptive_stats.csv")
    print(f"📊 Descriptive statistics saved to {save_path/'descriptive_stats.csv'}")
    return stats

def plot_correlation_matrix(df: pl.DataFrame, feature_cols: list[str], save_path: Path):
    df_pd = df.select(feature_cols).to_pandas()
    corr = df_pd.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix")

    fig_path = save_path / "correlation_matrix.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"📌 Correlation matrix saved to {fig_path}")

def plot_pairplot(df: pl.DataFrame, feature_cols: list[str], save_path: Path):
    sns.set(style="ticks")
    pairplot = sns.pairplot(df.to_pandas(), corner=True, plot_kws={'alpha': 0.3, 's': 10})
    fig_path = save_path / "pairplot.png"
    pairplot.savefig(fig_path, dpi=300, bbox_inches="tight")

    print(f"📌 Pairplot saved to {fig_path}")

def plot_correlation_heatmap(df: pl.DataFrame, feature_cols: list[str], save_path: Path, method='pearson'):
    corr = df.to_pandas().corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(f"{method.capitalize()} Correlation Heatmap")
    fig_path = save_path / f"correlation_heatmap_{method}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    print(f"📌 Correlation heatmap saved to {fig_path}")

def main():
    years = ["2021", "2022", "2023"]

    feature_cols = ['WSPD', 'VHM0', 'VTM02', 'corrected_VHM0', 'corrected_VTM02',
                    'U10', 'V10', 'wave_dir_sin', 'wave_dir_cos',
                    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
                    'sin_month', 'cos_month', 'lat_norm', 'lon_norm']
    feature_cols = ['WSPD_mean', 'VHM0_mean', 'VTM02_mean', 'corrected_VHM0_mean', 'corrected_VTM02_mean',
                    'U10_mean', 'V10_mean', 'wave_dir_sin_mean', 'wave_dir_cos_mean',
                    'sin_hour_mean', 'cos_hour_mean', 'sin_doy_mean', 'cos_doy_mean',
                    'sin_month_mean', 'cos_month_mean', 'lat_norm_mean', 'lon_norm_mean']
    data_origin = "augmented_with_labels"

    for year in years:
        main_parquet_data_path = f"/data/tsolis/AI_project/parquet/{data_origin}/hourly/"
        main_output_path = Path(f"outputs/eda/{data_origin}/{year}")
        main_output_path.mkdir(parents=True, exist_ok=True)

        parquet_file = f"{main_parquet_data_path}/WAVEAN{year}.parquet"
        # df = load_data(parquet_file)
        df_lazy = load_data_lazy(parquet_file)
        df = df_lazy.select(feature_cols).collect()
        compute_descriptive_stats(df, feature_cols, main_output_path)
        plot_correlation_matrix(df, feature_cols, main_output_path)
        plot_pairplot(df, feature_cols, main_output_path)

if __name__ == "__main__":
    main()
