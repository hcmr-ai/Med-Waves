import polars as pl

def hourly_timeseries(df: pl.DataFrame, value_col: str = "VHM0") -> pl.DataFrame:
    """
    Aggregate by hour to get mean per hour for the entire grid.
    Returns a Polars DataFrame with ['time', 'mean_value'].
    """
    return (
        df.group_by("time")
        .agg(pl.col(value_col).mean().alias("mean_value"))
        .sort("time")
    )

def flag_missing(df: pl.DataFrame, value_col: str = "VHM0") -> pl.DataFrame:
    """
    Adds a column 'is_missing' indicating missing values in value_col.
    """
    return df.with_columns(
        (pl.col(value_col).is_null()).alias("is_missing")
    )

def flag_outliers_zscore(df: pl.DataFrame, value_col: str = "VHM0", thresh: float = 3.0) -> pl.DataFrame:
    """
    Adds an 'is_outlier' column using z-score method.
    Only for non-missing rows!
    """
    stats = df.select([pl.col(value_col).mean(), pl.col(value_col).std()])
    mu = stats[0, 0]
    sigma = stats[0, 1]
    return df.with_columns(
        (
            ((pl.col(value_col) - mu).abs() / sigma > thresh)
        ).alias("is_outlier")
    )

def smooth_rolling(df: pl.DataFrame, value_col: str = "VHM0", window: int = 7) -> pl.DataFrame:
    """
    Adds a 'smoothed' column: rolling mean over specified window (centered).
    """
    return df.with_columns(
        pl.col(value_col).rolling_mean(window_size=window, min_periods=1).alias("smoothed")
    )

def difference_series(df: pl.DataFrame, value_col: str = "VHM0") -> pl.DataFrame:
    """
    Adds a 'differenced' column: first difference of the time series.
    """
    return df.with_columns(
        pl.col(value_col).diff().fill_null(0).alias("differenced")
    )

def scale_series(df: pl.DataFrame):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(df.values.reshape(-1, 1))