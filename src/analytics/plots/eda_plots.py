import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def plot_time_series(ts: pd.Series, save_path=None, show=True):
    plt.figure(figsize=(14,5))
    plt.plot(ts.index, ts, label="Actual")
    plt.title("Actual Time Series")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_outliers(ts: pd.Series, outlier_mask, save_path=None, show=True):
    plt.figure(figsize=(14,5))
    plt.plot(ts.index, ts, label="Actual")
    plt.scatter(ts.index[outlier_mask], ts[outlier_mask], color="crimson", marker="x", label="Outliers")
    plt.title("Actual TS vs. Outliers")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_smoothed_differenced(ts: pd.Series, smoothed: pd.Series, differenced: pd.Series, scaled_series: pd.Series, save_path=None, show=True):
    fig, axs = plt.subplots(4, 1, figsize=(14,8), sharex=True)
    axs[0].plot(ts.index, ts, label="Actual TS", color="blue")
    axs[0].set_title("Actual TS")
    axs[0].legend()
    axs[1].plot(ts.index, smoothed, label="Smoothed (rolling mean)", color="green")
    axs[1].set_title("Smoothed TS")
    axs[1].legend()
    axs[2].plot(ts.index, differenced, label="Differenced (1st diff)", color="purple")
    axs[2].set_title("Differenced TS")
    axs[2].legend()
    axs[3].plot(ts.index, scaled_series, label="Scaled TS", color="red")
    axs[3].set_title("Scaled TS")
    axs[3].legend()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_distribution(ts: pd.Series, feature_col_name: str, save_path=None, show=False):
    plt.figure(figsize=(10, 5))
    sns.histplot(ts, bins=40, kde=True, color="royalblue", stat="density")
    plt.title(f"Distribution of {feature_col_name}")
    plt.xlabel(feature_col_name)
    plt.ylabel("Density")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_monthly_seasonality(df: pd.Series, value_col: str, save_path=None):
    """
    Plot average value per month across years.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(df["month"], df["monthly_mean"], color='xkcd:amethyst', marker="o")
    plt.title(f"Monthly Mean {value_col}")
    plt.xlabel("Month")
    plt.ylabel(f"Mean {value_col}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_weekly_seasonality(df: pd.Series, value_col: str, save_path=None):
    """
    Plot average value per weekday (Monday=0) across years.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(df["weekday"], df["weekday_mean"], color="orange", marker="o")
    plt.title(f"Weekly Mean {value_col}")
    plt.xlabel("Weekday")
    plt.ylabel(f"Mean {value_col}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_weekly_month_seasonality(df: pd.Series, value_col: str, save_path=None):
    fig, ax = plt.subplots()
    for month in sorted(df["month"].unique()):
        monthly = df[df["month"] == month]
        weekly_means = monthly.groupby("weekday")[f"{value_col}_mean"].mean()
        x = range(7)
        line, = ax.plot(x, weekly_means, label=f"Month {month}")
        ax.annotate(
            f"{month}",
            xy=(x[-1], weekly_means.iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            color=line.get_color(),
            va='center',
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlabel("Day of Week (0=Mon)")
    ax.set_ylabel(f"Mean {value_col} (m)")
    ax.set_title(f"Weekly {value_col} Pattern by Month")
    ax.legend().set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_hourly_by_weekday(
    df: pd.DataFrame,
    value_col: str = "VHM0_mean",
    weekday_labels: list = None,
    save_path: str = None
):
    """
    Plots the average hourly values per weekday using seaborn lineplot.
    Args:
        df: DataFrame with 'hour', 'weekday', and value_col columns.
        value_col: Feature to plot (e.g., "VHM0_mean").
        weekday_labels: List of weekday names (optional).
        save_path: If given, save the plot to this path.
    """
    # Group and average
    df_plot = (
        df[["hour", "weekday", value_col]]
        .dropna()
        .groupby(["hour", "weekday"])
        .mean()
        .reset_index()
    )
    # Set weekday labels if not provided
    if weekday_labels is None:
        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def safe_weekday_label(w):
        try:
            return weekday_labels[int(w) - 1]
        except (IndexError, ValueError, TypeError):
            return f"Day {w}"

    # Convert weekday number to string for legend
    df_plot["weekday_str"] = df_plot["weekday"].map(safe_weekday_label)

    plt.figure(figsize=(10, 7))
    sns.lineplot(data=df_plot, x="hour", y=value_col, hue="weekday_str")
    plt.title(f"Hourly Pattern by Day of Week: {value_col}")
    plt.xlabel("Hour of Day")
    plt.ylabel(f"{value_col}")
    plt.legend(title="Weekday", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_time_series_decomposition(
    ts: pd.Series,
    freq: int = 24,  # e.g., 24 for hourly with daily seasonality, 365 for daily with yearly seasonality
    model: str = "additive",  # or "multiplicative"
    save_path: str = None,
):
    """
    Decompose a time series and plot its components.
    Args:
        ts: pandas Series with DateTimeIndex.
        freq: int, period for seasonal decomposition (24=hourly/daily, 12=monthly/yearly, etc).
        model: "additive" or "multiplicative".
        save_path: Path to save the plot (optional).
        show: Whether to show the plot interactively.
    """
    # Drop missing values for decomposition
    ts_clean = ts.dropna()

    # Decompose
    decomposition = seasonal_decompose(ts_clean, period=freq, model=model, extrapolate_trend="freq")
    # decomposition = seasonal_decompose(ts_clean, period=freq, model=model,model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # Plotting the decomposed components
    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(ts_clean, label='Original Time Series', color='blue')
    plt.title('Original Time Series')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend Component', color='orange')
    plt.title('Trend Component')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal Component', color='green')
    plt.title('Seasonal Component')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual Component', color='red')
    plt.title('Residual Component')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()
