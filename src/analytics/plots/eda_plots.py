import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
