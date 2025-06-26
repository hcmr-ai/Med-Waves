# %% importing libraries
import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm  # Import tqdm for progress bar

# %% File paths
# Path to NCDF files
file_path_X_TEST = (
    "/data/tsolis/AI_project/without_reduced/_reduced_For_Testing_Grid_25step_TEST/"
)
file_path_X_TRAIN = (
    "/data/tsolis/AI_project/without_reduced/_reduced_For_Testing_Grid_25step_TRAIN/"
)
file_path_Y_TEST = (
    "/data/tsolis/AI_project/with_reduced/_reduced_For_Testing_Grid_25step_TEST/"
)
file_path_Y_TRAIN = (
    "/data/tsolis/AI_project/with_reduced/_reduced_For_Testing_Grid_25step_TRAIN/"
)

output_directory = "/data/tsolis/AI_project/output"
save_dir = "/data/tsolis/AI_project/output/models"
os.makedirs(save_dir, exist_ok=True)
files_X_TEST = sorted(glob.glob(file_path_X_TEST + "WAVEAN20*.nc"))
files_X_TRAIN = sorted(glob.glob(file_path_X_TRAIN + "WAVEAN20*.nc"))
files_Y_TEST = sorted(glob.glob(file_path_Y_TEST + "WAVEAN20*.nc"))
files_Y_TRAIN = sorted(glob.glob(file_path_Y_TRAIN + "WAVEAN20*.nc"))


# Initialize the header for the CSV file
csv_header = [
    "Experiment – Description",
    "Bias",
    "RMSE",
    "MAE",
    "Pearson",
    "Lat",
    "Lon",
    "Season",
]
metrics_file = os.path.join(output_directory, "experiment_metrics.csv")
vmax = 0
global all_y_test, all_X_test, all_y_pred
all_y_test, all_X_test, all_y_pred = [], [], []
# Write the header to the CSV file initially
if not os.path.exists(metrics_file):
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# %% Open nc files
# Chunk the data to manage memory
# Open datasets
X_test = xr.open_mfdataset(files_X_TEST, chunks="auto")
print("X_test ready")
X_train = xr.open_mfdataset(files_X_TRAIN, chunks="auto")
print("X_train ready")
Y_test = xr.open_mfdataset(files_Y_TEST, chunks="auto")
print("Y_test ready")
Y_train = xr.open_mfdataset(files_Y_TRAIN, chunks="auto")
print("Y_train ready")

# %% csv valid locations
# Initialize an empty list to store the valid (latitude, longitude) tuples
valid_lat_lon_pairs = []
# Path to the CSV file
csv_file_path = "/home/n.tsolis/AI_project/valid_lat_lon_pairs_5.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Convert the DataFrame to a list of tuples (Latitude, Longitude)
valid_lat_lon_pairs = list(zip(df["Latitude"], df["Longitude"], strict=False))

dataset_map = []
Y_pred_ds = []
time_map = X_test.time.values[:]


# %% define functions
def plot_map(X_test, Y_test, df, time_map, output_directory):
    c = 0
    projection = ccrs.PlateCarree()

    # Loop through each specific time in time_map and gather vmax values from all datasets
    for specific_time in time_map:
        # Get the VHM0 data from X_test (xarray)
        vhm0_X_test = X_test.sel(time=specific_time)["VHM0"]
        vhm0_Y_test = Y_test.sel(time=specific_time)["VHM0"]

        # Convert Dask arrays to NumPy if needed (for max calculation)
        max_X_test = (
            da.nanmax(vhm0_X_test).compute()
            if isinstance(vhm0_X_test.data, da.Array)
            else np.nanmax(vhm0_X_test)
        )
        max_Y_test = (
            da.nanmax(vhm0_Y_test).compute()
            if isinstance(vhm0_Y_test.data, da.Array)
            else np.nanmax(vhm0_Y_test)
        )

        # Extract the VHM0 data from the DataFrame for the given time
        if isinstance(df, pd.DataFrame):
            vhm0_df = df[df["Time"] == specific_time]["Value"].values
            max_df = np.nanmax(vhm0_df) if vhm0_df.size > 0 else np.nan
        else:
            max_df = np.nan

        # Find the maximum value across all sources
        vmax = max(max_X_test, max_Y_test, max_df)
        print(f"vmax={vmax}")

        c = c + 1
        # Plotting
        fig = plt.figure(figsize=(10, 8))

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(
            [
                X_test.longitude.min(),
                X_test.longitude.max(),
                X_test.latitude.min(),
                X_test.latitude.max(),
            ],
            crs=ccrs.PlateCarree(),
        )

        # Add map features
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=1)

        ax.set_title(f"VHM0 Values of X_test at {specific_time}")
        file_name = f"{c}_VHM0_Values_of_X_test.png"
        file_path = os.path.join(output_directory, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close()
        fig = plt.figure(figsize=(10, 8))

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(
            [
                Y_test.longitude.min(),
                Y_test.longitude.max(),
                Y_test.latitude.min(),
                Y_test.latitude.max(),
            ],
            crs=ccrs.PlateCarree(),
        )

        # Add map features
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=1)

        ax.set_title(f"VHM0 Values of Y_test at {specific_time}")
        file_name = f"{c}_VHM0_Values_of_Y_test.png"
        file_path = os.path.join(output_directory, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close()

        # Filter the DataFrame for the current time step
        subset = df[df["Time"] == specific_time]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": projection})

        # Add map features (coastlines, gridlines, etc.)
        ax.set_extent(
            [
                Y_test.longitude.min(),
                Y_test.longitude.max(),
                Y_test.latitude.min(),
                Y_test.latitude.max(),
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.coastlines()
        # ax.gridlines(draw_labels=False)
        # Add map features
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=1)
        # Get unique Longitude and Latitude values
        lon_unique = np.unique(subset["Longitude"])
        lat_unique = np.unique(subset["Latitude"])

        # Create a mesh grid for interpolation
        lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)

        # Initialize grid with NaN values
        value_grid = np.full(lon_grid.shape, np.nan)

        # Convert Longitude and Latitude to a tuple for lookup
        coords = list(zip(subset["Longitude"], subset["Latitude"], strict=False))
        values = subset["Value"].values

        # Fill the grid with values, leave missing values as NaN
        for i, (lon, lat) in enumerate(coords):
            lon_idx = np.where(lon_unique == lon)[0][0]
            lat_idx = np.where(lat_unique == lat)[0][0]
            value_grid[lat_idx, lon_idx] = values[i]

        # Plot the data using pcolormesh
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            value_grid,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            vmin=0,
            vmax=vmax,
        )

        # Add color bar
        plt.colorbar(mesh, ax=ax, label="VHM0 (meters)")

        # Add title
        ax.set_title(f"VHM0 Map of Y_pred at Time {specific_time}")
        filename = f"{c}_VHM0_Values_of_Y_pred.png"

        file_path = os.path.join(output_directory, filename)
        plt.savefig(file_path)

        plt.close()  # Close the plot to avoid overlap between iterations


def plot_metrics_histogram(metrics_by_range):
    """
    Plots metrics for each range of VHM0 as grouped bar charts.

    Parameters:
    - metrics_by_range: List of metrics for each range, where each element contains:
      [Range Label, Samples, RMSE Before, RMSE After, MAE Before, MAE After, Bias Before, Bias After, Pearson Before, Pearson After]
    """
    # Convert metrics to a DataFrame for easier plotting
    df = pd.DataFrame(
        metrics_by_range,
        columns=[
            "Range",
            "Samples",
            "RMSE Before",
            "RMSE After",
            "MAE Before",
            "MAE After",
            "Bias Before",
            "Bias After",
            "Pearson Before",
            "Pearson After",
        ],
    )

    # Metrics to plot
    metrics_to_plot = ["RMSE", "MAE", "Bias", "Pearson"]

    # Create grouped bar plots
    x = np.arange(len(df))  # Label locations
    width = 0.35  # Bar width

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axs[i]
        ax.bar(x - width / 2, df[f"{metric} Before"], width, label=f"{metric} Before")
        ax.bar(x + width / 2, df[f"{metric} After"], width, label=f"{metric} After")

        ax.set_title(f"{metric} by VHM0 Range")
        ax.set_xlabel("VHM0 Range")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Range"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    file_name = "metrics_by_range_histogram.png"
    file_path = os.path.join(output_directory, file_name)
    plt.savefig(file_path)
    plt.close()
    print(f"Saved histogram of metrics by range to {file_path}")


def compute_metrics_by_range(y_test_combined, X_test_combined, y_pred_combined):
    """
    Compute metrics and sample size for different ranges of VHM0 values.
    """
    # Define ranges for VHM0 values
    bins = np.arange(
        0, np.max(y_test_combined) + 1, 1
    )  # Ranges: 0-1, 1-2, ..., up to max value
    bin_labels = [f"{int(b)}-{int(b + 1)}" for b in bins[:-1]]

    # Digitize data into bins
    y_test_bins = np.digitize(y_test_combined, bins) - 1  # Bin indices start at 0
    metrics_by_range = []

    print("\nMetrics by VHM0 Range:")
    for i, label in enumerate(bin_labels):
        # Select data in the current range
        mask = y_test_bins == i
        y_test_range = y_test_combined[mask]
        X_test_range = X_test_combined[mask]
        y_pred_range = y_pred_combined[mask]

        # Skip if no data in the range
        if len(y_test_range) == 0:
            continue
        if len(y_test_range) < 2 or len(X_test_range) < 2 or len(y_pred_range) < 2:
            print(
                f"Skipping range {label} due to insufficient data for Pearson correlation."
            )
            continue

        # Calculate metrics
        rmse_before = np.sqrt(mean_squared_error(y_test_range, X_test_range))
        rmse_after = np.sqrt(mean_squared_error(y_test_range, y_pred_range))

        mae_before = mean_absolute_error(y_test_range, X_test_range)
        mae_after = mean_absolute_error(y_test_range, y_pred_range)

        bias_before = np.mean(X_test_range - y_test_range)
        bias_after = np.mean(y_pred_range - y_test_range)

        pearson_before, _ = pearsonr(y_test_range, X_test_range)
        pearson_after, _ = pearsonr(y_test_range, y_pred_range)

        # Log metrics
        print(
            f"Range {label}: Samples: {len(y_test_range)}, "
            f"RMSE Before: {rmse_before:.4f}, RMSE After: {rmse_after:.4f}, "
            f"MAE Before: {mae_before:.4f}, MAE After: {mae_after:.4f}, "
            f"Bias Before: {bias_before:.4f}, Bias After: {bias_after:.4f}, "
            f"Pearson Before: {pearson_before:.4f}, Pearson After: {pearson_after:.4f}"
        )

        # Append metrics for saving
        metrics_by_range.append(
            [
                label,
                len(y_test_range),
                rmse_before,
                rmse_after,
                mae_before,
                mae_after,
                bias_before,
                bias_after,
                pearson_before,
                pearson_after,
            ]
        )

    # Save to CSV
    range_metrics_file = os.path.join(output_directory, "metrics_by_range.csv")
    with open(range_metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Range",
                "Samples",
                "RMSE Before",
                "RMSE After",
                "MAE Before",
                "MAE After",
                "Bias Before",
                "Bias After",
                "Pearson Before",
                "Pearson After",
            ]
        )
        writer.writerows(metrics_by_range)
    plot_metrics_histogram(metrics_by_range)
    print(f"Saved metrics by range to {range_metrics_file}")


def compute_overall_metrics():
    # print(all_X_test,all_y_pred,all_y_test)
    # Check if the lists contain data
    if not all_y_test or not all_X_test or not all_y_pred:
        print(
            "Error: No data available for computing overall metrics. Ensure all_y_test, all_X_test, and all_y_pred are populated."
        )
        return

    y_test_combined = np.concatenate(all_y_test)
    X_test_combined = np.concatenate(all_X_test)
    y_pred_combined = np.concatenate(all_y_pred)
    y_test_combined = y_test_combined.ravel()
    X_test_combined = X_test_combined.ravel()
    y_pred_combined = y_pred_combined.ravel()
    valid_indices = (
        ~np.isnan(y_test_combined)
        & ~np.isnan(X_test_combined)
        & ~np.isnan(y_pred_combined)
    )
    y_test_combined = y_test_combined[valid_indices]
    X_test_combined = X_test_combined[valid_indices]
    y_pred_combined = y_pred_combined[valid_indices]

    # Calculate metrics
    rmse_before = np.sqrt(mean_squared_error(y_test_combined, X_test_combined))
    rmse_after = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))

    mae_before = mean_absolute_error(y_test_combined, X_test_combined)
    mae_after = mean_absolute_error(y_test_combined, y_pred_combined)
    # change y_test - x_test/y_pred
    bias_before = np.mean(X_test_combined - y_test_combined)
    bias_after = np.mean(y_pred_combined - y_test_combined)

    pearson_before, _ = pearsonr(y_test_combined.flatten(), X_test_combined.flatten())
    pearson_after, _ = pearsonr(y_test_combined.flatten(), y_pred_combined.flatten())

    # Print results
    print("\nOverall Metrics:")
    print(f"RMSE Before: {rmse_before:.4f}, RMSE After: {rmse_after:.4f}")
    print(f"MAE Before: {mae_before:.4f}, MAE After: {mae_after:.4f}")
    print(f"Bias Before: {bias_before:.4f}, Bias After: {bias_after:.4f}")
    print(
        f"Pearson Correlation Before: {pearson_before:.4f}, Pearson After: {pearson_after:.4f}"
    )

    # Save to CSV
    metrics_row1 = [
        "Default case (without currents and obs)",
        bias_before,
        rmse_before,
        mae_before,
        pearson_before,
        "ALL",
        "ALL",
        "ALL",
    ]
    metrics_row2 = [
        "EXP1 – Bias correction (BC) - simple DNN",
        bias_after,
        rmse_after,
        mae_after,
        pearson_after,
        "ALL",
        "ALL",
        "ALL",
    ]

    with open(metrics_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(metrics_row1)
        writer.writerow(metrics_row2)
    compute_metrics_by_range(y_test_combined, X_test_combined, y_pred_combined)
    # Generate scatter plots
    plot_aggregated_scatter(y_test_combined, X_test_combined, y_pred_combined)


def plot_aggregated_scatter(y_test_combined, X_test_combined, y_pred_combined):
    """
    Creates scatter plots for aggregated data from all lat-lon pairs and includes a table of overall metrics.

    Parameters:
    - y_test_combined: numpy array of true VHM0 values (with currents).
    - X_test_combined: numpy array of predicted VHM0 values before model (without currents).
    - y_pred_combined: numpy array of predicted VHM0 values after model.
    """
    # Calculate overall metrics
    rmse_before = np.sqrt(mean_squared_error(y_test_combined, X_test_combined))
    rmse_after = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))

    mae_before = mean_absolute_error(y_test_combined, X_test_combined)
    mae_after = mean_absolute_error(y_test_combined, y_pred_combined)

    bias_before = np.mean(X_test_combined - y_test_combined)
    bias_after = np.mean(y_pred_combined - y_test_combined)

    pearson_before, _ = pearsonr(y_test_combined.flatten(), X_test_combined.flatten())
    pearson_after, _ = pearsonr(y_test_combined.flatten(), y_pred_combined.flatten())

    metrics = [
        ["Metric", "Before Model", "After Model"],
        ["RMSE", f"{rmse_before:.4f}", f"{rmse_after:.4f}"],
        ["MAE", f"{mae_before:.4f}", f"{mae_after:.4f}"],
        ["Bias", f"{bias_before:.4f}", f"{bias_after:.4f}"],
        ["Pearson", f"{pearson_before:.4f}", f"{pearson_after:.4f}"],
    ]

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Scatter plot before model
    axs[0].scatter(
        y_test_combined, X_test_combined, alpha=0.6, edgecolor="k", label="Data"
    )
    axs[0].plot(
        [min(y_test_combined), max(y_test_combined)],
        [min(y_test_combined), max(y_test_combined)],
        "r--",
        label="Ideal Fit",
    )
    axs[0].set_xlabel("True VHM0 With Currents")
    axs[0].set_ylabel("VHM0 Without Currents")
    axs[0].set_title("Scatter Plot Before Model")
    axs[0].legend()
    axs[0].grid(True)

    # Scatter plot after model
    axs[1].scatter(
        y_test_combined, y_pred_combined, alpha=0.6, edgecolor="k", label="Predictions"
    )
    axs[1].plot(
        [min(y_test_combined), max(y_test_combined)],
        [min(y_test_combined), max(y_test_combined)],
        "r--",
        label="Ideal Fit",
    )
    axs[1].set_xlabel("True VHM0 With Currents")
    axs[1].set_ylabel("Predicted VHM0")
    axs[1].set_title("Scatter Plot After Model")
    axs[1].legend()
    axs[1].grid(True)

    # Add a table with metrics below the plots
    fig.subplots_adjust(bottom=0.2)
    table_ax = fig.add_subplot(
        111, frame_on=False
    )  # Add a full-width table at the bottom
    table_ax.axis("tight")
    table_ax.axis("off")

    # Create the table
    table = plt.table(
        cellText=metrics,
        cellLoc="center",
        colLabels=None,
        loc="bottom",
        bbox=[0.2, -0.6, 0.6, 0.4],  # Adjust the bbox as needed
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(metrics[0]))))

    # Save the scatter plots with the table
    file_name = "aggregated_scatter_plots_with_metrics.png"
    file_path = os.path.join(output_directory, file_name)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Saved aggregated scatter plot with metrics table to {file_path}")


# Define the function to process a single location (lat, lon)
def process_location(
    lat_lon_pair,
    X_train,
    Y_train,
    X_test,
    Y_test,
    output_directory=output_directory,
    chunk_size=500,
):
    lat, lon = lat_lon_pair
    try:
        os.makedirs(
            output_directory, exist_ok=True
        )  # Create the directory if it doesn't exist
        # Extract Data for the Given Location
        X_train_loc = (
            X_train.sel(latitude=lat, longitude=lon)
            .to_array()
            .sel(variable=["VHM0", "WSPD"])
            .values
        )
        Y_train_loc = Y_train.sel(latitude=lat, longitude=lon)["VHM0"].values
        X_test_loc = (
            X_test.sel(latitude=lat, longitude=lon)
            .to_array()
            .sel(variable=["VHM0", "WSPD"])
            .values
        )
        Y_test_loc = Y_test.sel(latitude=lat, longitude=lon)["VHM0"].values

        # Reshape to match the time dimension
        X_train_loc = X_train_loc.reshape(-1, X_train.sizes["time"]).T
        Y_train_loc = Y_train_loc.reshape(-1, Y_train.sizes["time"]).T
        X_test_loc = X_test_loc.reshape(-1, X_test.sizes["time"]).T
        Y_test_loc = Y_test_loc.reshape(-1, Y_test.sizes["time"]).T

        # Feature Transformation
        poly = PolynomialFeatures(degree=2, include_bias=False)
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        """
        model = SGDRegressor(
            loss='epsilon_insensitive',
            epsilon=0,
            early_stopping=True,
            max_iter=50,
            tol=1e-3,
            shuffle=True,
            random_state=42,
            average=True  # Averages coefficients for stability
        )
        """
        model = LinearRegression()
        X_train_poly = poly.fit_transform(X_train_loc)
        X_train_scaled = scalerX.fit_transform(X_train_poly)

        Y_train_scaled = scalerY.fit_transform(
            Y_train_loc
        ).ravel()  # Ensure Y_train_scaled is 1D

        # Train Model
        model.fit(X_train_scaled, Y_train_scaled)
        # save model
        flnm = f"model_{lat}_{lon}.pkl"
        filepath = os.path.join(save_dir, flnm)
        joblib.dumb(model, filepath)
        # Prepare NetCDF file
        output_file = f"{output_directory}/Y_pred_map_{lat}_{lon}.nc"

        # Prepare lists to store results
        times = []
        preds_list = []

        # Process in Chunks
        for start in range(0, X_test_loc.shape[0], chunk_size):
            end = min(start + chunk_size, X_test_loc.shape[0])
            X_test_chunk = X_test_loc[start:end]

            X_test_poly_chunk = poly.transform(X_test_chunk)
            X_test_scaled_chunk = scalerX.transform(X_test_poly_chunk)

            Y_pred_chunk = model.predict(X_test_scaled_chunk)
            Y_pred_chunk = scalerY.inverse_transform(
                Y_pred_chunk.reshape(-1, 1)
            )  # Shape: (chunk_size, 1)

            # Store results
            time_indices = [
                list(X_test.time.values).index(t)
                for t in time_map[start:end]
                if t in X_test.time.values
            ]
            Y_pred_filtered_chunk = Y_pred_chunk[
                : len(time_indices)
            ]  # Shape: (filtered_size, 1)

            times.extend(time_map[start:end])
            preds_list.append(Y_pred_filtered_chunk)  # Store chunks separately

        # Convert list of arrays to a single NumPy array
        preds_np = np.vstack(preds_list)  # Shape: (num_time_steps, 1)
        preds_np = preds_np.astype(
            np.float32
        )  # Convert to float32 to reduce precision and memory usage

        # Convert to Dask array with proper chunking
        preds_da = da.from_array(preds_np, chunks=(chunk_size, 1))  # Ensure 2D chunking

        # Create an xarray Dataset
        ds = xr.Dataset(
            {
                "VHM0": (
                    ["time"],
                    preds_da[:, 0],
                )  # Use 1D variable to match expected format
            },
            coords={
                "latitude": lat,
                "longitude": lon,
                "time": np.array(times),
            },
        )

        # Save as NetCDF with compression
        encoding = {"VHM0": {"zlib": True, "complevel": 5}}
        ds.to_netcdf(output_file, mode="w", encoding=encoding)

        return (
            Y_test_loc,
            X_test_loc[:, 0],
            None,
            None,
            lat,
            lon,
        )  # Reduce return data to save memory

    except Exception as e:
        print(f"Error processing location {(lat, lon)}: {e}")
        return None


# %% parallel processing
# Define the function to handle parallel processing with progress bar
def parallel_processing(X_train, Y_train, X_test, Y_test, valid_lat_lon_pairs):
    # Create a partial function to pass fixed arguments (X_train, Y_train, etc.)
    func = partial(
        process_location, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test
    )

    # Initialize a Pool with tqdm for progress bar
    with Pool(processes=100) as pool:
        # Use 'tqdm' to create a progress bar for the map function
        results = list(
            tqdm(
                pool.imap(func, valid_lat_lon_pairs),
                total=len(valid_lat_lon_pairs),
                desc="Processing locations",
            )
        )

    # Filter out None results
    results = [result for result in results if result is not None]

    # Separate the results for the other outputs
    y_test_combined = np.concatenate([res[0] for res in results])
    X_test_combined = np.concatenate([res[1] for res in results])
    y_pred_combined = np.concatenate([res[2] for res in results])

    Y_pred_map = []
    for res in results:
        extracted_values = res[
            3
        ]  # This should be the list of dictionaries containing time, latitude, longitude, and value
        Y_pred_map.extend(extracted_values)

    # plot_map(X_test,Y_test,df, time_map,output_directory)
    # print("Maps saved for each time step.")
    # Append the results to the global lists (if this is part of a larger data pipeline)
    all_y_test.append(y_test_combined)
    all_X_test.append(X_test_combined)
    all_y_pred.append(y_pred_combined)

    return y_test_combined, X_test_combined, y_pred_combined, Y_pred_map


# %% MAIN
# Ensure the code below only runs if the script is executed as the main program
if __name__ == "__main__":
    # Call the parallel processing function and get results
    y_test_combined, X_test_combined, y_pred_combined, Y_pred_ds = parallel_processing(
        X_train, Y_train, X_test, Y_test, valid_lat_lon_pairs
    )

    # After parallel processing, compute the overall metrics
    compute_overall_metrics()
