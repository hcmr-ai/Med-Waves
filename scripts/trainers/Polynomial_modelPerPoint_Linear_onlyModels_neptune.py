# %% importing libraries
import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import dask.array as da
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

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
    with Pool(processes=10) as pool:
        # Use 'tqdm' to create a progress bar for the map function
        _ = list(
            tqdm(
                pool.imap(func, valid_lat_lon_pairs),
                total=len(valid_lat_lon_pairs),
                desc="Processing locations",
            )
        )

    return None


# %% MAIN
# Ensure the code below only runs if the script is executed as the main program
if __name__ == "__main__":
    # Call the parallel processing function and get results
    parallel_processing(X_train, Y_train, X_test, Y_test, valid_lat_lon_pairs)

    # After parallel processing, compute the overall metrics
    # compute_overall_metrics()
