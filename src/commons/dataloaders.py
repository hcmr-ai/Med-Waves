import random

import numpy as np
import pyarrow.parquet as pq
import s3fs
import torch
from torch.utils.data import Dataset


class WaveDatasetDNN(Dataset):
    def __init__(
        self,
        file_paths,
        fs=None,
        patch_size=None,
        normalizer=None,
        excluded_columns=None,
        target_column="corrected_VHM0",
        predict_bias=False,
        subsample_step=None,
    ):
        """
        Dataset for DNN training with excluded column handling.

        Args:
            file_paths: list of parquet file paths (S3 or local)
            fs: optional s3fs.S3FileSystem()
            patch_size: tuple (H, W) for random crops, None = full grid
            normalizer: WaveNormalizer instance (already fitted on training data)
            excluded_columns: list of columns to exclude from input (e.g., ["time", "latitude", "longitude", "timestamp"])
            target_column: target column name (e.g., "corrected_VHM0")
            predict_bias: if True, predict bias (corrected_VHM0 - VHM0), if False, predict corrected_VHM0
            subsample_step: step size for subsampling the data, None = use all data
        """
        self.file_paths = file_paths
        self.fs = fs or s3fs.S3FileSystem()
        self.patch_size = patch_size
        self.normalizer = normalizer
        self.excluded_columns = excluded_columns or [
            "time",
            "latitude",
            "longitude",
            "timestamp",
        ]
        self.target_column = target_column
        self.predict_bias = predict_bias
        self.subsample_step = subsample_step
        # Index map: (file_idx, hour_idx)
        self.index_map = [
            (f_idx, h) for f_idx in range(len(file_paths)) for h in range(24)
        ]

    def __len__(self):
        return len(self.index_map)

    def _load_parquet(self, path):
        """Load parquet file efficiently using PyArrow operations"""
        if self.fs is None or path.startswith("/"):
            # Local file
            table = pq.read_table(path)
        else:
            # S3 file
            with self.fs.open(path, "rb") as f:
                table = pq.read_table(f)

        # Get column names
        column_names = [field.name for field in table.schema]
        # Don't filter excluded columns yet - we need VHM0 for bias calculation
        feature_cols = [
            col for col in column_names if col not in ["time", "latitude", "longitude"]
        ]

        # Convert to numpy arrays directly from PyArrow
        time_data = table.column("time").to_numpy()
        lat_data = table.column("latitude").to_numpy()
        lon_data = table.column("longitude").to_numpy()

        # Get unique values and create mappings
        unique_times = np.unique(time_data)
        unique_lats = np.unique(lat_data)
        unique_lons = np.unique(lon_data)

        T, H, W = len(unique_times), len(unique_lats), len(unique_lons)

        # Create efficient mappings using searchsorted
        time_indices = np.searchsorted(unique_times, time_data)
        lat_indices = np.searchsorted(unique_lats, lat_data)
        lon_indices = np.searchsorted(unique_lons, lon_data)

        # Pre-allocate output array
        arr = np.full((T, H, W, len(feature_cols)), np.nan, dtype=np.float32)

        # Convert feature columns to numpy
        feature_arrays = []
        for col in feature_cols:
            feature_arrays.append(table.column(col).to_numpy())

        # Use advanced indexing to fill array efficiently
        for j, feature_array in enumerate(feature_arrays):
            arr[time_indices, lat_indices, lon_indices, j] = feature_array

        return arr, feature_cols

    def __getitem__(self, idx):
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]

        arr, feature_cols = self._load_parquet(path)

        # Get the specific hour data
        hour_data = arr[hour_idx]  # shape (H, W, C)

        # Get input columns (all features except excluded columns)
        input_col_indices = [
            i for i, col in enumerate(feature_cols) if col not in self.excluded_columns
        ]

        X = hour_data[..., input_col_indices]  # shape (H, W, C_in)

        if self.predict_bias:
            # Calculate bias on-the-fly: corrected_VHM0 - VHM0
            vhm0_index = feature_cols.index("VHM0")
            corrected_index = feature_cols.index("corrected_VHM0")

            vhm0 = hour_data[..., vhm0_index : vhm0_index + 1]  # shape (H, W, 1)
            corrected = hour_data[
                ..., corrected_index : corrected_index + 1
            ]  # shape (H, W, 1)
            y = corrected - vhm0  # Target is the bias (correction field)
        else:
            # Use target column directly
            target_col_index = feature_cols.index(self.target_column)
            y = hour_data[
                ..., target_col_index : target_col_index + 1
            ]  # shape (H, W, 1)

        # Optional patch sampling
        if self.patch_size is not None:
            H, W, _ = X.shape
            ph, pw = self.patch_size
            if H > ph and W > pw:
                i = random.randint(0, H - ph)
                j = random.randint(0, W - pw)
                X = X[i : i + ph, j : j + pw, :]
                y = y[i : i + ph, j : j + pw, :]

        # Apply normalization (on X only)
        if self.normalizer is not None:
            X = self.normalizer.transform(X[np.newaxis, ...])[0]

        if self.subsample_step is not None:
            X = X[:: self.subsample_step, :: self.subsample_step, :]
            y = y[:: self.subsample_step, :: self.subsample_step, :]

        # Convert to torch (C, H, W)
        X = torch.from_numpy(X).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()

        # Create mask for NaN values
        mask = ~torch.isnan(y)

        return X, y, mask
