import random

import numpy as np
import pyarrow.parquet as pq
import s3fs
import torch
from torch.utils.data import Dataset


class WaveDataset(Dataset):
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
        self._feature_names_logged = True  # Flag to log channel order only once
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
            # S3 file - create new filesystem instance to avoid fork-safety issues
            fs = s3fs.S3FileSystem()
            with fs.open(path, "rb") as f:
                table = pq.read_table(f)

        # Get column names
        column_names = [field.name for field in table.schema]
        # Filter out excluded columns (but keep VHM0 for bias calculation if needed)
        feature_cols = [
            col for col in column_names if col not in self.excluded_columns
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

        # Get input columns (exclude metadata and the target column)
        input_col_indices = [
            i
            for i, col in enumerate(feature_cols)
            if (col not in self.excluded_columns) and (col != self.target_column)
        ]

        X = hour_data[..., input_col_indices]  # shape (H, W, C_in)

        # Log channel order information only once
        if not self._feature_names_logged:
            import logging
            logger = logging.getLogger(__name__)
            input_feature_names = [feature_cols[i] for i in input_col_indices]
            logger.info("=" * 80)
            logger.info("CHANNEL ORDER IN INPUT TENSOR X:")
            for ch_idx, feat_name in enumerate(input_feature_names):
                logger.info(f"  Channel {ch_idx}: {feat_name}")
            logger.info("=" * 80)
            self._feature_names_logged = True

        if self.predict_bias:
            # Calculate bias on-the-fly: corrected_VHM0 - VHM0
            vhm0_index = feature_cols.index("VHM0")
            corrected_index = feature_cols.index("corrected_VHM0")

            vhm0 = hour_data[..., vhm0_index : vhm0_index + 1]  # shape (H, W, 1)
            corrected = hour_data[
                ..., corrected_index : corrected_index + 1
            ]  # shape (H, W, 1)
            logger.debug(f"VHM0 shape: {vhm0.shape}, stats: mean={vhm0[~np.isnan(vhm0)].mean():.3f}, "
                        f"std={vhm0[~np.isnan(vhm0)].std():.3f}, NaN count: {np.isnan(vhm0).sum()}")
            logger.debug(f"Corrected shape: {corrected.shape}, stats: mean={corrected[~np.isnan(corrected)].mean():.3f}, "
                        f"std={corrected[~np.isnan(corrected)].std():.3f}, NaN count: {np.isnan(corrected).sum()}")

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



from torch.utils.data import Dataset


class CachedWaveDataset(Dataset):
    def __init__(self, file_paths, target_column="corrected_VHM0",
                 excluded_columns=None, normalizer=None,
                 patch_size=None, subsample_step=None, predict_bias=False,
                 enable_profiler=False, use_cache=True, normalize_target=False):
        self.file_paths = file_paths
        # self.index_map = index_map   # list of (file_idx, hour_idx)
        self.target_column = target_column
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.subsample_step = subsample_step
        self.predict_bias = predict_bias
        self.enable_profiler = enable_profiler
        self.index_map = [
            (f_idx, h) for f_idx in range(len(file_paths)) for h in range(24)
        ]
        self.H, self.W = 380, 1307
        self.C_in = len(self.excluded_columns) + 1  # +1 for target column
        # worker-local cache
        self._cache = {}
        self.use_cache = use_cache
        self.normalize_target = normalize_target

    def _load_file(self, path):
        table = pq.read_table(path)

        column_names = [field.name for field in table.schema]
        feature_cols = [col for col in column_names if col not in self.excluded_columns]

        # Extract coords
        time_data = table.column("time").to_numpy()
        lat_data = table.column("latitude").to_numpy()
        lon_data = table.column("longitude").to_numpy()

        unique_times = np.unique(time_data)
        unique_lats = np.unique(lat_data)
        unique_lons = np.unique(lon_data)

        T, H, W = len(unique_times), len(unique_lats), len(unique_lons)

        time_idx = np.searchsorted(unique_times, time_data)
        lat_idx = np.searchsorted(unique_lats, lat_data)
        lon_idx = np.searchsorted(unique_lons, lon_data)

        arr = np.full((T, H, W, len(feature_cols)), np.nan, dtype=np.float32)

        for j, col in enumerate(feature_cols):
            arr[time_idx, lat_idx, lon_idx, j] = table.column(col).to_numpy()

        tensor = torch.from_numpy(arr)  # shape (T,H,W,C)
        return tensor, feature_cols

    def _load_file_pt(self, path):
        data = torch.load(path, map_location="cpu")   # {"tensor": (T,H,W,C), "feature_cols": [...]}
        return data["tensor"], data["feature_cols"]

    def _get_file_tensor(self, path):
        if self.use_cache and path not in self._cache:
            # Drop any previously cached file (keep only 1 in memory)
            # self._cache.clear()
            tensor, feature_cols = self._load_file_pt(path)
            self._cache[path] = (tensor, feature_cols)
            return tensor, feature_cols
        else:
            tensor, feature_cols = self._load_file_pt(path)
            return tensor, feature_cols

    def __getitem__(self, idx):
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]

        tensor, feature_cols = self._get_file_tensor(path)

        hour_data = tensor[hour_idx]
        # Select input cols
        input_col_indices = [
            i for i, col in enumerate(feature_cols)
            if (col not in self.excluded_columns) and (col != self.target_column)
        ]
        FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos', 'corrected_VHM0']
        
        # Select input features in FEATURES_ORDER to match scaler's stats_ indices
        # This ensures stats_[c] applies to channel c in X
        input_features = []
        input_col_indices = []
        
        for feat in FEATURES_ORDER:
            # Skip if excluded or is target
            if feat in self.excluded_columns or feat == self.target_column:
                continue
            
            # Find feature in feature_cols
            if feat in feature_cols:
                idx_in_feature_cols = feature_cols.index(feat)
                input_features.append(feat)
                input_col_indices.append(idx_in_feature_cols)
        # print(input_features)
        # print(input_col_indices)
        X = hour_data[..., input_col_indices]  # (H, W, C_in)

        vhm0 = hour_data[..., feature_cols.index("VHM0"):feature_cols.index("VHM0")+1]

        if self.predict_bias:
            corrected = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
            y = corrected - vhm0
        else:
            y = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]

        # Patch sampling
        if self.patch_size is not None:
            H, W, _ = X.shape
            ph, pw = self.patch_size
            if H > ph and W > pw:
                i = random.randint(0, H - ph)
                j = random.randint(0, W - pw)
                X = X[i:i+ph, j:j+pw, :]
                y = y[i:i+ph, j:j+pw, :]


        if self.enable_profiler:
            with torch.profiler.record_function("normalize_and_subsample"):
                # Subsample
                if self.subsample_step is not None:
                    X = X[::self.subsample_step, ::self.subsample_step, :]
                    y = y[::self.subsample_step, ::self.subsample_step, :]
                    vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]

                if self.normalizer is not None:
                    # X = self.normalizer.transform(X.numpy()[np.newaxis])[0]
                    # X = torch.from_numpy(X).float()
                    # X = self.normalizer.transform_torch(X)
                    if self.normalize_target:
                        # Debug: Check inputs before normalization
                        # print(f"Before normalization - X shape: {X.shape}, y shape: {y.shape}")
                        # print(f"Before normalization X stats: mean={X[~torch.isnan(X)].mean().item():.4f}, std={X[~torch.isnan(X)].std().item():.4f}, NaN count: {torch.isnan(X).sum().item()}")
                        # print(f"Before normalization y stats: mean={y[~torch.isnan(y)].mean().item():.4f}, std={y[~torch.isnan(y)].std().item():.4f}, NaN count: {torch.isnan(y).sum().item()}")
                        
                        X, y = self.normalizer.transform_torch(X, normalize_target=True, target=y)
                        
                        # Debug: Check outputs after normalization
                        # print(f"After normalization - X shape: {X.shape}, y shape: {y.shape}")
                        # print(f"After normalization X stats: mean={X[~torch.isnan(X)].mean().item():.4f}, std={X[~torch.isnan(X)].std().item():.4f}, NaN count: {torch.isnan(X).sum().item()}")
                        # print(f"After normalization y stats: mean={y[~torch.isnan(y)].mean().item():.4f}, std={y[~torch.isnan(y)].std().item():.4f}, NaN count: {torch.isnan(y).sum().item()}")
                        # exit()
                    else:
                        X = self.normalizer.transform_torch(X, normalize_target=False)
        else:
            # Without profiler
             # Subsample
            if self.subsample_step is not None:
                X = X[::self.subsample_step, ::self.subsample_step, :]
                y = y[::self.subsample_step, ::self.subsample_step, :]

            if self.normalizer is not None:
                # X = self.normalizer.transform(X.numpy()[np.newaxis])[0]
                # X = torch.from_numpy(X).float()
                # X = self.normalizer.transform_torch(X)
                if self.normalize_target:
                    X, y = self.normalizer.transform_torch(X, normalize_target=True, target=y)
                else:
                    X = self.normalizer.transform_torch(X, normalize_target=False)

        # Convert to (C, H, W)
        X = X.permute(2, 0, 1).contiguous()
        y = y.permute(2, 0, 1).contiguous()
        vhm0_for_batch = vhm0.permute(2, 0, 1).contiguous()  # Convert to (C, H, W) like y

        # Mask for NaNs
        mask = ~torch.isnan(y)
        if self.predict_bias:
            # Also return VHM0 for baseline metrics calculation
            return X, y, mask, vhm0_for_batch
        else:
            return X, y, mask, vhm0_for_batch

    def __len__(self):
        return len(self.index_map)
