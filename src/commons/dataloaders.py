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
    FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos', 'corrected_VHM0', 'corrected_VTM02']
    def __init__(self, file_paths, target_column="corrected_VHM0",
                 excluded_columns=None, normalizer=None,
                 patch_size=None, subsample_step=None, predict_bias=False,
                 enable_profiler=False, use_cache=True, normalize_target=False, fs=None):
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
        self.features_order = self.normalizer.feature_order_ if self.normalizer is not None else self.FEATURES_ORDER
        # S3 filesystem - will be lazy-initialized per worker (not fork-safe)
        self._fs = None
        
        if self.normalizer is not None:
            print(f"Features order mismatch: {self.normalizer.feature_order_ != self.FEATURES_ORDER}")
            print(f"Features order: {self.normalizer.feature_order_}")
            print(f"Features order expected: {self.FEATURES_ORDER}")

            print(f"\n=== NORMALIZER DEBUG ===")
            print(f"Target column in config: {self.target_column}")
            print(f"Normalizer target_feature_name_: {self.normalizer.target_feature_name_}")
            print(f"Feature order: {self.normalizer.feature_order_}")

            if self.normalize_target and self.normalizer.feature_order_:
                try:
                    target_idx = self.normalizer.feature_order_.index(self.target_column)
                    print(f"✓ Found '{self.target_column}' at index {target_idx}")
                    
                    # Check what stats exist
                    if target_idx in self.normalizer.stats_:
                        stats = self.normalizer.stats_[target_idx]
                        if isinstance(stats, tuple):
                            mean, std = stats
                            print(f"  Stats at index {target_idx}: mean={mean:.4f}, std={std:.4f}")
                        else:
                            print(f"  Stats at index {target_idx}: {type(stats)}")
                        
                        # Actually set it
                        self.normalizer.target_stats_ = stats
                        print(f"✓ Set target_stats_ to index {target_idx}")
                    else:
                        print(f"✗ Index {target_idx} not in stats_! Available: {list(self.normalizer.stats_.keys())}")
                    
                    if hasattr(stats, 'mean_') and hasattr(stats, 'scale_'):
                        print(f"  StandardScaler mean_: {stats.mean_[0]:.4f}")
                        print(f"  StandardScaler scale_ (std): {stats.scale_[0]:.4f}")
                    else:
                        print(f"  StandardScaler attributes: {dir(stats)}")
                except (ValueError, KeyError) as e:
                    print(f"✗ Error: {e}")
            print(f"======================\n")

    @property
    def fs(self):
        """Lazy-initialize S3FileSystem per worker process (not fork-safe)"""
        if self._fs is None:
            import os
            import time
            # Stagger S3 connection creation across workers to avoid thundering herd
            worker_id = os.getpid() % 8  # Simple worker ID based on PID
            if worker_id > 0:
                time.sleep(worker_id * 0.1)  # 0-0.7s delay
            print(f"[Worker PID {os.getpid()}] Initializing S3FileSystem...")
            self._fs = s3fs.S3FileSystem()
            print(f"[Worker PID {os.getpid()}] S3FileSystem initialized ✓")
        return self._fs

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
        import os
        # Handle S3 paths
        if isinstance(path, str) and path.startswith("s3://"):
            print(f"[Worker PID {os.getpid()}] Opening S3 file: {os.path.basename(path)}")
            with self.fs.open(path, "rb") as f:
                print(f"[Worker PID {os.getpid()}] File opened, loading with torch...")
                data = torch.load(f, map_location="cpu")
            print(f"[Worker PID {os.getpid()}] Torch load complete ✓")
        else:
            data = torch.load(path, map_location="cpu")
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
        import os
        if idx < 5:  # Only log first few calls to avoid spam
            print(f"[Worker PID {os.getpid()}] __getitem__ called with idx={idx}")
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]
        if idx < 5:
            print(f"[Worker PID {os.getpid()}] Loading file: {os.path.basename(path)}")

        tensor, feature_cols = self._get_file_tensor(path)

        hour_data = tensor[hour_idx]
        # Select input cols
        input_col_indices = [
            i for i, col in enumerate(feature_cols)
            if (col not in self.excluded_columns) and (col != self.target_column)
        ]
        
        # Select input features in FEATURES_ORDER to match scaler's stats_ indices
        # This ensures stats_[c] applies to channel c in X
        input_features = []
        input_col_indices = []

        for feat in self.features_order:
            if feat in self.excluded_columns or feat == self.target_column:
                continue
            
            # Find feature in feature_cols
            if feat in feature_cols:
                idx_in_feature_cols = feature_cols.index(feat)
                input_features.append(feat)
                input_col_indices.append(idx_in_feature_cols)
        X = hour_data[..., input_col_indices]  # (H, W, C_in)

        # vhm0 = hour_data[..., feature_cols.index("VHM0"):feature_cols.index("VHM0")+1]
        # Extract uncorrected version based on target_column
        # e.g., "corrected_VHM0" -> "VHM0", "corrected_VTM02" -> "VTM02"
        uncorrected_column = self.target_column.replace("corrected_", "")
        vhm0 = hour_data[..., feature_cols.index(uncorrected_column):feature_cols.index(uncorrected_column)+1]

# Same fix for GridPatchWaveDataset (around line 481)

        if self.predict_bias:
            corrected = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
            y = corrected - vhm0
        else:
            y = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
        mask = ~torch.isnan(y)
        # Patch sampling
        if self.patch_size is not None:
            H, W, _ = X.shape
            ph, pw = self.patch_size
            if H > ph and W > pw:
                i = random.randint(0, H - ph)
                j = random.randint(0, W - pw)
                X = X[i:i+ph, j:j+pw, :]
                y = y[i:i+ph, j:j+pw, :]

        # Subsample
        if self.subsample_step is not None:
            X = X[::self.subsample_step, ::self.subsample_step, :]
            y = y[::self.subsample_step, ::self.subsample_step, :]
            vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]

        if self.normalizer is not None:
            if not hasattr(self, '_target_stats_verified'):
                if self.normalize_target and self.normalizer.target_stats_ is None:
                    print(f"⚠️ Worker: target_stats_ was None, re-setting")
                    target_idx = self.normalizer.feature_order_.index(self.target_column)
                    self.normalizer.target_stats_ = self.normalizer.stats_[target_idx]
                self._target_stats_verified = True

            # if self.normalize_target and self.normalizer.feature_order_:
            #     target_idx = self.normalizer.feature_order_.index(self.target_column)
            #     self.normalizer.target_stats_ = self.normalizer.stats_[target_idx]

            if self.normalize_target:
                X, y = self.normalizer.transform_torch(X, normalize_target=True, target=y)
            else:
                X = self.normalizer.transform_torch(X, normalize_target=False)

        # Convert to (C, H, W)
        X = X.permute(2, 0, 1).contiguous()
        y = y.permute(2, 0, 1).contiguous()
        vhm0_for_batch = vhm0.permute(2, 0, 1).contiguous()  # Convert to (C, H, W) like y
        mask = mask.permute(2, 0, 1).contiguous()  # Convert mask to (C, H, W) to match y

        # Mask for NaNs
        if self.predict_bias:
            # Also return VHM0 for baseline metrics calculation
            return X, y, mask, vhm0_for_batch
        else:
            return X, y, mask, vhm0_for_batch

    def __len__(self):
        return len(self.index_map)


class GridPatchWaveDataset(Dataset):
    """Dataset that samples patches in a systematic grid to cover the entire image."""
    
    def __init__(self, file_paths, patch_size=(128, 128), stride=None,
                 target_column="corrected_VHM0", excluded_columns=None, 
                 normalizer=None, subsample_step=None, predict_bias=False,
                 use_cache=True, normalize_target=False, fs=None):
        """
        Args:
            patch_size: (H, W) size of each patch
            stride: step size for patch extraction. If None, uses patch_size (no overlap)
                   If smaller than patch_size, creates overlapping patches
        """
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.stride = stride or patch_size  # Default: non-overlapping
        self.target_column = target_column
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.subsample_step = subsample_step
        self.predict_bias = predict_bias
        self.use_cache = use_cache
        self.normalize_target = normalize_target
        self._cache = {}
        # S3 filesystem - will be lazy-initialized per worker (not fork-safe)
        self._fs = None
        
        # Image dimensions - detect from actual stored data
        # Load a sample file to get actual dimensions (handles pre-subsampled data)
        sample_tensor, _ = self._load_file_pt(file_paths[0])
        self.H_full, self.W_full = sample_tensor.shape[1], sample_tensor.shape[2]
        print(f"Detected data dimensions: {self.H_full} x {self.W_full}")
        
        # IMPORTANT: Reset S3FileSystem after loading sample file
        # This prevents fork-safety issues when workers are created
        self._fs = None
        
        # Calculate grid dimensions
        ph, pw = self.patch_size
        sh, sw = self.stride
        
        # Number of patches along each dimension
        self.n_patches_h = (self.H_full - ph) // sh + 1
        self.n_patches_w = (self.W_full - pw) // sw + 1
        
        # Total patches per hour
        patches_per_hour = self.n_patches_h * self.n_patches_w
        
        # Index map: (file_idx, hour_idx, patch_row, patch_col)
        self.index_map = [
            (f_idx, h, pi, pj)
            for f_idx in range(len(file_paths))
            for h in range(24)
            for pi in range(self.n_patches_h)
            for pj in range(self.n_patches_w)
        ]
        
        print(f"Grid Patch Dataset Info:")
        print(f"  Full image size: {self.H_full} x {self.W_full}")
        print(f"  Patch size: {ph} x {pw}")
        print(f"  Stride: {sh} x {sw}")
        print(f"  Grid: {self.n_patches_h} x {self.n_patches_w} = {patches_per_hour} patches/hour")
        print(f"  Total samples: {len(self.index_map)}")
    
    @property
    def fs(self):
        """Lazy-initialize S3FileSystem per worker process (not fork-safe)"""
        if self._fs is None:
            import os
            import time
            # Stagger S3 connection creation across workers to avoid thundering herd
            worker_id = os.getpid() % 8  # Simple worker ID based on PID
            if worker_id > 0:
                time.sleep(worker_id * 0.1)  # 0-0.7s delay
            print(f"[Worker PID {os.getpid()}] Initializing S3FileSystem...")
            self._fs = s3fs.S3FileSystem()
            print(f"[Worker PID {os.getpid()}] S3FileSystem initialized ✓")
        return self._fs
    
    def _load_file_pt(self, path):
        # Handle S3 paths
        if isinstance(path, str) and path.startswith("s3://"):
            with self.fs.open(path, "rb") as f:
                data = torch.load(f, map_location="cpu")
        else:
            data = torch.load(path, map_location="cpu")
        return data["tensor"], data["feature_cols"]
    
    def _get_file_tensor(self, path):
        if self.use_cache and path in self._cache:
            return self._cache[path]
        
        tensor, feature_cols = self._load_file_pt(path)
        if self.use_cache:
            self._cache.clear()  # Keep only 1 file in memory
            self._cache[path] = (tensor, feature_cols)
        return tensor, feature_cols
    
    def __getitem__(self, idx):
        file_idx, hour_idx, patch_i, patch_j = self.index_map[idx]
        path = self.file_paths[file_idx]
        
        # Load full day tensor
        tensor, feature_cols = self._get_file_tensor(path)
        hour_data = tensor[hour_idx]  # (H, W, C)
        
        # Select input features
        FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 
                         'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 
                         'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 
                         'wave_dir_cos', 'corrected_VHM0']
        
        input_col_indices = []
        for feat in FEATURES_ORDER:
            if feat in self.excluded_columns or feat == self.target_column:
                continue
            if feat in feature_cols:
                input_col_indices.append(feature_cols.index(feat))
        
        X = hour_data[..., input_col_indices]
        vhm0 = hour_data[..., feature_cols.index("VHM0"):feature_cols.index("VHM0")+1]
        
        # Get target
        if self.predict_bias:
            corrected = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
            y = corrected - vhm0
        else:
            y = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
        
        # Apply subsampling BEFORE patch extraction
        if self.subsample_step is not None:
            X = X[::self.subsample_step, ::self.subsample_step, :]
            y = y[::self.subsample_step, ::self.subsample_step, :]
            vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]
        
        # Extract patch at grid location
        ph, pw = self.patch_size
        sh, sw = self.stride
        i_start = patch_i * sh
        j_start = patch_j * sw
        
        X = X[i_start:i_start+ph, j_start:j_start+pw, :]
        y = y[i_start:i_start+ph, j_start:j_start+pw, :]
        vhm0 = vhm0[i_start:i_start+ph, j_start:j_start+pw, :]
        
        # Normalize
        if self.normalizer is not None:
            if self.normalize_target:
                X, y = self.normalizer.transform_torch(X, normalize_target=True, target=y)
            else:
                X = self.normalizer.transform_torch(X, normalize_target=False)
        
        # Convert to (C, H, W)
        X = X.permute(2, 0, 1).contiguous()
        y = y.permute(2, 0, 1).contiguous()
        vhm0 = vhm0.permute(2, 0, 1).contiguous()
        mask = ~torch.isnan(y).contiguous()  # y is already (C, H, W), no need to permute
        
        return X, y, mask, vhm0
    
    def __len__(self):
        return len(self.index_map)