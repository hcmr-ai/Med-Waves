from torch.utils.data import Dataset
import random
import numpy as np
import torch
import pyarrow.parquet as pq
import s3fs

class CachedWaveDataset(Dataset):
    """
    Cached wave dataset with multi-task learning support.

    Args:
        file_paths: List of file paths to load
        target_columns: Dict mapping task names to column names.
                       Single task: {'vhm0': 'corrected_VHM0'}
                       Multi-task: {'vhm0': 'corrected_VHM0', 'vtm02': 'corrected_VTM02'}
                       Default: {'vhm0': 'corrected_VHM0'}
        excluded_columns: Columns to exclude from input
        normalizer: WaveNormalizer instance
        patch_size: Tuple (H, W) for random crops
        subsample_step: Subsample step size
        predict_bias: If True, predict bias (corrected - uncorrected)
        enable_profiler: Enable profiling
        use_cache: Enable file caching
        normalize_target: Normalize target values
        fs: S3 filesystem instance
        max_cache_size: Maximum number of files to cache

    Returns:
        Single task: X, y_tensor, mask, vhm0
        Multi-task: X, targets_dict, mask, vhm0
    """
    FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos', 'corrected_VHM0', 'corrected_VTM02']
    def __init__(self, file_paths,
                 target_columns = None
,
                 excluded_columns=None, normalizer=None,
                 patch_size=None, subsample_step=None, predict_bias=False,
                 enable_profiler=False, use_cache=True, normalize_target=False, fs=None,
                 max_cache_size=20):
        if target_columns is None:
            target_columns = {"vhm0": "corrected_VHM0"}
        self.file_paths = file_paths

        # Default to single task 'vhm0' if not provided
        self.target_columns = target_columns

        # Determine if multi-task
        self.is_multi_task = len(self.target_columns) > 1

        # For backward compatibility and single-task scenarios
        self.target_column = list(self.target_columns.values())[0]
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.patch_size = patch_size
        self.subsample_step = subsample_step
        self.predict_bias = predict_bias
        self.enable_profiler = enable_profiler
        # self.index_map = [
        #     (f_idx, h) for f_idx in range(len(file_paths)) for h in range(24)
        # ]
        self.H, self.W = 380, 1307
        self.C_in = len(self.excluded_columns) + 1  # +1 for target column
        # worker-local cache with LRU eviction
        from collections import OrderedDict
        self._cache = OrderedDict()
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size  # Number of files to keep in cache
        self.normalize_target = normalize_target
        self.features_order = self.normalizer.feature_order_ if self.normalizer is not None else self.FEATURES_ORDER
        # S3 filesystem - will be lazy-initialized per worker (not fork-safe)
        self._fs = None

         # Auto-detect file type (hourly vs daily)
        if len(file_paths) > 0:
            print(f"Detecting file format from: {file_paths[0]}")
            sample_tensor, _ = self._load_file_pt(file_paths[0])
            self.is_hourly = (sample_tensor.ndim == 3)  # 3D=hourly, 4D=daily
            print(f"  Tensor shape: {sample_tensor.shape}")
            print(f"  File type: {'HOURLY' if self.is_hourly else 'DAILY'}")
            self._fs = None  # Reset after sample load
            self._cache.clear()
        else:
            self.is_hourly = False

        # Create index_map based on file type
        if self.is_hourly:
            # Hourly files: one sample per file
            self.index_map = [(f_idx, 0) for f_idx in range(len(file_paths))]
            print(f"  Index map: {len(self.index_map)} samples (1 per file)")
        else:
            # Daily files: 24 samples per file
            self.index_map = [
                (f_idx, h) for f_idx in range(len(file_paths)) for h in range(24)
            ]
            print(f"  Index map: {len(self.index_map)} samples (24 per file)")

        if self.normalizer is not None:
            print(f"Features order mismatch: {self.normalizer.feature_order_ != self.FEATURES_ORDER}")
            print(f"Features order: {self.normalizer.feature_order_}")
            print(f"Features order expected: {self.FEATURES_ORDER}")

            print("\n=== NORMALIZER DEBUG ===")
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
            print("======================\n")

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
            data = torch.load(path, map_location="cpu", weights_only=False)
        return data["tensor"], data["feature_cols"]

    def _get_file_tensor(self, path):
        if self.use_cache:
            # Check if file is in cache
            if path in self._cache:
                # Move to end (mark as recently used for LRU)
                self._cache.move_to_end(path)
                return self._cache[path]
            else:
                # Load new file
                tensor, feature_cols = self._load_file_pt(path)

                # Add to cache
                self._cache[path] = (tensor, feature_cols)

                # Evict oldest if cache is full (LRU eviction)
                if len(self._cache) > self.max_cache_size:
                    # Remove oldest (first) item
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                return tensor, feature_cols
        else:
            # No caching - reload every time
            tensor, feature_cols = self._load_file_pt(path)
            return tensor, feature_cols

    def __getitem__(self, idx):
        # if idx < 5:  # Only log first few calls to avoid spam
        #     print(f"[Worker PID {os.getpid()}] __getitem__ called with idx={idx}")
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]
        # if idx < 5:
        #     print(f"[Worker PID {os.getpid()}] Loading file: {os.path.basename(path)}")

        tensor, feature_cols = self._get_file_tensor(path)

         # Auto-detect file type (hourly vs daily)
        if self.is_hourly:
            hour_data = tensor  # Already single hour: (H, W, C)
        else:
            hour_data = tensor[hour_idx]

        # Select input features in FEATURES_ORDER to match scaler's stats_ indices
        # This ensures stats_[c] applies to channel c in X
        # Exclude ALL target columns from input
        all_target_columns = list(self.target_columns.values())
        input_features = []
        input_col_indices = []

        for feat in self.features_order:
            if feat in self.excluded_columns or feat in all_target_columns:
                continue

            # Find feature in feature_cols
            if feat in feature_cols:
                idx_in_feature_cols = feature_cols.index(feat)
                input_features.append(feat)
                input_col_indices.append(idx_in_feature_cols)
        X = hour_data[..., input_col_indices]  # (H, W, C_in)

        # Extract uncorrected version for reconstruction (use first target's uncorrected version)
        # e.g., "corrected_VHM0" -> "VHM0", "corrected_VTM02" -> "VTM02"
        uncorrected_column = self.target_column.replace("corrected_", "")
        vhm0 = hour_data[..., feature_cols.index(uncorrected_column):feature_cols.index(uncorrected_column)+1]

        # Extract targets for each task
        targets = {}
        for task_name, target_col in self.target_columns.items():
            if self.predict_bias:
                # Predict bias: corrected - uncorrected
                corrected = hour_data[..., feature_cols.index(target_col):feature_cols.index(target_col)+1]
                uncorr_col = target_col.replace("corrected_", "")
                uncorr = hour_data[..., feature_cols.index(uncorr_col):feature_cols.index(uncorr_col)+1]
                targets[task_name] = corrected - uncorr
            else:
                # Predict corrected value directly
                targets[task_name] = hour_data[..., feature_cols.index(target_col):feature_cols.index(target_col)+1]

        # Use first task's mask (all tasks should have same valid pixels)
        first_target = targets[list(self.target_columns.keys())[0]]
        mask = ~torch.isnan(first_target)

        # Subsample
        if self.subsample_step is not None:
            X = X[::self.subsample_step, ::self.subsample_step, :]
            vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]
            for task_name in targets:
                targets[task_name] = targets[task_name][::self.subsample_step, ::self.subsample_step, :]

         # Patch sampling
        if self.patch_size is not None:
            H, W, _ = X.shape
            ph, pw = self.patch_size
            if H > ph and W > pw:
                i = random.randint(0, H - ph)
                j = random.randint(0, W - pw)
                X = X[i:i+ph, j:j+pw, :]
                vhm0 = vhm0[i:i+ph, j:j+pw, :]
                mask = mask[i:i+ph, j:j+pw, :]
                for task_name in targets:
                    targets[task_name] = targets[task_name][i:i+ph, j:j+pw, :]

        if self.normalizer is not None:
            if self.normalize_target:
                # Normalize X once
                X = self.normalizer.transform_torch(X, normalize_target=False)

                # Normalize all targets individually with their respective stats
                for task_name, target_tensor in targets.items():
                    target_col = self.target_columns[task_name]

                    # Set target_stats_ for this specific task
                    if target_col in self.normalizer.feature_order_:
                        target_idx = self.normalizer.feature_order_.index(target_col)
                        if target_idx in self.normalizer.stats_:
                            self.normalizer.target_stats_ = self.normalizer.stats_[target_idx]

                    # Normalize this target with its own stats
                    _, normalized_target = self.normalizer.transform_torch(
                        X.clone(), normalize_target=True, target=target_tensor
                    )
                    targets[task_name] = normalized_target
            else:
                X = self.normalizer.transform_torch(X, normalize_target=False)

        # Convert to (C, H, W)
        X = X.permute(2, 0, 1).contiguous()
        vhm0_for_batch = vhm0.permute(2, 0, 1).contiguous()  # Convert to (C, H, W)
        mask = mask.permute(2, 0, 1).contiguous()  # Convert mask to (C, H, W)
        X = torch.nan_to_num(X, nan=0.0)
        vhm0_for_batch = torch.nan_to_num(vhm0_for_batch, nan=0.0)

        # Convert all targets to (C, H, W)
        for task_name in targets:
            targets[task_name] = targets[task_name].permute(2, 0, 1).contiguous()

        # Backward compatibility: return tensor for single-task, dict for multi-task
        if not self.is_multi_task:
            # Extract single target using actual task name
            task_name = list(self.target_columns.keys())[0]
            y = targets[task_name]
            return X, y, mask, vhm0_for_batch
        else:
            return X, targets, mask, vhm0_for_batch

    def __len__(self):
        return len(self.index_map)
