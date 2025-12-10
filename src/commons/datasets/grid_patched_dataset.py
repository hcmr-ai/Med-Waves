import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import s3fs

class GridPatchWaveDataset(Dataset):
    FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 'wave_dir_cos', 'corrected_VHM0', 'corrected_VTM02']
    
    def __init__(self, file_paths, patch_size=(128, 128), stride=None,
                 target_column="corrected_VHM0", excluded_columns=None, 
                 normalizer=None, subsample_step=None, predict_bias=False,
                 use_cache=True, normalize_target=False, wave_bins=[1, 2, 3, 4, 5],
                 min_valid_pixels=0.3, fs=None):  # Minimum fraction of valid (non-NaN) pixels
        
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.target_column = target_column
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.subsample_step = subsample_step
        self.predict_bias = predict_bias
        self.use_cache = use_cache
        self.normalize_target = normalize_target
        self.wave_bins = wave_bins  # m thresholds between bins
        self.min_valid_pixels = min_valid_pixels  # Filter patches with too much land
        self._cache = {}
        self.features_order = self.normalizer.feature_order_ if self.normalizer is not None else self.FEATURES_ORDER
        # S3 filesystem - will be lazy-initialized per worker (not fork-safe)
        self._fs = None
        if self.normalizer is not None:
            print(f"Features order mismatch: {self.normalizer.feature_order_ != self.FEATURES_ORDER}")
            print(f"Features order: {self.normalizer.feature_order_}")
            print(f"Features order expected: {self.FEATURES_ORDER}")
        # Load one file to infer dimensions
        sample_tensor, _ = self._load_file_pt(file_paths[0])
        self.H_full, self.W_full = sample_tensor.shape[1], sample_tensor.shape[2]

        ph, pw = self.patch_size
        sh, sw = self.stride
        
        # Patches per dimension
        self.n_patches_h = (self.H_full - ph) // sh + 1
        self.n_patches_w = (self.W_full - pw) // sw + 1
        
        # Build index map (will filter later if min_valid_pixels > 0)
        self.index_map = [
            (f_idx, h, pi, pj)
            for f_idx in range(len(file_paths))
            for h in range(24)
            for pi in range(self.n_patches_h)
            for pj in range(self.n_patches_w)
        ]
        
        # Initialize wave bins
        self.patch_bins = [None] * len(self.index_map)

        print(f"Loaded GridPatchWaveDataset: {len(self.index_map)} patches (before filtering)")
    
    def compute_all_bins(self):
        """Pre-compute wave bins for all patches and filter patches with insufficient sea pixels."""
        print(f"Pre-computing wave bins for {len(self.index_map)} patches...")
        
        # Collect some sample values to understand the distribution
        sample_vhm0_values = []
        valid_indices = []  # Track which patches have sufficient sea pixels
        
        # Process file by file to avoid redundant loading
        for file_idx, path in tqdm(enumerate(self.file_paths), total=len(self.file_paths), desc="Computing wave bins"):
            # Load file once
            tensor, feature_cols = self._get_file_tensor(path)
            vhm0_col_idx = feature_cols.index("VHM0")
            
            # Process all hours in this file
            for hour_idx in range(24):
                hour_data = tensor[hour_idx]
                vhm0 = hour_data[..., vhm0_col_idx:vhm0_col_idx+1]
                
                if self.subsample_step is not None:
                    vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]
                
                # VHM0 is already in raw meters (0-15m), no denormalization needed
                
                # Compute bins for all patches in this hour using vectorized operations
                ph, pw = self.patch_size
                sh, sw = self.stride
                
                for patch_i in range(self.n_patches_h):
                    for patch_j in range(self.n_patches_w):
                        i_start, j_start = patch_i * sh, patch_j * sw
                        vhm0_patch = vhm0[i_start:i_start+ph, j_start:j_start+pw, :]
                        
                        # Find corresponding index in index_map
                        idx = (file_idx * 24 * self.n_patches_h * self.n_patches_w + 
                               hour_idx * self.n_patches_h * self.n_patches_w + 
                               patch_i * self.n_patches_w + 
                               patch_j)
                        
                        # Check if patch has sufficient valid pixels
                        patch_size = vhm0_patch.numel()
                        valid_mask = ~torch.isnan(vhm0_patch)
                        n_valid = valid_mask.sum().item()
                        valid_fraction = n_valid / patch_size if patch_size > 0 else 0
                        
                        # Filter out patches with too much land
                        if valid_fraction < self.min_valid_pixels:
                            self.patch_bins[idx] = None  # Mark for removal
                            continue
                        
                        # Keep this patch
                        valid_indices.append(idx)
                        
                        # Filter out NaN values for computing bin
                        valid_vhm0 = vhm0_patch[valid_mask]
                        
                        if len(valid_vhm0) > 0:
                            max_vhm0 = valid_vhm0.max().item()
                        else:
                            max_vhm0 = 0.0
                        
                        # Collect samples for the first file to show distribution
                        if file_idx == 0 and len(sample_vhm0_values) < 1000:
                            sample_vhm0_values.append(max_vhm0)
                        
                        self.patch_bins[idx] = self.get_wave_bin(max_vhm0)
        
        # Filter index_map and patch_bins to only include valid patches
        print(f"Filtering patches: {len(valid_indices)}/{len(self.index_map)} have >{self.min_valid_pixels*100:.0f}% valid pixels")
        valid_indices_set = set(valid_indices)
        self.index_map = [self.index_map[i] for i in valid_indices]
        self.patch_bins = [self.patch_bins[i] for i in valid_indices]
        
        # Show VHM0 distribution
        if sample_vhm0_values:
            import numpy as np
            sample_arr = np.array(sample_vhm0_values)
            print(f"VHM0 distribution (valid patches, first 1000):")
            print(f"  Min: {sample_arr.min():.2f}m, Max: {sample_arr.max():.2f}m")
            print(f"  Mean: {sample_arr.mean():.2f}m, Median: {np.median(sample_arr):.2f}m")
            print(f"  Percentiles: 25%={np.percentile(sample_arr, 25):.2f}m, "
                  f"75%={np.percentile(sample_arr, 75):.2f}m, 90%={np.percentile(sample_arr, 90):.2f}m")
        
        print(f"Completed: {len(self.index_map)} valid patches after filtering")
    
    @property
    def fs(self):
        """Lazy-initialize S3FileSystem per worker process (not fork-safe)"""
        if self._fs is None:
            self._fs = s3fs.S3FileSystem()
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
            self._cache.clear()
            self._cache[path] = (tensor, feature_cols)
        return tensor, feature_cols

    def get_wave_bin(self, max_vhm0):
        """Return a bin index for a given max VHM0 value."""
        for i, thresh in enumerate(self.wave_bins):
            if max_vhm0 < thresh:
                return i
        return len(self.wave_bins)  # last bin

    def __getitem__(self, idx):
        # Unpack indices
        file_idx, hour_idx, patch_i, patch_j = self.index_map[idx]
        path = self.file_paths[file_idx]

        # Load full image
        tensor, feature_cols = self._get_file_tensor(path)
        hour_data = tensor[hour_idx]

        # Select input features
        input_col_indices = [
            feature_cols.index(feat) for feat in self.features_order
            if feat in feature_cols and feat not in self.excluded_columns and feat != self.target_column
        ]

        X = hour_data[..., input_col_indices]
        vhm0 = hour_data[..., feature_cols.index("VHM0"):feature_cols.index("VHM0")+1]

        if self.predict_bias:
            corrected = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]
            y = corrected - vhm0
        else:
            y = hour_data[..., feature_cols.index(self.target_column):feature_cols.index(self.target_column)+1]

        if self.subsample_step is not None:
            X = X[::self.subsample_step, ::self.subsample_step, :]
            y = y[::self.subsample_step, ::self.subsample_step, :]
            vhm0 = vhm0[::self.subsample_step, ::self.subsample_step, :]

        # Extract patch
        ph, pw = self.patch_size
        sh, sw = self.stride
        i_start, j_start = patch_i * sh, patch_j * sw
        X = X[i_start:i_start+ph, j_start:j_start+pw, :]
        y = y[i_start:i_start+ph, j_start:j_start+pw, :]
        vhm0 = vhm0[i_start:i_start+ph, j_start:j_start+pw, :]

        # Save wave bin on first access
        if self.patch_bins[idx] is None:
            self.patch_bins[idx] = self.get_wave_bin(vhm0.max().item())

        if self.normalizer is not None:
            if self.normalize_target:
                X, y = self.normalizer.transform_torch(X, normalize_target=True, target=y)
            else:
                X = self.normalizer.transform_torch(X, normalize_target=False)

        X = X.permute(2, 0, 1).contiguous()
        y = y.permute(2, 0, 1).contiguous()
        vhm0 = vhm0.permute(2, 0, 1).contiguous()
        mask = ~torch.isnan(y)

        return X, y, mask, vhm0, self.patch_bins[idx]  # Return bin!

    def __len__(self):
        return len(self.index_map)
