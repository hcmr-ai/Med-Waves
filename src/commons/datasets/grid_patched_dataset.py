import torch
from torch.utils.data import Dataset
import numpy as np

class GridPatchWaveDataset(Dataset):
    def __init__(self, file_paths, patch_size=(128, 128), stride=None,
                 target_column="corrected_VHM0", excluded_columns=None, 
                 normalizer=None, subsample_step=None, predict_bias=False,
                 use_cache=True, normalize_target=False, wave_bins=[1, 2, 3, 4, 5]):
        
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
        self._cache = {}

        # Load one file to infer dimensions
        sample_tensor, _ = self._load_file_pt(file_paths[0])
        self.H_full, self.W_full = sample_tensor.shape[1], sample_tensor.shape[2]

        ph, pw = self.patch_size
        sh, sw = self.stride
        
        # Patches per dimension
        self.n_patches_h = (self.H_full - ph) // sh + 1
        self.n_patches_w = (self.W_full - pw) // sw + 1
        
        # Build index map
        self.index_map = [
            (f_idx, h, pi, pj)
            for f_idx in range(len(file_paths))
            for h in range(24)
            for pi in range(self.n_patches_h)
            for pj in range(self.n_patches_w)
        ]
        
        # Initialize wave bins
        self.patch_bins = [None] * len(self.index_map)

        print(f"Loaded GridPatchWaveDataset: {len(self.index_map)} patches")

    def _load_file_pt(self, path):
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
        FEATURES_ORDER = ['VHM0', 'WSPD', 'VTM02', 'U10', 'V10', 'sin_hour', 
                         'cos_hour', 'sin_doy', 'cos_doy', 'sin_month', 
                         'cos_month', 'lat_norm', 'lon_norm', 'wave_dir_sin', 
                         'wave_dir_cos', 'corrected_VHM0']

        input_col_indices = [
            feature_cols.index(feat) for feat in FEATURES_ORDER
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
