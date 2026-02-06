import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


@dataclass
class PatchSamplingConfig:
    patch_size: Tuple[int, int] = (128, 256)   # (H, W) on the grid stored in the pt file
    max_tries: int = 30                        # for stratified sampling
    score: str = "p90"                         # "p90" | "max" | "mean"
    # For stratified patch sampling (grouped bins is better than 1m bins for sampling):
    # Example groups: 0-2, 2-4, 4-6, 6+
    bin_edges_m: Tuple[float, ...] = (2.0, 4.0, 6.0)
    min_valid_fraction: float = 0.3            # fraction of non-NaN pixels required in patch


class TimestepPatchWaveDataset(Dataset):
    """
    One item = (file_idx, hour_idx) -> returns one sampled patch.

    Expects each .pt to store:
        data["tensor"]      shape: (24, H, W, C)
        data["feature_cols"] list[str] length C

    Works for either full-res OR already-subsampled files:
        - pass file_paths_full or file_paths_subsampled
        - no subsampling is applied in __getitem__
    """

    def __init__(
        self,
        file_paths: List[str],
        target_columns: Optional[Dict[str, str]] = None,
        excluded_columns: Optional[List[str]] = None,
        normalizer=None,
        normalize_target: bool = False,
        predict_bias: bool = False,
        predict_log_correction: bool = False,  # recommended for bin issues
        eps: float = 1e-3,
        patch_cfg: PatchSamplingConfig = PatchSamplingConfig(),
        sampling_mode: str = "random",         # "random" | "stratified"
        # If stratified, you can pass a fixed bin_id to force a regime (useful later with a sampler)
        forced_bin_id: Optional[int] = None,
        use_cache: bool = True,
        max_cache_files: int = 2,
        features_order: Optional[List[str]] = None,
    ):
        self.file_paths = file_paths
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.normalize_target = normalize_target
        self.predict_bias = predict_bias
        self.predict_log_correction = predict_log_correction
        self.eps = eps

        self.patch_cfg = patch_cfg
        self.sampling_mode = sampling_mode
        self.forced_bin_id = forced_bin_id

        # Multi-task support
        self.target_columns = target_columns or {"vhm0": "corrected_VHM0"}
        self.is_multi_task = len(self.target_columns) > 1

        # Feature ordering
        self.features_order = (
            features_order
            if features_order is not None
            else (self.normalizer.feature_order_ if self.normalizer is not None else None)
        )

        # Simple small file cache (local NVMe)
        self.use_cache = use_cache
        self.max_cache_files = max_cache_files
        self._cache = {}          # path -> (tensor, feature_cols)
        self._cache_order = []    # LRU

        # Infer dims from one file
        tensor, feature_cols = self._get_file_tensor(self.file_paths[0])
        self.H, self.W = tensor.shape[1], tensor.shape[2]
        self.feature_cols = feature_cols  # not guaranteed identical across files, but typically is

        ph, pw = self.patch_cfg.patch_size
        if ph > self.H or pw > self.W:
            raise ValueError(
                f"Patch size {self.patch_cfg.patch_size} bigger than grid {(self.H, self.W)}"
            )

        # Build index over (file_idx, hour_idx)
        self.index_map = [(f_idx, h) for f_idx in range(len(self.file_paths)) for h in range(24)]

    def __len__(self):
        return len(self.index_map)

    # ---------------- IO ----------------
    def _load_file_pt(self, path: str):
        data = torch.load(path, map_location="cpu")
        return data["tensor"], data["feature_cols"]

    def _get_file_tensor(self, path: str):
        if not self.use_cache:
            return self._load_file_pt(path)

        if path in self._cache:
            return self._cache[path]

        tensor, feature_cols = self._load_file_pt(path)

        # LRU cache
        self._cache[path] = (tensor, feature_cols)
        self._cache_order.append(path)
        if len(self._cache_order) > self.max_cache_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return tensor, feature_cols

    # ---------------- sampling utils ----------------
    def _patch_score(self, vhm0_patch: torch.Tensor) -> float:
        # vhm0_patch: (ph, pw, 1)
        valid = ~torch.isnan(vhm0_patch)
        if valid.sum().item() == 0:
            return float("nan")
        vals = vhm0_patch[valid].flatten()

        if self.patch_cfg.score == "max":
            return float(vals.max().item())
        if self.patch_cfg.score == "mean":
            return float(vals.mean().item())

        # default p90
        k = max(1, int(math.ceil(0.90 * vals.numel())))
        return float(vals.kthvalue(k).values.item())

    def _bin_id(self, score_m: float) -> int:
        # bin_edges_m defines grouped bins (recommended)
        for i, edge in enumerate(self.patch_cfg.bin_edges_m):
            if score_m < edge:
                return i
        return len(self.patch_cfg.bin_edges_m)

    def _valid_fraction(self, patch: torch.Tensor) -> float:
        # patch: (ph, pw, 1)
        valid = ~torch.isnan(patch)
        return float(valid.sum().item() / patch.numel())

    def _sample_patch_coords(self, vhm0_full: torch.Tensor) -> Tuple[int, int, int]:
        """
        Returns (i_start, j_start, bin_id).
        vhm0_full is (H, W, 1) in meters (raw).
        """
        ph, pw = self.patch_cfg.patch_size
        max_i = self.H - ph
        max_j = self.W - pw

        # Random mode
        if self.sampling_mode == "random":
            i = random.randint(0, max_i)
            j = random.randint(0, max_j)
            patch = vhm0_full[i:i+ph, j:j+pw, :]
            score = self._patch_score(patch)
            return i, j, self._bin_id(score) if not math.isnan(score) else 0

        # Stratified mode: try to hit desired bin
        target_bin = self.forced_bin_id  # may be None

        best_i, best_j, best_bin = 0, 0, 0
        best_score = -1.0

        for _ in range(self.patch_cfg.max_tries):
            i = random.randint(0, max_i)
            j = random.randint(0, max_j)
            patch = vhm0_full[i:i+ph, j:j+pw, :]

            if self._valid_fraction(patch) < self.patch_cfg.min_valid_fraction:
                continue

            score = self._patch_score(patch)
            if math.isnan(score):
                continue

            b = self._bin_id(score)

            # If no specific target bin requested, prefer higher-sea-state patches a bit
            if target_bin is None:
                if score > best_score:
                    best_score, best_i, best_j, best_bin = score, i, j, b
                continue

            # If target bin requested, accept immediately when found
            if b == target_bin:
                return i, j, b

            # Otherwise keep best fallback
            if score > best_score:
                best_score, best_i, best_j, best_bin = score, i, j, b

        return best_i, best_j, best_bin

    # ---------------- main ----------------
    def __getitem__(self, idx: int):
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]

        tensor, feature_cols = self._get_file_tensor(path)
        hour_data = tensor[hour_idx]  # (H, W, C)

        # Determine input columns: take features_order if provided, else take all minus excluded/targets
        target_colnames = list(self.target_columns.values())

        if self.features_order is not None:
            input_col_indices = [
                feature_cols.index(feat) for feat in self.features_order
                if feat in feature_cols and feat not in self.excluded_columns and feat not in target_colnames
            ]
        else:
            input_col_indices = [
                i for i, name in enumerate(feature_cols)
                if name not in self.excluded_columns and name not in target_colnames
            ]

        X_full = hour_data[..., input_col_indices]  # (H,W,Cin)

        vhm0_idx = feature_cols.index("VHM0")
        vhm0_full = hour_data[..., vhm0_idx:vhm0_idx+1]  # (H,W,1) raw meters

        # Sample patch coordinates (optionally stratified)
        i0, j0, patch_bin = self._sample_patch_coords(vhm0_full)

        ph, pw = self.patch_cfg.patch_size
        X = X_full[i0:i0+ph, j0:j0+pw, :]
        vhm0 = vhm0_full[i0:i0+ph, j0:j0+pw, :]

        # Build targets
        targets = {}
        for task_name, tgt_col in self.target_columns.items():
            tgt_idx = feature_cols.index(tgt_col)
            y_full = hour_data[..., tgt_idx:tgt_idx+1]
            y = y_full[i0:i0+ph, j0:j0+pw, :]

            if self.predict_log_correction:
                # z = log(DA+eps) - log(raw+eps)
                # Here y is "corrected/DA", vhm0 is raw
                y = torch.log(y + self.eps) - torch.log(vhm0 + self.eps)
            elif self.predict_bias:
                y = y - vhm0

            targets[task_name] = y

        # Normalize inputs (and optionally targets)
        if self.normalizer is not None:
            X = self.normalizer.transform_torch(X, normalize_target=False)

            if self.normalize_target:
                # Safer pattern: normalize target with precomputed stats externally if possible.
                # Here we keep it simple: normalize target using its own stats if normalizer supports it.
                for task_name, y in targets.items():
                    _, y_norm = self.normalizer.transform_torch(
                        X.clone(), normalize_target=True, target=y
                    )
                    targets[task_name] = y_norm

        # Convert to (C,H,W)
        X = X.permute(2, 0, 1).contiguous()
        vhm0 = vhm0.permute(2, 0, 1).contiguous()
        for k in targets:
            targets[k] = targets[k].permute(2, 0, 1).contiguous()

        # Mask from first target (assumes NaNs mark invalid/land)
        first_task = next(iter(targets.keys()))
        mask = ~torch.isnan(targets[first_task])

        if not self.is_multi_task:
            y = targets[next(iter(self.target_columns.keys()))]
            return X, y, mask, vhm0, patch_bin
        else:
            return X, targets, mask, vhm0, patch_bin


if __name__ == "__main__":
    # full-res training
    full_pt_paths = [f"/opt/dlami/nvme/preprocessed/WAVEAN{year}_h{hour:03d}.pt" for year in [2019, 2020, 2021] for hour in range(24)]
    ds_full = TimestepPatchWaveDataset(
        file_paths=full_pt_paths,
        patch_cfg=PatchSamplingConfig(patch_size=(128, 256)),
        sampling_mode="stratified",          # important for bins
        predict_log_correction=True,         # strongly recommended
    )
    print(ds_full[0])