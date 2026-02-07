import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PatchSamplingConfig:
    patch_size: Tuple[int, int] = (32, 96)        # multiples of 16 are best for your TransformerBranch
    max_tries: int = 50                            # more tries helps coastal domains
    score: str = "p90"                             # "p90" | "max" | "mean"
    bin_edges_m: Tuple[float, ...] = (2.0, 4.0)    # grouped bins for sampling (3 bins total)
    min_valid_fraction: float = 0.6                # coastal domains: require enough sea pixels
    precompute_valid_anchors: bool = True          # avoids wasting tries on land-heavy patches


class TimestepPatchWaveDataset(Dataset):
    """
    Patch-per-timestep dataset (Option A):

    - One item corresponds to a (file_idx, hour_idx).
    - Returns ONE sampled patch from that hour.
    - Works for both full-res and subsampled .pt files (no on-the-fly subsampling).
    - Fills NaNs in inputs to prevent NaN propagation through Conv/Transformer.
    - Keeps NaNs in targets and provides a mask for loss masking.
    - Optionally appends a sea_mask channel to inputs.

    Expected .pt file format:
        data["tensor"]      shape (24, H, W, C)
        data["feature_cols"] list[str] length C

    Returns:
        Single task:
            X:        (Cin(+1 if sea_mask), ph, pw)
            y:        (1, ph, pw)
            mask:     (1, ph, pw) bool
            vhm0:     (1, ph, pw) raw VHM0 with NaNs filled (for reconstruction)
            patch_bin: int
            coords:   (i0, j0) top-left of patch
        Multi-task:
            X, targets_dict, mask, vhm0, patch_bin, coords
    """

    def __init__(
        self,
        file_paths: List[str],
        target_columns: Optional[Dict[str, str]] = None,
        excluded_columns: Optional[List[str]] = None,
        normalizer=None,
        normalize_target: bool = False,
        predict_bias: bool = False,
        predict_log_correction: bool = True,  # best default for bin issues
        eps: float = 1e-3,
        patch_cfg: PatchSamplingConfig = None,
        sampling_mode: str = "random",        # "random" | "stratified"
        # If None and sampling_mode="stratified", we round-robin target bins via idx % n_bins
        forced_bin_id: Optional[int] = None,
        use_cache: bool = True,
        max_cache_files: int = 2,
        features_order: Optional[List[str]] = None,
        add_sea_mask_channel: bool = True,    # recommended
        seed: Optional[int] = None,
        return_coords: bool = True,
    ):
        self.file_paths = file_paths
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.normalize_target = normalize_target

        self.predict_bias = predict_bias
        self.predict_log_correction = predict_log_correction
        if predict_bias and predict_log_correction:
            raise ValueError("Choose only one of predict_bias or predict_log_correction.")
        self.eps = eps

        self.patch_cfg = patch_cfg or PatchSamplingConfig()
        self.sampling_mode = sampling_mode
        self.forced_bin_id = forced_bin_id

        self.add_sea_mask_channel = add_sea_mask_channel
        self.return_coords = return_coords

        # Multi-task support
        self.target_columns = target_columns or {"vhm0": "corrected_VHM0"}
        self.is_multi_task = len(self.target_columns) > 1

        # Feature ordering
        self.features_order = (
            features_order
            if features_order is not None
            else (self.normalizer.feature_order_ if self.normalizer is not None else None)
        )

        # Cache (local NVMe)
        self.use_cache = use_cache
        self.max_cache_files = max_cache_files
        self._cache = {}        # path -> (tensor, feature_cols)
        self._cache_order = []  # LRU order

        # RNG
        self._rng = random.Random(seed)

        # Infer dims from one file
        tensor, feature_cols = self._get_file_tensor(self.file_paths[0])
        self.H, self.W = tensor.shape[1], tensor.shape[2]
        self.feature_cols_ref = feature_cols

        ph, pw = self.patch_cfg.patch_size
        if ph > self.H or pw > self.W:
            raise ValueError(f"Patch size {self.patch_cfg.patch_size} bigger than grid {(self.H, self.W)}")
        if ph < 32 or pw < 64:
            # because you downsample by 16x in UNet path; tiny patches create 1x? bottlenecks
            print(f"[WARN] Very small patch {self.patch_cfg.patch_size}; consider >= (32, 64).")

        # Build index over (file_idx, hour_idx)
        self.index_map = [(f_idx, h) for f_idx in range(len(self.file_paths)) for h in range(24)]

        # Precompute valid anchors once from a sample frame (land mask is static across time)
        self.valid_anchors: Optional[List[Tuple[int, int]]] = None
        if self.patch_cfg.precompute_valid_anchors:
            self.valid_anchors = self._precompute_valid_anchors()

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

    # ---------------- anchors ----------------
    def _precompute_valid_anchors(self) -> List[Tuple[int, int]]:
        """
        Precompute top-left (i,j) anchors where patch has enough sea pixels.
        Uses VHM0 NaN mask from the first file, hour 0.
        """
        tensor, feature_cols = self._get_file_tensor(self.file_paths[0])
        hour0 = tensor[0]  # (H,W,C)
        vhm0_idx = feature_cols.index("VHM0")
        vhm0 = hour0[..., vhm0_idx:vhm0_idx+1]  # (H,W,1)

        sea = (~torch.isnan(vhm0)).float()  # (H,W,1) 1 sea, 0 land

        ph, pw = self.patch_cfg.patch_size
        max_i = self.H - ph
        max_j = self.W - pw

        anchors: List[Tuple[int, int]] = []
        # Simple scan (fast enough for your sizes)
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                patch = sea[i:i+ph, j:j+pw, :]
                sea_frac = float(patch.mean().item())
                if sea_frac >= self.patch_cfg.min_valid_fraction:
                    anchors.append((i, j))

        if not anchors:
            raise RuntimeError(
                "No valid anchors found. Lower min_valid_fraction or reduce patch_size."
            )

        print(f"[TimestepPatchWaveDataset] Precomputed {len(anchors)} valid anchors "
              f"(min_valid_fraction={self.patch_cfg.min_valid_fraction}).")
        return anchors

    # ---------------- sampling utils ----------------
    def _patch_score(self, vhm0_patch: torch.Tensor) -> float:
        """
        vhm0_patch: (ph, pw, 1) raw meters with NaNs over land.
        """
        valid = ~torch.isnan(vhm0_patch)
        if valid.sum().item() == 0:
            return float("nan")
        vals = vhm0_patch[valid].flatten()

        mode = self.patch_cfg.score
        if mode == "max":
            return float(vals.max().item())
        if mode == "mean":
            return float(vals.mean().item())

        # default p90 (approx via kthvalue)
        k = max(1, int(math.ceil(0.90 * vals.numel())))
        return float(vals.kthvalue(k).values.item())

    def _bin_id(self, score_m: float) -> int:
        for i, edge in enumerate(self.patch_cfg.bin_edges_m):
            if score_m < edge:
                return i
        return len(self.patch_cfg.bin_edges_m)

    def _sample_anchor(self) -> Tuple[int, int]:
        if self.valid_anchors is None:
            # fallback random top-left anywhere
            ph, pw = self.patch_cfg.patch_size
            return self._rng.randint(0, self.H - ph), self._rng.randint(0, self.W - pw)
        return self._rng.choice(self.valid_anchors)

    def _sample_patch_coords(self, vhm0_full: torch.Tensor, idx: int) -> Tuple[int, int, int]:
        """
        Returns (i_start, j_start, patch_bin).
        If stratified and no forced_bin_id, uses round-robin bin targets based on idx.
        """
        ph, pw = self.patch_cfg.patch_size

        # Decide target bin
        n_bins = len(self.patch_cfg.bin_edges_m) + 1
        if self.sampling_mode == "stratified":
            target_bin = self.forced_bin_id
            if target_bin is None:
                target_bin = idx % n_bins
        else:
            target_bin = None

        best_i, best_j, best_bin = 0, 0, 0
        best_score = -1.0

        # Random sampling without stratification
        if target_bin is None and self.sampling_mode == "random":
            i0, j0 = self._sample_anchor()
            patch = vhm0_full[i0:i0+ph, j0:j0+pw, :]
            score = self._patch_score(patch)
            b = self._bin_id(score) if not math.isnan(score) else 0
            return i0, j0, b

        # Otherwise try to match target bin
        for _ in range(self.patch_cfg.max_tries):
            i0, j0 = self._sample_anchor()
            patch = vhm0_full[i0:i0+ph, j0:j0+pw, :]
            score = self._patch_score(patch)
            if math.isnan(score):
                continue
            b = self._bin_id(score)

            if b == target_bin:
                return i0, j0, b

            # best fallback (prefer higher score)
            if score > best_score:
                best_score, best_i, best_j, best_bin = score, i0, j0, b

        return best_i, best_j, best_bin

    # ---------------- main ----------------
    def __getitem__(self, idx: int):
        file_idx, hour_idx = self.index_map[idx]
        path = self.file_paths[file_idx]

        tensor, feature_cols = self._get_file_tensor(path)
        hour_data = tensor[hour_idx]  # (H, W, C)

        # Input columns: features_order if provided, else all minus excluded/targets
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
        vhm0_full = hour_data[..., vhm0_idx:vhm0_idx+1]  # (H,W,1) raw meters (NaN on land)

        # Sample patch coords
        i0, j0, patch_bin = self._sample_patch_coords(vhm0_full, idx)

        ph, pw = self.patch_cfg.patch_size
        X = X_full[i0:i0+ph, j0:j0+pw, :]
        vhm0 = vhm0_full[i0:i0+ph, j0:j0+pw, :]

        # Sea mask
        sea_mask = (~torch.isnan(vhm0)).float()  # (ph,pw,1), 1 sea, 0 land

        # Build targets (keep NaNs on land!)
        targets: Dict[str, torch.Tensor] = {}
        for task_name, tgt_col in self.target_columns.items():
            tgt_idx = feature_cols.index(tgt_col)
            y_full = hour_data[..., tgt_idx:tgt_idx+1]
            y = y_full[i0:i0+ph, j0:j0+pw, :]

            if self.predict_log_correction:
                # z = log(DA+eps) - log(raw+eps)
                y = torch.log(y + self.eps) - torch.log(vhm0 + self.eps)
            elif self.predict_bias:
                y = y - vhm0

            targets[task_name] = y

        # ----- IMPORTANT: fill NaNs in inputs only -----
        X = torch.nan_to_num(X, nan=0.0)
        vhm0_filled = torch.nan_to_num(vhm0, nan=0.0)  # useful for reconstruction / logging
        if self.add_sea_mask_channel:
            X = torch.cat([X, sea_mask], dim=-1)

        # Normalize inputs (targets remain masked by NaNs)
        if self.normalizer is not None:
            X = self.normalizer.transform_torch(X, normalize_target=False)
            # If your normalizer might introduce NaNs (shouldn't), clamp again:
            X = torch.nan_to_num(X, nan=0.0)

            if self.normalize_target:
                # Not recommended unless your normalizer supports stable per-target normalization.
                for task_name, y in targets.items():
                    _, y_norm = self.normalizer.transform_torch(X.clone(), normalize_target=True, target=y)
                    targets[task_name] = y_norm

        # Convert to (C,H,W)
        X = X.permute(2, 0, 1).contiguous()
        vhm0_filled = vhm0_filled.permute(2, 0, 1).contiguous()
        sea_mask.permute(2, 0, 1).contiguous()
        for k in targets:
            targets[k] = targets[k].permute(2, 0, 1).contiguous()

        # Mask from first target (NaNs mark invalid/land)
        first_task = next(iter(targets.keys()))
        mask = ~torch.isnan(targets[first_task])  # (1,ph,pw) bool

        if not self.is_multi_task:
            y = targets[next(iter(self.target_columns.keys()))]
            if self.return_coords:
                return X, y, mask, vhm0_filled, patch_bin, (i0, j0)
            return X, y, mask, vhm0_filled, patch_bin
        else:
            if self.return_coords:
                return X, targets, mask, vhm0_filled, patch_bin, (i0, j0)
            return X, targets, mask, vhm0_filled, patch_bin
