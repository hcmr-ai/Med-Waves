import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PatchSamplingConfig:
    patch_size: Tuple[int, int] = (
        32,
        96,
    )  # multiples of 16 are best for your TransformerBranch
    max_tries: int = 50  # more tries helps coastal domains
    score: str = "p90"  # "p90" | "max" | "mean"
    bin_edges_m: Tuple[float, ...] = (
        2.0,
        4.0,
    )  # grouped bins for sampling (3 bins total)
    min_valid_fraction: float = 0.6  # coastal domains: require enough sea pixels
    precompute_valid_anchors: bool = True  # avoids wasting tries on land-heavy patches


class TimestepPatchWaveDataset(Dataset):
    """
    Patch-based dataset with multiple sampling strategies.

    Sampling modes:
        "random"     - One random patch per (file, hour). Natural bin distribution.
        "stratified" - One patch per (file, hour), targeted to a specific wave-height bin.
        "exhaustive" - All non-overlapping tiles per (file, hour). Full spatial coverage.

    Works for both full-res and subsampled .pt files (no on-the-fly subsampling).
    Fills NaNs in inputs to prevent NaN propagation through Conv/Transformer.
    Keeps NaNs in targets and provides a mask for loss masking.
    Optionally appends a sea_mask channel to inputs.

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
        sampling_mode: str = "random",  # "random" | "stratified" | "exhaustive"
        # If None and sampling_mode="stratified", we round-robin target bins via idx % n_bins
        forced_bin_id: Optional[int] = None,
        use_cache: bool = True,
        max_cache_files: int = 2,
        features_order: Optional[List[str]] = None,
        add_sea_mask_channel: bool = True,  # recommended
        seed: Optional[int] = None,
        return_coords: bool = True,
        region_filter: Optional[
            str
        ] = None,  # "atlantic", "mediterranean", "eastern_med", or None
    ):
        self.file_paths = file_paths
        self.excluded_columns = excluded_columns or []
        self.normalizer = normalizer
        self.normalize_target = normalize_target

        self.predict_bias = predict_bias
        self.predict_log_correction = predict_log_correction
        if predict_bias and predict_log_correction:
            raise ValueError(
                "Choose only one of predict_bias or predict_log_correction."
            )
        self.eps = eps

        self.patch_cfg = patch_cfg or PatchSamplingConfig()
        self.sampling_mode = sampling_mode
        self.forced_bin_id = forced_bin_id

        self.add_sea_mask_channel = add_sea_mask_channel
        self.return_coords = return_coords
        self.region_filter = region_filter

        # Multi-task support
        self.target_columns = target_columns or {"vhm0": "corrected_VHM0"}
        self.is_multi_task = len(self.target_columns) > 1

        # Feature ordering
        self.features_order = (
            features_order
            if features_order is not None
            else (
                self.normalizer.feature_order_ if self.normalizer is not None else None
            )
        )

        # Cache (local NVMe)
        self.use_cache = use_cache
        self.max_cache_files = max_cache_files
        self._cache = {}  # path -> (tensor, feature_cols)
        self._cache_order = []  # LRU order

        # RNG
        self._rng = random.Random(seed)

        # Infer dims from one file
        tensor, feature_cols = self._get_file_tensor(self.file_paths[0])
        self.H, self.W = tensor.shape[1], tensor.shape[2]
        self.feature_cols_ref = feature_cols

        ph, pw = self.patch_cfg.patch_size
        if ph > self.H or pw > self.W:
            raise ValueError(
                f"Patch size {self.patch_cfg.patch_size} bigger than grid {(self.H, self.W)}"
            )
        if ph < 32 or pw < 64:
            # because you downsample by 16x in UNet path; tiny patches create 1x? bottlenecks
            print(
                f"[WARN] Very small patch {self.patch_cfg.patch_size}; consider >= (32, 64)."
            )

        # Precompute spatial crop indices for region filtering
        self.crop_h_indices = None
        self.crop_w_indices = None
        if self.region_filter is not None:
            print("\n=== REGION FILTERING ACTIVE (TimestepPatchWaveDataset) ===")
            print(f"  Filtering to: {self.region_filter.upper()}")
            print("  Boundary: Gibraltar Strait (lon=-5.5°)")

            # Extract coordinates from first file
            lat_idx = feature_cols.index("latitude")
            lon_idx = feature_cols.index("longitude")
            lat_data = tensor[0, ..., lat_idx]  # (H, W) - first timestep
            lon_data = tensor[0, ..., lon_idx]  # (H, W)

            # Gibraltar boundary
            GIBRALTAR_LON = -5.5

            # Find which columns (longitude) and rows (latitude) to keep
            if self.region_filter == "atlantic":
                region_condition = lon_data < GIBRALTAR_LON
                print("  Keeping pixels: lon < -5.5° (West of Gibraltar)")
            elif self.region_filter == "mediterranean":
                region_condition = lon_data >= GIBRALTAR_LON
                print("  Keeping pixels: lon >= -5.5° (East of Gibraltar)")
            else:
                raise ValueError(f"Unknown region_filter: {self.region_filter}")

            # Find columns/rows with at least one valid pixel in target region
            valid_coords = (
                region_condition & ~torch.isnan(lat_data) & ~torch.isnan(lon_data)
            )
            cols_with_region = valid_coords.any(dim=0)  # Check each column
            rows_with_region = valid_coords.any(dim=1)  # Check each row

            # Get indices of columns/rows to keep
            self.crop_w_indices = torch.where(cols_with_region)[0]
            self.crop_h_indices = torch.where(rows_with_region)[0]

            # Store cropped coordinate grids for evaluation
            self.cropped_lat_grid = lat_data[self.crop_h_indices, :][
                :, self.crop_w_indices
            ]
            self.cropped_lon_grid = lon_data[self.crop_h_indices, :][
                :, self.crop_w_indices
            ]

            # Update H, W to reflect cropped dimensions
            old_H, old_W = self.H, self.W
            self.H = len(self.crop_h_indices)
            self.W = len(self.crop_w_indices)

            print(f"  Spatial cropping: ({old_H}, {old_W}) → ({self.H}, {self.W})")
            original_size = old_H * old_W
            cropped_size = self.H * self.W
            print(
                f"  Removed {original_size - cropped_size} pixels ({(1 - cropped_size / original_size) * 100:.1f}% reduction)"
            )
            print("============================================================\n")
        else:
            self.cropped_lat_grid = None
            self.cropped_lon_grid = None

        # Build index map (H, W are final after region cropping)
        ph, pw = self.patch_cfg.patch_size
        self.tile_grid: Optional[List[Tuple[int, int]]] = None

        if self.sampling_mode == "exhaustive":
            tile_rows = self.H // ph
            tile_cols = self.W // pw
            self.tile_grid = [
                (r * ph, c * pw) for r in range(tile_rows) for c in range(tile_cols)
            ]
            n_tiles = len(self.tile_grid)
            self.index_map = [
                (f_idx, h, t_idx)
                for f_idx in range(len(self.file_paths))
                for h in range(24)
                for t_idx in range(n_tiles)
            ]
            coverage = (n_tiles * ph * pw) / (self.H * self.W) * 100
            print(
                f"[TimestepPatchWaveDataset] Exhaustive tiling: {tile_rows}x{tile_cols} = {n_tiles} tiles/frame"
            )
            print(f"  Coverage: {coverage:.1f}%  |  Samples/file: {24 * n_tiles}")
            print(f"  Total samples: {len(self.index_map)}")
        else:
            self.index_map = [
                (f_idx, h) for f_idx in range(len(self.file_paths)) for h in range(24)
            ]

        # Precompute valid anchors once from a sample frame (land mask is static across time)
        # Only needed for random/stratified modes
        self.valid_anchors: Optional[List[Tuple[int, int]]] = None
        if (
            self.sampling_mode != "exhaustive"
            and self.patch_cfg.precompute_valid_anchors
        ):
            self.valid_anchors = self._precompute_valid_anchors()

    def __len__(self):
        return len(self.index_map)

    def get_coordinates(self):
        """
        Get coordinate grids (lat, lon) for the dataset.

        Returns:
            tuple: (lat_grid, lon_grid) as numpy arrays
                  - If region filtering is active, returns cropped coordinates
                  - Otherwise, returns full coordinates from first file
        """
        if self.cropped_lat_grid is not None and self.cropped_lon_grid is not None:
            # Return cropped coordinates (for region filtering)
            return self.cropped_lat_grid.numpy(), self.cropped_lon_grid.numpy()
        else:
            # Load coordinates from first file (no region filtering)
            tensor, feature_cols = self._get_file_tensor(self.file_paths[0])
            lat_idx = feature_cols.index("latitude")
            lon_idx = feature_cols.index("longitude")

            # Use first timestep
            lat_grid = tensor[0, ..., lat_idx].numpy()
            lon_grid = tensor[0, ..., lon_idx].numpy()

            return lat_grid, lon_grid

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

        # Apply spatial cropping if region filtering is enabled
        if self.crop_h_indices is not None and self.crop_w_indices is not None:
            hour0 = hour0[self.crop_h_indices, :, :][:, self.crop_w_indices, :]

        vhm0_idx = feature_cols.index("VHM0")
        vhm0 = hour0[..., vhm0_idx : vhm0_idx + 1]  # (H,W,1)

        sea = (~torch.isnan(vhm0)).float()  # (H,W,1) 1 sea, 0 land

        ph, pw = self.patch_cfg.patch_size
        max_i = self.H - ph
        max_j = self.W - pw

        anchors: List[Tuple[int, int]] = []
        # Simple scan (fast enough for your sizes)
        for i in range(max_i + 1):
            for j in range(max_j + 1):
                patch = sea[i : i + ph, j : j + pw, :]
                sea_frac = float(patch.mean().item())
                if sea_frac >= self.patch_cfg.min_valid_fraction:
                    anchors.append((i, j))

        if not anchors:
            raise RuntimeError(
                "No valid anchors found. Lower min_valid_fraction or reduce patch_size."
            )

        print(
            f"[TimestepPatchWaveDataset] Precomputed {len(anchors)} valid anchors "
            f"(min_valid_fraction={self.patch_cfg.min_valid_fraction})."
        )
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

    def _sample_patch_coords(
        self, vhm0_full: torch.Tensor, idx: int, forced_bin: int | None = None
    ) -> Tuple[int, int, int]:
        """
        Returns (i_start, j_start, patch_bin).
        If stratified and no forced_bin, uses round-robin bin targets based on idx.
        """
        ph, pw = self.patch_cfg.patch_size

        # Decide target bin
        n_bins = len(self.patch_cfg.bin_edges_m) + 1
        if self.sampling_mode == "stratified":
            target_bin = forced_bin or self.forced_bin_id
            if target_bin is None:
                target_bin = idx % n_bins
        else:
            target_bin = None

        best_i, best_j, best_bin = 0, 0, 0
        best_score = -1.0

        # Random sampling without stratification
        if target_bin is None and self.sampling_mode == "random":
            i0, j0 = self._sample_anchor()
            patch = vhm0_full[i0 : i0 + ph, j0 : j0 + pw, :]
            score = self._patch_score(patch)
            b = self._bin_id(score) if not math.isnan(score) else 0
            return i0, j0, b

        # Otherwise try to match target bin
        for _ in range(self.patch_cfg.max_tries):
            i0, j0 = self._sample_anchor()
            patch = vhm0_full[i0 : i0 + ph, j0 : j0 + pw, :]
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
    def __getitem__(self, idx: int | tuple):
        forced_bin = None
        if isinstance(idx, tuple):
            idx, forced_bin = idx

        # Unpack index — exhaustive has 3 elements, random/stratified have 2
        if self.sampling_mode == "exhaustive":
            file_idx, hour_idx, tile_idx = self.index_map[idx]
        else:
            file_idx, hour_idx = self.index_map[idx]

        path = self.file_paths[file_idx]

        tensor, feature_cols = self._get_file_tensor(path)
        hour_data = tensor[hour_idx]  # (H, W, C)

        # Apply spatial cropping if region filtering is enabled
        if self.crop_h_indices is not None and self.crop_w_indices is not None:
            hour_data = hour_data[self.crop_h_indices, :, :][:, self.crop_w_indices, :]

        # Input columns: features_order if provided, else all minus excluded/targets
        target_colnames = list(self.target_columns.values())
        if self.features_order is not None:
            input_col_indices = [
                feature_cols.index(feat)
                for feat in self.features_order
                if feat in feature_cols
                and feat not in self.excluded_columns
                and feat not in target_colnames
            ]
        else:
            input_col_indices = [
                i
                for i, name in enumerate(feature_cols)
                if name not in self.excluded_columns and name not in target_colnames
            ]

        X_full = hour_data[..., input_col_indices]  # (H,W,Cin)

        vhm0_idx = feature_cols.index("VHM0")
        vhm0_full = hour_data[
            ..., vhm0_idx : vhm0_idx + 1
        ]  # (H,W,1) raw meters (NaN on land)

        # Get patch coordinates
        ph, pw = self.patch_cfg.patch_size
        if self.sampling_mode == "exhaustive":
            i0, j0 = self.tile_grid[tile_idx]
            score = self._patch_score(vhm0_full[i0 : i0 + ph, j0 : j0 + pw, :])
            patch_bin = self._bin_id(score) if not math.isnan(score) else 0
        else:
            i0, j0, patch_bin = self._sample_patch_coords(vhm0_full, idx, forced_bin)

        X = X_full[i0 : i0 + ph, j0 : j0 + pw, :]
        vhm0 = vhm0_full[i0 : i0 + ph, j0 : j0 + pw, :]

        # Sea mask
        # Note: Spatial cropping already removed non-target region if filtering is enabled
        sea_mask = (~torch.isnan(vhm0)).float()  # (ph,pw,1), 1 sea, 0 land

        # Build targets (keep NaNs on land!)
        targets: Dict[str, torch.Tensor] = {}
        for task_name, tgt_col in self.target_columns.items():
            tgt_idx = feature_cols.index(tgt_col)
            y_full = hour_data[..., tgt_idx : tgt_idx + 1]
            y = y_full[i0 : i0 + ph, j0 : j0 + pw, :]

            if self.predict_log_correction:
                # z = log(DA+eps) - log(raw+eps)
                y = torch.log(y + self.eps) - torch.log(vhm0 + self.eps)
            elif self.predict_bias:
                y = y - vhm0

            targets[task_name] = y

        # ----- IMPORTANT: fill NaNs in inputs only -----
        X = torch.nan_to_num(X, nan=0.0)
        vhm0_filled = torch.nan_to_num(
            vhm0, nan=0.0
        )  # useful for reconstruction / logging
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
                    _, y_norm = self.normalizer.transform_torch(
                        X.clone(), normalize_target=True, target=y
                    )
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


if __name__ == "__main__":
    # Quick sanity check. For full debug with maps, run:
    #   poetry run python scripts/debug_patch_dataset.py
    import glob

    data_dir = "/opt/dlami/nvme/preprocessed_subsampled_step_5/"
    pt_files = sorted(glob.glob(f"{data_dir}/WAVEAN*.pt"))[:1]

    ds = TimestepPatchWaveDataset(
        file_paths=pt_files,
        target_columns={"vhm0": "corrected_VHM0"},
        excluded_columns=[
            "time",
            "latitude",
            "longitude",
            "timestamp",
            "corrected_VTM02",
            "WDIR",
            "VMDR",
        ],
        region_filter="mediterranean",
        return_coords=True,
        predict_bias=True,
        predict_log_correction=False,
    )

    X, y, mask, vhm0, patch_bin, (i0, j0) = ds[0]
    y_sea = y[mask]
    print(f"X: {X.shape}  y: {y.shape}  mask: {mask.shape}  bin: {patch_bin}")
    print(f"anchor: ({i0},{j0})  sea: {mask.sum()}/{mask.numel()}")
    print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"y range (sea): [{y_sea.min():.4f}, {y_sea.max():.4f}]")
