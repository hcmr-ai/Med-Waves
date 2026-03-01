"""
Unit tests for CachedWaveDataset and TimestepPatchWaveDataset.

Uses synthetic .pt files so tests run without real data.
Run with: poetry run python -m pytest tests/unit/test_datasets.py -v
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.commons.datasets.cache_wave_dataset import CachedWaveDataset
from src.commons.datasets.time_step_patch_dataset import (
    PatchSamplingConfig,
    TimestepPatchWaveDataset,
)


# ------------------------------------------------------------------ fixtures

FEATURE_COLS = [
    "VHM0", "WSPD", "VTM02", "U10", "V10",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "sin_month", "cos_month", "lat_norm", "lon_norm",
    "wave_dir_sin", "wave_dir_cos",
    "corrected_VHM0", "corrected_VTM02",
    "time", "latitude", "longitude", "timestamp",
    "WDIR", "VMDR",
]

EXCLUDED = ["time", "latitude", "longitude", "timestamp", "corrected_VTM02", "WDIR", "VMDR"]


def _make_synthetic_tensor(H=76, W=262):
    """Create a synthetic (24, H, W, C) tensor mimicking real data."""
    C = len(FEATURE_COLS)
    tensor = torch.randn(24, H, W, C)

    vhm0_idx = FEATURE_COLS.index("VHM0")
    corrected_idx = FEATURE_COLS.index("corrected_VHM0")
    lat_idx = FEATURE_COLS.index("latitude")
    lon_idx = FEATURE_COLS.index("longitude")

    # VHM0: positive values with some structure
    vhm0 = torch.abs(torch.randn(24, H, W)) * 2.0 + 0.5
    tensor[:, :, :, vhm0_idx] = vhm0
    tensor[:, :, :, corrected_idx] = vhm0 + torch.randn(24, H, W) * 0.1

    # Lat/lon: regular grid spanning Atlantic + Mediterranean
    lats = torch.linspace(30.0, 46.0, H)
    lons = torch.linspace(-18.0, 36.0, W)
    lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing="ij")
    for t in range(24):
        tensor[t, :, :, lat_idx] = lat_grid
        tensor[t, :, :, lon_idx] = lon_grid

    # Add land mask (NaN) for top-left quadrant to simulate coastline
    land_mask = (lat_grid > 43.0) & (lon_grid < 0.0)
    for t in range(24):
        tensor[t, land_mask, vhm0_idx] = float("nan")
        tensor[t, land_mask, corrected_idx] = float("nan")

    return tensor


@pytest.fixture
def synthetic_pt_files(tmp_path):
    """Create two synthetic .pt files in a temp directory."""
    paths = []
    for name in ["WAVEAN20170101.pt", "WAVEAN20170102.pt"]:
        path = str(tmp_path / name)
        tensor = _make_synthetic_tensor()
        torch.save({"tensor": tensor, "feature_cols": FEATURE_COLS}, path)
        paths.append(path)
    return paths


# ================================================================
#  CachedWaveDataset tests
# ================================================================

class TestCachedWaveDataset:

    def test_basic_shape_and_types(self, synthetic_pt_files):
        ds = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
        )
        X, y, mask, vhm0 = ds[0]

        assert X.ndim == 3, "X should be (C, H, W)"
        assert y.ndim == 3, "y should be (1, H, W)"
        assert mask.ndim == 3, "mask should be (1, H, W)"
        assert vhm0.ndim == 3, "vhm0 should be (1, H, W)"
        assert mask.dtype == torch.bool
        assert X.dtype == torch.float32

    def test_no_nan_in_inputs(self, synthetic_pt_files):
        ds = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
        )
        X, y, mask, vhm0 = ds[0]

        assert not torch.isnan(X).any(), "X should have no NaN (filled with 0)"
        assert not torch.isnan(vhm0).any(), "vhm0 should have no NaN (filled with 0)"

    def test_mask_matches_target_nans(self, synthetic_pt_files):
        ds = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
        )
        X, y, mask, vhm0 = ds[0]

        y_nan = torch.isnan(y)
        assert (mask == ~y_nan).all(), "mask should be True where y is valid, False where NaN"

    def test_dataset_length(self, synthetic_pt_files):
        ds = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
        )
        assert len(ds) == 2 * 24, "2 daily files x 24 hours = 48 samples"

    def test_region_filter_reduces_width(self, synthetic_pt_files):
        ds_full = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            region_filter=None,
        )
        ds_med = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            region_filter="mediterranean",
        )
        X_full, _, _, _ = ds_full[0]
        X_med, _, _, _ = ds_med[0]

        assert X_med.shape[2] < X_full.shape[2], (
            f"Mediterranean width ({X_med.shape[2]}) should be less than full ({X_full.shape[2]})"
        )

    def test_predict_bias_vs_direct(self, synthetic_pt_files):
        ds_bias = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
        )
        ds_direct = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=False,
        )
        _, y_bias, mask_b, _ = ds_bias[0]
        _, y_direct, mask_d, _ = ds_direct[0]

        # Bias values should be small (corrected - raw), direct values should be larger
        y_bias_sea = y_bias[mask_b]
        y_direct_sea = y_direct[mask_d]
        assert y_bias_sea.abs().mean() < y_direct_sea.abs().mean(), (
            "Bias target should have smaller magnitude than direct target"
        )

    def test_excluded_columns_reduce_channels(self, synthetic_pt_files):
        ds_few = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
        )
        ds_none = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=[],
        )
        X_few, _, _, _ = ds_few[0]
        X_none, _, _, _ = ds_none[0]

        assert X_few.shape[0] < X_none.shape[0], (
            f"Excluding columns should reduce channels: {X_few.shape[0]} vs {X_none.shape[0]}"
        )

    def test_patch_crop(self, synthetic_pt_files):
        ph, pw = 32, 96
        ds = CachedWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            patch_size=(ph, pw),
        )
        X, y, mask, vhm0 = ds[0]

        assert X.shape[1] == ph and X.shape[2] == pw, (
            f"Patch crop should produce ({ph}, {pw}), got ({X.shape[1]}, {X.shape[2]})"
        )


# ================================================================
#  TimestepPatchWaveDataset tests
# ================================================================

class TestTimestepPatchWaveDataset:

    def test_basic_shape_and_types(self, synthetic_pt_files):
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            return_coords=True,
        )
        X, y, mask, vhm0, patch_bin, (i0, j0) = ds[0]
        ph, pw = ds.patch_cfg.patch_size

        assert X.shape[1] == ph and X.shape[2] == pw
        assert y.shape == (1, ph, pw)
        assert mask.shape == (1, ph, pw)
        assert mask.dtype == torch.bool
        assert isinstance(patch_bin, int)
        assert isinstance(i0, int) and isinstance(j0, int)

    def test_no_nan_in_inputs(self, synthetic_pt_files):
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
        )
        X, y, mask, vhm0, *_ = ds[0]

        assert not torch.isnan(X).any(), "X should have no NaN"
        assert not torch.isnan(vhm0).any(), "vhm0 should have no NaN"

    def test_dataset_length_random(self, synthetic_pt_files):
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode="random",
        )
        assert len(ds) == 2 * 24, "2 files x 24 hours = 48 samples for random mode"

    def test_dataset_length_exhaustive(self, synthetic_pt_files):
        patch_cfg = PatchSamplingConfig(patch_size=(32, 96), min_valid_fraction=0.0)
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode="exhaustive",
            patch_cfg=patch_cfg,
        )
        H, W = ds.H, ds.W
        tile_rows = H // 32
        tile_cols = W // 96
        expected_tiles = tile_rows * tile_cols
        expected_len = 2 * 24 * expected_tiles

        assert len(ds) == expected_len, (
            f"Exhaustive: 2 files x 24h x {expected_tiles} tiles = {expected_len}, got {len(ds)}"
        )

    def test_exhaustive_tiles_are_deterministic(self, synthetic_pt_files):
        patch_cfg = PatchSamplingConfig(patch_size=(32, 96), min_valid_fraction=0.0)
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode="exhaustive",
            patch_cfg=patch_cfg,
            return_coords=True,
        )

        # Same index should always return same coords
        _, _, _, _, _, (i0_a, j0_a) = ds[0]
        _, _, _, _, _, (i0_b, j0_b) = ds[0]
        assert i0_a == i0_b and j0_a == j0_b, "Exhaustive mode should return deterministic coords"

    def test_exhaustive_covers_different_locations(self, synthetic_pt_files):
        patch_cfg = PatchSamplingConfig(patch_size=(32, 96), min_valid_fraction=0.0)
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode="exhaustive",
            patch_cfg=patch_cfg,
            return_coords=True,
        )
        n_tiles = len(ds.tile_grid)

        # First n_tiles samples (same file, same hour) should have unique coords
        coords = set()
        for idx in range(n_tiles):
            _, _, _, _, _, (i0, j0) = ds[idx]
            coords.add((i0, j0))

        assert len(coords) == n_tiles, (
            f"Expected {n_tiles} unique tile positions, got {len(coords)}"
        )

    def test_region_filter_reduces_width(self, synthetic_pt_files):
        ds_full = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            region_filter=None,
        )
        ds_med = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            region_filter="mediterranean",
        )
        assert ds_med.W < ds_full.W, (
            f"Med W ({ds_med.W}) should be less than full W ({ds_full.W})"
        )

    def test_stratified_mode_respects_forced_bin(self, synthetic_pt_files):
        ds = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            sampling_mode="stratified",
            return_coords=True,
        )
        # Calling with a tuple (idx, forced_bin) should not crash
        X, y, mask, vhm0, patch_bin, (i0, j0) = ds[(0, 1)]
        assert X.ndim == 3, "Should return valid tensor even with forced bin"

    def test_sea_mask_channel(self, synthetic_pt_files):
        ds_with = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            add_sea_mask_channel=True,
        )
        ds_without = TimestepPatchWaveDataset(
            file_paths=synthetic_pt_files,
            target_columns={"vhm0": "corrected_VHM0"},
            excluded_columns=EXCLUDED,
            predict_bias=True,
            predict_log_correction=False,
            add_sea_mask_channel=False,
        )
        X_with, *_ = ds_with[0]
        X_without, *_ = ds_without[0]

        assert X_with.shape[0] == X_without.shape[0] + 1, (
            "Sea mask channel should add exactly 1 channel"
        )
