"""
Unit tests for WaveBinBalancedSampler and BalancedBinBatchSampler.

Run with: poetry run python -m pytest tests/unit/test_samplers.py -v
"""

import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.commons.datasets.samplers import (
    BalancedBinBatchSampler,
    WaveBinBalancedSampler,
)


# ================================================================
#  BalancedBinBatchSampler tests
# ================================================================

class TestBalancedBinBatchSampler:

    def test_batch_count(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=12
        )
        assert len(sampler) == 100 // 12, (
            f"Expected {100 // 12} steps, got {len(sampler)}"
        )

    def test_custom_steps_per_epoch(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=12, steps_per_epoch=5
        )
        assert len(sampler) == 5

    def test_batch_size_matches(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=12
        )
        for batch in sampler:
            assert len(batch) == 12, f"Expected batch of 12, got {len(batch)}"

    def test_bin_distribution_is_balanced(self):
        n_bins = 3
        batch_size = 12
        sampler = BalancedBinBatchSampler(
            dataset_len=200, n_bins=n_bins, batch_size=batch_size
        )
        for batch in sampler:
            bin_counts = Counter(bin_id for _, bin_id in batch)
            # With 12 items and 3 bins, each bin should get exactly 4
            for b in range(n_bins):
                assert bin_counts[b] == batch_size // n_bins, (
                    f"Bin {b} got {bin_counts[b]} samples, expected {batch_size // n_bins}"
                )

    def test_yields_tuples_of_idx_and_bin(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=50, n_bins=2, batch_size=4
        )
        batch = next(iter(sampler))
        for item in batch:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2, f"Expected 2-tuple, got length {len(item)}"
            base_idx, bin_id = item
            assert 0 <= base_idx < 50, f"base_idx {base_idx} out of range [0, 50)"
            assert 0 <= bin_id < 2, f"bin_id {bin_id} out of range [0, 2)"

    def test_set_epoch_changes_sequence(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=6, seed=42
        )
        sampler.set_epoch(0)
        batches_e0 = [batch[:] for batch in sampler]

        sampler.set_epoch(1)
        batches_e1 = [batch[:] for batch in sampler]

        # Different epochs should produce different sequences (with high probability)
        assert batches_e0 != batches_e1, "Different epochs should produce different batches"

    def test_deterministic_with_same_seed(self):
        kwargs = dict(dataset_len=100, n_bins=3, batch_size=6, seed=42)

        s1 = BalancedBinBatchSampler(**kwargs)
        s1.set_epoch(0)
        batches_1 = [batch[:] for batch in s1]

        s2 = BalancedBinBatchSampler(**kwargs)
        s2.set_epoch(0)
        batches_2 = [batch[:] for batch in s2]

        assert batches_1 == batches_2, "Same seed + epoch should produce identical batches"

    def test_custom_bins_per_batch(self):
        # 6-sample batch: 4 from bin 0, 2 from bin 1
        bins_per_batch = [0, 0, 0, 0, 1, 1]
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=2, batch_size=6,
            bins_per_batch=bins_per_batch,
        )
        for batch in sampler:
            bin_counts = Counter(bin_id for _, bin_id in batch)
            assert bin_counts[0] == 4
            assert bin_counts[1] == 2

    def test_uneven_division(self):
        # 7 items across 3 bins: first bin gets 3, others get 2 each
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=7
        )
        for batch in sampler:
            bin_counts = Counter(bin_id for _, bin_id in batch)
            counts = sorted(bin_counts.values(), reverse=True)
            assert counts == [3, 2, 2], f"Expected [3,2,2] with 7//3, got {counts}"

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="dataset_len must be > 0"):
            BalancedBinBatchSampler(dataset_len=0, n_bins=3, batch_size=4)

        with pytest.raises(ValueError, match="n_bins must be > 0"):
            BalancedBinBatchSampler(dataset_len=10, n_bins=0, batch_size=4)

        with pytest.raises(ValueError, match="batch_size must be > 0"):
            BalancedBinBatchSampler(dataset_len=10, n_bins=3, batch_size=0)

    def test_bins_per_batch_length_mismatch(self):
        with pytest.raises(ValueError, match="bins_per_batch must have length == batch_size"):
            BalancedBinBatchSampler(
                dataset_len=10, n_bins=3, batch_size=4,
                bins_per_batch=[0, 0, 1],  # length 3 != batch_size 4
            )

    def test_bins_per_batch_invalid_bin_ids(self):
        with pytest.raises(ValueError, match="bins_per_batch has invalid bin ids"):
            BalancedBinBatchSampler(
                dataset_len=10, n_bins=2, batch_size=4,
                bins_per_batch=[0, 0, 2, 1],  # bin 2 is invalid when n_bins=2
            )

    def test_small_dataset_gets_at_least_one_step(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=3, n_bins=2, batch_size=8
        )
        assert len(sampler) == 1, "Small dataset should still get 1 step"

    def test_iteration_count_matches_len(self):
        sampler = BalancedBinBatchSampler(
            dataset_len=100, n_bins=3, batch_size=12
        )
        batch_count = sum(1 for _ in sampler)
        assert batch_count == len(sampler)


# ================================================================
#  WaveBinBalancedSampler tests
# ================================================================

class _FakeDataset:
    """Minimal dataset mock for WaveBinBalancedSampler."""

    def __init__(self, patch_bins):
        self.patch_bins = patch_bins


class TestWaveBinBalancedSampler:

    def test_basic_iteration(self):
        bins = np.array([0] * 20 + [1] * 20 + [2] * 20)
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(ds, batch_size=9)

        indices = list(sampler)
        assert len(indices) > 0, "Sampler should yield at least some indices"

    def test_batch_size_integrity(self):
        bins = np.array([0] * 30 + [1] * 30)
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(ds, batch_size=6)

        indices = list(sampler)
        # Total should be multiple of batch_size
        assert len(indices) % 6 == 0, (
            f"Total indices {len(indices)} not a multiple of batch_size 6"
        )

    def test_total_batches_bounded_by_smallest_bin(self):
        # Bin 0 has 10, bin 1 has 100 → only 10//(10//2)=2 batches possible
        bins = np.array([0] * 10 + [1] * 100)
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(ds, batch_size=10)

        # Each bin gets 5 per batch; bin 0 has 10 → 2 batches max
        assert sampler.total_batches == 2

    def test_indices_in_range(self):
        N = 60
        bins = np.array([i % 3 for i in range(N)])
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(ds, batch_size=6)

        for idx in sampler:
            assert 0 <= idx < N, f"Index {idx} out of range [0, {N})"

    def test_custom_bins_per_batch(self):
        bins = np.array([0] * 40 + [1] * 40 + [2] * 40)
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(
            ds, batch_size=10,
            bins_per_batch={0: 5, 1: 3, 2: 2},
        )

        # 40 samples, need 5 → 8 batches from bin 0
        # 40 samples, need 3 → 13 batches from bin 1
        # 40 samples, need 2 → 20 batches from bin 2
        # Min = 8
        assert sampler.total_batches == 8

    def test_len_equals_total_batches_times_bs(self):
        bins = np.array([0] * 20 + [1] * 20)
        ds = _FakeDataset(bins)
        sampler = WaveBinBalancedSampler(ds, batch_size=8)

        assert len(sampler) == sampler.total_batches * 8
