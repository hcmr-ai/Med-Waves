from torch.utils.data import Sampler
import numpy as np
import random
from typing import Iterator, List, Optional, Sequence, Tuple
from torch.utils.data import Sampler

class WaveBinBalancedSampler(Sampler):
    def __init__(self, dataset, batch_size, bins_per_batch=None):
        """
        dataset: GridPatchWaveDataset instance (must have patch_bins computed)
        bins_per_batch: how many samples to draw from each bin per batch
                        e.g. {0: 8, 1: 8, 2: 4, 3: 2, 4: 2}
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Default: equal distribution across bins
        if bins_per_batch is None:
            unique_bins = np.unique(dataset.patch_bins)
            n_bins = len(unique_bins)
            n_each = batch_size // n_bins
            print(f"Found {n_bins} unique bins: {unique_bins}")
            # Use actual bin IDs, not range(n_bins)
            self.bins_per_batch = {int(b): n_each for b in unique_bins}
        else:
            self.bins_per_batch = bins_per_batch
        
        # Build idx lists per bin
        self.bin_to_idxs = {b: [] for b in self.bins_per_batch.keys()}
        for idx, bin_id in enumerate(self.dataset.patch_bins):
            if bin_id in self.bin_to_idxs:
                self.bin_to_idxs[bin_id].append(idx)

        # Convert to numpy arrays for fast choice
        for b in self.bin_to_idxs:
            self.bin_to_idxs[b] = np.array(self.bin_to_idxs[b])

        # Compute number of batches per epoch
        # For each bin, calculate how many batches we can make given the samples we need per batch
        batches_per_bin = []
        for b, n_samples_needed in self.bins_per_batch.items():
            if n_samples_needed > 0:
                n_available = len(self.bin_to_idxs[b])
                batches_possible = n_available // n_samples_needed
                batches_per_bin.append(batches_possible)
                print(f"  Bin {b}: {n_available} samples available, need {n_samples_needed} per batch -> {batches_possible} batches possible")
        
        self.total_batches = min(batches_per_bin) if batches_per_bin else 0
        print(f"Total batches per epoch: {self.total_batches} (batch_size={self.batch_size})")

    def __len__(self):
        return self.total_batches * self.batch_size

    def __iter__(self):
        for _ in range(self.total_batches):
            batch_idxs = []
            for b, n in self.bins_per_batch.items():
                batch_idxs.append(
                    np.random.choice(
                        self.bin_to_idxs[b], size=n, replace=False
                    )
                )
            batch_idxs = np.concatenate(batch_idxs)
            np.random.shuffle(batch_idxs)
            yield from batch_idxs

IndexType = Tuple[int, int]  # (base_idx, bin_id)

class BalancedBinBatchSampler(Sampler[List[IndexType]]):
    """
    Yields batches of indices where each index is (base_idx, bin_id).
    This lets the dataset sample a patch from the requested bin_id.

    - Guarantees fixed bin composition per batch.
    - Samples base_idx uniformly (with replacement) across the dataset.
    - Works with multi-worker DataLoader (sampler runs in main process).
    """

    def __init__(
        self,
        dataset_len: int,
        n_bins: int,
        batch_size: int,
        bins_per_batch: Optional[Sequence[int]] = None,
        drop_last: bool = True,
        seed: int = 123,
        steps_per_epoch: Optional[int] = None,
    ):
        """
        Args:
            dataset_len: len(dataset)
            n_bins: number of bins = len(bin_edges_m) + 1
            batch_size: number of samples per batch
            bins_per_batch:
                Optional explicit list of bin_ids of length batch_size.
                Example for n_bins=3, batch_size=12:
                    [0,0,0,0, 1,1,1,1, 2,2,2,2]
                If None: uses equal allocation (or as equal as possible).
            drop_last: whether to drop last incomplete batch (recommended True)
            seed: RNG seed
            steps_per_epoch: number of batches per epoch. If None, defaults to floor(dataset_len/batch_size)
        """
        if dataset_len <= 0:
            raise ValueError("dataset_len must be > 0")
        if n_bins <= 0:
            raise ValueError("n_bins must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset_len = dataset_len
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if bins_per_batch is None:
            # Equal-ish distribution across bins
            base = batch_size // n_bins
            rem = batch_size % n_bins
            pattern = []
            for b in range(n_bins):
                pattern.extend([b] * (base + (1 if b < rem else 0)))
            # If batch_size not divisible by n_bins, first bins get +1
            self.bins_per_batch = pattern
        else:
            if len(bins_per_batch) != batch_size:
                raise ValueError("bins_per_batch must have length == batch_size")
            if any((b < 0 or b >= n_bins) for b in bins_per_batch):
                raise ValueError("bins_per_batch has invalid bin ids")
            self.bins_per_batch = list(bins_per_batch)

        if steps_per_epoch is None:
            self.steps_per_epoch = dataset_len // batch_size
            if self.steps_per_epoch == 0:
                self.steps_per_epoch = 1
        else:
            self.steps_per_epoch = int(steps_per_epoch)

        self._rng = random.Random(self.seed)

    def set_epoch(self, epoch: int):
        """Call this at each epoch start for deterministic shuffling across epochs."""
        self.epoch = int(epoch)
        self._rng = random.Random(self.seed + self.epoch * 10007)

    def __iter__(self) -> Iterator[List[IndexType]]:
        for _ in range(self.steps_per_epoch):
            batch: List[IndexType] = []
            # Shuffle bin pattern each batch for variety (optional but nice)
            bins = self.bins_per_batch[:]
            self._rng.shuffle(bins)

            for b in bins:
                base_idx = self._rng.randrange(self.dataset_len)
                batch.append((base_idx, b))

            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch
