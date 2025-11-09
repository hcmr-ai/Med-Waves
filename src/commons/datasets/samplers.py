import torch
from torch.utils.data import Sampler
import numpy as np

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
