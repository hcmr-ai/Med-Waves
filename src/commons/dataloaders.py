import s3fs

from src.commons.datasets.time_step_patch_dataset import TimestepPatchWaveDataset, PatchSamplingConfig
from src.commons.datasets.cache_wave_dataset import CachedWaveDataset
from src.commons.helpers import DNNConfig
from src.commons.helpers import get_file_list, split_files_by_year
from torch.utils.data import DataLoader
from src.commons.datasets.samplers import BalancedBinBatchSampler
from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer
from logging import getLogger
logger = getLogger(__name__)

def create_data_loaders(config: DNNConfig, fs: s3fs.S3FileSystem) -> tuple:
    """Create train and validation data loaders
    
    Supports region filtering via data_config["region_filter"]:
        - "atlantic": lon < -6.0
        - "mediterranean": -6.0 <= lon <= 25.0
        - "eastern_med": lon > 25.0
        - None: No filtering (default)
    """
    data_config = config.config["data"]
    training_config = config.config["training"]

    # Get file list
    files = get_file_list(
        data_config["data_path"], data_config["file_pattern"], data_config["max_files"]
    )

    logger.info(f"Found {len(files)} files")

    # Split files by year (2021=train, 2022=val, 2023=test)
    train_files, val_files, test_files = split_files_by_year(
        files,
        train_year=data_config.get("train_year", 2021),
        val_year=data_config.get("val_year", 2022),
        test_year=data_config.get("test_year", 2023),
        val_months=data_config.get("val_months", []),
        test_months=data_config.get("test_months", []),
    )

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Val files: {len(val_files)}")
    logger.info(f"Test files: {len(test_files)}")

    # Create datasets
    patch_size = tuple(data_config["patch_size"]) if data_config["patch_size"] else None
    excluded_columns = data_config.get(
        "excluded_columns", ["time", "latitude", "longitude", "timestamp"]
    )
    target_columns = data_config.get("target_columns", {"vhm0": "corrected_VHM0"})
    predict_bias = data_config.get("predict_bias", False)
    subsample_step = data_config.get("subsample_step", None)
    region_filter = data_config.get("region_filter", None)  # None, "atlantic", "mediterranean", "eastern_med"

    normalizer = WaveNormalizer.load_from_s3("medwav-dev-data",data_config["normalizer_path"])
    # normalizer = WaveNormalizer.load_from_disk(data_config["normalizer_path"])
    logger.info(f"Normalizer: {normalizer.mode}")
    logger.info(f"Normalizer stats: {normalizer.stats_}")
    logger.info(f"Loaded normalizer from {data_config['normalizer_path']}")
    if data_config.get("use_patch_sampling") is not None:
        # Create patch sampling configuration
        patch_cfg = PatchSamplingConfig(
            patch_size=tuple(patch_size) if patch_size else (32, 96),
            max_tries=data_config.get("max_tries", 50),
            score=data_config.get("patch_score", "p90"),
            bin_edges_m=tuple(data_config.get("bin_edges_m", [2.0, 4.0])),
            min_valid_fraction=data_config.get("min_valid_pixels", 0.6),
            precompute_valid_anchors=data_config.get("precompute_valid_anchors", True)
        )
        
        train_dataset = TimestepPatchWaveDataset(
            train_files,
            target_columns=target_columns,
            excluded_columns=excluded_columns,
            normalizer=normalizer,
            normalize_target=data_config.get("normalize_target", False),
            predict_bias=predict_bias,
            predict_log_correction=data_config.get("predict_log_correction", False),
            eps=data_config.get("eps", 1e-3),
            patch_cfg=patch_cfg,
            sampling_mode=data_config.get("sampling_mode", "random"),
            forced_bin_id=data_config.get("forced_bin_id", None),
            use_cache=data_config.get("use_cache", True),
            max_cache_files=data_config.get("max_cache_size", 2),
            features_order=data_config.get("features_order", None),
            add_sea_mask_channel=data_config.get("add_sea_mask_channel", False),
            seed=data_config.get("random_seed", 42),
            return_coords=data_config.get("return_coords", True)
        )
    else:
        train_dataset = CachedWaveDataset(
            train_files,
            patch_size=patch_size,
            excluded_columns=excluded_columns,
            target_columns=target_columns,
            predict_bias=predict_bias,
            subsample_step=subsample_step,
            normalizer=normalizer,
            enable_profiler=True,
            use_cache=data_config.get("use_cache", False),
            normalize_target=data_config.get("normalize_target", False),
            fs=fs,
            max_cache_size=data_config.get("max_cache_size", 20)
        )

    # if data_config.get("patch_size_deactivate", None) is not None:
    #     # Create patch sampling configuration for validation
    #     val_patch_cfg = PatchSamplingConfig(
    #         patch_size=tuple(patch_size) if patch_size else (32, 96),
    #         max_tries=data_config.get("max_tries", 50),
    #         score=data_config.get("patch_score", "p90"),
    #         bin_edges_m=tuple(data_config.get("bin_edges_m", [2.0, 4.0])),
    #         min_valid_fraction=data_config.get("min_valid_pixels", 0.6),
    #         precompute_valid_anchors=data_config.get("precompute_valid_anchors", True)
    #     )
        
    #     val_dataset = TimestepPatchWaveDataset(
    #         val_files,
    #         target_columns=target_columns,
    #         excluded_columns=excluded_columns,
    #         normalizer=normalizer,
    #         normalize_target=data_config.get("normalize_target", False),
    #         predict_bias=predict_bias,
    #         predict_log_correction=data_config.get("predict_log_correction", True),
    #         eps=data_config.get("eps", 1e-3),
    #         patch_cfg=val_patch_cfg,
    #         sampling_mode=data_config.get("sampling_mode", "random"),
    #         forced_bin_id=data_config.get("forced_bin_id", None),
    #         use_cache=data_config.get("use_cache", True),
    #         max_cache_files=data_config.get("max_cache_size", 2),
    #         features_order=data_config.get("features_order", None),
    #         add_sea_mask_channel=data_config.get("add_sea_mask_channel", True),
    #         seed=data_config.get("random_seed", 42),
    #         return_coords=data_config.get("return_coords", True)
    #     )
    # else:
    val_dataset = CachedWaveDataset(
        val_files,
        patch_size=patch_size,
        excluded_columns=excluded_columns,
        target_columns=target_columns,
        predict_bias=predict_bias,
        subsample_step=subsample_step,
        normalizer=normalizer,
        enable_profiler=True,
        use_cache=data_config.get("use_cache", False),
        normalize_target=data_config.get("normalize_target", False),
        fs=fs,
        max_cache_size=data_config.get("max_cache_size", 20)
    )

    # # Pre-compute wave bins and filter patches (if using patched dataset)
    # use_balanced_sampling = data_config.get("use_balanced_sampling", False)  # Set to False for uniform random sampling

    # if patch_size is not None:
    #     # ALWAYS compute bins to filter out invalid patches (regardless of balanced sampling)
    #     # logger.info("Computing wave bins and filtering invalid patches...")
    #     # train_dataset.compute_all_bins()
    #     # logger.info(f"Training dataset after filtering: {len(train_dataset)} patches")

    #     # Also filter validation dataset
    #     # val_dataset.compute_all_bins()
    #     logger.info(f"Validation dataset after filtering: {len(val_dataset)} patches")

    #     if len(val_dataset) == 0:
    #         raise ValueError("Validation dataset is empty after filtering! Check val_year/val_months or lower min_valid_pixels threshold.")

    #     if use_balanced_sampling:
    #         logger.info("Using balanced sampling (equal samples per wave height bin)")
    #     else:
    #         logger.info("Using uniform random sampling (shuffle=True, no sampler)")

    n_bins = len(train_dataset.patch_cfg.bin_edges_m) + 1

    if data_config.get("use_patch_sampling", False):
        n_hours = len(train_files) * 24  # or compute from len(file_paths)*24
        steps_per_epoch = int(n_hours / training_config["batch_size"])
        
        # Get bin sampling distribution from config or use default
        bin_sampling_weights = data_config.get("bin_sampling_weights", None)
        
        if bin_sampling_weights is not None:
            # User-specified weights: [weight_bin0, weight_bin1, weight_bin2, ...]
            total_weight = sum(bin_sampling_weights)
            bins_per_batch = []
            for bin_id, weight in enumerate(bin_sampling_weights):
                count = int(round(weight / total_weight * training_config["batch_size"]))
                bins_per_batch.extend([bin_id] * count)
            # Adjust if rounding caused mismatch
            while len(bins_per_batch) < training_config["batch_size"]:
                bins_per_batch.append(0)  # Add to first bin
            bins_per_batch = bins_per_batch[:training_config["batch_size"]]  # Trim if over
            logger.info(f"Using custom bin sampling weights: {bin_sampling_weights}")
            logger.info(f"Resulting bins_per_batch distribution: {[bins_per_batch.count(i) for i in range(n_bins)]}")
        else:
            # Default: None (equal distribution handled by BalancedBinBatchSampler)
            bins_per_batch = None
            logger.info(f"Using equal bin sampling distribution across {n_bins} bins")
        
        batch_sampler = BalancedBinBatchSampler(
            dataset_len=len(train_dataset),
            n_bins=n_bins,
            batch_size=training_config["batch_size"],
            bins_per_batch=bins_per_batch,
            steps_per_epoch=steps_per_epoch,
            seed=123,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,   # ✅ only batch_sampler
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
            persistent_workers=training_config.get(
                "persistent_workers", training_config["num_workers"] > 0
            ),
            prefetch_factor=training_config["prefetch_factor"],
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
            persistent_workers=training_config.get(
                "persistent_workers", training_config["num_workers"] > 0
            ),
            prefetch_factor=training_config["prefetch_factor"],
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=training_config["pin_memory"],
        persistent_workers=training_config.get("persistent_workers", training_config["num_workers"] > 0),
        prefetch_factor=None,
        sampler=None,
        # timeout=300  # 5 minute timeout for S3 loading
    )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, normalizer