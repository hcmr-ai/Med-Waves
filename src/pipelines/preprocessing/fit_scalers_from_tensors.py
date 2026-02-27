"""
Fit WaveNormalizer scalers from .pt tensor files.

Loads (T, H, W, C) tensors, computes tensor-level spatial/temporal features
(grad_mag, dVHM0, dWSPD) that require the full grid, then fits the normalizer
on all channels including the derived ones.

Usage:
    poetry run python -m src.pipelines.preprocessing.fit_scalers_from_tensors
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import fsspec
import numpy as np
import torch
from tqdm import tqdm

from src.commons.preprocessing.bu_net_preprocessing import WaveNormalizer

S3_BUCKET = "medwav-dev-data"
S3_PREFIX = "scalers/"
LOCAL_TMP = "data/scalers/"


def compute_tensor_features(tensor: torch.Tensor, feature_cols: list[str]) -> tuple[torch.Tensor, list[str]]:
    """Compute spatial/temporal features on the (T, H, W, C) tensor.

    Adds:
        dVHM0:    temporal diff of VHM0 along time axis per grid point
        dWSPD:    temporal diff of WSPD along time axis per grid point
        grad_mag: 2D spatial gradient magnitude of VHM0 (sqrt(dx² + dy²))

    Returns:
        augmented tensor (T, H, W, C+3), updated feature_cols
    """
    vhm0_idx = feature_cols.index("VHM0")
    wspd_idx = feature_cols.index("WSPD")

    vhm0 = tensor[..., vhm0_idx]  # (T, H, W)
    wspd = tensor[..., wspd_idx]  # (T, H, W)

    # Temporal diffs along time axis: diff at t=0 is 0
    dVHM0 = torch.zeros_like(vhm0)
    dVHM0[1:] = vhm0[1:] - vhm0[:-1]

    dWSPD = torch.zeros_like(wspd)
    dWSPD[1:] = wspd[1:] - wspd[:-1]

    # 2D spatial gradient of VHM0: finite differences with zero-padding
    # dx: diff along W (longitude) axis
    dx = torch.zeros_like(vhm0)
    dx[..., 1:] = vhm0[..., 1:] - vhm0[..., :-1]

    # dy: diff along H (latitude) axis
    dy = torch.zeros_like(vhm0)
    dy[:, 1:, :] = vhm0[:, 1:, :] - vhm0[:, :-1, :]

    grad_mag = torch.sqrt(dx ** 2 + dy ** 2)

    # Stack new channels: (T, H, W) -> (T, H, W, 1)
    new_channels = torch.stack([dVHM0, dWSPD, grad_mag], dim=-1)  # (T, H, W, 3)
    augmented = torch.cat([tensor, new_channels], dim=-1)

    new_feature_cols = feature_cols + ["dVHM0", "dWSPD", "grad_mag"]
    return augmented, new_feature_cols


GIBRALTAR_LON = -5.5


def _extract_year(filename: str) -> int | None:
    """Extract year from WAVEAN{YYYYMMDD}.pt filename."""
    import re
    m = re.search(r"WAVEAN(\d{4})\d{4}", filename)
    return int(m.group(1)) if m else None


def _apply_region_filter(
    tensor: torch.Tensor,
    feature_cols: list[str],
    region: str,
) -> torch.Tensor:
    """Crop tensor (T, H, W, C) to Atlantic or Mediterranean based on longitude."""
    lon_idx = feature_cols.index("longitude")
    lon_grid = tensor[0, :, :, lon_idx]  # (H, W) — static across time

    if region == "atlantic":
        col_mask = (lon_grid < GIBRALTAR_LON).any(dim=0)  # keep cols west of Gibraltar
    elif region == "mediterranean":
        col_mask = (lon_grid >= GIBRALTAR_LON).any(dim=0)  # keep cols east of Gibraltar
    else:
        raise ValueError(f"Unknown region: {region}. Use 'atlantic' or 'mediterranean'.")

    keep_cols = torch.where(col_mask)[0]
    return tensor[:, :, keep_cols, :]


def list_pt_files(
    data_dir: str,
    years: list[int] | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Discover WAVEAN*.pt files in a directory (local or s3://)."""
    is_s3 = data_dir.startswith("s3://")

    if is_s3:
        fs = fsspec.filesystem("s3")
        files = sorted(
            f if f.startswith("s3://") else f"s3://{f}"
            for f in fs.glob(data_dir.rstrip("/") + "/WAVEAN*.pt")
        )
    else:
        files = sorted(str(p) for p in Path(data_dir).glob("WAVEAN*.pt"))

    if years is not None:
        year_set = set(years)
        files = [f for f in files if _extract_year(f.split("/")[-1]) in year_set]
        print(f"Filtered to years {sorted(year_set)}: {len(files)} files")

    if max_files:
        files = files[:max_files]

    return files


def process_single_file(
    filepath: str,
    region_filter: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load one .pt file, compute derived features, flatten to (N, C).

    Returns valid (non-NaN VHM0) rows as float32 and the feature column names.
    """
    is_s3 = filepath.startswith("s3://")
    if is_s3:
        with fsspec.open(filepath, "rb") as fh:
            data = torch.load(fh, map_location="cpu", weights_only=False)
    else:
        data = torch.load(filepath, map_location="cpu", weights_only=False)

    tensor = data["tensor"]
    fcols = data["feature_cols"]

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    if region_filter:
        tensor = _apply_region_filter(tensor, fcols, region_filter)

    tensor, fcols = compute_tensor_features(tensor, fcols)

    flat = tensor.reshape(-1, tensor.shape[-1]).numpy()
    vhm0_col = fcols.index("VHM0")
    valid_mask = ~np.isnan(flat[:, vhm0_col])
    return flat[valid_mask].astype(np.float32), fcols


def load_pt_files(
    data_dir: str,
    years: list[int] | None = None,
    region_filter: str | None = None,
    max_files: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load all .pt files into memory, compute derived features, concatenate to (N, C).

    This loads everything at once -- see ``fit_streaming`` for a low-memory alternative.
    """
    files = list_pt_files(data_dir, years=years, max_files=max_files)

    print(f"Loading {len(files)} .pt files from {data_dir}")
    if region_filter:
        print(f"Region filter: {region_filter} (Gibraltar boundary lon={GIBRALTAR_LON})")

    n_workers = min(32, len(files))
    all_flat: list[np.ndarray] = [None] * len(files)  # type: ignore[list-item]
    feature_cols_ref: list[str] | None = None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_idx = {
            pool.submit(process_single_file, f, region_filter): i
            for i, f in enumerate(files)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(files), desc="Loading .pt files"):
            idx = future_to_idx[future]
            flat, fcols = future.result()
            if feature_cols_ref is None:
                feature_cols_ref = fcols
            elif fcols != feature_cols_ref:
                raise ValueError(
                    f"Feature mismatch in {files[idx]}: expected {feature_cols_ref}, got {fcols}"
                )
            all_flat[idx] = flat

    X = np.concatenate(all_flat, axis=0).astype(np.float32)
    print(f"Total valid samples: {X.shape[0]:,}, channels: {X.shape[1]}")
    return X, feature_cols_ref


RESERVOIR_SIZE = 1_000_000


def fit_streaming(
    data_dirs: list[str],
    mode: str,
    years: list[int] | None = None,
    region_filter: str | None = None,
    max_files: int | None = None,
    target_feature: str | None = None,
    reservoir_size: int = RESERVOIR_SIZE,
    n_quantiles: int = 1000,
    random_state: int = 42,
) -> tuple[WaveNormalizer, list[str]]:
    """Fit a WaveNormalizer by streaming files one at a time (low-memory).

    For zscore: accumulates running sum/sum_sq/count per channel (exact).
    For quantile: reservoir-samples rows, then fits QuantileTransformer at the end.
    """
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    all_files: list[str] = []
    for data_dir in data_dirs:
        files = list_pt_files(data_dir, years=years, max_files=max_files)
        print(f"Found {len(files)} .pt files in {data_dir}")
        if region_filter:
            print(f"Region filter: {region_filter} (Gibraltar boundary lon={GIBRALTAR_LON})")
        all_files.extend(files)

    print(f"Streaming {len(all_files)} files total (mode={mode})")

    feature_cols: list[str] | None = None
    n_channels: int = 0

    ch_sum: np.ndarray | None = None
    ch_sum_sq: np.ndarray | None = None
    ch_count: np.ndarray | None = None

    reservoir: np.ndarray | None = None
    seen = 0
    rng = np.random.default_rng(random_state)

    def _update_zscore(flat: np.ndarray) -> None:
        assert ch_sum is not None and ch_sum_sq is not None and ch_count is not None
        for c in range(n_channels):
            col = flat[:, c].astype(np.float64)
            mask = ~np.isnan(col)
            col = col[mask]
            ch_sum[c] += col.sum()
            ch_sum_sq[c] += (col**2).sum()
            ch_count[c] += len(col)

    def _update_reservoir(flat: np.ndarray) -> None:
        nonlocal seen
        assert reservoir is not None
        n_new = len(flat)
        if seen + n_new <= reservoir_size:
            reservoir[seen : seen + n_new] = flat
        else:
            space = max(0, reservoir_size - seen)
            if space > 0:
                reservoir[seen:reservoir_size] = flat[:space]
            remaining = flat[space:]
            n_remaining = len(remaining)
            if n_remaining > 0:
                indices = np.arange(seen + space, seen + space + n_remaining, dtype=np.float64)
                probs = reservoir_size / (indices + 1)
                mask = rng.random(n_remaining) < probs
                replace_count = int(mask.sum())
                if replace_count > 0:
                    replace_idx = rng.integers(0, reservoir_size, size=replace_count)
                    reservoir[replace_idx] = remaining[mask]
        seen += n_new

    n_download_workers = min(32, len(all_files))
    with ThreadPoolExecutor(max_workers=n_download_workers) as pool:
        future_to_file = {
            pool.submit(process_single_file, f, region_filter): f
            for f in all_files
        }
        for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Streaming files"):
            f = future_to_file.pop(future)
            flat, fcols = future.result()
            del future

            if feature_cols is None:
                feature_cols = fcols
                n_channels = flat.shape[1]
                if mode == "zscore":
                    ch_sum = np.zeros(n_channels, dtype=np.float64)
                    ch_sum_sq = np.zeros(n_channels, dtype=np.float64)
                    ch_count = np.zeros(n_channels, dtype=np.int64)
                else:
                    reservoir = np.empty((reservoir_size, n_channels), dtype=np.float32)
            elif fcols != feature_cols:
                raise ValueError(f"Feature mismatch in {f}: expected {feature_cols}, got {fcols}")

            if mode == "zscore":
                _update_zscore(flat)
            else:
                _update_reservoir(flat)

            del flat

    if feature_cols is None:
        raise RuntimeError("No .pt files found in the provided directories")

    normalizer = WaveNormalizer(mode=mode, n_quantiles=n_quantiles, random_state=random_state)
    normalizer.feature_order_ = feature_cols
    normalizer.target_feature_name_ = target_feature

    if mode == "zscore":
        assert ch_sum is not None and ch_sum_sq is not None and ch_count is not None
        total_samples = 0
        for c in range(n_channels):
            mean = ch_sum[c] / ch_count[c]
            var = ch_sum_sq[c] / ch_count[c] - mean**2
            ss = StandardScaler()
            ss.mean_ = np.array([mean])
            ss.scale_ = np.array([max(np.sqrt(var), 1e-10)])
            ss.var_ = np.array([var])
            ss.n_samples_seen_ = int(ch_count[c])
            ss.n_features_in_ = 1
            normalizer.stats_[c] = ss
            total_samples = max(total_samples, int(ch_count[c]))
        print(f"Total valid samples: ~{total_samples:,}, channels: {n_channels}")
    else:
        assert reservoir is not None
        actual_size = min(seen, reservoir_size)
        sample = reservoir[:actual_size]
        print(f"Reservoir: {actual_size:,} samples from {seen:,} total, channels: {n_channels}")
        for c in range(n_channels):
            col = sample[:, c]
            col = col[~np.isnan(col)]
            qt = QuantileTransformer(
                n_quantiles=min(n_quantiles, len(col)),
                output_distribution="normal",
                random_state=random_state,
            )
            qt.fit(col.reshape(-1, 1))
            normalizer.stats_[c] = qt

    print(f"Features: {feature_cols}")
    return normalizer, feature_cols


def save_to_s3(local_path: str, bucket: str, key: str):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded to s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit scalers from .pt tensor files")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            "s3://medwav-dev-data/preprocessed_extended/",
        ],
        help="Directories containing .pt files",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Training years to include, e.g. --years 2018 2019 2020 2021. None = all.",
    )
    parser.add_argument(
        "--region-filter",
        choices=["atlantic", "mediterranean"],
        default=None,
        help="Region filter: 'atlantic' (lon < -5.5) or 'mediterranean' (lon >= -5.5). None = full domain.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--scaler-name", default="BU24h_zscore_target_19_21_all_with_corrected_extended")
    parser.add_argument("--mode", default="zscore", choices=["zscore", "quantile"])
    parser.add_argument("--target-feature", default="corrected_VHM0")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Low-memory mode: compute stats incrementally instead of loading all data at once.",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print(f"Fitting normalizer: {args.scaler_name}")
    print(f"{'=' * 80}")

    if args.streaming:
        normalizer, feature_cols = fit_streaming(
            data_dirs=args.data_dirs,
            mode=args.mode,
            years=args.years,
            region_filter=args.region_filter,
            max_files=args.max_files,
            target_feature=args.target_feature,
        )
    else:
        X_parts = []
        feature_cols = None
        for data_dir in args.data_dirs:
            X_part, fc = load_pt_files(
                data_dir,
                years=args.years,
                region_filter=args.region_filter,
                max_files=args.max_files,
            )
            if feature_cols is None:
                feature_cols = fc
            elif fc != feature_cols:
                raise ValueError("Feature mismatch across directories")
            X_parts.append(X_part)

        X = np.concatenate(X_parts, axis=0)
        print(f"\nCombined data: {X.shape[0]:,} samples, {X.shape[1]} channels")
        print(f"Features: {feature_cols}")

        if X.ndim == 2:
            X = X.reshape(-1, 1, 1, X.shape[-1])

        normalizer = WaveNormalizer(mode=args.mode)

        print(f"Data shape: {X.shape}")
        print(f"Number of features: {len(feature_cols)}")

        if len(feature_cols) != X.shape[-1]:
            raise ValueError(
                f"Mismatch: feature_cols has {len(feature_cols)} items but data has {X.shape[-1]} channels"
            )

        normalizer.fit(X, feature_order=feature_cols, target_feature_name=args.target_feature)
        del X, X_parts

    # Validation
    print("\nNormalizer metadata:")
    print(f"  Feature order: {normalizer.feature_order_}")
    print(f"  Target feature: {normalizer.target_feature_name_}")
    print(f"  Number of stats channels: {len(normalizer.stats_)}")

    if normalizer.feature_order_ and normalizer.target_feature_name_:
        try:
            target_idx = normalizer.feature_order_.index(normalizer.target_feature_name_)
            print(f"  Target '{normalizer.target_feature_name_}' at index: {target_idx}")
            if target_idx in normalizer.stats_:
                stats = normalizer.stats_[target_idx]
                if hasattr(stats, "mean_"):
                    print(f"  Target stats: mean={stats.mean_[0]:.6f}, scale={stats.scale_[0]:.6f}")
        except ValueError:
            print(f"  WARNING: Target '{normalizer.target_feature_name_}' not found in feature_order!")

    print(f"{'=' * 80}\n")

    # Save locally
    local_path = os.path.join(LOCAL_TMP, f"{args.scaler_name}.pkl")
    os.makedirs(LOCAL_TMP, exist_ok=True)
    normalizer.save(local_path)
    print(f"Saved normalizer to {local_path}")

    # Upload to S3
    if not args.no_s3:
        s3_key = f"{S3_PREFIX}{args.scaler_name}.pkl"
        normalizer.save_to_s3(local_path, S3_BUCKET, s3_key)
        print(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
    else:
        print("Skipping S3 upload (--no-s3)")
