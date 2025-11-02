import logging

import boto3
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)

class WaveNormalizer:
    def __init__(self, mode="zscore", n_quantiles=1000, random_state=42):
        """
        mode: "zscore" or "quantile"
        n_quantiles: used for quantile transform
        """
        self.mode = mode
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.stats_ = {}  # per-channel mean/std or quantile model
        self.target_stats_ = None  # Optional target-specific scaler stats
        self.feature_order_ = None  # Optional: list of feature names in the order they were fitted
        self.target_feature_name_ = None  # Optional: name of target feature (e.g., "corrected_VHM0")

    def fit(self, X, feature_order=None, target_feature_name=None):
        """
        Fit normalizer on training data.
        X shape: (N, H, W, C) or (N, C) flattened per channel

        Args:
            X: Training data with shape (N, H, W, C) or (N, C)
            feature_order: Optional list of feature names in the order they appear in X
                          (should match the channel order). If None, target lookup by name won't work.
            target_feature_name: Optional name of the target feature (e.g., "corrected_VHM0").
                                If provided along with feature_order, enables automatic target stats lookup.
        """
        n_channels = X.shape[-1]

        # Store feature order and target name if provided
        if feature_order is not None:
            if len(feature_order) != n_channels:
                raise ValueError(
                    f"feature_order length ({len(feature_order)}) must match number of channels ({n_channels})"
                )
            self.feature_order_ = feature_order
            self.target_feature_name_ = target_feature_name
        elif target_feature_name is not None:
            # Warn if target_feature_name is provided but feature_order is not
            logger.warning(
                "target_feature_name provided without feature_order. "
                "Target lookup by name will not work. Please provide feature_order or set it manually."
            )

        for c in range(n_channels):
            data_c = X[..., c].ravel()
            data_c = data_c[~np.isnan(data_c)]  # drop NaNs

            if self.mode == "zscore":
                ss = StandardScaler()
                ss.fit(data_c.reshape(-1, 1))
                self.stats_[c] = ss
            elif self.mode == "quantile":
                qt = QuantileTransformer(
                    n_quantiles=min(self.n_quantiles, len(data_c)),
                    output_distribution="normal",
                    random_state=self.random_state,
                )
                qt.fit(data_c.reshape(-1, 1))
                self.stats_[c] = qt
            else:
                raise ValueError("Mode must be 'zscore' or 'quantile'")
        return self

    def transform(self, X):
        """
        Apply normalization
        X shape: (N, H, W, C)
        """
        X_out = np.empty_like(X, dtype=np.float32)
        n_channels = X.shape[-1]
        for c in range(n_channels):
            data_c = X[..., c]
            if self.mode == "zscore":
                if isinstance(self.stats_[c], tuple):
                    # Handle tuple format (mean, std)
                    mean, std = self.stats_[c]
                    X_out[..., c] = (data_c - mean) / (std + 1e-6)
                else:
                    # Handle StandardScaler object
                    scaler = self.stats_[c]
                    flat = data_c.reshape(-1, 1)
                    trans = scaler.transform(flat)
                    X_out[..., c] = trans.reshape(data_c.shape)
            else:  # quantile
                qt = self.stats_[c]
                flat = data_c.reshape(-1, 1)
                trans = qt.transform(flat)
                X_out[..., c] = trans.reshape(data_c.shape)
        return X_out

    def transform_torch(self, X, normalize_target=False, target=None, target_channel_index=0):
        """
        PyTorch-optimized transform that avoids numpy conversions.
        X shape: (H, W, C) or (C, H, W) - torch tensor
        target: Optional target tensor to normalize (H, W) or (H, W, 1) or (1, H, W)
        target_channel_index: Which channel index in stats_ to use for target normalization (default: 0)
        Returns: torch tensor of same shape as X, and optionally normalized target if provided
        """
        import torch

        # Handle different input shapes
        if X.dim() == 4:  # (N, H, W, C) or (N, C, H, W)
            if X.shape[1] == self.stats_.__len__():  # (N, C, H, W)
                needs_permute = True
                X = X.permute(0, 2, 3, 1)  # (N, H, W, C)
            else:  # (N, H, W, C)
                needs_permute = False
            batch_size = X.shape[0]
        elif X.dim() == 3:
            if X.shape[0] == self.stats_.__len__():  # (C, H, W)
                needs_permute = True
                X = X.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
                batch_size = 1
            else:  # (H, W, C)
                X = X.unsqueeze(0)  # (1, H, W, C)
                needs_permute = False
                batch_size = 1
        else:
            raise ValueError(f"Unsupported tensor shape: {X.shape}")

        X_out = X.clone()
        n_channels = X.shape[-1]

        for c in range(n_channels):
            data_c = X[..., c]  # (N, H, W) or (H, W)

            if self.mode == "zscore":
                if isinstance(self.stats_[c], tuple):
                    # Fast path: direct computation
                    mean, std = self.stats_[c]
                    X_out[..., c] = (data_c - mean) / (std + 1e-6)
                else:
                    # Fallback: convert to numpy, transform, convert back
                    data_np = data_c.detach().cpu().numpy()
                    flat = data_np.reshape(-1, 1)
                    scaler = self.stats_[c]
                    trans = scaler.transform(flat)
                    X_out[..., c] = torch.from_numpy(
                        trans.reshape(data_np.shape)
                    ).to(X.device).type_as(X)
            else:  # quantile - must use numpy
                data_np = data_c.detach().cpu().numpy()
                flat = data_np.reshape(-1, 1)
                qt = self.stats_[c]
                trans = qt.transform(flat)
                X_out[..., c] = torch.from_numpy(
                    trans.reshape(data_np.shape)
                ).to(X.device).type_as(X)

        # Remove batch dimension if it was added
        if batch_size == 1:
            X_out = X_out.squeeze(0)

        # Permute back if needed
        if needs_permute:
            if X_out.dim() == 4:  # (N, H, W, C)
                X_out = X_out.permute(0, 3, 1, 2)  # (N, C, H, W)
            else:  # (H, W, C)
                X_out = X_out.permute(2, 0, 1)  # (C, H, W)

        # Normalize target if requested
        target_out = None
        if normalize_target and target is not None:
            # Determine target stats to use
            target_stats = None

            # Priority 1: Use explicitly set target_stats_
            if self.target_stats_ is not None:
                target_stats = self.target_stats_

            # Priority 2: Look up by feature name if feature_order_ is available
            elif self.feature_order_ is not None and self.target_feature_name_ is not None:
                try:
                    target_idx = self.feature_order_.index(self.target_feature_name_)
                    if target_idx in self.stats_:
                        target_stats = self.stats_[target_idx]
                except ValueError:
                    pass

            # Priority 3: Use explicitly provided channel index
            elif target_channel_index in self.stats_:
                target_stats = self.stats_[target_channel_index]

            # Priority 4: Fallback to last channel (common case where corrected_VHM0 is last in fit_scalers.py)
            elif len(self.stats_) > 0:
                target_stats = self.stats_[len(self.stats_) - 1]

            if target_stats is None:
                raise ValueError(
                    "Target stats not found. Options:\n"
                    "  1. Set target_stats_ attribute\n"
                    "  2. Set feature_order_ and target_feature_name_ attributes\n"
                    "  3. Pass target_channel_index that exists in stats_\n"
                    "  4. Ensure stats_ is populated (will default to last channel)"
                )

            # Handle target shape - ensure it's (H, W) for normalization
            target_shape = target.shape
            if target.dim() == 3:
                if target.shape[0] == 1:  # (1, H, W)
                    target_2d = target.squeeze(0)  # (H, W)
                    needs_target_unsqueeze = True
                elif target.shape[-1] == 1:  # (H, W, 1)
                    target_2d = target.squeeze(-1)  # (H, W)
                    needs_target_unsqueeze = True
                else:
                    raise ValueError(f"Unsupported target shape: {target.shape}")
            elif target.dim() == 2:  # (H, W)
                target_2d = target
                needs_target_unsqueeze = False
            else:
                raise ValueError(f"Unsupported target shape: {target.shape}")

            # Normalize target
            target_2d_np = target_2d.detach().cpu().numpy()
            flat_target = target_2d_np.reshape(-1, 1)

            if self.mode == "zscore":
                if isinstance(target_stats, tuple):
                    # Fast path: direct computation
                    mean, std = target_stats
                    target_normalized = (target_2d_np - mean) / (std + 1e-6)
                else:
                    # Fallback: use StandardScaler
                    trans = target_stats.transform(flat_target)
                    target_normalized = trans.reshape(target_2d_np.shape)
            else:  # quantile
                trans = target_stats.transform(flat_target)
                target_normalized = trans.reshape(target_2d_np.shape)

            # Convert back to torch and restore original shape
            target_out = torch.from_numpy(target_normalized).to(target.device).type_as(target)
            if needs_target_unsqueeze:
                if target_shape[0] == 1:  # Was (1, H, W)
                    target_out = target_out.unsqueeze(0)  # (1, H, W)
                elif target_shape[-1] == 1:  # Was (H, W, 1)
                    target_out = target_out.unsqueeze(-1)  # (H, W, 1)

        if target_out is not None:
            return X_out, target_out
        else:
            return X_out

    def inverse_transform_torch(self, target, target_channel_index=0):
        """
        PyTorch-optimized inverse transform for targets.
        target: torch tensor (H, W) or (1, H, W) or (H, W, 1)
        target_channel_index: Which channel index in stats_ to use for target denormalization
        Returns: denormalized target tensor of same shape
        """
        import torch

        # Find target stats
        target_stats = None

        # Priority 1: Use explicitly set target_stats_
        if self.target_stats_ is not None:
            target_stats = self.target_stats_

        # Priority 2: Look up by feature name if feature_order_ is available
        elif self.feature_order_ is not None and self.target_feature_name_ is not None:
            try:
                target_idx = self.feature_order_.index(self.target_feature_name_)
                if target_idx in self.stats_:
                    target_stats = self.stats_[target_idx]
            except ValueError:
                pass

        # Priority 3: Use explicitly provided channel index
        elif target_channel_index in self.stats_:
            target_stats = self.stats_[target_channel_index]

        # Priority 4: Fallback to last channel
        elif len(self.stats_) > 0:
            target_stats = self.stats_[len(self.stats_) - 1]

        if target_stats is None:
            raise ValueError(
                "Target stats not found for inverse transform. Options:\n"
                "  1. Set target_stats_ attribute\n"
                "  2. Set feature_order_ and target_feature_name_ attributes\n"
                "  3. Pass target_channel_index that exists in stats_\n"
                "  4. Ensure stats_ is populated (will default to last channel)"
            )

        # Handle target shape - ensure it's (H, W) for normalization
        target_shape = target.shape
        if target.dim() == 3:
            if target.shape[0] == 1:  # (1, H, W)
                target_2d = target.squeeze(0)  # (H, W)
                needs_target_unsqueeze = True
            elif target.shape[-1] == 1:  # (H, W, 1)
                target_2d = target.squeeze(-1)  # (H, W)
                needs_target_unsqueeze = True
            else:
                raise ValueError(f"Unsupported target shape: {target.shape}")
        elif target.dim() == 2:  # (H, W)
            target_2d = target
            needs_target_unsqueeze = False
        else:
            raise ValueError(f"Unsupported target shape: {target.shape}")

        # Denormalize target
        target_2d_np = target_2d.detach().cpu().numpy()
        flat_target = target_2d_np.reshape(-1, 1)

        if self.mode == "zscore":
            if isinstance(target_stats, tuple):
                # Fast path: direct computation
                mean, std = target_stats
                target_denormalized = target_2d_np * std + mean
            else:
                # Fallback: use StandardScaler
                inv = target_stats.inverse_transform(flat_target)
                target_denormalized = inv.reshape(target_2d_np.shape)
        else:  # quantile
            inv = target_stats.inverse_transform(flat_target)
            target_denormalized = inv.reshape(target_2d_np.shape)

        # Convert back to torch and restore original shape
        target_out = torch.from_numpy(target_denormalized).to(target.device).type_as(target)
        if needs_target_unsqueeze:
            if target_shape[0] == 1:  # Was (1, H, W)
                target_out = target_out.unsqueeze(0)  # (1, H, W)
            elif target_shape[-1] == 1:  # Was (H, W, 1)
                target_out = target_out.unsqueeze(-1)  # (H, W, 1)

        return target_out

    def inverse_transform(self, X):
        """Undo normalization (useful for predictions)"""
        X_out = np.empty_like(X, dtype=np.float32)
        n_channels = X.shape[-1]
        for c in range(n_channels):
            data_c = X[..., c]
            if self.mode == "zscore":
                if isinstance(self.stats_[c], tuple):
                    # Handle tuple format (mean, std)
                    mean, std = self.stats_[c]
                    X_out[..., c] = data_c * std + mean
                else:
                    # Handle StandardScaler object
                    scaler = self.stats_[c]
                    flat = data_c.reshape(-1, 1)
                    inv = scaler.inverse_transform(flat)
                    X_out[..., c] = inv.reshape(data_c.shape)
            else:  # quantile
                qt = self.stats_[c]
                flat = data_c.reshape(-1, 1)
                inv = qt.inverse_transform(flat)
                X_out[..., c] = inv.reshape(data_c.shape)
        return X_out

    def save(self, path):
        obj = {
            "mode": self.mode,
            "stats": self.stats_,
        }
        # Save optional metadata if available
        if self.feature_order_ is not None:
            obj["feature_order"] = self.feature_order_
        if self.target_feature_name_ is not None:
            obj["target_feature_name"] = self.target_feature_name_
        if self.target_stats_ is not None:
            obj["target_stats"] = self.target_stats_
        joblib.dump(obj, path)

    def load(self, path):
        obj = joblib.load(path)
        self.mode = obj["mode"]
        self.stats_ = obj["stats"]
        # Load optional metadata if available
        self.feature_order_ = obj.get("feature_order", None)
        self.target_feature_name_ = obj.get("target_feature_name", None)
        self.target_stats_ = obj.get("target_stats", None)
        return self

    def save_to_s3(self, local_path, bucket, key):
        import boto3

        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        print(f"✓ Uploaded to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"

    @classmethod
    def load_from_s3(cls, bucket, key, tmp_path="/tmp/scaler.pkl"):
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, tmp_path)
        normalizer = cls()
        return normalizer.load(tmp_path)
