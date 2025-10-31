import boto3
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler


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

    def fit(self, X):
        """
        Fit normalizer on training data.
        X shape: (N, H, W, C) or (N, C) flattened per channel
        """
        n_channels = X.shape[-1]
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
    
    def transform_torch(self, X):
        """
        PyTorch-optimized transform that avoids numpy conversions.
        X shape: (H, W, C) or (C, H, W) - torch tensor
        Returns: torch tensor of same shape
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
        
        return X_out

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
        joblib.dump({"mode": self.mode, "stats": self.stats_}, path)

    def load(self, path):
        obj = joblib.load(path)
        self.mode = obj["mode"]
        self.stats_ = obj["stats"]
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
