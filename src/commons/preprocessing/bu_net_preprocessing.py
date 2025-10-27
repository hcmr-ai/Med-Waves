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
