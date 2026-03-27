import numpy as np
import polars as pl
from scipy.interpolate import interp1d


class EDCDFCorrector:
    def __init__(self):
        self.cdf_models = {}

    def fit(
        self, df: pl.DataFrame, variables: list[str], corrected_suffix="corrected_"
    ):
        for var in variables:
            model_raw = df[var].to_numpy().astype(float)
            obs_raw = df[corrected_suffix + var].to_numpy().astype(float)

            mask = np.isfinite(model_raw) & np.isfinite(obs_raw)
            model_clean = model_raw[mask]
            obs_clean = obs_raw[mask]

            model_sorted = np.sort(model_clean)
            obs_sorted = np.sort(obs_clean)
            n = len(model_sorted)

            if n < 2:
                raise ValueError(
                    f"Variable '{var}': need at least 2 finite paired samples, got {n}"
                )

            p = (np.arange(n) + 0.5) / n

            # F_model: value -> probability (mid-rank CDF for ties)
            model_unique, inverse = np.unique(model_sorted, return_inverse=True)
            p_model_unique = np.bincount(inverse, weights=p) / np.bincount(inverse)
            f_model_cdf = interp1d(
                model_unique, p_model_unique, bounds_error=False, fill_value=(p[0], p[-1])
            )

            # Q_obs: probability -> observed value
            f_obs_quantile = interp1d(
                p, obs_sorted, bounds_error=False, fill_value=(obs_sorted[0], obs_sorted[-1])
            )

            self.cdf_models[var] = (f_model_cdf, f_obs_quantile, p[0], p[-1])

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        for var, (f_model_cdf, f_obs_quantile, p_min, p_max) in self.cdf_models.items():
            x = df[var].to_numpy().astype(float)
            prob = np.clip(f_model_cdf(x), p_min, p_max)
            corrected = f_obs_quantile(prob)
            df = df.with_columns(pl.Series(name=f"predicted_{var}", values=corrected))
        return df
