import numpy as np
import polars as pl
from scipy.interpolate import interp1d


class EDCDFCorrector:
    def __init__(self):
        self.cdf_models = {}

    def fit(self, df: pl.DataFrame, variables: list[str], corrected_suffix="corrected_"):
        for var in variables:
            model_train = df[var].to_numpy()
            obs_train = df[corrected_suffix + var].to_numpy()

            # Sort values and compute empirical CDFs
            model_sorted = np.sort(model_train)
            obs_sorted = np.sort(obs_train)
            cdf_model = np.linspace(0, 1, len(model_sorted))
            cdf_obs = np.linspace(0, 1, len(obs_sorted))

            # Store interpolators
            f_model_inv = interp1d(cdf_model, model_sorted, bounds_error=False, fill_value="extrapolate")
            f_obs_inv = interp1d(cdf_obs, obs_sorted, bounds_error=False, fill_value="extrapolate")
            f_model_cdf = interp1d(model_sorted, cdf_model, bounds_error=False, fill_value=(0,1))

            self.cdf_models[var] = (f_model_inv, f_obs_inv, f_model_cdf)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        for var, (f_model_inv, f_obs_inv, f_model_cdf) in self.cdf_models.items():
            x = df[var].to_numpy()
            F = f_model_cdf(x)
            corrected = x + f_obs_inv(F) - f_model_inv(F)
            df = df.with_columns(pl.Series(name=f"predicted_{var}", values=corrected))
        return df
