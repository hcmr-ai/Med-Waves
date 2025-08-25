import polars as pl


class DeltaCorrector:
    def __init__(self):
        self.bias_per_variable = {}

    def fit(self, df: pl.DataFrame, variables: list[str], corrected_suffix="corrected_"):
        for var in variables:
            model_mean = df[var].mean()
            obs_mean = df[corrected_suffix + var].mean()
            self.bias_per_variable[var] = (model_mean - obs_mean)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        for var, bias in self.bias_per_variable.items():
            df = df.with_columns((pl.col(var) - bias).alias(f"predicted_{var}"))
        return df
