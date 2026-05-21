"""Load med_simple23.csv into a pandas DataFrame (pandas required)."""

from pathlib import Path

import pandas as pd

# DATA_PATH = Path(__file__).resolve().parent / "med_simple23.csv"
DATA_PATH = Path(__file__).resolve().parent / "grid_point_timeseries_300.csv"
# DATA_PATH = Path(__file__).resolve().parent / "mlp_points_300.csv"


def load_data(csv_path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())
    print(df.dtypes)
