import numpy as np
import polars as pl
import xarray as xr


def extract_features_from_file(x_path, y_path, use_dask=False):
    xds = xr.open_dataset(x_path, chunks="auto" if use_dask else None)
    yds = xr.open_dataset(y_path, chunks="auto" if use_dask else None)

    x_vhm0 = xds["VHM0"].values
    wspd = xds["WSPD"].values
    y_vhm0 = yds["VHM0"].values

    # All should have the same shape: (time, lat, lon)
    assert x_vhm0.shape == wspd.shape == y_vhm0.shape

    # Create coordinate grids
    time_len, lat_len, lon_len = x_vhm0.shape
    time = np.repeat(xds.time.values, lat_len * lon_len)
    lat = np.tile(np.repeat(xds.latitude.values, lon_len), time_len)
    lon = np.tile(np.tile(xds.longitude.values, lat_len), time_len)

    df = pl.DataFrame(
        {
            "vhm0_x": x_vhm0.flatten(),
            "wspd": wspd.flatten(),
            "vhm0_y": y_vhm0.flatten(),
            "lat": lat,
            "lon": lon,
            "time": time,
        }
    ).drop_nulls()

    return df
