import glob
import os
from pathlib import Path

import polars as pl
import xarray as xr
from tqdm import tqdm


def convert_netcdf_to_parquet_hourly(netcdf_file_path: Path, output_dir: Path, log_steps: bool = False) -> None:
    """
    Convert a NetCDF file into a flat hourly Parquet file using Polars.

    Parameters
    ----------
    netcdf_file_path : Path
        Path to the input NetCDF file.
    output_file_path : Path
        Path to the output Parquet file where the hourly data will be stored.

    Notes
    -----
    - Assumes the NetCDF contains hourly measurements.
    - Converts all available variables and metadata into a flat DataFrame.
    """
    ds = xr.open_dataset(netcdf_file_path)
    df = ds.to_dataframe().reset_index()
    pl_df = pl.DataFrame(df)

    pl_df.write_parquet(output_dir)
    if log_steps:
        print(f"✅ Saved: {output_dir}")


def main():
    """
    Main function to batch-convert all NetCDF files in a directory into hourly Parquet files.

    Steps:
    - Scans the input directory for `.nc` files.
    - Converts each file individually and stores the result in the output directory.
    - Displays progress using tqdm.
    """
    netcdf_dir = Path("/data/tsolis/AI_project/with_reduced")
    output_dir = Path("/data/tsolis/AI_project/parquet/with_reduced/hourly")
    output_dir.mkdir(parents=True, exist_ok=True)

    netcdf_files = sorted(glob.glob(os.path.join(netcdf_dir, "*202306*.nc")))
    print(f"Found {len(netcdf_files)} NetCDF files.")

    for nc_file in tqdm(netcdf_files, desc="Converting NetCDF to Parquet"):
        date_str = nc_file.split("/")[-1].split(".nc")[0]
        out_path = f"{output_dir}/{date_str}.parquet"

        convert_netcdf_to_parquet_hourly(nc_file, out_path, True)


if __name__ == "__main__":
    main()
