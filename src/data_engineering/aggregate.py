from pathlib import Path

import polars as pl


def aggregate_hourly_mean(
    year: str,
    main_parquet_data_path: str,
    exclude_cols: set | None = None
):
    """
    Aggregates the hourly mean for all columns in the specified Parquet files for a given year, excluding the columns 'time', 'latitude', and 'longitude'.

    This function scans all Parquet files matching the pattern 'WAVEAN{year}*.parquet' in the '*/hourly' subdirectory of the main data path.
    It computes the mean for each column (except 'time', 'latitude', and 'longitude') grouped by the 'time' column, and saves the resulting DataFrame as a
    Parquet file in the 'main_parquet_data_path/hourly_mean' subdirectory. If the output file already exists, the function skips computation.

    Parameters
    ----------
    year : str, optional
        The year for which to aggregate data. Used to match input files and name the output file. Default is '2023'.
    main_parquet_data_path : str, optional
        The root directory containing the Parquet data. Default is '/data/tsolis/AI_project/parquet'.

    Side Effects
    ------------
    - Reads Parquet files from disk.
    - Writes the aggregated hourly mean as a Parquet file to disk if it does not already exist.
    - Prints progress and status messages to stdout.

    Returns
    -------
    None
        This function does not return a value. The result is saved to disk.

    Example
    -------
    >>> aggregate_hourly_mean(year="2022", main_parquet_data_path="/data/myproject/parquet")
    """
    if exclude_cols is None:
        exclude_cols = {"time", "latitude", "longitude"}
    parquet_files_path = f"{main_parquet_data_path}/hourly/WAVEAN{year}*.parquet"
    output_path = f"{main_parquet_data_path}/hourly_mean/WAVEAN{year}.parquet"
    data_path = Path(output_path)

    if data_path.exists():
        print(f"File {data_path} already exists. Skipping aggregation.")
    else:
        print(f"Loading raw data from {parquet_files_path} and aggregating hourly means...")
        ts_scanned = pl.scan_parquet(parquet_files_path)
        ts_schema = ts_scanned.collect_schema()
        agg_cols = [col for col in ts_schema if col not in exclude_cols]
        cols_to_check = ["time"] + agg_cols

        agg_exprs = [pl.col(col).mean().alias(f"{col}_mean") for col in agg_cols]
        ts_df = (
            ts_scanned
            .drop_nulls(cols_to_check)
            .group_by("time")
            .agg(agg_exprs)
            .sort("time")
            .collect()
        )
        print(f"Saving hourly mean for {year} to {output_path}...")
        ts_df.write_parquet(output_path)
        print(f"Saved hourly mean for {year}.")

    print("Data aggregation process completed.")

if __name__ == "__main__":
    for year in ["2023"]:
        aggregate_hourly_mean(year=year, main_parquet_data_path="/data/tsolis/AI_project/parquet/augmented_with_labels")
