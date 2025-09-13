import polars as pl

# Path to your converted .parquet file
file_path = "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20231231.parquet"

# Load the file
df = pl.read_parquet(file_path)

print(df.shape)
print(df.schema)
print(df.head)

# Select only relevant columns
df_selected = df.select(["VHM0", "VTM02", "corrected_VHM0", "corrected_VTM02"])

# Print shape and schema
print(df_selected.shape)
print(df_selected.schema)

# Show first few rows
print(df_selected.head(n=20))

# print((df_selected["VHM0"] - df_selected["corrected_VHM0"]).abs().max())
# print((df_selected["VHM0"] - df_selected["corrected_VHM0"]).abs().mean())


# file_path_wo = "/data/tsolis/AI_project/parquet/without_reduced/hourly/WAVEAN20210101*"
# file_path_with = "/data/tsolis/AI_project/parquet/without_reduced/hourly/WAVEAN20210101*"

# df_wo = pl.read_parquet(file_path_wo)
# df_with = pl.read_parquet(file_path_with)
# # Select only relevant columns
# df_wo = df_wo.select(["VHM0", "VTM02"])
# df_with = df_with.select(["VHM0", "VTM02"])

# # Print shape and schema
# print(df_selected.shape)
# print(df_selected.schema)

# # Show first few rows
# print(df_with.head(n=20))
# print(df_wo.head(n=20))


# ds = xr.open_dataset("/data/tsolis/AI_project/with_reduced/WAVEAN20210101.nc")
# df = ds.to_dataframe().reset_index()
# pl_df = pl.DataFrame(df)
# print(pl_df.head(n=20))
