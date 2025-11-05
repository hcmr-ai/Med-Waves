import os
import glob
import torch
import pyarrow.parquet as pq
import numpy as np

# 📂 Input/output paths
INPUT_DIR = "/mnt/ebs/year=2021"   # folder with WAVEAN*.parquet
OUTPUT_DIR = "/opt/dlami/nvme/preprocessed_subsampled_step_5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🛠️ Helper to load parquet into dense tensor (T,H,W,C)
def load_parquet_as_tensor(path, excluded_columns=None, subsample_step=5):
    excluded_columns = excluded_columns or []
    table = pq.read_table(path)

    column_names = [field.name for field in table.schema]
    feature_cols = [col for col in column_names if col not in excluded_columns]

    time_data = table.column("time").to_numpy()
    lat_data = table.column("latitude").to_numpy()
    lon_data = table.column("longitude").to_numpy()

    unique_times = np.unique(time_data)
    unique_lats = np.unique(lat_data)
    unique_lons = np.unique(lon_data)

    T, H, W = len(unique_times), len(unique_lats), len(unique_lons)

    time_idx = np.searchsorted(unique_times, time_data)
    lat_idx = np.searchsorted(unique_lats, lat_data)
    lon_idx = np.searchsorted(unique_lons, lon_data)

    arr = np.full((T, H, W, len(feature_cols)), np.nan, dtype=np.float32)

    for j, col in enumerate(feature_cols):
        arr[time_idx, lat_idx, lon_idx, j] = table.column(col).to_numpy()

    tensor = torch.from_numpy(arr)  # (T,H,W,C)

    if subsample_step > 1:
        tensor = tensor[:, ::subsample_step, ::subsample_step, :].clone()

    return tensor, feature_cols


from multiprocessing import Pool, cpu_count

def process_file(path):
    base = os.path.basename(path).replace(".parquet", ".pt")
    # base = "subsampled_step_5_" + base
    out_path = os.path.join(OUTPUT_DIR, base)
    if os.path.exists(out_path):
        return f"Skipping {base}"
    tensor, feature_cols = load_parquet_as_tensor(path)
    torch.save({"tensor": tensor, "feature_cols": feature_cols}, out_path)
    return f"Done {base}"

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "WAVEAN2021*.parquet")))
    with Pool(processes=cpu_count()//2) as pool:  # half CPU cores
        for msg in pool.imap_unordered(process_file, files):
            print(msg)

# if __name__ == "__main__":
#     main()
