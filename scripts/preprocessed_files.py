import os
import glob
import torch
import pyarrow.parquet as pq
import numpy as np
import fsspec
import sys

# 📂 Input/output paths
INPUT_DIR = "s3://medwav-dev-data/parquet/hourly/year=2020/"   # folder with WAVEAN*.parquet
OUTPUT_DIR = "s3://medwav-dev-data/preprocessed/"
# Create directory only for local output paths
if not OUTPUT_DIR.startswith("s3://"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🛠️ Helper to load parquet into dense tensor (T,H,W,C)
def load_parquet_as_tensor(path, excluded_columns=None, subsample_step=5, is_s3_input=False):
    import gc
    excluded_columns = excluded_columns or []
    if is_s3_input:
        # Open the S3 object to a real file-like handle for pyarrow
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh)
    else:
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

    # Clean up intermediate objects
    del table, time_data, lat_data, lon_data, arr
    del unique_times, unique_lats, unique_lons
    del time_idx, lat_idx, lon_idx
    gc.collect()

    return tensor, feature_cols


from multiprocessing import Pool, cpu_count

def process_file(path):
    import gc
    base = os.path.basename(path).replace(".parquet", ".pt")
    # base = "subsampled_step_5_" + base
    
    # Properly construct output path for S3 or local
    if OUTPUT_DIR.startswith("s3://"):
        out_path = OUTPUT_DIR.rstrip("/") + "/" + base
    else:
        out_path = os.path.join(OUTPUT_DIR, base)
    
    # Check if output file exists (handle both local and S3)
    try:
        if out_path.startswith("s3://"):
            fs = fsspec.filesystem("s3")
            if fs.exists(out_path):
                print(f"✓ Skipping (already exists): {base}")
                return f"Skipped {base}"
        elif os.path.exists(out_path):
            print(f"✓ Skipping (already exists): {base}")
            return f"Skipped {base}"
    except Exception as e:
        print(f"⚠ Warning: Could not check if {base} exists: {e}")
    
    print(f"⚙ Processing: {path}")
    is_s3_input = path.startswith("s3://")
    is_s3_output = out_path.startswith("s3://")
    tensor, feature_cols = load_parquet_as_tensor(path, excluded_columns=None, subsample_step=1, is_s3_input=is_s3_input)
    if is_s3_output:
        with fsspec.open(out_path, "wb") as f:
            torch.save({"tensor": tensor, "feature_cols": feature_cols}, f)
    else:
        torch.save({"tensor": tensor, "feature_cols": feature_cols}, out_path)
    
    # Clean up memory aggressively
    del tensor, feature_cols
    gc.collect()
    
    # Clear PyTorch cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return f"Done {base}"

if __name__ == "__main__":
    if INPUT_DIR.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        files = sorted(fs.glob(INPUT_DIR.rstrip("/") + "/WAVEAN*.parquet"))
        # Ensure S3 scheme; some fsspec/S3FS methods may return paths without scheme
        files = [p if p.startswith("s3://") else f"s3://{p}" for p in files]
    else:
        files = sorted(glob.glob(os.path.join(INPUT_DIR, "WAVEAN2021*.parquet")))
    
    print(f"\n{'='*60}")
    print(f"Found {len(files)} total files to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    skipped_count = 0
    processed_count = 0
    
    # maxtasksperchild=1 restarts workers after each file to prevent memory leaks
    pool = Pool(processes=cpu_count()//2, maxtasksperchild=1)
    try:
        for i, msg in enumerate(pool.imap_unordered(process_file, files), 1):
            if "Skipped" in msg:
                skipped_count += 1
            elif "Done" in msg:
                processed_count += 1
            print(f"[{i}/{len(files)}] {msg}")
            sys.stdout.flush()  # Force immediate output
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user. Terminating workers...")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        pool.terminate()
        pool.join()
        raise
    else:
        # Normal completion - close pool gracefully
        print("\n✓ All files processed. Closing worker pool...")
        sys.stdout.flush()
        pool.close()
        pool.join()
        print("✓ Worker pool closed successfully.")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Total:     {len(files)}")
    print(f"{'='*60}\n")

# if __name__ == "__main__":
#     main()
