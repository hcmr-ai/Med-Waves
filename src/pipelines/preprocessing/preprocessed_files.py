import glob
import os
import sys

import fsspec
import numpy as np
import pyarrow.parquet as pq
import torch

# 📂 Configuration
INPUT_DIR = "s3://medwav-dev-data/parquet/hourly/year=2022/"   # folder with WAVEAN*.parquet
OUTPUT_DIR = "s3://medwav-dev-data/preprocessed_hourly/"        # output directory

# 🔧 SAVE_HOURLY option:
#   True:  Save each hour as separate file (e.g., WAVEAN20200101_h00.pt ... _h23.pt)
#          File size: ~46 MB per file, 17,520 total files (730 days × 24 hours)
#          Benefits: Better for multi-worker DataLoader, less disk I/O contention
#   False: Save entire day as one file (e.g., WAVEAN20200101.pt)
#          File size: ~1.1 GB per file, 730 total files
SAVE_HOURLY = True

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


from multiprocessing import Pool


def process_file(path):
    import gc
    base = os.path.basename(path).replace(".parquet", "")  # Remove .parquet only, no .pt yet

    print(f"⚙ Processing: {path}")
    is_s3_input = path.startswith("s3://")
    is_s3_output = OUTPUT_DIR.startswith("s3://")
    tensor, feature_cols = load_parquet_as_tensor(path, excluded_columns=None, subsample_step=1, is_s3_input=is_s3_input)
    T = tensor.shape[0]

    if SAVE_HOURLY:
        # Save each hour as a separate file
        saved_files = []
        for hour_idx in range(T):
            # Extract single hour and clone to create independent storage: (H, W, C)
            # .clone() is critical - without it, sliced tensor shares storage with original!
            hour_tensor = tensor[hour_idx].clone()

            # Create filename: WAVEAN20200101_h00.pt, WAVEAN20200101_h01.pt, etc.
            hour_filename = f"{base}_h{hour_idx:02d}.pt"

            if OUTPUT_DIR.startswith("s3://"):
                out_path = OUTPUT_DIR.rstrip("/") + "/" + hour_filename
            else:
                out_path = os.path.join(OUTPUT_DIR, hour_filename)

            # Check if already exists
            try:
                if out_path.startswith("s3://"):
                    fs = fsspec.filesystem("s3")
                    if fs.exists(out_path):
                        print(f"Skipping {hour_filename} because it already exists")
                        saved_files.append(f"skipped {hour_filename}")
                        continue
                elif os.path.exists(out_path):
                    saved_files.append(f"skipped {hour_filename}")
                    continue
            except Exception:
                pass

            # Save hour with compression (pickle protocol 4 is more efficient)
            data = {"tensor": hour_tensor, "feature_cols": feature_cols, "hour": hour_idx}
            if is_s3_output:
                with fsspec.open(out_path, "wb") as f:
                    torch.save(data, f, pickle_protocol=4)
            else:
                torch.save(data, out_path, pickle_protocol=4)

            saved_files.append(f"saved {hour_filename}")

        # Clean up memory
        del tensor, feature_cols
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = f"Done {base}: {len([s for s in saved_files if 'saved' in s])}/{T} hours"
        return result
    else:
        # Original: Save entire day as one file
        out_filename = f"{base}.pt"
        if OUTPUT_DIR.startswith("s3://"):
            out_path = OUTPUT_DIR.rstrip("/") + "/" + out_filename
        else:
            out_path = os.path.join(OUTPUT_DIR, out_filename)

        # Check if exists
        try:
            if out_path.startswith("s3://"):
                fs = fsspec.filesystem("s3")
                if fs.exists(out_path):
                    return f"Skipped {out_filename}"
            elif os.path.exists(out_path):
                return f"Skipped {out_filename}"
        except Exception:
            pass

        # Save daily file
        if is_s3_output:
            with fsspec.open(out_path, "wb") as f:
                torch.save({"tensor": tensor, "feature_cols": feature_cols}, f)
        else:
            torch.save({"tensor": tensor, "feature_cols": feature_cols}, out_path)

        # Clean up memory
        del tensor, feature_cols
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return f"Done {out_filename}"

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
    pool = Pool(processes=14, maxtasksperchild=1)
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
    print("Summary:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Total:     {len(files)}")
    print(f"{'='*60}\n")

# if __name__ == "__main__":
#     main()
