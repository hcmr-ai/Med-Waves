import os

from azure.storage.blob import BlobServiceClient

conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if not conn_str:
    raise SystemExit(
        "Set AZURE_STORAGE_CONNECTION_STRING (full Azure Storage connection string)."
    )
client = BlobServiceClient.from_connection_string(conn_str)
container = client.get_container_client("medwav-data")

# def fmt(b):
#     return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

# live_folders = defaultdict(int)
# deleted_folders = defaultdict(int)

# total = 0
# for blob in container.list_blobs(include=["deleted"]):
#     parts = blob.name.split("/")
#     key = parts[0]

#     size = blob.size or 0
#     if blob.deleted:
#         deleted_folders[key] += size
#     else:
#         live_folders[key] += size
#     total += 1
#     if total % 5000 == 0:
#         print(f"  scanned {total}...")

# print(f"\nLIVE ({len(live_folders)} folders):")
# for f, s in sorted(live_folders.items(), key=lambda x: -x[1]):
#     print(f"  /{f}: {fmt(s)}")

# print(f"\nDELETED ({len(deleted_folders)} folders):")
# for f, s in sorted(deleted_folders.items(), key=lambda x: -x[1]):
#     print(f"  /{f}: {fmt(s)}")


def fmt(b):
    return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

# Get subfolders in checkpoints/ (excluding checkpoints/checkpoints/)
# root_ckpt = defaultdict(int)
# for blob in container.list_blobs(name_starts_with="checkpoints/"):
#     if not blob.name.startswith("checkpoints/checkpoints/"):
#         parts = blob.name.split("/")
#         if len(parts) >= 2:
#             root_ckpt[parts[1]] += blob.size or 0

# # Get subfolders in checkpoints/checkpoints/
# nested_ckpt = defaultdict(int)
# for blob in container.list_blobs(name_starts_with="checkpoints/checkpoints/"):
#     parts = blob.name.split("/")
#     if len(parts) >= 3:
#         nested_ckpt[parts[2]] += blob.size or 0

# common = set(root_ckpt.keys()) & set(nested_ckpt.keys())
# only_root = set(root_ckpt.keys()) - set(nested_ckpt.keys())
# only_nested = set(nested_ckpt.keys()) - set(root_ckpt.keys())

# print(f"Common folders: {len(common)}")
# for f in sorted(common):
#     print(f"  {f}: root={fmt(root_ckpt[f])} nested={fmt(nested_ckpt[f])}")

# print(f"\nOnly in root: {len(only_root)}")
# for f in sorted(only_root):
#     print(f"  {f}: {fmt(root_ckpt[f])}")

# print(f"\nOnly in nested: {len(only_nested)}")
# for f in sorted(only_nested):
#     print(f"  {f}: {fmt(nested_ckpt[f])}")

to_delete = [
    "checkpoints/dnn_training_extended_subsampled_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_cos_delta",
    "checkpoints/dnn_training_extended_subsampled_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_cos_delta_atlantic",
    "checkpoints/dnn_training_extended_subsampled_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_cos_delta_mediterranean",
    "checkpoints/dnn_training_full_19-21_mse_64_lambda_lr_multitask",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mbw_64_lambda_lr",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_atlantic_filtered",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_atlantic_filtered_patch_sampling_exhaustive",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_mediterranean",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_bias_correction_mediterranean_filtered",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_64_lambda_lr_norm_vhm0_correction_mediterranean_filtered",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_huber_tail_64_lambda_lr_bias_correction_mediterranean_filtered",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_patch_sampling_64_lambda_lr",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_patch_sampling_64_lambda_lr_bias_correction",
    "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_18-21_mse_patch_sampling_64_lambda_lr_bias_correction_bin_sampling_weights",
]

total_blobs = 0
total_size = 0

for prefix in to_delete:
    count = 0
    size = 0
    for blob in container.list_blobs(name_starts_with=prefix + "/"):
        size += blob.size or 0
        container.get_blob_client(blob.name).delete_blob()
        count += 1
    print(f"  {prefix.split('/')[-1][:60]}: {count} blobs, {fmt(size)}")
    total_blobs += count
    total_size += size

print(f"\nTotal deleted: {total_blobs} blobs, {fmt(total_size)}")
print(f"Monthly saving: ~€{total_size / 1024**3 * 0.018:.2f}")



# Get filenames from nested folder
# nested = set()
# for blob in container.list_blobs(name_starts_with="preprocessed_extended/preprocessed_extended/"):
#     filename = blob.name.split("/")[-1]
#     nested.add(filename)

# # Get filenames from root folder (excluding the nested subfolder)
# root = set()
# for blob in container.list_blobs(name_starts_with="preprocessed_extended/"):
#     if not blob.name.startswith("preprocessed_extended/preprocessed_extended/"):
#         filename = blob.name.split("/")[-1]
#         root.add(filename)

# print(f"Nested folder: {len(nested)} files")
# print(f"Root folder (excl. nested): {len(root)} files")
# print(f"Files in both: {len(nested & root)}")
# print(f"Only in nested: {len(nested - root)}")
# print(f"Only in root: {len(root - nested)}")

# # Show sample overlap
# sample = list(nested & root)[:5]
# print(f"\nSample duplicates: {sample}")


# def fmt(b):
#     return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

# nested_count, nested_size = 0, 0
# root_count, root_size = 0, 0

# for blob in container.list_blobs(name_starts_with="preprocessed_extended/"):
#     s = blob.size or 0
#     if blob.name.startswith("preprocessed_extended/preprocessed_extended/"):
#         nested_count += 1
#         nested_size += s
#     else:
#         root_count += 1
#         root_size += s

# print(f"Root:   {root_count} files, {fmt(root_size)}, ~€{root_size / 1024**3 * 0.018:.2f}/month")
# print(f"Nested: {nested_count} files, {fmt(nested_size)}, ~€{nested_size / 1024**3 * 0.018:.2f}/month")
# print(f"Total:  {root_count+nested_count} files, {fmt(root_size+nested_size)}, ~€{(root_size+nested_size) / 1024**3 * 0.018:.2f}/month")


# nested = {}
# for blob in container.list_blobs(name_starts_with="preprocessed_extended/preprocessed_extended/"):
#     filename = blob.name.split("/")[-1]
#     nested[filename] = blob

# root = set()
# for blob in container.list_blobs(name_starts_with="preprocessed_extended/"):
#     if not blob.name.startswith("preprocessed_extended/preprocessed_extended/"):
#         root.add(blob.name.split("/")[-1])

# only_nested = {k: v for k, v in nested.items() if k not in root}
# print(f"10 files only in nested:")
# for name, blob in sorted(only_nested.items()):
#     print(f"  {name} — {blob.size/1024**2:.0f} MB — last modified: {blob.last_modified}")

# for name in sorted(only_nested.keys()):
#     print(f"  Missing from root: {name}")


# def fmt(b):
#     return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

# count = 0
# size = 0
# for blob in container.list_blobs(name_starts_with="preprocessed_extended/"):
#     if not blob.name.startswith("preprocessed_extended/preprocessed_extended/"):
#         size += blob.size or 0
#         container.get_blob_client(blob.name).delete_blob()
#         count += 1
#         if count % 100 == 0:
#             print(f"Deleted {count} blobs...")

# print(f"\nTotal deleted: {count} blobs, {fmt(size)}")
# print(f"Monthly saving: ~€{size / 1024**3 * 0.018:.2f}")