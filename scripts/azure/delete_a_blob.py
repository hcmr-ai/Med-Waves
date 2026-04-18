import os

from azure.storage.blob import BlobServiceClient
from collections import defaultdict

conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
if not conn_str:
    raise SystemExit(
        "Set AZURE_STORAGE_CONNECTION_STRING (full Azure Storage connection string)."
    )
client = BlobServiceClient.from_connection_string(conn_str)
container = client.get_container_client("medwav-data")

# # Target just this one folder
# target_prefix = "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_64_19-22_debug"

# count = 0
# for blob in container.list_blobs(name_starts_with="raw/"):
#     container.get_blob_client(blob.name).delete_blob()
#     count += 1
#     if count % 100 == 0:
#         print(f"Deleted {count} blobs...")

# print(f"Done. Deleted {count} blobs.")



# to_delete = [
#     "checkpoints/checkpoints_100_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual",
#     "checkpoints/checkpoints_200_val_test_23_nick_17-22_large_filters_mse_64_enhanced",
#     "checkpoints/checkpoints_full_20-21_huber_64_lambda_lr_256",
#     "checkpoints/dnn_training_full_20-21_mse_64_lambda_lr_128",
#     "checkpoints/dnn_training_full_20-21_mse_64_lambda_lr_256",
#     "checkpoints/dnn_training_subsample_step_5",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_mse_64_lambda_lr_swinunet_VHM0",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_64_lambda_lr",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_64_lambda_lr_VTM02",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_mdn_64_lambda_lr",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_mse_perceptual_64_lambda_lr",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_17-21_weighted_mse_64_lambda_lr_VHM0_adamw",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_gan_17-21_mse_64_lambda_lr",
#     "checkpoints/dnn_training_subsample_step_5_100_val_22_test_23_transunet_gan_17-21_mse_gan_64_lambda_lr_VHM0",
#     "checkpoints/dnn_training_subsample_step_5_100_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual",
#     "checkpoints/dnn_training_subsample_step_5_100_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_patch_bin_balanced",
#     "checkpoints/dnn_training_subsample_step_5_100_val_test_23_nick_17-22_light_weighted_smooth_l1_64_enhanced_no_residual",
#     "checkpoints/dnn_training_subsample_step_5_100_val_test_23_transunet_17-22_mse_64",
#     "checkpoints/dnn_training_subsample_step_5_20_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_finetune_pixel_switch",
#     "checkpoints/dnn_training_subsample_step_5_20_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_finetune_pixel_switch_freeze",
#     "checkpoints/dnn_training_subsample_step_5_20_val_test_23_nick_17-22_light_mse_64_enhanced_no_residual_finetune_pixel_switch_stable_freeze",
#     "checkpoints/dnn_training_subsample_step_5_30_val_test_23_nick_18-22_large_filters_smooth_l1_128_step_lr_group_norm",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_bias",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_bias_100",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_bias_100_val_test_23",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_19-22_large_filters",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_19-22_large_filters_smooth_l1_128",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_19-22_large_filters_smooth_l1_128_step_lr",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_64_19-22",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_nick_64_19-22_debug",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout_no_early_stopping",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout_no_early_stopping_nick",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout_no_early_stopping_nick_48",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout_no_early_stopping_nick_48_19-22",
#     "checkpoints/dnn_training_subsample_step_5_small_filters_normalize_target_100_val_test_23_no_dropout_no_early_stopping_nick_weight_decay_1e-4",
# ]

# def fmt(b):
#     return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

# total_blobs = 0
# total_size = 0

# for prefix in to_delete:
#     count = 0
#     size = 0
#     for blob in container.list_blobs(name_starts_with=prefix + "/"):
#         size += blob.size or 0
#         container.get_blob_client(blob.name).delete_blob()
#         count += 1
#     print(f"  {prefix.split('/')[-1]}: {count} blobs, {fmt(size)}")
#     total_blobs += count
#     total_size += size

# print(f"\nTotal deleted: {total_blobs} blobs, {fmt(total_size)}")
# print(f"Monthly saving: ~€{total_size / 1024**3 * 0.018:.2f}")

def fmt(b):
    return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"

total_blobs = 0
total_size = 0

for blob in container.list_blobs(name_starts_with="preprocessed/"):
    total_size += blob.size or 0
    container.get_blob_client(blob.name).delete_blob()
    total_blobs += 1
    if total_blobs % 100 == 0:
        print(f"Deleted {total_blobs} blobs...")

print(f"\nTotal deleted: {total_blobs} blobs, {fmt(total_size)}")
print(f"Monthly saving: ~€{total_size / 1024**3 * 0.018:.2f}")