import os
from datetime import datetime, timezone
from pathlib import Path

import boto3


def get_file_metadata(file_path):
    """Get local file size and modification time."""
    if not os.path.exists(file_path):
        return None, None

    stat = os.stat(file_path)
    size = stat.st_size
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return size, mtime

def should_download(s3_obj, local_file_path):
    """
    Determine if file should be downloaded based on size and modification time.
    Returns True if file should be downloaded.
    """
    local_size, local_mtime = get_file_metadata(local_file_path)

    # File doesn't exist locally, download it
    if local_size is None:
        return True

    s3_size = s3_obj['Size']
    s3_mtime = s3_obj['LastModified']

    # Download if size differs
    if local_size != s3_size:
        return True

    # Download if S3 file is newer
    if s3_mtime > local_mtime:
        return True

    return False

def sync_s3_folder(bucket_name, s3_folder, local_dir, delete_extra=False):
    """
    Sync S3 folder to local directory (like aws s3 sync).

    Args:
        bucket_name: Name of the S3 bucket (without s3:// prefix)
        s3_folder: Path to the folder in S3 (e.g., 'path/to/folder/')
        local_dir: Local directory path where files will be synced
        delete_extra: If True, delete local files not present in S3
    """
    # Clean bucket name - remove s3:// prefix if present
    bucket_name = bucket_name.replace('s3://', '').strip('/')

    # If bucket_name still contains '/', extract bucket and update s3_folder
    if '/' in bucket_name:
        parts = bucket_name.split('/', 1)
        bucket_name = parts[0]
        if len(parts) > 1:
            s3_folder = parts[1] + ('/' + s3_folder if s3_folder else '')

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Ensure s3_folder ends with /
    if s3_folder and not s3_folder.endswith('/'):
        s3_folder += '/'

    # Create local directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # List all objects in the S3 folder
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

    downloaded_count = 0
    skipped_count = 0
    s3_files = set()

    print(f"Syncing s3://{bucket_name}/{s3_folder} to {local_dir}\n")

    for page in pages:
        if 'Contents' not in page:
            print(f"No files found in s3://{bucket_name}/{s3_folder}")
            break

        for obj in page['Contents']:
            s3_key = obj['Key']

            # Skip if it's the folder itself
            if s3_key.endswith('/'):
                continue

            # Get relative path from the s3_folder
            relative_path = s3_key[len(s3_folder):]
            local_file_path = os.path.join(local_dir, relative_path)
            s3_files.add(local_file_path)

            # Check if download is needed
            if should_download(obj, local_file_path):
                # Create subdirectories if needed
                local_file_dir = os.path.dirname(local_file_path)
                if local_file_dir:
                    Path(local_file_dir).mkdir(parents=True, exist_ok=True)

                # Download the file
                print(f"Downloading: {s3_key}")
                s3.download_file(bucket_name, s3_key, local_file_path)

                # Set the modification time to match S3
                mtime = obj['LastModified'].timestamp()
                os.utime(local_file_path, (mtime, mtime))

                downloaded_count += 1
            else:
                print(f"Skipping (up to date): {relative_path}")
                skipped_count += 1

    # Delete extra local files if requested
    deleted_count = 0
    if delete_extra:
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file = os.path.join(root, file)
                if local_file not in s3_files:
                    print(f"Deleting (not in S3): {local_file}")
                    os.remove(local_file)
                    deleted_count += 1

        # Remove empty directories
        for root, dirs, _ in os.walk(local_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError:
                    pass

    print(f"\n{'='*60}")
    print("Sync complete!")
    print(f"Downloaded: {downloaded_count} files")
    print(f"Skipped (up to date): {skipped_count} files")
    if delete_extra:
        print(f"Deleted (not in S3): {deleted_count} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Configuration
    BUCKET_NAME = "s3://medwav-dev-data/"
    S3_FOLDER = "preprocessed_subsampled_step_5/"  # e.g., "data/reports/2024/"
    LOCAL_DIR = "/data/tsolis/AI_project/preprocessed_subsampled_step_5"    # Local destination folder
    DELETE_EXTRA = False                # Set to True to delete local files not in S3

    # Run the sync
    sync_s3_folder(BUCKET_NAME, S3_FOLDER, LOCAL_DIR, DELETE_EXTRA)
