"""
Download a checkpoint folder from Azure Blob Storage.

Usage:
    python scripts/azure/download_checkpoint.py \
        --folder checkpoints/<run_name> \
        --output /local/path/to/save

    # list available checkpoint folders first:
    python scripts/azure/download_checkpoint.py --list
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

from azure.storage.blob import BlobServiceClient

CONTAINER = "medwav-data"
CHECKPOINTS_PREFIX = "checkpoints/"


def fmt(b: int) -> str:
    return f"{b / 1024**3:.2f} GB" if b < 1024**4 else f"{b / 1024**4:.2f} TiB"


def list_checkpoint_folders(container) -> None:
    """Print all top-level checkpoint folders with their sizes."""
    folders: dict[str, int] = defaultdict(int)
    for blob in container.list_blobs(name_starts_with=CHECKPOINTS_PREFIX):
        parts = blob.name.split("/")
        # parts[0] = "checkpoints", parts[1] = run folder name
        if len(parts) >= 2:
            folders[parts[1]] += blob.size or 0

    print(f"{'Folder':<80} {'Size':>10}")
    print("-" * 92)
    for name, size in sorted(folders.items(), key=lambda x: x[0]):
        print(f"{name:<80} {fmt(size):>10}")
    print(f"\n{len(folders)} checkpoint folders found.")


def download_folder(container, blob_prefix: str, output_dir: Path) -> None:
    """Download all blobs under blob_prefix to output_dir, preserving structure."""
    blobs = list(container.list_blobs(name_starts_with=blob_prefix.rstrip("/") + "/"))

    if not blobs:
        print(f"No blobs found under prefix: {blob_prefix}")
        return

    total_size = sum(b.size or 0 for b in blobs)
    print(f"Found {len(blobs)} blobs ({fmt(total_size)}) under '{blob_prefix}'")
    print(f"Downloading to: {output_dir}\n")

    downloaded = 0
    for blob in blobs:
        # Preserve the full blob path relative to output_dir
        local_path = output_dir / blob.name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size == blob.size:
            print(f"  [skip] {blob.name}")
            downloaded += 1
            continue

        blob_client = container.get_blob_client(blob.name)
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

        downloaded += 1
        if downloaded % 10 == 0 or downloaded == len(blobs):
            print(f"  [{downloaded}/{len(blobs)}] {blob.name}")

    print(f"\nDone. {downloaded} files saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download checkpoint folder from Azure Blob Storage")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=(
            "Blob prefix to download, e.g. 'checkpoints/<neptune_run_name>'. "
            "If only the run name is given (no leading 'checkpoints/'), it is prepended automatically."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./downloads",
        help="Local directory to save files into (default: ./downloads)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all checkpoint folders with their sizes and exit",
    )
    args = parser.parse_args()

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise SystemExit(
            "Set AZURE_STORAGE_CONNECTION_STRING (full Azure Storage connection string)."
        )
    client = BlobServiceClient.from_connection_string(conn_str)
    container = client.get_container_client(CONTAINER)

    if args.list:
        list_checkpoint_folders(container)
        return

    if not args.folder:
        parser.error("--folder is required unless --list is used")

    folder = args.folder
    if not folder.startswith("checkpoints/"):
        folder = f"checkpoints/{folder}"

    output_dir = Path(args.output)
    download_folder(container, folder, output_dir)


if __name__ == "__main__":
    main()
