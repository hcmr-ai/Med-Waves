#!/usr/bin/env python3
"""
Optimized upload for long-distance Azure connections (e.g., Greece -> East US).
Uses chunked uploads with aggressive retry and proper block size tuning.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobBlock, BlobServiceClient
from tqdm import tqdm


class OptimizedAzureBlobUploader:
    """Optimized uploader for long-distance connections."""

    def __init__(self, connection_string: str, max_workers: int = 5):
        """
        Initialize with optimized settings for long-distance uploads.

        Args:
            connection_string: Azure Storage connection string
            max_workers: Parallel upload threads (lower for stability)
        """
        # Use optimized connection settings
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string,
            connection_timeout=300,  # 5 minute connection timeout
            read_timeout=300,  # 5 minute read timeout
        )
        self.max_workers = max_workers
        # Use smaller blocks for better network handling (4MB instead of default)
        self.block_size = 4 * 1024 * 1024  # 4MB blocks

    def create_container_if_not_exists(self, container_name: str):
        """Create container if it doesn't exist."""
        try:
            self.blob_service_client.create_container(container_name)
            print(f"✓ Created container: {container_name}")
        except ResourceExistsError:
            print(f"✓ Container '{container_name}' exists")
        except AzureError as e:
            print(f"✗ Container error: {e}")

    def upload_file_chunked(self,
                           local_file_path: Path,
                           container_name: str,
                           blob_name: str,
                           overwrite: bool = True) -> tuple[bool, str]:
        """
        Upload file in chunks with retry logic optimized for slow connections.
        """
        max_retries = 5

        for attempt in range(max_retries):
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )

                # Check if blob exists and skip if not overwriting
                if not overwrite:
                    try:
                        blob_client.get_blob_properties()
                        return True, f"↷ {local_file_path.name} (exists, skipped)"
                    except ResourceNotFoundError:
                        pass  # Blob doesn't exist, proceed with upload

                file_size = local_file_path.stat().st_size

                # For small files, upload directly
                if file_size < self.block_size:
                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(
                            data,
                            overwrite=overwrite,
                            timeout=600,
                            max_concurrency=1
                        )
                    return True, f"✓ {local_file_path.name}"

                # For large files, use staged block upload
                block_list = []
                with open(local_file_path, "rb") as f:
                    block_num = 0
                    while True:
                        chunk = f.read(self.block_size)
                        if not chunk:
                            break

                        # Generate block ID
                        block_id = f"{block_num:08d}"
                        block_id_encoded = hashlib.md5(block_id.encode()).hexdigest()

                        # Upload block with retry
                        for block_attempt in range(3):
                            try:
                                blob_client.stage_block(
                                    block_id=block_id_encoded,
                                    data=chunk,
                                    length=len(chunk),
                                    timeout=300
                                )
                                break
                            except Exception as block_error:
                                if block_attempt == 2:
                                    raise block_error
                                time.sleep(2 ** block_attempt)

                        block_list.append(BlobBlock(block_id=block_id_encoded))
                        block_num += 1

                # Commit all blocks
                blob_client.commit_block_list(block_list, timeout=300)
                return True, f"✓ {local_file_path.name}"

            except AzureError as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # Cap at 30 seconds
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower():
                        return False, f"✗ {local_file_path.name}: TIMEOUT (connection too slow)"
                    return False, f"✗ {local_file_path.name}: {error_msg[:80]}"
            except Exception as e:
                return False, f"✗ {local_file_path.name}: {str(e)[:80]}"

        return False, f"✗ {local_file_path.name}: Max retries"

    def upload_directory_parallel(self,
                                  local_directory: Path,
                                  container_name: str,
                                  blob_prefix: str = "",
                                  pattern: str = "*",
                                  recursive: bool = True,
                                  overwrite: bool = True,
                                  resume_file: Optional[Path] = None) -> tuple[int, int]:
        """Upload directory with parallel processing."""

        if not local_directory.exists():
            print(f"✗ Directory not found: {local_directory}")
            return 0, 0

        # Get files
        if recursive:
            files = list(local_directory.rglob(pattern))
        else:
            files = list(local_directory.glob(pattern))

        files = [f for f in files if f.is_file()]

        if not files:
            print("✗ No files found")
            return 0, 0

        # Load resume state
        uploaded_files = set()
        if resume_file and resume_file.exists():
            with open(resume_file, 'r') as f:
                uploaded_files = set(json.load(f))
            print(f"✓ Resume: {len(uploaded_files)} files already uploaded")

        # Build upload list
        files_to_upload = []
        for file_path in files:
            relative_path = file_path.relative_to(local_directory)
            blob_name = str(Path(blob_prefix) / relative_path) if blob_prefix else str(relative_path)
            blob_name = blob_name.replace("\\", "/")

            if blob_name not in uploaded_files:
                files_to_upload.append((file_path, blob_name))

        if not files_to_upload:
            print("✓ All files uploaded!")
            return len(uploaded_files), 0

        total_size_gb = sum(f[0].stat().st_size for f in files_to_upload) / 1024 / 1024 / 1024
        print(f"Uploading {len(files_to_upload)} files ({total_size_gb:.2f} GB)")
        print(f"Workers: {self.max_workers}, Block size: {self.block_size / 1024 / 1024:.0f}MB")

        successful = len(uploaded_files)
        failed = 0
        failed_files = []
        start_time = time.time()

        # Parallel upload
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self.upload_file_chunked,
                    file_path,
                    container_name,
                    blob_name,
                    overwrite
                ): (file_path, blob_name)
                for file_path, blob_name in files_to_upload
            }

            with tqdm(total=len(files_to_upload), desc="Upload", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path, blob_name = future_to_file[future]

                    try:
                        success, message = future.result()

                        if success:
                            successful += 1
                            uploaded_files.add(blob_name)

                            # Save progress every 5 files
                            if resume_file and successful % 5 == 0:
                                with open(resume_file, 'w') as f:
                                    json.dump(list(uploaded_files), f)

                            # Show speed estimate
                            elapsed = time.time() - start_time
                            rate = (successful - len(uploaded_files) + len(files_to_upload)) / elapsed if elapsed > 0 else 0

                            pbar.set_postfix({
                                'ok': successful,
                                'fail': failed,
                                'rate': f'{rate:.1f}/min'
                            })
                        else:
                            failed += 1
                            failed_files.append(message)
                            if "TIMEOUT" in message:
                                tqdm.write(f"⚠ {message}")

                        pbar.update(1)

                    except Exception as e:
                        failed += 1
                        failed_files.append(f"✗ {file_path.name}: {str(e)[:60]}")
                        pbar.update(1)

        # Save final state
        if resume_file:
            with open(resume_file, 'w') as f:
                json.dump(list(uploaded_files), f)

        # Summary
        if failed_files:
            print(f"\n{'='*60}")
            print(f"Failed uploads ({len(failed_files)}):")
            for msg in failed_files[:15]:
                print(f"  {msg}")
            if len(failed_files) > 15:
                print(f"  ... and {len(failed_files) - 15} more")

        return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Optimized Azure upload for slow connections")
    parser.add_argument("--directory", "-d", required=True, help="Directory to upload")
    parser.add_argument("--container", "-c", required=True, help="Container name")
    parser.add_argument("--prefix", "-p", default="", help="Blob prefix")
    parser.add_argument("--pattern", default="*", help="File pattern")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--create-container", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Parallel workers (default: 5)")
    parser.add_argument("--resume-file", type=str)

    args = parser.parse_args()

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("✗ Set AZURE_STORAGE_CONNECTION_STRING")
        sys.exit(1)

    try:
        uploader = OptimizedAzureBlobUploader(conn_str, max_workers=args.workers)

        if args.create_container:
            uploader.create_container_if_not_exists(args.container)

        resume_file = Path(args.resume_file) if args.resume_file else Path(f".resume_{args.container.replace('/', '_')}.json")

        start = time.time()
        successful, failed = uploader.upload_directory_parallel(
            local_directory=Path(args.directory),
            container_name=args.container,
            blob_prefix=args.prefix,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            overwrite=not args.no_overwrite,
            resume_file=resume_file
        )

        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print(f"Summary: {successful} ok, {failed} failed, {elapsed/60:.1f} min")
        if successful > 0:
            print(f"Average: {successful/(elapsed/60):.1f} files/min")
        print(f"{'='*60}")

        if failed == 0 and resume_file.exists():
            resume_file.unlink()

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n✗ Interrupted - resume file saved")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
