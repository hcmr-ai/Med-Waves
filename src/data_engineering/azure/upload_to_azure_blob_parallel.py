#!/usr/bin/env python3
"""
Parallel upload script for Azure Blob Storage with retry logic and resume capability.

This version is optimized for uploading many large files efficiently.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from azure.core.exceptions import AzureError, ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContainerClient
from tqdm import tqdm


class ParallelAzureBlobUploader:
    """Handle parallel uploads to Azure Blob Storage."""

    def __init__(self, connection_string: Optional[str] = None,
                 account_name: Optional[str] = None,
                 account_key: Optional[str] = None,
                 max_workers: int = 10):
        """
        Initialize the Azure Blob uploader with parallel support.

        Args:
            connection_string: Azure Storage connection string
            account_name: Azure Storage account name
            account_key: Azure Storage account key
            max_workers: Maximum number of parallel upload threads
        """
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=account_key
            )
        else:
            raise ValueError(
                "Either provide connection_string or both account_name and account_key"
            )

        self.max_workers = max_workers

    def create_container_if_not_exists(self, container_name: str) -> ContainerClient:
        """Create container if it doesn't exist."""
        try:
            container_client = self.blob_service_client.create_container(container_name)
            print(f"✓ Created container: {container_name}")
            return container_client
        except ResourceExistsError:
            print(f"✓ Container '{container_name}' already exists")
            return self.blob_service_client.get_container_client(container_name)
        except AzureError as e:
            print(f"✗ Error creating container: {e}")
            raise

    def upload_file_with_retry(self,
                                local_file_path: Path,
                                container_name: str,
                                blob_name: str,
                                overwrite: bool = True,
                                max_retries: int = 3,
                                timeout: int = 600) -> tuple[bool, str]:
        """
        Upload a single file with retry logic.

        Returns:
            Tuple of (success: bool, message: str)
        """
        for attempt in range(max_retries):
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )

                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(
                        data,
                        overwrite=overwrite,
                        timeout=timeout,
                        max_concurrency=4  # Parallel block uploads for large files
                    )

                return True, f"✓ {local_file_path.name}"

            except AzureError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return False, f"✗ {local_file_path.name}: {str(e)[:100]}"
            except Exception as e:
                return False, f"✗ {local_file_path.name}: {str(e)[:100]}"

        return False, f"✗ {local_file_path.name}: Max retries exceeded"

    def upload_directory_parallel(self,
                                   local_directory: Path,
                                   container_name: str,
                                   blob_prefix: str = "",
                                   pattern: str = "*",
                                   recursive: bool = True,
                                   overwrite: bool = True,
                                   resume_file: Optional[Path] = None) -> tuple[int, int]:
        """
        Upload directory with parallel processing.

        Args:
            resume_file: Path to resume file tracking uploaded files
        """
        if not local_directory.exists() or not local_directory.is_dir():
            print(f"✗ Directory not found: {local_directory}")
            return 0, 0

        # Get all matching files
        if recursive:
            files = list(local_directory.rglob(pattern))
        else:
            files = list(local_directory.glob(pattern))

        files = [f for f in files if f.is_file()]

        if not files:
            print(f"✗ No files matching pattern '{pattern}' found")
            return 0, 0

        # Load resume state
        uploaded_files = set()
        if resume_file and resume_file.exists():
            with open(resume_file, 'r') as f:
                uploaded_files = set(json.load(f))
            print(f"✓ Loaded resume file: {len(uploaded_files)} files already uploaded")

        # Filter out already uploaded files
        files_to_upload = []
        for file_path in files:
            relative_path = file_path.relative_to(local_directory)
            blob_name = str(Path(blob_prefix) / relative_path) if blob_prefix else str(relative_path)
            blob_name = blob_name.replace("\\", "/")

            if blob_name not in uploaded_files:
                files_to_upload.append((file_path, blob_name))

        if not files_to_upload:
            print("✓ All files already uploaded!")
            return len(uploaded_files), 0

        print(f"Found {len(files_to_upload)} files to upload ({len(uploaded_files)} already done)")

        # Calculate total size
        total_size = sum(f[0].stat().st_size for f in files_to_upload)
        print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

        successful = len(uploaded_files)
        failed = 0
        failed_files = []

        # Parallel upload with progress bar
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self.upload_file_with_retry,
                    file_path,
                    container_name,
                    blob_name,
                    overwrite
                ): (file_path, blob_name)
                for file_path, blob_name in files_to_upload
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(files_to_upload), desc="Uploading", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path, blob_name = future_to_file[future]
                    try:
                        success, message = future.result()

                        if success:
                            successful += 1
                            uploaded_files.add(blob_name)

                            # Save resume state periodically (every 10 files)
                            if resume_file and successful % 10 == 0:
                                with open(resume_file, 'w') as f:
                                    json.dump(list(uploaded_files), f)
                        else:
                            failed += 1
                            failed_files.append(message)
                            tqdm.write(message)

                        pbar.update(1)
                        pbar.set_postfix({
                            'success': successful,
                            'failed': failed
                        })

                    except Exception as e:
                        failed += 1
                        failed_files.append(f"✗ {file_path.name}: {str(e)[:100]}")
                        tqdm.write(f"✗ Error processing {file_path.name}: {e}")
                        pbar.update(1)

        # Save final resume state
        if resume_file:
            with open(resume_file, 'w') as f:
                json.dump(list(uploaded_files), f)
            print(f"\n✓ Resume file saved: {resume_file}")

        # Print failed files summary
        if failed_files:
            print("\n" + "=" * 60)
            print("Failed uploads:")
            for msg in failed_files[:20]:  # Show first 20
                print(f"  {msg}")
            if len(failed_files) > 20:
                print(f"  ... and {len(failed_files) - 20} more")

        return successful, failed

    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> list[str]:
        """List all blobs in a container."""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except AzureError as e:
            print(f"✗ Error listing blobs: {e}")
            return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel upload to Azure Blob Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--directory", "-d", type=str, required=True,
                       help="Path to directory to upload")
    parser.add_argument("--container", "-c", type=str, required=True,
                       help="Azure container name")
    parser.add_argument("--prefix", "-p", type=str, default="",
                       help="Blob prefix/folder path")
    parser.add_argument("--pattern", type=str, default="*",
                       help="File pattern to match (default: *)")
    parser.add_argument("--no-recursive", action="store_true",
                       help="Don't recurse into subdirectories")
    parser.add_argument("--no-overwrite", action="store_true",
                       help="Don't overwrite existing blobs")
    parser.add_argument("--create-container", action="store_true",
                       help="Create container if it doesn't exist")
    parser.add_argument("--workers", "-w", type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    parser.add_argument("--resume-file", type=str,
                       help="Path to resume file (auto-generated if not specified)")
    parser.add_argument("--connection-string", type=str,
                       help="Azure Storage connection string")
    parser.add_argument("--account-name", type=str,
                       help="Azure Storage account name")
    parser.add_argument("--account-key", type=str,
                       help="Azure Storage account key")

    args = parser.parse_args()

    # Get credentials
    connection_string = (
        args.connection_string or
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    account_name = (
        args.account_name or
        os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    )
    account_key = (
        args.account_key or
        os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    )

    if not connection_string and not (account_name and account_key):
        print("✗ Error: Azure credentials not provided")
        print("   Set AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY)")
        sys.exit(1)

    try:
        # Initialize uploader
        uploader = ParallelAzureBlobUploader(
            connection_string=connection_string,
            account_name=account_name,
            account_key=account_key,
            max_workers=args.workers
        )

        # Create container if requested
        if args.create_container:
            uploader.create_container_if_not_exists(args.container)

        # Set up resume file
        resume_file = None
        if args.resume_file:
            resume_file = Path(args.resume_file)
        else:
            # Auto-generate resume file name
            safe_container_name = args.container.replace('/', '_')
            resume_file = Path(f".azure_upload_resume_{safe_container_name}.json")

        print(f"Using {args.workers} parallel workers")
        print(f"Resume file: {resume_file}")

        # Upload directory
        directory_path = Path(args.directory)
        start_time = time.time()

        successful, failed = uploader.upload_directory_parallel(
            local_directory=directory_path,
            container_name=args.container,
            blob_prefix=args.prefix,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            overwrite=not args.no_overwrite,
            resume_file=resume_file
        )

        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("Upload Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Time: {elapsed_time / 60:.1f} minutes")
        if successful > 0:
            print(f"  Speed: {successful / (elapsed_time / 60):.1f} files/min")
        print(f"{'='*60}")

        # Clean up resume file if all succeeded
        if failed == 0 and resume_file.exists():
            resume_file.unlink()
            print("✓ Removed resume file (all uploads successful)")

        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n✗ Upload interrupted by user")
        print("  Resume file saved. Run the same command again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
