import logging
import os
import re
from typing import List, Optional

import boto3
import s3fs
from botocore.config import Config

logger = logging.getLogger(__name__)


def list_s3_parquet_files(
    bucket: str,
    prefix: str = "",
    aws_profile: Optional[str] = None,
    filter_months: Optional[List[int]] = None,
    train_end_year: Optional[int] = None,
    test_start_year: Optional[int] = None,
) -> List[str]:
    """
    List all parquet files in an S3 bucket with the given prefix.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to filter files (e.g., "parquet/hourly/")
        aws_profile: AWS profile to use (optional)
        filter_months: Optional list of months (1-12) to filter files by
        train_end_year: Year up to which to include all months for training
        test_start_year: Year from which to apply month filtering for evaluation

    Returns:
        List of S3 URIs (s3://bucket/key format)
    """
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3_client = session.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            s3={"addressing_style": "virtual"},
        ),
    )

    parquet_files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".parquet"):
                        # Apply year-aware filtering
                        if _matches_year_aware_filter(
                            key, filter_months, train_end_year, test_start_year
                        ):
                            parquet_files.append(f"s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

    return sorted(parquet_files)


def _matches_year_aware_filter(
    s3_key: str,
    filter_months: Optional[List[int]],
    train_end_year: Optional[int],
    test_start_year: Optional[int],
) -> bool:
    """
    Check if an S3 key matches the year-aware filter.

    Logic:
    - For years <= train_end_year: Include all months (no filtering)
    - For years >= test_start_year: Apply month filtering if specified

    Args:
        s3_key: S3 key (e.g., "parquet/hourly/year=2023/WAVEAN20231231.parquet")
        filter_months: List of months (1-12) to filter by for test years
        train_end_year: Year up to which to include all months
        test_start_year: Year from which to apply month filtering

    Returns:
        True if the file matches the year-aware filter
    """
    # Extract filename from S3 key
    filename = s3_key.split("/")[-1]

    # Match pattern like WAVEAN20231231.parquet
    match = re.search(r"WAVEAN(\d{4})(\d{2})(\d{2})\.parquet$", filename)
    if not match:
        return False

    year, month, day = match.groups()
    year_int = int(year)
    month_int = int(month)

    # For training years (<= train_end_year): include all months
    if train_end_year is not None and year_int <= train_end_year:
        return True

    # For test years (>= test_start_year): apply month filtering if specified
    if test_start_year is not None and year_int >= test_start_year:
        if filter_months is not None:
            return month_int in filter_months
        else:
            return True  # Include all months if no month filter specified

    # Default: include the file
    return True


def download_s3_checkpoint(s3_path: str, local_dir: str) -> str:
    """Download checkpoint from S3 to local directory"""
    try:
        fs = s3fs.S3FileSystem()

        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Extract filename from S3 path
        filename = os.path.basename(s3_path)
        local_path = os.path.join(local_dir, filename)

        # Download from S3
        fs.get(s3_path, local_path)
        logger.info(f"Downloaded checkpoint from S3: {s3_path} -> {local_path}")

        return local_path

    except Exception as e:
        logger.error(f"Failed to download S3 checkpoint {s3_path}: {e}")
        return None
