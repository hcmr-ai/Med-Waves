from typing import List, Optional

import boto3
from botocore.config import Config


def list_s3_parquet_files(bucket: str, prefix: str = "", aws_profile: Optional[str] = None) -> List[str]:
    """
    List all parquet files in an S3 bucket with the given prefix.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix to filter files (e.g., "parquet/hourly/")
        aws_profile: AWS profile to use (optional)

    Returns:
        List of S3 URIs (s3://bucket/key format)
    """
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3_client = session.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            s3={"addressing_style": "virtual"}
        )
    )

    parquet_files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.parquet'):
                        parquet_files.append(f"s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

    return sorted(parquet_files)
