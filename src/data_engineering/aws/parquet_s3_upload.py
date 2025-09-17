import re
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from tqdm import tqdm

# --- config you change ---
SRC_DIR = Path("/data/tsolis/AI_project/parquet/augmented_with_labels/hourly")
BUCKET  = "medwav-dev-data"
PREFIX  = "parquet/hourly"   # base prefix in S3
PARTITION_BY_YEAR = True     # put files under year=YYYY/
SKIP_IF_SAME_SIZE = True     # fast resume: skip if object exists with same size
MAX_CONCURRENCY = 16         # threads for multipart uploads
CHUNK_MB = 64                # multipart chunk size
# -------------------------

date_rx = re.compile(r"WAVEAN(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})\.parquet$")

def s3_key_for(path: Path) -> str:
    m = date_rx.search(path.name)
    if PARTITION_BY_YEAR and m:
        y = m.group("y")
        return f"{PREFIX}/year={y}/{path.name}"
    return f"{PREFIX}/{path.name}"

def object_exists_with_size(s3, bucket: str, key: str, size: int) -> bool:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp.get("ContentLength") == size
    except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        return False
    except Exception:
        return False

def main() -> int:
    files = sorted(SRC_DIR.glob("WAVEAN2021*.parquet"))
    if not files:
        print(f"No files found under {SRC_DIR}")
        return 1

    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            s3={"addressing_style": "virtual"}
        ),
    )

    tcfg = TransferConfig(
        multipart_threshold=CHUNK_MB * 1024 * 1024,
        multipart_chunksize=CHUNK_MB * 1024 * 1024,
        max_concurrency=MAX_CONCURRENCY,
        use_threads=True,
    )

    uploaded = skipped = 0
    with tqdm(total=len(files), unit="file") as pbar:
        for f in files:
            key = s3_key_for(f)
            size = f.stat().st_size
            do_skip = SKIP_IF_SAME_SIZE and object_exists_with_size(s3_client, BUCKET, key, size)
            if do_skip:
                skipped += 1
                pbar.set_postfix_str(f"skip {f.name}")
                pbar.update(1)
                continue
            s3_client.upload_file(str(f), BUCKET, key, Config=tcfg)
            uploaded += 1
            pbar.set_postfix_str(f"up {f.name}")
            pbar.update(1)

    print(f"\nDone. Uploaded: {uploaded}, Skipped: {skipped}, Total: {len(files)}")
    print(f"s3://{BUCKET}/{PREFIX}/")
    return 0

if __name__ == "__main__":
    sys.exit(main())
