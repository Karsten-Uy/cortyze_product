"""Upload a local file to R2 and print a 24h presigned GET URL.

Usage:
    cortyze-secrets
    uv run python scripts/upload_to_r2.py /path/to/file.mp4
    uv run python scripts/upload_to_r2.py /path/to/file.mp4 --key custom/path.mp4
    uv run python scripts/upload_to_r2.py /path/to/file.mp4 --bucket cortyze-predictions

Default key is `uploads/<basename>`. The presigned URL expires in 24h.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local file to upload")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("R2_BUCKET_UPLOADS", "cortyze-uploads"),
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Object key in the bucket (default: uploads/<basename>)",
    )
    parser.add_argument("--expires", type=int, default=86400)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        return 1

    key = args.key or f"uploads/{args.path.name}"
    account = os.environ["R2_ACCOUNT_ID"]
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    print(f"Uploading {args.path} -> {args.bucket}/{key}", file=sys.stderr)
    s3.upload_file(str(args.path), args.bucket, key)

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": args.bucket, "Key": key},
        ExpiresIn=args.expires,
    )
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
