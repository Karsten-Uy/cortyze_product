"""Mint a 24h presigned GET URL for an object already in R2.

Usage:
    cortyze-secrets
    uv run python scripts/presign_r2.py <key>
    uv run python scripts/presign_r2.py <key> --bucket cortyze-predictions

Defaults to the uploads bucket. Object key is whatever the R2 dashboard
shows under "Object key" — e.g. "uploads/pepsi ad read.m4a" (spaces ok,
no URL-encoding needed; boto3 handles it).
"""

from __future__ import annotations

import argparse
import os
import sys

import boto3
from botocore.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="Object key, e.g. 'uploads/voice.m4a'")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("R2_BUCKET_UPLOADS", "cortyze-uploads"),
    )
    parser.add_argument("--expires", type=int, default=86400, help="seconds (default 24h)")
    args = parser.parse_args()

    account = os.environ["R2_ACCOUNT_ID"]
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": args.bucket, "Key": args.key},
        ExpiresIn=args.expires,
    )
    print(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
