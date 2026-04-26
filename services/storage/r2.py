"""S3-compatible object storage client (Cloudflare R2 in prod, MinIO in dev).

The API surface is identical for both; only the endpoint URL changes.
Set `S3_ENDPOINT_URL` to override (e.g. `http://localhost:9000` for local
MinIO via scripts/dev_minio.sh); leave it unset to derive R2's URL from
`R2_ACCOUNT_ID`. The client is constructed lazily on first call so the
API can boot in mock mode without any credentials present.

Two buckets per IMPLEMENTATION_PLAN.md §5.2:
- R2_BUCKET_UPLOADS:     user content, 7-day TTL via lifecycle rule
- R2_BUCKET_PREDICTIONS: persisted (T, 20484) arrays, indefinite TTL
"""

import io
import os
from datetime import timedelta
from uuid import uuid4

import numpy as np


class R2Client:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        endpoint_url = os.environ.get("S3_ENDPOINT_URL")
        if not endpoint_url:
            account_id = os.environ["R2_ACCOUNT_ID"]
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        self.uploads_bucket = os.environ["R2_BUCKET_UPLOADS"]
        self.predictions_bucket = os.environ["R2_BUCKET_PREDICTIONS"]

    def mint_upload_url(self, content_type: str = "video/mp4") -> dict[str, str]:
        """Create a presigned PUT URL for direct browser upload.

        Returns {put_url, get_url, content_url}. Frontend PUTs the file to
        put_url, then sends content_url to /analyze.
        """
        key = f"uploads/{uuid4()}"
        put_url = self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.uploads_bucket,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=int(timedelta(minutes=5).total_seconds()),
        )
        get_url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.uploads_bucket, "Key": key},
            ExpiresIn=int(timedelta(hours=1).total_seconds()),
        )
        return {"put_url": put_url, "get_url": get_url, "content_url": get_url}

    def store_predictions(self, request_id: str, predictions: np.ndarray) -> str:
        """Upload (T, 20484) array as float16 NPZ. Returns the R2 URI."""
        buf = io.BytesIO()
        np.savez_compressed(buf, preds=predictions.astype(np.float16))
        buf.seek(0)
        key = f"predictions/{request_id}.npz"
        self._client.put_object(
            Bucket=self.predictions_bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        return f"r2://{self.predictions_bucket}/{key}"


_client: R2Client | None = None


def get_client() -> R2Client | None:
    """Return R2Client if env is configured; None otherwise (mock mode)."""
    global _client
    if _client is not None:
        return _client
    if not os.environ.get("R2_ACCESS_KEY"):
        return None
    _client = R2Client()
    return _client
