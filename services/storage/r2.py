"""S3-compatible object storage client (Cloudflare R2 in prod, MinIO in dev).

The API surface is identical for both; only the endpoint URL changes.
`STORAGE_MODE` is the explicit switch:

    STORAGE_MODE=off    (default) — no client; uploads disabled, mock mode.
    STORAGE_MODE=minio  — local MinIO via S3_ENDPOINT_URL (default http://localhost:9000).
    STORAGE_MODE=r2     — Cloudflare R2; endpoint derived from R2_ACCOUNT_ID.

The client is constructed lazily on first call so the API can boot in
mock mode without any credentials present.

Two buckets per IMPLEMENTATION_PLAN.md §5.2:
- R2_BUCKET_UPLOADS:     user content, 7-day TTL via lifecycle rule
- R2_BUCKET_PREDICTIONS: persisted (T, 20484) arrays, indefinite TTL
"""

import io
import os
from datetime import timedelta
from uuid import uuid4

import numpy as np


def _resolve_mode() -> str:
    """Read STORAGE_MODE, with legacy fallback to S3_ENDPOINT_URL/R2_ACCESS_KEY presence."""
    mode = os.environ.get("STORAGE_MODE", "").strip().lower()
    if mode:
        return mode
    # Legacy auto-detect: kept so existing .env files don't break silently.
    if not os.environ.get("R2_ACCESS_KEY"):
        return "off"
    return "minio" if os.environ.get("S3_ENDPOINT_URL") else "r2"


class R2Client:
    def __init__(self) -> None:
        import boto3
        from botocore.config import Config

        mode = _resolve_mode()
        if mode == "minio":
            endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
        elif mode == "r2":
            account_id = os.environ["R2_ACCOUNT_ID"]
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        else:
            raise RuntimeError(
                f"R2Client constructed with STORAGE_MODE={mode!r}; expected 'minio' or 'r2'."
            )
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
    """Return R2Client per STORAGE_MODE; None when off (mock mode)."""
    global _client
    if _client is not None:
        return _client
    mode = _resolve_mode()
    if mode == "off":
        return None
    if mode not in ("minio", "r2"):
        raise RuntimeError(
            f"STORAGE_MODE={mode!r} not recognized. Use 'off', 'minio', or 'r2'."
        )
    _client = R2Client()
    return _client
