"""POST /upload-url — mint a presigned R2 PUT for direct browser upload.

Flow:
  1. Frontend POSTs the filename + mime + size to this route.
  2. We validate (whitelist mime types, size cap), then ask the R2 client
     for a fresh presigned PUT URL plus a matching presigned GET URL
     valid for the run's lifetime.
  3. Browser PUTs the file directly to R2 (no API hop for the bytes).
  4. Browser submits the GET URL as `media_url` when creating a run.

Mock mode (`STORAGE_MODE=off`, the default in dev) returns 503 — the
frontend renders this as "Upload disabled in mock mode" so a developer
without R2 configured still gets a clear signal rather than a 500.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.storage.r2 import get_client as get_r2_client

from ..auth import optional_user
from ..limiter import limiter

_log = logging.getLogger("cortyze.routes.upload")

router = APIRouter()

# Mime allowlist. Match the frontend's UI hint ("MP4, MOV, JPG, PNG…")
# and the same set of extensions the legacy /analyze accepts. Anything
# outside this list is a 415 — TRIBE can't process it anyway.
_ALLOWED_MIMES: frozenset[str] = frozenset({
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-m4v",
    "image/jpeg",
    "image/png",
    "image/webp",
})

# 100 MiB. Same as the frontend hint.
_MAX_BYTES: int = 100 * 1024 * 1024


class UploadUrlRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str = Field(min_length=1, max_length=100)
    size: int = Field(ge=0)


class UploadUrlResponse(BaseModel):
    put_url: str       # PUT here with the file bytes
    get_url: str       # send this as `media_url` on POST /runs
    object_key: str    # stable R2 key — round-trip to /runs so /runs/:id can re-presign
    content_type: str  # echo back so the browser sets the matching header


@router.post(
    "/upload-url",
    response_model=UploadUrlResponse,
    status_code=status.HTTP_200_OK,
)
@limiter.limit("60/minute")
async def mint_upload_url(
    request: Request,
    body: UploadUrlRequest,
    user_id: str | None = Depends(optional_user),
) -> UploadUrlResponse:
    del user_id  # auth-gated for rate-limit purposes; not yet keyed into the R2 path

    if body.content_type not in _ALLOWED_MIMES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"unsupported content_type {body.content_type!r}; "
                f"expected one of {sorted(_ALLOWED_MIMES)}"
            ),
        )
    if body.size > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"file too large ({body.size} bytes); max is {_MAX_BYTES}",
        )

    r2 = get_r2_client()
    if r2 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "object storage is disabled (STORAGE_MODE=off). "
                "Set STORAGE_MODE=r2 + R2_* env vars to enable uploads."
            ),
        )

    minted = r2.mint_upload_url(content_type=body.content_type)
    return UploadUrlResponse(
        put_url=minted["put_url"],
        get_url=minted["get_url"],
        object_key=minted["object_key"],
        content_type=body.content_type,
    )
