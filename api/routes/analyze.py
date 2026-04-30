"""POST /analyze, GET /report/{request_id}, GET /reports, POST /upload-url.

`/analyze` runs synchronously against the configured RunPod client (mock by
default). All routes that touch a user's data are gated by Supabase auth;
unauth'd callers get 401. `/upload-url` is also gated since presigned PUTs
shouldn't be issued anonymously.

The Stage 3 listing endpoint `/reports` returns paginated `ReportSummary`
objects for the sidebar. The full BrainReport is fetched separately via
`/report/{id}` only when the user clicks into a specific run.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from core.schemas import AnalyzeRequest, BrainReport, ReportSummary
from services.persistence.reports import get_store
from services.storage.r2 import get_client as get_r2

from ..auth import require_user
from ..predict import predict_brain_report

router = APIRouter()


@router.post("/analyze", response_model=BrainReport)
def analyze(
    req: AnalyzeRequest,
    user_id: str = Depends(require_user),
) -> BrainReport:
    """Run the analysis. All callers must be authenticated — the verified
    `sub` from the Supabase JWT overrides any user_id the client passed,
    so the frontend can't impersonate other users.
    """
    req = req.model_copy(update={"user_id": user_id})
    return predict_brain_report(req)


@router.get("/report/{request_id}", response_model=BrainReport)
def get_report(
    request_id: str,
    user_id: str = Depends(require_user),
) -> BrainReport:
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=501,
            detail="Reports persistence not configured. Set DATABASE_URL env var.",
        )
    report = store.get(request_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")
    # Defense in depth on top of RLS — refuse to return a report owned by
    # someone else even if the service role can read it.
    if report.user_id and report.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")

    # If the report was persisted with a brain image, mint a fresh
    # presigned URL on every load (the one stored at create-time may have
    # expired). The frontend prefers brain_image_uri over the inline b64
    # for past runs, so this is what makes "view past run" show the image.
    if report.brain_image_uri:
        r2 = get_r2()
        if r2 is not None:
            try:
                report.brain_image_uri = r2.presign_brain_image(
                    request_id, expires=24 * 3600
                )
            except Exception:
                # Presign failure (image deleted, R2 down, etc.) — fall
                # through with the stale URL; frontend will fail to load
                # the image but the rest of the report still renders.
                pass
    return report


@router.get("/reports", response_model=dict)
def list_reports(
    campaign_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    user_id: str = Depends(require_user),
) -> dict:
    """Paginated sidebar listing. Returns
    `{"items": [ReportSummary, ...], "next_cursor": str | null}`.

    Why a wrapper dict instead of a bare list: cursor pagination needs a
    place to put the cursor, and adding it to a header is harder for
    Next.js client code to reach than a JSON field.
    """
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=501,
            detail="Reports persistence not configured. Set DATABASE_URL env var.",
        )
    items, next_cursor = store.list_for_user(
        user_id, campaign_id=campaign_id, limit=limit, cursor=cursor
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "next_cursor": next_cursor,
    }


@router.post("/upload-url")
def upload_url(
    content_type: str = "video/mp4",
    user_id: str = Depends(require_user),
) -> dict[str, str]:
    r2 = get_r2()
    if r2 is None:
        raise HTTPException(
            status_code=501,
            detail="Object storage not configured. Set R2_ACCOUNT_ID/R2_ACCESS_KEY/R2_SECRET_KEY/R2_BUCKET_UPLOADS env vars.",
        )
    return r2.mint_upload_url(content_type=content_type)
