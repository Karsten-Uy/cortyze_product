"""POST /analyze, GET /report/{request_id}, POST /upload-url.

`/analyze` runs synchronously against the configured RunPod client (mock by
default). The other two endpoints auto-degrade to 501 when their
respective infrastructure (R2, Supabase) is not configured via env vars,
so the API always boots even on a fresh dev machine.
"""

from fastapi import APIRouter, HTTPException

from core.schemas import AnalyzeRequest, BrainReport
from services.persistence.reports import get_store
from services.storage.r2 import get_client as get_r2

from ..predict import predict_brain_report

router = APIRouter()


@router.post("/analyze", response_model=BrainReport)
def analyze(req: AnalyzeRequest) -> BrainReport:
    return predict_brain_report(req)


@router.get("/report/{request_id}", response_model=BrainReport)
def get_report(request_id: str) -> BrainReport:
    store = get_store()
    if store is None:
        raise HTTPException(
            status_code=501,
            detail="Reports persistence not configured. Set DATABASE_URL env var.",
        )
    report = store.get(request_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {request_id} not found")
    return report


@router.post("/upload-url")
def upload_url(content_type: str = "video/mp4") -> dict[str, str]:
    r2 = get_r2()
    if r2 is None:
        raise HTTPException(
            status_code=501,
            detail="Object storage not configured. Set R2_ACCOUNT_ID/R2_ACCESS_KEY/R2_SECRET_KEY/R2_BUCKET_UPLOADS env vars.",
        )
    return r2.mint_upload_url(content_type=content_type)
