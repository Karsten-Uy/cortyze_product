"""Core prediction pipeline: AnalyzeRequest -> BrainReport.

Plain function (per IMPLEMENTATION_PLAN.md §6.6) so Stage 4's batch backfill
worker can import and loop without a FastAPI request lifecycle.

Persistence is opt-in: if R2 is configured, the (T, 20484) array is uploaded
and `raw_predictions_uri` is populated. If Supabase is configured, the
report row is inserted. Either can be missing; the function still returns
a valid BrainReport.
"""

import time

from core.atlas.mapper import aggregate
from core.schemas import AnalyzeRequest, BrainReport
from core.scoring.goals import overall_score
from core.scoring.normalize import normalize
from services.persistence.reports import get_store
from services.storage.r2 import get_client as get_r2

from .clients.runpod import get_client

MODEL_VERSION = "tribev2-mock-2026-04"


def predict_brain_report(req: AnalyzeRequest) -> BrainReport:
    t0 = time.monotonic()

    client = get_client()
    raw_predictions = client.predict(req.content_url, req.content_type)

    raw_region_activations = aggregate(raw_predictions)
    region_scores = normalize(raw_region_activations)
    overall = overall_score(region_scores, req.goal)

    raw_predictions_uri: str | None = None
    r2 = get_r2()
    if r2 is not None:
        raw_predictions_uri = r2.store_predictions(req.request_id, raw_predictions)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    report = BrainReport(
        request_id=req.request_id,
        region_scores=region_scores,
        overall_score=overall,
        goal=req.goal,
        user_id=req.user_id,
        model_version=MODEL_VERSION,
        raw_predictions_uri=raw_predictions_uri,
        elapsed_ms=elapsed_ms,
    )

    store = get_store()
    if store is not None:
        store.insert(report)

    return report
