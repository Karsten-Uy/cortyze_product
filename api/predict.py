"""Core prediction pipeline: AnalyzeRequest -> BrainReport.

Plain function (per IMPLEMENTATION_PLAN.md §6.6) so Stage 4's batch backfill
worker can import and loop without a FastAPI request lifecycle.

Persistence is opt-in: if R2 is configured, the (T, 20484) array is uploaded
and `raw_predictions_uri` is populated. If Supabase is configured, the
report row is inserted. Either can be missing; the function still returns
a valid BrainReport.

Stage 2 prep: the per-timestep / moments pipeline runs alongside the
averaged-score pipeline so dip and peak windows are detected. They're not
yet attached to the BrainReport (waiting for the suggestion engine), but
are precomputed here so wiring Claude in later is just a function call.
"""

import logging
import time

from core.atlas.mapper import aggregate
from core.atlas.temporal import aggregate_per_timestep, normalize_per_timestep
from core.schemas import AnalyzeRequest, BrainReport
from core.scoring.goals import overall_score
from core.scoring.normalize import normalize
from services.persistence.reports import get_store
from services.storage.r2 import get_client as get_r2
from services.suggestions import diagnose, is_enabled as suggestions_enabled
from services.suggestions.moments import annotate_moments, find_moments
from services.visualization.brain_plot import render_brain_png

from .clients.runpod import get_client

MODEL_VERSION = "tribev2-mock-2026-04"

_log = logging.getLogger(__name__)


def predict_brain_report(req: AnalyzeRequest) -> BrainReport:
    t0 = time.monotonic()

    client = get_client()
    response = client.predict(req.content_url, req.content_type)
    raw_predictions = response.predictions

    raw_region_activations = aggregate(raw_predictions)
    region_scores = normalize(raw_region_activations)
    overall = overall_score(region_scores, req.goal)

    # Per-timestep pipeline. Powers (a) the dip/peak chips on the frontend
    # region cards, (b) the per-region sparkline, and (c) — once Stage 2 lands
    # — the LLM prompt context.
    per_t_raw = aggregate_per_timestep(raw_predictions)
    per_t_scores = normalize_per_timestep(per_t_raw)
    moments = annotate_moments(find_moments(per_t_scores), response.events)
    region_timeseries = {
        region: [round(float(v), 2) for v in series]
        for region, series in per_t_scores.items()
    }
    if moments:
        _log.info(
            "request_id=%s detected %d moments (dips=%d peaks=%d)",
            req.request_id,
            len(moments),
            sum(1 for m in moments if m.type == "dip"),
            sum(1 for m in moments if m.type == "peak"),
        )

    raw_predictions_uri: str | None = None
    r2 = get_r2()
    if r2 is not None:
        raw_predictions_uri = r2.store_predictions(req.request_id, raw_predictions)

    try:
        brain_image_b64: str | None = render_brain_png(raw_predictions)
    except Exception:
        brain_image_b64 = None

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    report = BrainReport(
        request_id=req.request_id,
        region_scores=region_scores,
        overall_score=overall,
        goal=req.goal,
        user_id=req.user_id,
        model_version=MODEL_VERSION,
        raw_predictions_uri=raw_predictions_uri,
        brain_image_b64=brain_image_b64,
        elapsed_ms=elapsed_ms,
        region_timeseries=region_timeseries,
        moments=moments,
    )

    # Stage 2: per-region diagnosis. Opt-in via ENABLE_SUGGESTIONS so flipping
    # SUGGESTION_LLM_MODE to a paid provider doesn't auto-fire on every call.
    if suggestions_enabled():
        try:
            report.suggestions = diagnose(report)
            _log.info(
                "request_id=%s generated %d suggestions",
                req.request_id,
                len(report.suggestions),
            )
        except Exception as e:
            _log.warning(
                "request_id=%s suggestion engine errored: %s", req.request_id, e
            )

    store = get_store()
    if store is not None:
        store.insert(report)

    return report
