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
from core.scoring.goals import Goal, overall_score
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
    response = client.predict(
        content_url=req.content_url,
        content_type=req.content_type,
        image_urls=req.image_urls,
        audio_url=req.audio_url,
        caption=req.caption,
        seconds_per_image=req.seconds_per_image,
    )
    raw_predictions = response.predictions

    raw_region_activations = aggregate(raw_predictions)
    region_scores = normalize(raw_region_activations)
    overall = overall_score(region_scores, req.goal)
    # Cache all four goal-weighted overalls upfront. Region scores are
    # goal-independent, so this is just four weighted sums on the same
    # cached vector — basically free. Lets the frontend swap the goal
    # lens without hitting the backend at all.
    overall_by_goal = {
        g.value: overall_score(region_scores, g) for g in Goal
    }

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

    # Persist the rendered PNG to R2 so past runs (and regoals) can re-
    # render it without recomputing. The b64 is also returned inline
    # for the immediate response so the frontend doesn't need a second
    # round-trip on first paint.
    brain_image_uri: str | None = None
    brain_image_request_id: str | None = None
    if r2 is not None and brain_image_b64:
        try:
            import base64
            png_bytes = base64.b64decode(brain_image_b64)
            brain_image_uri = r2.store_brain_image(req.request_id, png_bytes)
            brain_image_request_id = req.request_id
        except Exception as e:
            _log.warning(
                "request_id=%s failed to persist brain image to R2: %s",
                req.request_id,
                e,
            )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    report = BrainReport(
        request_id=req.request_id,
        region_scores=region_scores,
        overall_score=overall,
        goal=req.goal,
        content_type=req.content_type,
        user_id=req.user_id,
        model_version=MODEL_VERSION,
        raw_predictions_uri=raw_predictions_uri,
        brain_image_b64=brain_image_b64,
        brain_image_uri=brain_image_uri,
        brain_image_request_id=brain_image_request_id,
        elapsed_ms=elapsed_ms,
        region_timeseries=region_timeseries,
        moments=moments,
        additional_context=req.additional_context,
        campaign_id=req.campaign_id,
        title=req.title,
        caption_text=req.caption,
        overall_by_goal=overall_by_goal,
        audio_url=req.audio_url,
        image_count=len(req.image_urls or []) if req.content_type == "post" else 0,
        seconds_per_image=req.seconds_per_image,
    )

    # Stage 2: per-region diagnosis. Opt-in via ENABLE_SUGGESTIONS so flipping
    # SUGGESTION_LLM_MODE to a paid provider doesn't auto-fire on every call.
    # The suggestion engine reads `report.content_type` to pick the right
    # system prompt (video vs post vs gallery) — gallery params let it
    # translate dip windows into image-range form.
    if suggestions_enabled():
        try:
            report.suggestions = diagnose(
                report,
                image_count=len(req.image_urls or []),
                seconds_per_image=req.seconds_per_image,
                has_audio=bool(req.audio_url),
                additional_context=req.additional_context,
            )
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
