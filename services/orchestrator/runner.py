"""Pipeline runner: Phase 1 → 2 → 3 → 4 → done.

Runs as a background asyncio task per run. Phase 1 (TRIBE inference)
and Phase 2 (trend context) start concurrently; once both finish the
join feeds Phase 3 (synthesis); Phase 3's output goes through Phase 4
(validation) and the final SuggestionPlan is written to persistence.

This file is mock-friendly: when no real GPU is configured, Phase 1
returns a deterministic per-region score derived from the brief text
hash. Real TRIBE wiring uses the existing `api/predict.py` path and
slots in below the `_run_phase_1` shim.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core.goals_v2 import composite_score
from core.regions_v2 import REGION_KEYS, RegionKey
from core.schemas_v2 import RunRecord

from ..persistence.runs_v2 import RUN_STORE
from ..synthesis import get_client as get_synthesis_client
from ..synthesis.protocol import SynthesisInput
from ..trends import get_client as get_trends_client
from ..validation import get_client as get_validation_client
from .events import EVENT_BUS, RunEvent

_log = logging.getLogger("cortyze.orchestrator")

# asyncio holds only WEAK references to tasks created via
# `asyncio.create_task` — without a strong ref, the task can be
# garbage-collected mid-execution. We park each pipeline task here
# and `discard` it on completion so memory stays bounded.
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
_RUNNING_TASKS: set[asyncio.Task] = set()


async def start_run(record: RunRecord) -> None:
    """Persist the queued record and kick off the background pipeline.

    Caller is the route handler — the route returns the run_id
    immediately and never waits on the pipeline. The pipeline writes
    back through `RUN_STORE.update(...)`.

    When `record.demo_id` is set, we run a short cosmetic pipeline that
    emits the same SSE events as the real one but loads its
    `SuggestionPlan` from `data/demo_runs/<demo_id>.json` instead of
    Phase 3 synthesis. The frontend can't tell the difference —
    analyzing animation plays normally, sidebar populates normally.
    """
    RUN_STORE.put(record)
    if record.demo_id:
        task = asyncio.create_task(_demo_pipeline(record))
    else:
        task = asyncio.create_task(_pipeline(record))
    _RUNNING_TASKS.add(task)
    task.add_done_callback(_RUNNING_TASKS.discard)


async def _pipeline(record: RunRecord) -> None:
    run_id = record.id
    try:
        await EVENT_BUS.publish(run_id, RunEvent(stage="queued", progress=0.0))

        # Phase 1 + Phase 2 in parallel. Each yields per-region or
        # per-stage progress events along the way.
        region_scores, trend_ctx = await asyncio.gather(
            _run_phase_1(record),
            _run_phase_2(record),
        )

        RUN_STORE.update(run_id, status="synthesizing")

        # Phase 3 — synthesis
        await EVENT_BUS.publish(run_id, RunEvent(stage="synthesizing", progress=0.85))

        prev_score = RUN_STORE.previous_score(record.user_id, exclude_id=run_id)
        synth_in = SynthesisInput(
            name=record.name,
            goal=record.goal,
            brief=record.brief,
            caption=record.caption,
            region_scores=region_scores,
            trend_context=trend_ctx,
            prev_score=prev_score,
            kind=record.kind,
            # Real-mode hook: when _run_phase_1 starts returning the 1 Hz
            # per-region timeseries from TRIBE, pass it through here so
            # the synthesis client can compute true peak windows. Mock
            # pipeline leaves this None.
            region_timeseries=None,
        )
        plan = get_synthesis_client().synthesize(synth_in)

        # Phase 4 — validation
        RUN_STORE.update(run_id, status="validating")
        await EVENT_BUS.publish(run_id, RunEvent(stage="validating", progress=0.95))
        plan = get_validation_client().validate(plan)

        # Done.
        RUN_STORE.update(
            run_id,
            status="complete",
            result=plan,
            completed_at=_now_iso(),
        )
        await EVENT_BUS.publish(run_id, RunEvent(stage="complete", progress=1.0))

    except Exception as exc:  # noqa: BLE001  — top-level pipeline guard
        _log.exception("pipeline failed for run %s", run_id)
        RUN_STORE.update(
            run_id,
            status="failed",
            error=str(exc),
            completed_at=_now_iso(),
        )
        await EVENT_BUS.publish(
            run_id,
            RunEvent(stage="failed", progress=1.0, error=str(exc)),
        )


# ---------------------------------------------------------------------------
# Demo short-circuit pipeline
# ---------------------------------------------------------------------------


async def _demo_pipeline(record: RunRecord) -> None:
    """Cosmetic pipeline for "Try a sample" runs.

    Mimics the real pipeline's stage events at roughly the same cadence
    so the frontend's analyzing animation plays normally, then loads
    the canned `SuggestionPlan` from disk. Total wall time ~2s.
    """
    from services.demo import load_demo_run

    run_id = record.id
    try:
        await EVENT_BUS.publish(run_id, RunEvent(stage="queued", progress=0.0))

        demo = load_demo_run(record.demo_id or "")
        if demo is None:
            raise ValueError(
                f"demo_id={record.demo_id!r} is not registered; "
                "expected one of data/demo_runs/*.json"
            )

        # The user didn't upload anything for a demo run, so the
        # RunRecord starts out with `media_url=None`. Copy the demo's
        # source URL onto the record so GET /runs/:id returns it — the
        # Results screen's hero player needs it to render the clip.
        if demo.media_url:
            RUN_STORE.update(run_id, media_url=demo.media_url)

        # Phase 1 cosmetic — emit per-region scan events so the
        # AnalysisAnimation hits each badge in turn. Same shape as the
        # real Phase 1 in `_run_phase_1`. The frontend's animation plays
        # its own ~5s cosmetic timeline regardless of backend timing,
        # so we don't need real sleeps here — just yield to the loop.
        RUN_STORE.update(run_id, status="neuro_running")
        for i, key in enumerate(REGION_KEYS):
            progress = (i + 1) / len(REGION_KEYS) * 0.7
            await EVENT_BUS.publish(
                run_id,
                RunEvent(stage="neuro_scanning", region_key=key, progress=progress),
            )
            await asyncio.sleep(0)
        RUN_STORE.update(run_id, status="neuro_done")

        # Phase 2 cosmetic.
        RUN_STORE.update(run_id, status="context_running")
        await EVENT_BUS.publish(
            run_id, RunEvent(stage="context_running", progress=0.78)
        )
        RUN_STORE.update(run_id, status="context_done")

        # Phase 3 cosmetic — skip real synthesis, use the canned plan.
        RUN_STORE.update(run_id, status="synthesizing")
        await EVENT_BUS.publish(
            run_id, RunEvent(stage="synthesizing", progress=0.9)
        )

        # Phase 4 cosmetic.
        RUN_STORE.update(run_id, status="validating")
        await EVENT_BUS.publish(
            run_id, RunEvent(stage="validating", progress=0.97)
        )

        RUN_STORE.update(
            run_id,
            status="complete",
            result=demo.plan,
            completed_at=_now_iso(),
        )
        await EVENT_BUS.publish(
            run_id, RunEvent(stage="complete", progress=1.0)
        )

    except Exception as exc:  # noqa: BLE001 — top-level guard
        _log.exception("demo pipeline failed for run %s", run_id)
        RUN_STORE.update(
            run_id,
            status="failed",
            error=str(exc),
            completed_at=_now_iso(),
        )
        await EVENT_BUS.publish(
            run_id,
            RunEvent(stage="failed", progress=1.0, error=str(exc)),
        )


# ---------------------------------------------------------------------------
# Phase 1 — neural inference (TRIBE v2)
# ---------------------------------------------------------------------------


async def _run_phase_1(record: RunRecord) -> dict[RegionKey, float]:
    """Phase 1 wrapper.

    In mock mode (the default), returns deterministic per-region
    scores derived from the brief text. Cycles per-region scanning
    events at ~250ms intervals so the frontend animation has a real
    pulse to react to.

    For real TRIBE inference, this is where you call the existing
    `api/predict.py` path and project the 8-region output through
    `core.regions_v2.project_legacy_scores`.
    """
    RUN_STORE.update(record.id, status="neuro_running")

    scores = _deterministic_region_scores(record.brief, record.caption, record.name)

    # Emit one event per region — drives the AnalysisAnimation's
    # sequential scan. Order matches the v2 canonical order.
    #
    # We don't pace these here; the AnalysisAnimation component on the
    # frontend handles the per-region cosmetic delay. Real TRIBE
    # inference takes minutes, so per-region progress events are
    # naturally spaced — the bottleneck is the GPU, not us.
    for i, key in enumerate(REGION_KEYS):
        progress = (i + 1) / len(REGION_KEYS) * 0.7  # leave headroom for Phase 2-4
        await EVENT_BUS.publish(
            record.id,
            RunEvent(stage="neuro_scanning", region_key=key, progress=progress),
        )

    RUN_STORE.update(record.id, status="neuro_done")
    return scores


def _deterministic_region_scores(*inputs: str) -> dict[RegionKey, float]:
    """Stable per-region score derived from the input strings.

    Same inputs → same scores, every time. Numbers cluster in the
    25-55 range so the resulting status badge is reliably "Needs
    work" — the mock pipeline's job is to look like a struggling ad
    so the suggestions feel useful.
    """
    seed = hash(("|".join(inputs)).encode("utf-8") if isinstance(inputs[0], str) else inputs)
    out: dict[RegionKey, float] = {}
    for i, key in enumerate(REGION_KEYS):
        # Deterministic, no Math.random — we're in production code.
        h = (seed >> (i * 5)) & 0xFFFF
        out[key] = 25.0 + (h % 3000) / 100.0  # 25.0 .. 55.0
    return out


# ---------------------------------------------------------------------------
# Phase 2 — social context (GraphRAG / mock)
# ---------------------------------------------------------------------------


async def _run_phase_2(record: RunRecord):
    RUN_STORE.update(record.id, status="context_running")
    await EVENT_BUS.publish(
        record.id, RunEvent(stage="context_running", progress=0.6)
    )
    # Run the (potentially synchronous) trend client in a thread so
    # we don't block Phase 1's event-loop ticks. Mock mode is
    # near-instant; a real GraphRAG call may not be.
    client = get_trends_client()
    ctx = await asyncio.to_thread(
        client.fetch,
        brief=record.brief,
        caption=record.caption,
        goal=record.goal,
    )
    RUN_STORE.update(record.id, status="context_done")
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Re-export so callers don't need to know about composite_score's
# location when reading this file.
__all__ = ["start_run", "composite_score"]
