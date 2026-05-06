"""v2 (`/runs`) endpoints: POST, GET, list, SSE event stream.

The four endpoints in this router match the contract documented in
`docs/architecture/architecture_v2.md` §5 and consumed by the new
Cortyze frontend at `cortyze_frontend/app/(authed)/page.tsx`.

  POST /runs              → create + queue a new run; returns run_id
  GET  /runs              → sidebar list (most recent N for the user)
  GET  /runs/{id}         → full RunRecord (status + result if ready)
  GET  /runs/{id}/events  → SSE stream of progress events

Auth: `optional_user` (matches the legacy /analyze pattern). When auth
is configured the user_id is enforced as the join key on list/get.
When AUTH_DISABLED=true the dev sentinel UUID is used so local dev
without Supabase still works end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from core.schemas_v2 import (
    PastRun,
    RunCreatedResponse,
    RunRecord,
    RunRequest,
)

from ..auth import optional_user
from ..limiter import limiter

# Importing the orchestrator/persistence singletons here is a deliberate
# coupling — the route is the only caller, and the alternative
# (dependency-injected factories) buys us nothing for a single-file
# router.
from services.orchestrator import EVENT_BUS, start_run
from services.persistence.runs_v2 import RUN_STORE


router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_kind(media_url: str | None) -> str:
    """Default Image vs Video guess from the URL extension."""
    if not media_url:
        return "Video"
    lowered = media_url.lower()
    if any(lowered.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".m4v")):
        return "Video"
    if any(lowered.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif")):
        return "Image"
    return "Video"


@router.post(
    "/runs",
    response_model=RunCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
@limiter.limit("30/minute")
async def create_run(
    request: Request,
    body: RunRequest,
    user_id: str | None = Depends(optional_user),
) -> RunCreatedResponse:
    """Queue a new run and return its id immediately.

    The pipeline runs in a background asyncio task; the frontend
    polls GET /runs/{id} or subscribes to /runs/{id}/events.
    """
    record = RunRecord(
        user_id=user_id,
        name=body.name,
        goal=body.goal,
        brief=body.brief,
        caption=body.caption,
        media_url=body.media_url,
        kind=body.kind or _infer_kind(body.media_url),  # type: ignore[arg-type]
        status="queued",
        created_at=_now_iso(),
    )
    await start_run(record)
    return RunCreatedResponse(run_id=record.id)


@router.get("/runs", response_model=list[PastRun])
async def list_runs(
    request: Request,
    limit: int = 20,
    user_id: str | None = Depends(optional_user),
) -> list[PastRun]:
    """Sidebar list. Most recent `limit` runs for the caller."""
    return RUN_STORE.list_for_user(user_id, limit=limit)


@router.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(
    run_id: str,
    request: Request,
    user_id: str | None = Depends(optional_user),
) -> RunRecord:
    record = RUN_STORE.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    if record.user_id is not None and record.user_id != user_id:
        # Don't leak existence — same 404 as a missing run.
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return record


@router.get("/runs/{run_id}/events")
async def run_events(
    run_id: str,
    request: Request,
    user_id: str | None = Depends(optional_user),
) -> StreamingResponse:
    """Server-Sent Events stream for the AnalysisAnimation view.

    The frontend opens this with `new EventSource(...)` and listens
    for `event: stage` frames. Stream closes after the terminal
    `complete` or `failed` event.
    """
    record = RUN_STORE.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    if record.user_id is not None and record.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

    async def streamer() -> AsyncIterator[bytes]:
        try:
            async for event in EVENT_BUS.subscribe(run_id):
                # If the client disconnects we stop publishing — the
                # `await client.is_disconnected()` check on `request` is
                # cheap and avoids a stale subscriber forever holding
                # the queue open.
                if await request.is_disconnected():
                    break
                yield event.to_sse().encode("utf-8")
                if event.stage in ("complete", "failed"):
                    break
        finally:
            EVENT_BUS.clear(run_id)

    return StreamingResponse(
        streamer(),
        media_type="text/event-stream",
        headers={
            # Keep proxies / browsers from buffering the stream.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
