"""In-process event bus for run progress.

Each run gets its own asyncio.Queue. The orchestrator publishes
progress events; the SSE endpoint subscribes and forwards them to the
client. No external broker — single-process FastAPI is fine for this
build, and the ergonomics of an in-process queue keep the code
testable without spinning up Redis.

When the API moves to multiple workers (uvicorn `--workers > 1`), this
needs to become a real broker (Redis pub/sub, NATS, etc.). Until then,
the single-process assumption is documented at the call site.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from core.schemas_v2 import RegionKey


@dataclass
class RunEvent:
    """One progress tick. The frontend's AnalysisAnimation consumes
    `stage` + `regionKey` + `progress` to drive its scan visualisation.

    Stages:
      * `queued`             — job created, not yet running.
      * `neuro_scanning`     — Phase 1 in flight; `regionKey` cycles.
      * `context_running`    — Phase 2 (parallel with neuro).
      * `synthesizing`       — Phase 3 (Claude / mock).
      * `validating`         — Phase 4 (MiroFish / mock).
      * `complete`           — final event; subscriber should close.
      * `failed`             — terminal; `error` populated.
    """

    stage: str
    progress: float = 0.0  # 0..1
    region_key: RegionKey | None = None
    error: str | None = None

    def to_sse(self) -> str:
        """Serialize to a Server-Sent Events frame."""
        import json

        payload: dict[str, object] = {"stage": self.stage, "progress": self.progress}
        if self.region_key:
            payload["regionKey"] = self.region_key
        if self.error:
            payload["error"] = self.error
        return f"event: stage\ndata: {json.dumps(payload)}\n\n"


class _EventBus:
    """Per-run-id queue registry. Singleton."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[RunEvent | None]] = {}
        # Kept-around tail of the last completed event per run, so a
        # subscriber that connects late still gets the terminal state
        # rather than a hang. None == no terminal yet.
        self._terminal: dict[str, RunEvent] = {}

    def queue_for(self, run_id: str) -> asyncio.Queue[RunEvent | None]:
        if run_id not in self._queues:
            self._queues[run_id] = asyncio.Queue()
        return self._queues[run_id]

    async def publish(self, run_id: str, event: RunEvent) -> None:
        q = self.queue_for(run_id)
        await q.put(event)
        if event.stage in ("complete", "failed"):
            self._terminal[run_id] = event
            # Sentinel so subscribers can break out of their loop.
            await q.put(None)

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        # If the run already terminated, replay just the terminal
        # event and exit — late subscribers see the final state, not
        # silence.
        if run_id in self._terminal and run_id not in self._queues:
            yield self._terminal[run_id]
            return

        q = self.queue_for(run_id)
        while True:
            event = await q.get()
            if event is None:
                break
            yield event

    def clear(self, run_id: str) -> None:
        """Drop the queue for a run. Called after the SSE subscriber
        finishes streaming so old runs don't leak memory.
        """
        self._queues.pop(run_id, None)
        # Keep `_terminal` around — it's tiny and lets late subscribers
        # still see the final state. A real implementation would TTL
        # this; for now it lives until process restart.


EVENT_BUS = _EventBus()
