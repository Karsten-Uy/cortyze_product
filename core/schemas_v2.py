"""Pydantic schemas for the v2 (`/runs`) pipeline.

These match the contract documented in
`docs/architecture/architecture_v2.md` §3.3 and the TypeScript types
in `cortyze_frontend/lib/cortyze-data.ts`. Field names must stay in
sync with the frontend — the Results view type-checks against them.
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .goals_v2 import GoalKey
from .regions_v2 import RegionKey


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Body of POST /runs.

    Mirrors the Lab Bench form on the frontend. `media_url` is a R2
    presigned URL (or any HTTPS URL the GPU worker can fetch) — the API
    doesn't accept multipart uploads directly, the frontend uses
    `/upload-url` first.
    """

    name: str
    goal: GoalKey
    brief: str = ""
    caption: str = ""
    media_url: str | None = None
    # Image vs video kind. Inferred from the file extension when missing.
    kind: Literal["Video", "Image"] | None = None


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class RegionScore(BaseModel):
    key: RegionKey
    score: float  # 0..100
    benchmark: float  # 0..100


class Reference(BaseModel):
    """Optional 'reference campaign' card under each suggestion."""

    brand: str
    campaign: str
    note: str
    scoreA: int
    labelA: str
    scoreB: int
    labelB: str


Priority = Literal["critical", "high", "medium"]


class Suggestion(BaseModel):
    """One actionable improvement attached to a region.

    `lift` is the predicted % uplift from applying this change. Phase 4
    (validation swarm) is what populates the final value; Phase 3
    (Claude / mock synthesis) seeds it with a heuristic.
    """

    id: int
    priority: Priority
    title: str  # imperative, ≤ 50 chars
    area: RegionKey
    lift: float  # e.g. 8.2 = +8.2% expected lift
    explanation: str
    reference: Reference | None = None
    # Slugs of registered library examples (services/examples/library.py).
    # The frontend lazy-fetches GET /examples/{name} on card expand and
    # renders the manifest's display_name, scores, and thumbnail. Empty
    # list = library had no good match → frontend falls back to `reference`.
    examples: list[str] = []


Status = Literal["Needs work", "Solid", "Strong", "Hero"]


class SuggestionPlan(BaseModel):
    """The result payload returned to the frontend Results view."""

    score: float  # composite 0..100
    benchmark: float  # category benchmark 0..100
    delta: float  # signed delta vs the user's previous run
    status: Status
    regions: list[RegionScore]
    suggestions: list[Suggestion]


# ---------------------------------------------------------------------------
# Run lifecycle / list views
# ---------------------------------------------------------------------------


RunStatus = Literal[
    "queued",
    "neuro_running",
    "neuro_done",
    "context_running",
    "context_done",
    "synthesizing",
    "plan_done",
    "validating",
    "complete",
    "failed",
]


class RunRecord(BaseModel):
    """Full server-side record for a run. Returned by GET /runs/:id."""

    id: str = Field(default_factory=lambda: f"r-{uuid4().hex[:8]}")
    user_id: str | None = None
    name: str
    goal: GoalKey
    brief: str = ""
    caption: str = ""
    media_url: str | None = None
    kind: Literal["Video", "Image"] = "Video"
    status: RunStatus = "queued"
    created_at: str  # ISO-8601, server-set
    completed_at: str | None = None
    # Populated once status >= plan_done. The frontend reads this once
    # status == "complete" (or "plan_done", if Phase 4 is still running).
    result: SuggestionPlan | None = None
    # Set when status == "failed". Surfaced in the Results view.
    error: str | None = None


class PastRun(BaseModel):
    """Compact projection used by the sidebar list endpoint."""

    id: str
    name: str
    date: str  # pre-formatted "Apr 28" — server picks the locale
    kind: Literal["Video", "Image"]
    score: float


class RunCreatedResponse(BaseModel):
    run_id: str
