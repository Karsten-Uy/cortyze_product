"""GET /demos — list of "Try a sample" cards for the Lab bench.

Public read access — these are static curated samples, not user data.
The full canned `SuggestionPlan` is loaded server-side when a demo
run is created via `POST /runs` with `demo_id=<id>`; the listing
endpoint only returns the lightweight summary the Lab bench needs to
render its 3 cards (label, tagline, thumbnail, kind).

Two extra endpoints back the Compare page:
- `GET /demos/comparison?a=<id>&b=<id>` — pairwise hand-written narrative.
- `GET /demos/{demo_id}` — full demo payload (incl. plan) for a single id.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.demo import (
    ComparisonNarrative,
    DemoRun,
    DemoSummary,
    list_demos,
    load_comparison_narrative,
    load_demo_run,
)

router = APIRouter()


@router.get("/demos", response_model=list[DemoSummary])
def list_demo_runs() -> list[DemoSummary]:
    return list_demos()


# Declared before `/demos/{demo_id}` so the static "comparison" path wins
# the route match instead of being treated as a demo_id.
@router.get("/demos/comparison", response_model=ComparisonNarrative)
def get_comparison_narrative(
    a: str = Query(..., description="First demo_id"),
    b: str = Query(..., description="Second demo_id"),
) -> ComparisonNarrative:
    if a == b:
        raise HTTPException(
            status_code=400, detail="comparison requires two distinct demo_ids"
        )
    narrative = load_comparison_narrative(a, b)
    if narrative is None:
        raise HTTPException(
            status_code=404,
            detail=f"no comparison narrative for pair ({a}, {b})",
        )
    return narrative


@router.get("/demos/{demo_id}", response_model=DemoRun)
def get_demo_run(demo_id: str) -> DemoRun:
    record = load_demo_run(demo_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown demo_id: {demo_id}")
    return record
