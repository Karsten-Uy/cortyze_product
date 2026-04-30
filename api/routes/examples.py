"""GET /examples — reference-ad library access for the SuggestionCard expand UX.

When a user clicks a suggestion to expand, the frontend fetches the
top reference example for that region. The Suggestion ships only the
ad's `name` to keep payloads small; full details (display name,
thumbnail, score, description) come from this endpoint.

Public read access — reference ads are curated content, not user data,
so no auth is required. Listed alongside auth'd routes for symmetry with
the other route modules.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ExampleAd(BaseModel):
    name: str
    display_name: str
    description: str
    source_url: str
    license: str
    region_scores: dict[str, float]
    overall_by_goal: dict[str, float]
    # Stage 2 enrichments — older manifests may omit these.
    thumbnail_url: str | None = None
    tags: list[str] = []
    content_type: str | None = None
    caption: str | None = None


@router.get("/examples", response_model=list[ExampleAd])
def list_examples() -> list[ExampleAd]:
    from services.examples.library import all_ads

    return [_to_response(ad) for ad in all_ads()]


def _to_response(ad: dict) -> ExampleAd:
    """Translate a raw manifest dict to the API response shape, defaulting
    Stage 2 enrichments for older manifests written before they existed."""
    return ExampleAd(
        name=ad["name"],
        display_name=ad["display_name"],
        description=ad.get("description", ""),
        source_url=ad.get("source_url", ""),
        license=ad.get("license", ""),
        region_scores=ad.get("region_scores", {}),
        overall_by_goal=ad.get("overall_by_goal", {}),
        thumbnail_url=ad.get("thumbnail_url"),
        tags=ad.get("tags", []) or [],
        content_type=ad.get("content_type"),
        caption=ad.get("caption"),
    )


@router.get("/examples/{name}", response_model=ExampleAd)
def get_example(name: str) -> ExampleAd:
    from services.examples.library import get_by_name

    ad = get_by_name(name)
    if ad is None:
        raise HTTPException(status_code=404, detail=f"reference ad '{name}' not found")
    return _to_response(ad)
