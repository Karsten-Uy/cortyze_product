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

from services.synthesis.peaks import fake_peak_window

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
    # Total clip duration in seconds. Derived from `predictions_shape[0]`
    # because TRIBE v2 outputs at 1 Hz; `None` for image-only ads.
    duration_s: float | None = None
    # Per-region peak window (seconds) — `{region_key: [start_s, end_s]}`.
    # The frontend uses these to seek the embedded player to the moment
    # in the example ad where this region peaks. Today a deterministic
    # fake_peak_window stand-in (mirrors the user-clip mock); when real
    # per-timestep timeseries land in the manifest we'll argmax over them.
    peak_windows: dict[str, list[float]] = {}


@router.get("/examples", response_model=list[ExampleAd])
def list_examples() -> list[ExampleAd]:
    from services.examples.library import all_ads

    return [_to_response(ad) for ad in all_ads()]


def _to_response(ad: dict) -> ExampleAd:
    """Translate a raw manifest dict to the API response shape, defaulting
    Stage 2 enrichments for older manifests written before they existed."""
    duration_s = _duration_from_shape(ad.get("predictions_shape"))
    region_keys = list((ad.get("region_scores") or {}).keys())
    # Use ad name as the seed so the same ad always yields the same
    # peak window for a given region; hashlib via fake_peak_window's
    # signature wants an int, so derive one stably from the name.
    name_seed = abs(hash(ad.get("name", ""))) & 0xFFFF
    peak_windows: dict[str, list[float]] = {}
    if duration_s and duration_s >= 4.0 and region_keys:
        for region in region_keys:
            start, end = fake_peak_window(
                name_seed, region, duration_s=duration_s, window_s=4.0
            )
            peak_windows[region] = [start, end]
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
        duration_s=duration_s,
        peak_windows=peak_windows,
    )


def _duration_from_shape(shape: list[int] | None) -> float | None:
    """TRIBE v2 outputs at 1 Hz, so the time axis (`shape[0]`) is the
    duration in seconds. `None` when shape is missing / malformed
    (typical for image-only manifests)."""
    if not shape or len(shape) < 1:
        return None
    try:
        return float(shape[0])
    except (TypeError, ValueError):
        return None


@router.get("/examples/{name}", response_model=ExampleAd)
def get_example(name: str) -> ExampleAd:
    from services.examples.library import get_by_name

    ad = get_by_name(name)
    if ad is None:
        raise HTTPException(status_code=404, detail=f"reference ad '{name}' not found")
    return _to_response(ad)
