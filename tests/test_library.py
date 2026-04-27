"""Tests for services.examples.library.

Validates the query API the suggestion engine will consume in Stage 2.
Uses synthetic in-memory ads where possible; falls back to whatever
manifests are committed under data/reference_ads/ for the real-data path.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

from core.atlas.regions import REGIONS
from core.scoring.goals import Goal
from services.examples import library
from services.examples.library import ReferenceAd


def _make_ad(name: str, region: str, score: float) -> ReferenceAd:
    """Create a synthetic reference ad with `score` in `region`, 0 elsewhere."""
    region_scores = {r: 0.0 for r in REGIONS}
    region_scores[region] = score
    return cast(
        ReferenceAd,
        {
            "name": name,
            "display_name": name,
            "source_url": f"https://example.com/{name}",
            "description": "",
            "license": "",
            "predictions_path": "",
            "predictions_shape": [10, 20484],
            "region_scores": region_scores,
            "overall_by_goal": {g.value: score / 8 for g in Goal},
            "registered_at": "2026-01-01T00:00:00Z",
        },
    )


def _patch_ads(ads: list[ReferenceAd]):
    """Replace the cached ads list."""
    library.reload()
    return patch.object(library._load_all, "__wrapped__", lambda: ads)


def test_top_n_for_region_orders_by_region_score():
    ads = [
        _make_ad("a", "visual_cortex", 30.0),
        _make_ad("b", "visual_cortex", 90.0),
        _make_ad("c", "visual_cortex", 60.0),
    ]
    library.reload()
    with patch.object(library, "_load_all", lambda: ads):
        top = library.top_n_for_region("visual_cortex", n=2)
    assert [a["name"] for a in top] == ["b", "c"]


def test_top_n_for_region_handles_missing_region():
    ads = [_make_ad("a", "visual_cortex", 50.0)]
    with patch.object(library, "_load_all", lambda: ads):
        top = library.top_n_for_region("nonexistent_region", n=3)
    assert len(top) == 1
    assert top[0]["name"] == "a"


def test_top_n_for_goal_orders_by_overall():
    ads = [
        _make_ad("low", "amygdala", 10.0),
        _make_ad("high", "amygdala", 90.0),
        _make_ad("mid", "amygdala", 50.0),
    ]
    with patch.object(library, "_load_all", lambda: ads):
        top = library.top_n_for_goal(Goal.ENGAGEMENT, n=2)
    assert [a["name"] for a in top] == ["high", "mid"]


def test_get_by_name_returns_match_or_none():
    ads = [_make_ad("foo", "visual_cortex", 50.0)]
    with patch.object(library, "_load_all", lambda: ads):
        assert library.get_by_name("foo") is not None
        assert library.get_by_name("missing") is None


def test_all_ads_returns_a_copy():
    """Mutating the returned list must not affect the cache."""
    ads = [_make_ad("a", "motor", 50.0)]
    with patch.object(library, "_load_all", lambda: ads):
        out = library.all_ads()
    out.clear()
    with patch.object(library, "_load_all", lambda: ads):
        assert len(library.all_ads()) == 1


def test_real_library_loads_and_has_sintel():
    """If a real library exists on disk (sintel_trailer.json), it loads."""
    library.reload()
    real = library.all_ads()
    if not real:
        # No reference ads registered yet — skip rather than fail
        import pytest
        pytest.skip("no manifests in data/reference_ads/")
    names = {a["name"] for a in real}
    assert "sintel_trailer" in names, f"got: {names}"
    sintel = library.get_by_name("sintel_trailer")
    assert sintel is not None
    assert set(sintel["region_scores"].keys()) == set(REGIONS.keys())
    assert all(0.0 <= s <= 100.0 for s in sintel["region_scores"].values())
