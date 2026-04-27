"""Stage 2 reference ad library.

Loads all manifests from data/reference_ads/ at import. Provides the
query API the suggestion engine needs: "given that the user's visual
cortex scored 32, find ads that scored top-N in visual cortex." Used by
services/suggestions/ in Stage 2 to pair every suggestion with a
concrete, brain-validated example.

Today: JSON-on-disk manifests written by scripts/register_reference_ad.py.

# TODO(stage 2): when reference ads come from production /analyze calls
# (saved to R2 + Postgres), replace _load_all() with a Postgres query
# against the reports table filtered by `is_reference=true`. The query
# API stays unchanged.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from core.scoring.goals import Goal


class ReferenceAd(TypedDict):
    name: str
    display_name: str
    source_url: str
    description: str
    license: str
    predictions_path: str
    predictions_shape: list[int]
    region_scores: dict[str, float]
    overall_by_goal: dict[str, float]
    registered_at: str


_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "reference_ads"


@lru_cache(maxsize=1)
def _load_all() -> list[ReferenceAd]:
    if not _DATA_DIR.exists():
        return []
    out: list[ReferenceAd] = []
    for p in sorted(_DATA_DIR.glob("*.json")):
        if p.name == "manifest.json":
            continue  # reserved for future bulk-index file
        out.append(json.loads(p.read_text()))
    return out


def reload() -> None:
    """Drop the cache so the next call reloads from disk. Useful in tests."""
    _load_all.cache_clear()


def all_ads() -> list[ReferenceAd]:
    return list(_load_all())


def get_by_name(name: str) -> ReferenceAd | None:
    for ad in _load_all():
        if ad["name"] == name:
            return ad
    return None


def top_n_for_region(region: str, n: int = 3) -> list[ReferenceAd]:
    """Highest-scoring reference ads for a given brain region key.

    Used by the suggestion engine: when a user's region scored low,
    surface examples that scored high there.
    """
    return sorted(
        _load_all(),
        key=lambda ad: ad["region_scores"].get(region, 0.0),
        reverse=True,
    )[:n]


def top_n_for_goal(goal: Goal, n: int = 3) -> list[ReferenceAd]:
    """Highest-scoring reference ads for a goal's weighted overall score."""
    return sorted(
        _load_all(),
        key=lambda ad: ad["overall_by_goal"].get(goal.value, 0.0),
        reverse=True,
    )[:n]
