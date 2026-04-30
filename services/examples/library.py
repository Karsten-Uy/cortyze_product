"""Stage 2 reference ad library.

Loads all manifests from data/reference_ads/ at import. Provides the
query API the suggestion engine needs: "given that the user's visual
cortex scored 32 on a Conversion-goal Pepsi post, find ads that
(a) scored high in visual cortex, (b) score well for Conversion overall,
and (c) are topically relevant to the user's content."

Today: JSON-on-disk manifests written by scripts/register_reference_ad.py.

# TODO(stage 2): when reference ads come from production /analyze calls
# (saved to R2 + Postgres), replace _load_all() with a Postgres query
# against the reports table filtered by `is_reference=true`. The query
# API stays unchanged.

# TODO(stage 2-3): replace `_lexical_relevance` with real embedding
# similarity (CLIP for visuals, text-embedding-3 for captions). The
# function-level interface stays the same; only the scoring
# implementation changes.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from core.scoring.goals import Goal


class ReferenceAd(TypedDict, total=False):
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
    # Optional Stage 2 enrichments — older manifests don't carry them.
    thumbnail_url: str  # public URL to a still preview (16:9 ideal)
    tags: list[str]  # category keywords used for relevance ranking
    content_type: str  # "video" | "post" | "image"
    caption: str  # the original caption / copy that accompanied the ad


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

    Pure region-score ranking — kept for backward compat and tests. New
    callers should prefer `best_examples` which factors in goal + content
    relevance too.
    """
    return sorted(
        _load_all(),
        key=lambda ad: ad.get("region_scores", {}).get(region, 0.0),
        reverse=True,
    )[:n]


def top_n_for_goal(goal: Goal, n: int = 3) -> list[ReferenceAd]:
    """Highest-scoring reference ads for a goal's weighted overall score."""
    return sorted(
        _load_all(),
        key=lambda ad: ad.get("overall_by_goal", {}).get(goal.value, 0.0),
        reverse=True,
    )[:n]


# ---- Stage 2 ranking ---------------------------------------------------


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "your", "you",
        "are", "but", "not", "have", "has", "had", "was", "were", "will",
        "would", "could", "should", "into", "out", "off", "all", "any",
        "more", "less", "than", "then", "also", "just", "now", "when", "what",
        "how", "why", "where", "they", "them", "our", "use", "using", "via",
        "via", "post", "video", "image", "ads", "ad", "content", "see", "get",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip stopwords + short tokens. Returns a set."""
    if not text:
        return set()
    tokens = {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}
    return tokens


def _lexical_relevance(ad: ReferenceAd, user_text: str) -> float:
    """Cheap word-overlap relevance between user content and an ad's
    description / caption / tags. Returns a multiplier in [0.5, 1.5]:
    no overlap = 0.5 (penalize off-topic), strong overlap = 1.5.

    Stand-in for real CLIP / text-embedding similarity, which is the
    Stage 2-3 plan. Behavior:
      - 0 overlap → 0.5 (50% damping)
      - 1 overlap → 0.7
      - 2 overlap → 0.9
      - 5+ overlap → 1.5 (50% boost, capped)
    """
    if not user_text:
        return 1.0  # no signal → neutral

    user_tokens = _tokenize(user_text)
    ad_text = " ".join(
        [
            ad.get("description", ""),
            ad.get("caption", ""),
            " ".join(ad.get("tags", []) or []),
            ad.get("display_name", ""),
        ]
    )
    ad_tokens = _tokenize(ad_text)
    if not user_tokens or not ad_tokens:
        return 1.0

    overlap = len(user_tokens & ad_tokens)
    return min(1.5, 0.5 + 0.2 * overlap)


def best_examples(
    *,
    region: str,
    goal: Goal,
    n: int = 3,
    user_text: str | None = None,
    user_content_type: str | None = None,
) -> list[ReferenceAd]:
    """Pick the best reference ads to surface alongside a region-X
    suggestion in a goal-Y user run.

    Ranking blends three signals:

      1. **Region score** (primary, 70% weight). The user's region X
         scored low; we want examples that scored HIGH on that same region.
      2. **Goal-weighted overall** (secondary, 30% weight). Tie-breaker
         when multiple ads score similarly in the region — prefer the
         one that's a stronger overall fit for the user's goal.
      3. **Lexical relevance** (multiplier, 0.5x–1.5x). Boost ads whose
         description/tags/caption share keywords with the user's caption
         + brand context. Lightweight stopgap until we wire embeddings.

    A small content-type bonus (+5) goes to ads that match the user's
    shape (post-vs-video) so a static-image user sees static-image
    examples first.
    """
    ads = _load_all()
    if not ads:
        return []

    def score(ad: ReferenceAd) -> float:
        region_s = ad.get("region_scores", {}).get(region, 0.0)
        goal_s = ad.get("overall_by_goal", {}).get(goal.value, 0.0)
        base = 0.7 * region_s + 0.3 * goal_s
        rel = _lexical_relevance(ad, user_text or "")
        # Content-type bonus is additive and small — it nudges ties but
        # doesn't override a much-better ad of the wrong shape.
        ct_bonus = (
            5.0
            if user_content_type and ad.get("content_type") == user_content_type
            else 0.0
        )
        return base * rel + ct_bonus

    return sorted(ads, key=score, reverse=True)[:n]
