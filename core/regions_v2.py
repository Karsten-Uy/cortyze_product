"""Six-region grouping for the v2 (`/runs`) pipeline.

The v2 frontend exposes six brain regions, not the eight of the legacy
`/analyze` flow. The new keys are a strict projection of the existing
eight: each new region maps 1:1 to one of the old keys, and the two
old regions that aren't surfaced (`prefrontal`, `motor`) are dropped.

Single source of truth — the same keys appear in:
  * `core.schemas_v2.RegionScore.key`
  * `core.goals_v2.GOAL_WEIGHTS_V2`
  * `cortyze_frontend/lib/cortyze-data.ts` (must match exactly)
"""

from __future__ import annotations

from typing import Literal

RegionKey = Literal[
    "memory",
    "emotion",
    "attention",
    "language",
    "face",
    "reward",
]

REGION_KEYS: tuple[RegionKey, ...] = (
    "memory",
    "emotion",
    "attention",
    "language",
    "face",
    "reward",
)

# Display label, scientific name, and a one-line role description per
# region. The frontend pulls equivalent strings from its own data
# module; these mirror them for backend-side rendering / logging.
REGION_META: dict[RegionKey, dict[str, str]] = {
    "memory":    {"label": "Memory",           "sci": "Hippocampus",         "role": "Memory encoding"},
    "emotion":   {"label": "Emotion",          "sci": "Amygdala",            "role": "Emotional salience"},
    "attention": {"label": "Attention",        "sci": "Visual cortex",       "role": "Visual processing"},
    "language":  {"label": "Language",         "sci": "Temporal lobe",       "role": "Language processing"},
    "face":      {"label": "Face recognition", "sci": "Fusiform face area",  "role": "Face detection"},
    "reward":    {"label": "Reward",           "sci": "NAcc / VTA",          "role": "Reward & motivation"},
}

# Mapping from the legacy 8-region keys (`core.atlas.regions.REGIONS`)
# to the v2 6-region keys. Unmapped legacy regions (`prefrontal`,
# `motor`) are intentionally dropped — they aren't surfaced in v2.
LEGACY_TO_V2: dict[str, RegionKey] = {
    "hippocampus":       "memory",
    "amygdala":          "emotion",
    "visual_cortex":     "attention",
    "temporal_language": "language",
    "fusiform_face":     "face",
    "reward":            "reward",
}


def project_legacy_scores(
    legacy_scores: dict[str, float],
) -> dict[RegionKey, float]:
    """Project an 8-region score dict onto the 6 v2 keys.

    Missing entries default to 0.0 so callers can pass partial dicts
    (useful in tests). Anything in `legacy_scores` whose key isn't in
    `LEGACY_TO_V2` is silently ignored — that's the intended behaviour
    for `prefrontal` / `motor`.
    """
    out: dict[RegionKey, float] = {k: 0.0 for k in REGION_KEYS}
    for legacy_key, v2_key in LEGACY_TO_V2.items():
        if legacy_key in legacy_scores:
            out[v2_key] = float(legacy_scores[legacy_key])
    return out


# Per-region category benchmarks. These are the dotted-line markers in
# the Results breakdown bars. Numbers come from the frontend prototype
# (`cortyze-data.ts`); calibrate against real ad-corpus data once Phase 2
# starts ingesting real reference campaigns.
BENCHMARKS: dict[RegionKey, float] = {
    "memory":    62.0,
    "emotion":   58.0,
    "attention": 55.0,
    "language":  52.0,
    "face":      50.0,
    "reward":    45.0,
}
