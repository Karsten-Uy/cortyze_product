"""Five-goal weighting for the v2 (`/runs`) pipeline.

The v2 frontend offers five goals (vs. four in the legacy /analyze flow):
brand recall, purchase intent, emotional resonance, trust, attention.

Each goal column re-weights the six v2 region scores into a single
composite. Weights come from a neuromarketing-driven default (memory
matters most for brand recall, reward for purchase intent, etc.) — they
are deliberately opinionated. Each column sums to 1.0; verified in
tests/test_goals_v2.py.

Goal keys are snake_case identifiers used over the wire. The frontend
maps display strings ("Brand recall") to these keys before POSTing.
"""

from __future__ import annotations

from typing import Literal

from .regions_v2 import REGION_KEYS, RegionKey

GoalKey = Literal[
    "brand_recall",
    "purchase_intent",
    "emotional_resonance",
    "trust",
    "attention",
]

GOAL_KEYS: tuple[GoalKey, ...] = (
    "brand_recall",
    "purchase_intent",
    "emotional_resonance",
    "trust",
    "attention",
)

# Display strings shown in the Lab Bench dropdown. Mirrors GOAL_OPTIONS
# in `cortyze_frontend/lib/cortyze-data.ts` so the API can echo a
# user-friendly label back in responses if needed.
GOAL_DISPLAY: dict[GoalKey, str] = {
    "brand_recall":        "Brand recall",
    "purchase_intent":     "Purchase intent",
    "emotional_resonance": "Emotional resonance",
    "trust":               "Trust",
    "attention":           "Attention",
}


GOAL_WEIGHTS_V2: dict[GoalKey, dict[RegionKey, float]] = {
    "brand_recall": {
        "memory":    0.30,
        "emotion":   0.20,
        "attention": 0.15,
        "language":  0.12,
        "face":      0.13,
        "reward":    0.10,
    },
    "purchase_intent": {
        "memory":    0.10,
        "emotion":   0.20,
        "attention": 0.20,
        "language":  0.05,
        "face":      0.15,
        "reward":    0.30,
    },
    "emotional_resonance": {
        "memory":    0.15,
        "emotion":   0.40,
        "attention": 0.05,
        "language":  0.05,
        "face":      0.25,
        "reward":    0.10,
    },
    "trust": {
        "memory":    0.20,
        "emotion":   0.15,
        "attention": 0.10,
        "language":  0.20,
        "face":      0.30,
        "reward":    0.05,
    },
    "attention": {
        "memory":    0.05,
        "emotion":   0.20,
        "attention": 0.40,
        "language":  0.10,
        "face":      0.15,
        "reward":    0.10,
    },
}


def composite_score(
    region_scores: dict[RegionKey, float],
    goal: GoalKey,
) -> float:
    """Weighted sum of the six region scores under the selected goal.

    Inputs and output are on a 0..100 scale. Missing region keys are
    treated as 0 — the v2 pipeline always emits all six, but unit
    tests sometimes pass partial dicts.
    """
    weights = GOAL_WEIGHTS_V2[goal]
    return sum(region_scores.get(k, 0.0) * w for k, w in weights.items())


def status_label(score: float) -> str:
    """Map a composite score (0..100) to the badge string shown on the
    ScoreCard. Thresholds match the frontend's tone palette: anything
    sub-50 is coral ("Needs work"), 50-69 lands a neutral "Solid",
    70-84 is "Strong", and 85+ is "Hero".
    """
    if score < 50:
        return "Needs work"
    if score < 70:
        return "Solid"
    if score < 85:
        return "Strong"
    return "Hero"


# Re-export for callers that want the canonical region order.
__all__ = [
    "GoalKey",
    "GOAL_KEYS",
    "GOAL_DISPLAY",
    "GOAL_WEIGHTS_V2",
    "composite_score",
    "status_label",
    "REGION_KEYS",
]
