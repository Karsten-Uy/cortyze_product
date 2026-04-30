"""Threshold rule engine — decides which regions deserve a suggestion.

Pure logic. No LLM, no I/O.

Two knobs control firing (both env-tunable so you can A/B without code edits):

  SUGGESTION_SCORE_THRESHOLD   region score must be < this to fire.
                               Default 70.0. Was 50 (sigmoid midpoint);
                               raised so even "below average" regions fire.
                               Lower it to be stricter (fewer suggestions).

  SUGGESTION_MIN_WEIGHT        region's goal weight must be >= this.
                               Default 0.10 (the "important" tier). Was
                               0.05 (the "minor" tier). Raised to skip
                               low-weight regions where the marginal cost
                               of a suggestion isn't worth the goal impact.

Tier classification (after both filters pass):
  weight >= 0.20 → critical (red flag, fix first)
  weight >= 0.10 → important
  weight >= 0.05 → minor   (only reachable if SUGGESTION_MIN_WEIGHT < 0.10)

Sorted output: critical first, then by deficit × weight so the most-urgent
fix is at the top.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from core.scoring.goals import GOAL_WEIGHTS, Goal

Priority = Literal["critical", "important", "minor"]


@dataclass(frozen=True)
class TriggeredRule:
    region: str
    score: float
    weight: float
    priority: Priority


_PRIORITY_ORDER: dict[Priority, int] = {
    "critical": 0,
    "important": 1,
    "minor": 2,
}


# Env-resolved at import time. Restart uvicorn to apply changes.
DEFAULT_SCORE_THRESHOLD = float(
    os.environ.get("SUGGESTION_SCORE_THRESHOLD", "70.0")
)
DEFAULT_MIN_WEIGHT = float(os.environ.get("SUGGESTION_MIN_WEIGHT", "0.10"))


def _classify(weight: float) -> Priority | None:
    if weight >= 0.20:
        return "critical"
    if weight >= 0.10:
        return "important"
    if weight >= 0.05:
        return "minor"
    return None


def trigger_rules(
    region_scores: dict[str, float],
    goal: Goal,
    *,
    score_threshold: float | None = None,
    min_weight: float | None = None,
) -> list[TriggeredRule]:
    """Return the regions that warrant a suggestion for this goal.

    Both thresholds default to env-resolved values; pass explicit floats
    to override (used in tests to pin behavior across env changes).
    """
    sthresh = (
        DEFAULT_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )
    mw = DEFAULT_MIN_WEIGHT if min_weight is None else min_weight

    weights = GOAL_WEIGHTS[goal]
    out: list[TriggeredRule] = []
    for region, score in region_scores.items():
        if score >= sthresh:
            continue
        weight = weights.get(region, 0.0)
        if weight < mw:
            continue
        priority = _classify(weight)
        if priority is None:
            continue
        out.append(
            TriggeredRule(
                region=region, score=score, weight=weight, priority=priority
            )
        )
    out.sort(
        key=lambda r: (_PRIORITY_ORDER[r.priority], -((50.0 - r.score) * r.weight))
    )
    return out
