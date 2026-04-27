"""Threshold rule engine — decides which regions deserve a suggestion.

Pure logic. No LLM, no I/O. Per the strategy doc:

  weight ≥ 0.20 → Critical (red flag, fix first)
  weight ≥ 0.10 → Important
  weight ≥ 0.05 → Minor
  weight < 0.05 → Ignore

A region must ALSO score below `score_threshold` (default 50, which is
the sigmoid-midpoint by construction). Sorted output puts Critical first,
then by deficit × weight so the most-urgent fix is at the top.
"""

from __future__ import annotations

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
    score_threshold: float = 50.0,
) -> list[TriggeredRule]:
    """Return the regions that warrant a suggestion for this goal."""
    weights = GOAL_WEIGHTS[goal]
    out: list[TriggeredRule] = []
    for region, score in region_scores.items():
        if score >= score_threshold:
            continue
        weight = weights.get(region, 0.0)
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
