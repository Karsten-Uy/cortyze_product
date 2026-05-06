"""Protocol + input shape for Phase 3 (suggestion synthesis).

Synthesis takes the joined output of Phases 1 (region scores) and 2
(trend context) plus the user's stated goal, and emits a structured
SuggestionPlan. Composite scoring is deterministic and lives outside
the LLM (see `core.goals_v2.composite_score`); the LLM only decides
which suggestions to surface and how to rank them.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from core.goals_v2 import GoalKey
from core.regions_v2 import RegionKey
from core.schemas_v2 import SuggestionPlan

from ..trends.protocol import TrendContext


class SynthesisInput(BaseModel):
    """Joined Phase 1 + Phase 2 payload handed to the LLM.

    `region_scores` is the already-projected v2 6-region dict.
    `prev_score` is the user's most recent composite (used to compute
    `delta` in the response). `None` means this is the user's first run.
    """

    name: str
    goal: GoalKey
    brief: str
    caption: str
    region_scores: dict[RegionKey, float]
    trend_context: TrendContext
    prev_score: float | None = None


class SynthesisClient(Protocol):
    """Single-method interface for Phase 3 implementations."""

    def synthesize(self, payload: SynthesisInput) -> SuggestionPlan:
        ...
