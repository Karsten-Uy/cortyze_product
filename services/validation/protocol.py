"""Protocol for Phase 4 (validation swarm).

Validation takes a SuggestionPlan and returns the same plan with each
suggestion's `lift` value updated based on a multi-agent simulation.
The composite score / regions / status / suggestion content stay
unchanged — Phase 4 only re-grades expected lift.

Suggestions whose validated lift falls below `LIFT_FLOOR` (default
1.5%) may be filtered out by the orchestrator; that policy lives at
the orchestrator layer, not here.
"""

from __future__ import annotations

from typing import Protocol

from core.schemas_v2 import SuggestionPlan


class ValidationClient(Protocol):
    """Single-method interface for Phase 4 implementations."""

    def validate(self, plan: SuggestionPlan) -> SuggestionPlan:
        ...
