"""Mock validation clients.

`MockValidationClient` deterministically nudges each suggestion's lift
up or down by a small amount so the output looks like it was simulated
rather than handed straight from Phase 3. The perturbation is keyed off
the suggestion id and priority so it's stable across runs (no
Math.random()-style render flicker — same input, same output).

`PassthroughValidationClient` does nothing. Useful when shipping v1
without a working Phase 4.
"""

from __future__ import annotations

from core.schemas_v2 import Suggestion, SuggestionPlan


# Per-priority bias — critical suggestions tend to validate slightly
# higher than the heuristic predicted; medium tend to validate slightly
# lower. Magnitudes are small (single-digit %) so the ordering is
# preserved.
_PRIORITY_BIAS: dict[str, float] = {
    "critical": +0.6,
    "high":      0.0,
    "medium":   -0.4,
}


class MockValidationClient:
    """Deterministic lift perturbation. Free, no I/O."""

    def validate(self, plan: SuggestionPlan) -> SuggestionPlan:
        new_suggestions: list[Suggestion] = []
        for s in plan.suggestions:
            # Bias by priority + a small id-keyed jitter (stays stable
            # across reruns of the same plan, never negative).
            jitter = ((s.id * 37) % 7 - 3) * 0.1  # -0.3 .. +0.3
            adjusted = s.lift + _PRIORITY_BIAS.get(s.priority, 0.0) + jitter
            new_suggestions.append(s.model_copy(update={"lift": round(max(0.0, adjusted), 1)}))

        return plan.model_copy(update={"suggestions": new_suggestions})


class PassthroughValidationClient:
    """No-op. Returns the plan unchanged."""

    def validate(self, plan: SuggestionPlan) -> SuggestionPlan:
        return plan
