"""Templated SuggestionPlan generator for free local development.

Inspects the per-region gap (region_score vs benchmark), picks the
worst-performing regions, and emits one or two suggestions per region
from a static template table. Deterministic given the same inputs.

The output shape is the contract — match
`docs/architecture/architecture_v2.md` §3.3 exactly so the frontend
Results view renders without translation.
"""

from __future__ import annotations

from core.goals_v2 import composite_score, status_label
from core.regions_v2 import BENCHMARKS, REGION_KEYS
from core.schemas_v2 import (
    Reference,
    RegionScore,
    Suggestion,
    SuggestionPlan,
)

from ..trends.protocol import TrendContext, TrendReference
from .protocol import SynthesisInput


# One or two templates per region. `lift` here is a heuristic seed —
# Phase 4 (validation) overwrites it with the simulated value.
_TEMPLATES: dict[str, list[dict[str, object]]] = {
    "memory": [
        {
            "priority": "critical",
            "title": "Add a story arc to your video",
            "lift": 8.2,
            "explanation": (
                "Your content presents facts in sequence. The brain encodes "
                "stories 2-3× more effectively than lists — a beginning, "
                "tension, and resolution triggers the memory circuit and "
                "creates lasting recall. Try restructuring around a single "
                "character or moment of change."
            ),
        },
        {
            "priority": "high",
            "title": "Repeat your brand name at 0:03 and 0:11",
            "lift": 4.1,
            "explanation": (
                "Spaced repetition (vs. clustered) lifts unaided recall "
                "meaningfully on memory benchmarks. Your brand name appears "
                "once near the end. Add a second mention in the first three "
                "seconds and reinforce in the back half."
            ),
        },
    ],
    "emotion": [
        {
            "priority": "critical",
            "title": "Open with something unexpected",
            "lift": 7.5,
            "explanation": (
                "The first 1.5 seconds barely register on emotional pathways. "
                "The amygdala responds to surprise and pattern violations — "
                "start with an unexpected visual, sound, or claim that breaks "
                "the viewer's prediction. Even a small jolt buys you the next "
                "5 seconds of attention."
            ),
        },
    ],
    "attention": [
        {
            "priority": "high",
            "title": "Increase visual contrast by 40%+",
            "lift": 6.1,
            "explanation": (
                "Visual cortex activation correlates strongly with luminance "
                "contrast and edge density. Your current creative reads as "
                "low-contrast across most frames. Push subject-vs-background "
                "separation, deepen shadows, or shift palette toward "
                "complementary hues."
            ),
        },
    ],
    "language": [
        {
            "priority": "medium",
            "title": "Simplify caption to grade 6 reading level",
            "lift": 3.2,
            "explanation": (
                "Your caption uses several multi-clause sentences. The "
                "temporal lobe processes simpler syntax 30% faster, leaving "
                "more cognitive bandwidth for emotional encoding. Aim for "
                "short, declarative sentences."
            ),
        },
    ],
    "face": [
        {
            "priority": "high",
            "title": "Show a real face in the first 1.5 seconds",
            "lift": 4.4,
            "explanation": (
                "The fusiform face area lights up within 170ms of seeing a "
                "face — earlier than almost any other recognition pathway. A "
                "human face in the opening frame anchors attention and "
                "builds parasocial trust before the message even lands."
            ),
        },
    ],
    "reward": [
        {
            "priority": "medium",
            "title": "Add a payoff beat in the final 2 seconds",
            "lift": 3.5,
            "explanation": (
                "The end-card resolves visually but not viscerally. A brief "
                "tactile cue (water bead, breath, fabric flex) before the "
                "logo activates reward circuitry without extending runtime."
            ),
        },
    ],
}


# Per-region preference for which trend reference best fits. Maps the
# v2 region key to the `label_region` string the trend ref uses.
_REGION_TO_REFERENCE_LABEL: dict[str, list[str]] = {
    "memory":    ["Memory"],
    "emotion":   ["Emotion"],
    "attention": ["Attention"],
    "face":      ["Engagement", "Face"],
    "reward":    ["Reward"],
    "language":  ["Language"],
}


class MockSynthesisClient:
    """Templated SuggestionPlan. Deterministic, free, always succeeds."""

    def synthesize(self, payload: SynthesisInput) -> SuggestionPlan:
        # Build the regions list (always all 6, in canonical order).
        regions = [
            RegionScore(
                key=k,
                score=float(payload.region_scores.get(k, 0.0)),
                benchmark=BENCHMARKS[k],
            )
            for k in REGION_KEYS
        ]

        # Composite + status are deterministic, not LLM-decided.
        composite = composite_score(payload.region_scores, payload.goal)
        # `benchmark` here is the goal-weighted benchmark (so
        # delta-vs-benchmark is fair under any goal).
        from core.goals_v2 import GOAL_WEIGHTS_V2

        weights = GOAL_WEIGHTS_V2[payload.goal]
        weighted_benchmark = sum(BENCHMARKS[k] * w for k, w in weights.items())

        delta = (
            composite - payload.prev_score if payload.prev_score is not None else 0.0
        )
        status = status_label(composite)

        # Pick suggestions: rank regions by gap-to-benchmark (largest
        # negative gaps first). Take templates from the worst regions
        # until we've got 4-6 suggestions.
        gaps = sorted(
            REGION_KEYS,
            key=lambda k: (BENCHMARKS[k] - payload.region_scores.get(k, 0.0)),
            reverse=True,
        )

        suggestions: list[Suggestion] = []
        next_id = 1
        for region in gaps:
            for template in _TEMPLATES.get(region, []):
                suggestions.append(
                    Suggestion(
                        id=next_id,
                        priority=template["priority"],  # type: ignore[arg-type]
                        title=template["title"],  # type: ignore[arg-type]
                        area=region,
                        lift=template["lift"],  # type: ignore[arg-type]
                        explanation=template["explanation"],  # type: ignore[arg-type]
                        reference=_match_reference(region, payload.trend_context),
                    )
                )
                next_id += 1
            if len(suggestions) >= 6:
                break

        return SuggestionPlan(
            score=round(composite, 1),
            benchmark=round(weighted_benchmark, 1),
            delta=round(delta, 1),
            status=status,
            regions=regions,
            suggestions=suggestions,
        )


def _match_reference(
    region: str, ctx: TrendContext
) -> Reference | None:
    """Pick the trend reference whose `label_region` best matches the
    given v2 region key. Returns None if nothing in the trend context
    fits — which the frontend renders as a suggestion without a
    reference card."""
    candidates = _REGION_TO_REFERENCE_LABEL.get(region, [])
    for ref in ctx.references:
        if ref.label_region in candidates:
            return _to_api_reference(ref)
    return None


def _to_api_reference(ref: TrendReference) -> Reference:
    return Reference(
        brand=ref.brand,
        campaign=ref.campaign,
        note=ref.note,
        scoreA=ref.score_region,
        labelA=ref.label_region,
        scoreB=ref.score_overall,
        labelB=ref.label_overall,
    )
