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
from core.regions_v2 import BENCHMARKS, LEGACY_TO_V2, REGION_KEYS
from core.schemas_v2 import (
    Reference,
    RegionScore,
    Suggestion,
    SuggestionPlan,
)

from ..trends.protocol import TrendContext, TrendReference
from .peaks import fake_peak_window
from .protocol import SynthesisInput

# Inverse of LEGACY_TO_V2 — v2 region key → legacy 8-region key the
# examples library indexes by. Used to look up reference manifests for
# a v2 suggestion.
_V2_TO_LEGACY: dict[str, str] = {v: k for k, v in LEGACY_TO_V2.items()}


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
        is_video = payload.kind == "Video"
        for region in gaps:
            for template in _TEMPLATES.get(region, []):
                peak_start: float | None = None
                peak_end: float | None = None
                if is_video:
                    peak_start, peak_end = fake_peak_window(next_id, region)
                suggestions.append(
                    Suggestion(
                        id=next_id,
                        priority=template["priority"],  # type: ignore[arg-type]
                        title=template["title"],  # type: ignore[arg-type]
                        area=region,
                        lift=template["lift"],  # type: ignore[arg-type]
                        explanation=template["explanation"],  # type: ignore[arg-type]
                        # `reference` is the legacy trends-mock card (Aesop /
                        # Apple / Nike / Dove). Intentionally left None — the
                        # frontend now exclusively renders library examples
                        # via `examples` slugs and lazy `/examples/{name}`
                        # fetches. Keeping the field in the schema for
                        # backward-compat with old DB-cached runs.
                        reference=None,
                        examples=_examples_for_region(region, payload.goal),
                        peak_start_s=peak_start,
                        peak_end_s=peak_end,
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


def _examples_for_region(v2_region: str, goal: str, n: int = 2) -> list[str]:
    """Return library example slugs for a v2 region + run goal.

    Best-effort with logging: if the library can't load or the goal value
    isn't representable in the v1 Goal enum, log the cause at WARNING and
    return []. We don't crash the whole synthesis over a missing example.
    """
    import logging

    log = logging.getLogger(__name__)

    print(
        f"[DEBUG _examples_for_region] v2_region={v2_region!r} goal={goal!r}",
        flush=True,
    )

    legacy_region = _V2_TO_LEGACY.get(v2_region)
    if not legacy_region:
        log.warning("no legacy region mapping for v2 region=%r", v2_region)
        print(f"[DEBUG] -> no legacy mapping, returning []", flush=True)
        return []

    # The v2 surface has 5 goals (brand_recall, purchase_intent,
    # emotional_resonance, trust, attention); the v1 Goal enum the
    # library indexes by has 4 (conversion, awareness, engagement,
    # brand_recall). Translate v2 → v1 via a small static map; goals
    # absent from v1 fall back to the closest match.
    from core.scoring.goals import Goal

    v2_to_v1_goal = {
        "brand_recall":        Goal.BRAND_RECALL,
        "purchase_intent":     Goal.CONVERSION,
        "emotional_resonance": Goal.ENGAGEMENT,
        "trust":               Goal.AWARENESS,
        "attention":           Goal.AWARENESS,
    }
    v1_goal = v2_to_v1_goal.get(goal)
    if v1_goal is None:
        # Unknown v2 goal — try the raw string in case caller passed a
        # v1 value directly (back-compat with tests).
        try:
            v1_goal = Goal(goal)
        except ValueError:
            log.warning("no v1 Goal mapping for v2 goal=%r", goal)
            return []

    try:
        from services.examples.library import best_examples

        ads = best_examples(region=legacy_region, goal=v1_goal, n=n)
        names = [ad["name"] for ad in ads]
        print(
            f"[DEBUG] -> region={legacy_region} goal={v1_goal.value} "
            f"returned {len(ads)} ads: {names}",
            flush=True,
        )
        return names
    except Exception as exc:
        log.warning(
            "best_examples failed for region=%s goal=%s: %s",
            legacy_region, v1_goal, exc,
        )
        print(f"[DEBUG] -> EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
        return []
