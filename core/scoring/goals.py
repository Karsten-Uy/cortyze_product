"""Goal enum + per-goal region weights + goal-weighted overall score.

Stage 1 uses these for the overall-score computation. Stage 2 will reuse the
same `Goal` enum for suggestion threshold rules; Stage 4 reuses it again for
audience-profile weighting. Single source of truth — keep stringly-typed
goals out of every other layer.

Weights match the Cortyze marketing spec verbatim. Each goal column sums to
1.0; verified in tests/test_goals.py.
"""

from enum import Enum


class Goal(str, Enum):
    CONVERSION = "conversion"
    AWARENESS = "awareness"
    ENGAGEMENT = "engagement"
    BRAND_RECALL = "brand_recall"


GOAL_WEIGHTS: dict[Goal, dict[str, float]] = {
    Goal.CONVERSION: {
        "visual_cortex": 0.12,
        "prefrontal": 0.25,
        "amygdala": 0.10,
        "fusiform_face": 0.02,
        "temporal_language": 0.08,
        "hippocampus": 0.05,
        "motor": 0.20,
        "reward": 0.18,
    },
    Goal.AWARENESS: {
        "visual_cortex": 0.25,
        "prefrontal": 0.05,
        "amygdala": 0.25,
        "fusiform_face": 0.10,
        "temporal_language": 0.05,
        "hippocampus": 0.20,
        "motor": 0.02,
        "reward": 0.08,
    },
    Goal.ENGAGEMENT: {
        "visual_cortex": 0.18,
        "prefrontal": 0.02,
        "amygdala": 0.25,
        "fusiform_face": 0.15,
        "temporal_language": 0.05,
        "hippocampus": 0.10,
        "motor": 0.05,
        "reward": 0.20,
    },
    Goal.BRAND_RECALL: {
        "visual_cortex": 0.15,
        "prefrontal": 0.03,
        "amygdala": 0.20,
        "fusiform_face": 0.10,
        "temporal_language": 0.12,
        "hippocampus": 0.30,
        "motor": 0.02,
        "reward": 0.08,
    },
}


def overall_score(region_scores: dict[str, float], goal: Goal) -> float:
    """Weighted sum of the 8 region scores by the goal's column.

    Inputs and output are on a 0-100 scale.
    """
    weights = GOAL_WEIGHTS[goal]
    return sum(region_scores[k] * w for k, w in weights.items())
