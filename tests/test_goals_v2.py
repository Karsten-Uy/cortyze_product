"""Unit tests for core/goals_v2.py and core/regions_v2.py."""

from __future__ import annotations

import math

from core.goals_v2 import (
    GOAL_KEYS,
    GOAL_WEIGHTS_V2,
    composite_score,
    status_label,
)
from core.regions_v2 import (
    BENCHMARKS,
    LEGACY_TO_V2,
    REGION_KEYS,
    project_legacy_scores,
)


def test_each_goal_column_sums_to_one():
    for goal in GOAL_KEYS:
        weights = GOAL_WEIGHTS_V2[goal]
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6), (
            f"{goal}: weights sum to {sum(weights.values()):.4f}"
        )


def test_every_goal_covers_all_six_regions():
    for goal in GOAL_KEYS:
        assert set(GOAL_WEIGHTS_V2[goal].keys()) == set(REGION_KEYS)


def test_composite_score_bounds():
    # All 100s → composite is 100.
    high = {k: 100.0 for k in REGION_KEYS}
    for goal in GOAL_KEYS:
        assert math.isclose(composite_score(high, goal), 100.0, abs_tol=1e-6)

    # All 0s → composite is 0.
    low = {k: 0.0 for k in REGION_KEYS}
    for goal in GOAL_KEYS:
        assert composite_score(low, goal) == 0.0


def test_status_label_thresholds():
    assert status_label(0) == "Needs work"
    assert status_label(49.9) == "Needs work"
    assert status_label(50) == "Solid"
    assert status_label(69.9) == "Solid"
    assert status_label(70) == "Strong"
    assert status_label(84.9) == "Strong"
    assert status_label(85) == "Hero"
    assert status_label(100) == "Hero"


def test_legacy_to_v2_projection():
    legacy = {
        "hippocampus": 80.0,
        "amygdala": 70.0,
        "visual_cortex": 60.0,
        "temporal_language": 50.0,
        "fusiform_face": 40.0,
        "reward": 30.0,
        # `prefrontal` and `motor` should be silently dropped
        "prefrontal": 99.0,
        "motor": 99.0,
    }
    v2 = project_legacy_scores(legacy)
    assert v2["memory"]    == 80.0
    assert v2["emotion"]   == 70.0
    assert v2["attention"] == 60.0
    assert v2["language"]  == 50.0
    assert v2["face"]      == 40.0
    assert v2["reward"]    == 30.0


def test_legacy_to_v2_partial_input_zero_fills():
    v2 = project_legacy_scores({"hippocampus": 80.0})
    assert v2["memory"] == 80.0
    for k in ("emotion", "attention", "language", "face", "reward"):
        assert v2[k] == 0.0


def test_benchmarks_cover_all_regions():
    assert set(BENCHMARKS.keys()) == set(REGION_KEYS)
    for v in BENCHMARKS.values():
        assert 0 < v < 100


def test_legacy_mapping_is_total_for_v2():
    """Every v2 region key has at least one legacy region mapped to it."""
    mapped = set(LEGACY_TO_V2.values())
    assert mapped == set(REGION_KEYS)
