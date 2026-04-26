"""Tests for core.scoring.goals.

Validates the goal weight table (typed from the Cortyze spec) is internally
consistent and stays in sync with core.atlas.regions.
"""

import pytest

from core.atlas.regions import REGIONS
from core.scoring.goals import Goal, GOAL_WEIGHTS, overall_score


def test_goal_enum_values():
    assert {g.value for g in Goal} == {
        "conversion",
        "awareness",
        "engagement",
        "brand_recall",
    }


def test_every_goal_has_weights():
    assert set(GOAL_WEIGHTS.keys()) == set(Goal)


@pytest.mark.parametrize("goal", list(Goal))
def test_each_goal_column_sums_to_one(goal: Goal):
    total = sum(GOAL_WEIGHTS[goal].values())
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("goal", list(Goal))
def test_goal_weights_match_region_keys(goal: Goal):
    assert set(GOAL_WEIGHTS[goal].keys()) == set(REGIONS.keys())


@pytest.mark.parametrize("goal", list(Goal))
def test_overall_score_bounded_when_inputs_bounded(goal: Goal):
    assert overall_score({k: 0.0 for k in REGIONS}, goal) == 0.0
    assert overall_score({k: 50.0 for k in REGIONS}, goal) == pytest.approx(50.0)
    assert overall_score({k: 100.0 for k in REGIONS}, goal) == pytest.approx(100.0)


@pytest.mark.parametrize("goal", list(Goal))
def test_overall_score_monotonic(goal: Goal):
    base = {k: 50.0 for k in REGIONS}
    base_score = overall_score(base, goal)
    for region in REGIONS:
        if GOAL_WEIGHTS[goal][region] > 0:
            bumped = {**base, region: 100.0}
            assert overall_score(bumped, goal) > base_score
