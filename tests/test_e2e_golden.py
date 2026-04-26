"""End-to-end pipeline regression test against the real fixture.

Runs the full back-half of the pipeline -- aggregate -> normalize ->
overall_score -- against tests/fixtures/golden_pred_*.npy and compares
the BrainReport-shaped output against a committed snapshot. Catches
silent changes to atlas/scoring/goals math.

First run with no snapshot: writes tests/fixtures/golden_report_sintel.json
and passes. Subsequent runs compare. To regenerate after an intentional
math change, delete the snapshot file and re-run.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.atlas.mapper import aggregate
from core.atlas.regions import REGIONS
from core.scoring.goals import GOAL_WEIGHTS, Goal, overall_score
from core.scoring.normalize import normalize

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_SNAPSHOT = _FIXTURES / "golden_report_sintel.json"
_TOLERANCE = 0.01  # absolute, on a 0-100 scale; well below human-perceptible difference


def _load_fixture() -> np.ndarray | None:
    candidates = sorted(_FIXTURES.glob("golden_pred_*.npy"))
    if not candidates:
        return None
    return np.load(candidates[0])


def _compute_report(preds: np.ndarray) -> dict:
    region_scores = normalize(aggregate(preds))
    overall_by_goal = {
        g.value: overall_score(region_scores, g) for g in Goal
    }
    return {
        "region_scores": region_scores,
        "overall_by_goal": overall_by_goal,
    }


@pytest.fixture(scope="module")
def report() -> dict:
    preds = _load_fixture()
    if preds is None:
        pytest.skip("no golden_pred_*.npy fixture present (run scripts/build_fixture.py)")
    return _compute_report(preds)


def test_report_has_all_eight_regions(report: dict):
    assert set(report["region_scores"].keys()) == set(REGIONS.keys())


def test_report_has_all_four_goals(report: dict):
    assert set(report["overall_by_goal"].keys()) == {g.value for g in Goal}


def test_all_scores_in_range(report: dict):
    for v in report["region_scores"].values():
        assert 0.0 <= v <= 100.0
    for v in report["overall_by_goal"].values():
        assert 0.0 <= v <= 100.0


def test_overall_matches_weight_application(report: dict):
    """overall_by_goal[g] must equal sum(region_scores[r] * GOAL_WEIGHTS[g][r])."""
    for goal in Goal:
        expected = sum(
            report["region_scores"][r] * w
            for r, w in GOAL_WEIGHTS[goal].items()
        )
        assert abs(report["overall_by_goal"][goal.value] - expected) < 1e-6


def test_matches_snapshot(report: dict):
    """Compare against the committed snapshot; first run writes it."""
    if not _SNAPSHOT.exists():
        _SNAPSHOT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote new snapshot to {_SNAPSHOT.name}; re-run to compare")

    expected = json.loads(_SNAPSHOT.read_text())

    for region, score in report["region_scores"].items():
        delta = abs(score - expected["region_scores"][region])
        assert delta < _TOLERANCE, (
            f"{region}: got {score:.4f}, expected {expected['region_scores'][region]:.4f} "
            f"(delta {delta:.4f} > tolerance {_TOLERANCE}). "
            f"If math changed intentionally, delete {_SNAPSHOT.name} and rerun."
        )

    for goal, score in report["overall_by_goal"].items():
        delta = abs(score - expected["overall_by_goal"][goal])
        assert delta < _TOLERANCE, (
            f"overall[{goal}]: got {score:.4f}, expected {expected['overall_by_goal'][goal]:.4f}"
        )
