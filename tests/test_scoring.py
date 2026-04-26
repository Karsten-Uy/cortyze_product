"""Tests for core.scoring.normalize.

Validates the calibration table is well-formed and that the sigmoid
normalization has the properties the rest of the system relies on:
bounded output, monotonic in input, and a known fixed point at the mean.
"""

import pytest

from core.atlas.regions import REGIONS
from core.scoring.normalize import CALIBRATION, normalize


def test_calibration_has_all_regions():
    assert set(CALIBRATION.keys()) == set(REGIONS.keys())


@pytest.mark.parametrize("region", list(REGIONS))
def test_calibration_entry_well_formed(region: str):
    entry = CALIBRATION[region]
    assert set(entry.keys()) == {"mu", "sigma"}
    assert isinstance(entry["mu"], (int, float))
    assert isinstance(entry["sigma"], (int, float))
    assert entry["sigma"] > 0


def test_normalize_keys_match_input():
    raw = {k: 0.5 for k in REGIONS}
    out = normalize(raw)
    assert set(out.keys()) == set(raw.keys())


def test_normalize_bounded_zero_to_hundred():
    for raw_value in [-1000.0, -10.0, 0.0, 10.0, 1000.0]:
        out = normalize({k: raw_value for k in REGIONS})
        for score in out.values():
            assert 0.0 <= score <= 100.0


def test_normalize_at_mu_returns_fifty():
    """When raw equals each region's mu, sigmoid(0) = 0.5, score = 50."""
    raw = {region: cal["mu"] for region, cal in CALIBRATION.items()}
    out = normalize(raw)
    for score in out.values():
        assert score == pytest.approx(50.0)


def test_normalize_monotonic():
    """Increasing raw activation strictly increases the score (sigma > 0)."""
    low_scores = normalize({k: -1.0 for k in REGIONS})
    high_scores = normalize({k: 1.0 for k in REGIONS})
    for region in REGIONS:
        assert high_scores[region] > low_scores[region]


def test_normalize_extreme_values_saturate():
    """100 sigmas below mu → near 0; 100 sigmas above → near 100.

    Phrased relative to mu/sigma so the test is robust to any future
    calibration values, not just the placeholder mu=0/sigma=1.
    """
    very_low = {r: c["mu"] - 100 * c["sigma"] for r, c in CALIBRATION.items()}
    very_high = {r: c["mu"] + 100 * c["sigma"] for r, c in CALIBRATION.items()}
    low_scores = normalize(very_low)
    high_scores = normalize(very_high)
    for region in REGIONS:
        assert low_scores[region] < 1.0
        assert high_scores[region] > 99.0
