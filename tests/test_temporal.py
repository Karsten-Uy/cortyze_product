"""Tests for core.atlas.temporal — per-timestep aggregation + normalization."""

import numpy as np
import pytest

from core.atlas.mapper import REGION_VERTICES
from core.atlas.regions import REGIONS
from core.atlas.temporal import aggregate_per_timestep, normalize_per_timestep


def test_aggregate_per_timestep_shape():
    preds = np.zeros((10, 20484), dtype=np.float32)
    out = aggregate_per_timestep(preds)
    assert set(out.keys()) == set(REGIONS.keys())
    for region, series in out.items():
        assert series.shape == (10,), f"{region}: {series.shape}"


def test_aggregate_per_timestep_constant_input_constant_output():
    preds = np.full((5, 20484), 0.42, dtype=np.float32)
    out = aggregate_per_timestep(preds)
    for series in out.values():
        # rtol generous: float32 + summing 1000s of values introduces
        # accumulated rounding error well above the default 1e-5
        np.testing.assert_allclose(series, 0.42, rtol=1e-4)


def test_aggregate_per_timestep_isolates_visual_cortex():
    """Activation only at visual_cortex vertices → only visual_cortex series moves."""
    preds = np.zeros((4, 20484), dtype=np.float32)
    visual_idx = REGION_VERTICES["visual_cortex"]
    preds[:, visual_idx] = 5.0
    out = aggregate_per_timestep(preds)
    np.testing.assert_allclose(out["visual_cortex"], 5.0)
    for region, series in out.items():
        if region != "visual_cortex":
            np.testing.assert_allclose(series, 0.0)


def test_aggregate_per_timestep_rejects_bad_shape():
    with pytest.raises(ValueError):
        aggregate_per_timestep(np.zeros((10, 100), dtype=np.float32))


def test_normalize_per_timestep_bounds():
    """Score must stay in [0, 100] for any raw value, including extremes."""
    raw = {region: np.array([-1000, -10, 0, 10, 1000], dtype=np.float32) for region in REGIONS}
    out = normalize_per_timestep(raw)
    for series in out.values():
        assert (series >= 0).all() and (series <= 100).all()


def test_normalize_per_timestep_monotonic():
    """Each region's series should be monotonic in raw input value."""
    raw = {region: np.array([-2, -1, 0, 1, 2], dtype=np.float32) for region in REGIONS}
    out = normalize_per_timestep(raw)
    for series in out.values():
        assert (np.diff(series) >= 0).all()


def test_normalize_per_timestep_keys_match():
    raw = {region: np.zeros(3, dtype=np.float32) for region in REGIONS}
    assert set(normalize_per_timestep(raw).keys()) == set(REGIONS.keys())
