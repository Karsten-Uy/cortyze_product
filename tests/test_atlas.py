"""Tests for core.atlas.mapper.

Validates the precomputed fsaverage5 DK label array and the per-region
aggregation: every region has vertices, constant inputs produce constant
outputs, and selectively activating one region's vertices isolates the
signal to that region.
"""

import numpy as np
import pytest

from core.atlas.mapper import REGION_VERTICES, VERTEX_LABELS, aggregate
from core.atlas.regions import REGIONS


def test_labels_array_shape():
    assert VERTEX_LABELS.shape == (20484,)


def test_every_region_has_vertices():
    for region in REGIONS:
        assert region in REGION_VERTICES
        assert len(REGION_VERTICES[region]) > 0, f"{region} resolves to zero vertices"


def test_aggregate_keys_match_regions():
    preds = np.zeros((10, 20484), dtype=np.float32)
    assert set(aggregate(preds).keys()) == set(REGIONS.keys())


def test_aggregate_zero_input_returns_zero():
    preds = np.zeros((10, 20484), dtype=np.float32)
    for value in aggregate(preds).values():
        assert value == 0.0


def test_aggregate_constant_input_returns_constant():
    preds = np.full((5, 20484), 3.7, dtype=np.float32)
    for value in aggregate(preds).values():
        assert value == pytest.approx(3.7)


def test_aggregate_isolates_to_visual_cortex():
    """All-ones at visual_cortex vertices, zero elsewhere -> only that region activates.

    Marketing regions are anatomically disjoint in our DK mapping, so any
    other region with non-zero output would indicate a vertex-index leak.
    """
    preds = np.zeros((1, 20484), dtype=np.float32)
    preds[0, REGION_VERTICES["visual_cortex"]] = 1.0
    out = aggregate(preds)
    assert out["visual_cortex"] == pytest.approx(1.0)
    for region, value in out.items():
        if region != "visual_cortex":
            assert value == pytest.approx(0.0), f"{region} leaked: {value}"


def test_aggregate_time_axis_averaging():
    """A region active only on the first frame averages to 1/T."""
    T = 4
    preds = np.zeros((T, 20484), dtype=np.float32)
    preds[0, REGION_VERTICES["motor"]] = 1.0
    out = aggregate(preds)
    assert out["motor"] == pytest.approx(1.0 / T)


def test_aggregate_rejects_bad_shape():
    with pytest.raises(ValueError):
        aggregate(np.zeros((10, 100), dtype=np.float32))


def test_aggregate_rejects_1d():
    with pytest.raises(ValueError):
        aggregate(np.zeros(20484, dtype=np.float32))
