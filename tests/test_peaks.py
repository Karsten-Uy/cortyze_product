"""Unit tests for the mock peak-window helper used by the v2 mock
synthesis path. Real peak detection (from per-region timeseries) is
covered separately by `tests/test_moments.py`."""

from __future__ import annotations

from services.synthesis.peaks import fake_peak_window


def test_peak_window_is_within_clip_duration():
    start, end = fake_peak_window(1, "memory", duration_s=30.0, window_s=4.0)
    assert 0.0 <= start
    assert end <= 30.0
    assert end > start


def test_peak_window_length_matches_request():
    start, end = fake_peak_window(7, "emotion", duration_s=30.0, window_s=4.0)
    assert end - start == 4.0


def test_peak_window_is_deterministic():
    a = fake_peak_window(3, "attention")
    b = fake_peak_window(3, "attention")
    assert a == b


def test_peak_window_varies_across_inputs():
    seen = {
        fake_peak_window(i, region)
        for i in range(1, 8)
        for region in ("memory", "emotion", "attention", "language", "face", "reward")
    }
    # 42 (id, region) pairs should land on more than a handful of
    # distinct windows — otherwise the hash isn't doing its job.
    assert len(seen) >= 12


def test_peak_window_handles_short_clip():
    # Window equals duration: degenerate, but should not crash and
    # must still return a span starting at zero.
    start, end = fake_peak_window(1, "memory", duration_s=4.0, window_s=4.0)
    assert start == 0.0
    assert end >= 4.0
