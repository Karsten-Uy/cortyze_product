"""Tests for services.suggestions.moments — dip/peak detection + annotation."""

import numpy as np

from services.suggestions.moments import (
    DEFAULT_DIP_THRESHOLD,
    DEFAULT_MIN_WINDOW_S,
    DEFAULT_PEAK_THRESHOLD,
    annotate_moments,
    find_moments,
)
from services.suggestions.schemas import Event, Moment


def test_find_moments_no_signal_returns_empty():
    """A flat 50 signal triggers neither dip (<40) nor peak (>70)."""
    series = {"amygdala": np.full(20, 50.0, dtype=np.float32)}
    assert find_moments(series) == []


def test_find_moments_detects_a_dip():
    series = {"amygdala": np.array([60, 60, 30, 25, 28, 60, 60], dtype=np.float32)}
    moments = find_moments(series, min_window_s=2)
    assert len(moments) == 1
    m = moments[0]
    assert m.region == "amygdala"
    assert m.type == "dip"
    assert m.start_s == 2.0
    assert m.end_s == 5.0
    assert 25 <= m.avg_score <= 30


def test_find_moments_detects_a_peak():
    series = {"motor": np.array([50, 75, 80, 90, 50], dtype=np.float32)}
    moments = find_moments(series, min_window_s=2)
    assert any(m.type == "peak" and m.region == "motor" for m in moments)


def test_find_moments_filters_short_windows():
    """A 1-second dip is below default min_window_s=2 and gets filtered."""
    series = {"reward": np.array([60, 30, 60, 60], dtype=np.float32)}
    assert find_moments(series, min_window_s=2) == []


def test_find_moments_keeps_short_windows_when_threshold_lowered():
    series = {"reward": np.array([60, 30, 60, 60], dtype=np.float32)}
    moments = find_moments(series, min_window_s=1)
    assert len(moments) == 1
    assert moments[0].type == "dip"


def test_find_moments_handles_multiple_dips_per_region():
    series = {
        "amygdala": np.array(
            [60, 30, 25, 60, 60, 30, 28, 35, 60], dtype=np.float32
        )
    }
    dips = [m for m in find_moments(series, min_window_s=2) if m.type == "dip"]
    assert len(dips) == 2
    assert dips[0].start_s == 1.0 and dips[0].end_s == 3.0
    assert dips[1].start_s == 5.0 and dips[1].end_s == 8.0


def test_find_moments_thresholds_are_strict_inequalities():
    """A value exactly at threshold is NOT a dip."""
    series = {
        "amygdala": np.array([60, DEFAULT_DIP_THRESHOLD, 60], dtype=np.float32)
    }
    assert find_moments(series, min_window_s=1) == []


def test_annotate_moments_with_no_events_marks_silent():
    moments = [
        Moment(region="amygdala", type="dip", start_s=2.0, end_s=5.0, avg_score=28.0)
    ]
    annotate_moments(moments, None)
    assert moments[0].context == "[silent]"
    assert moments[0].events == []


def test_annotate_moments_attaches_overlapping_words():
    moments = [
        Moment(region="amygdala", type="dip", start_s=10.0, end_s=15.0, avg_score=30.0),
    ]
    events = [
        Event(type="Word", start_s=2.0, duration_s=0.4, text="early"),
        Event(type="Word", start_s=11.0, duration_s=0.5, text="available"),
        Event(type="Word", start_s=13.5, duration_s=0.3, text="now"),
        Event(type="Word", start_s=20.0, duration_s=0.4, text="late"),
        Event(type="Audio", start_s=0.0, duration_s=30.0),
    ]
    annotate_moments(moments, events)
    m = moments[0]
    assert "available" in m.context and "now" in m.context
    assert "early" not in m.context and "late" not in m.context
    word_events = [e for e in m.events if e.type == "Word"]
    assert len(word_events) == 2


def test_annotate_moments_audio_only_window_marked_no_speech():
    moments = [
        Moment(region="reward", type="dip", start_s=5.0, end_s=10.0, avg_score=30.0)
    ]
    events = [Event(type="Audio", start_s=0.0, duration_s=30.0)]
    annotate_moments(moments, events)
    assert moments[0].context == "[audio, no speech]"


def test_annotate_moments_silent_window_marked_silent():
    moments = [
        Moment(region="reward", type="dip", start_s=5.0, end_s=10.0, avg_score=30.0)
    ]
    annotate_moments(moments, [])
    assert moments[0].context == "[silent]"


def test_default_thresholds_make_sense():
    assert DEFAULT_DIP_THRESHOLD < 50 < DEFAULT_PEAK_THRESHOLD
    assert DEFAULT_MIN_WINDOW_S >= 1
