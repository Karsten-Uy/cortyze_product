"""Peak-window helpers for the per-suggestion clip player.

Real TRIBE inference produces a 1 Hz per-region timeseries from which
`services.suggestions.moments.find_moments` already extracts true peak
windows. The v2 mock pipeline has no timeseries, so for the "playable
section" UI we mint a deterministic fake window keyed by the suggestion
id and region — that way re-renders don't shift the timestamp and unit
tests stay stable.
"""

from __future__ import annotations

import hashlib

from core.regions_v2 import RegionKey


def fake_peak_window(
    suggestion_id: int,
    region: RegionKey,
    *,
    duration_s: float = 30.0,
    window_s: float = 4.0,
) -> tuple[float, float]:
    """Return a deterministic `(start_s, end_s)` peak window.

    The window is `window_s` seconds long and lies fully inside
    `[0, duration_s)`. Same `(suggestion_id, region)` always yields the
    same window — important so the clip player doesn't seek to a new
    spot on every React re-render.
    """
    if window_s >= duration_s:
        return (0.0, max(window_s, duration_s))
    digest = hashlib.sha256(f"{suggestion_id}|{region}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big")
    # 0.5s granularity — feels less robotic than integer seconds while
    # staying visually "snapped" to a frame on the seek bar.
    max_start_half = int((duration_s - window_s) * 2)
    start = (bucket % (max_start_half + 1)) / 2.0
    return (round(start, 2), round(start + window_s, 2))
