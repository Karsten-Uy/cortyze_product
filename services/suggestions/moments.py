"""Detect dip/peak windows in per-region time series and annotate with events.

The Stage 2 suggestion engine consumes these `Moment` objects to build
timestamp-anchored prompts ("fix the dip at 0:14-0:19") instead of the
generic "your amygdala is low" advice that averaged scoring produces.

Two functions:
  - `find_moments(per_region_scores)` — pure numpy, no events needed
  - `annotate_moments(moments, events)` — glue: links each Moment to the
    audio/video/word events that overlap it
"""

from __future__ import annotations

import numpy as np

from .schemas import Event, Moment

DEFAULT_DIP_THRESHOLD = 40.0
DEFAULT_PEAK_THRESHOLD = 70.0
DEFAULT_MIN_WINDOW_S = 2.0


def find_moments(
    per_region_scores: dict[str, np.ndarray],
    *,
    dip_threshold: float = DEFAULT_DIP_THRESHOLD,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    min_window_s: float = DEFAULT_MIN_WINDOW_S,
) -> list[Moment]:
    """Find consecutive timesteps where region score < dip or > peak threshold.

    Input: dict mapping region keys to (T,) arrays in [0, 100], one value
    per second. Output: list of `Moment` objects (sorted region-wise, then
    by start_s). Windows shorter than `min_window_s` are filtered out so
    one-second blips don't pollute the report.
    """
    moments: list[Moment] = []
    for region, series in per_region_scores.items():
        moments.extend(
            _find_runs(
                series, region, dip_threshold, "dip", min_window_s, below=True
            )
        )
        moments.extend(
            _find_runs(
                series, region, peak_threshold, "peak", min_window_s, below=False
            )
        )
    return moments


def _find_runs(
    series: np.ndarray,
    region: str,
    threshold: float,
    type_: str,
    min_window_s: float,
    *,
    below: bool,
) -> list[Moment]:
    n = len(series)
    if n == 0:
        return []
    mask = (series < threshold) if below else (series > threshold)
    out: list[Moment] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        # Run is [i, j); length j - i seconds (TRIBE v2 outputs at 1 Hz).
        if (j - i) >= min_window_s:
            window = series[i:j]
            out.append(
                Moment(
                    region=region,
                    type=type_,  # type: ignore[arg-type]
                    start_s=float(i),
                    end_s=float(j),
                    avg_score=float(window.mean()),
                )
            )
        i = j
    return out


def annotate_moments(
    moments: list[Moment], events: list[Event] | None
) -> list[Moment]:
    """Attach the events overlapping each Moment + a one-line context summary.

    Mutates and returns the input list. With no events (e.g. mock client
    without event synthesis), the context is "[silent]".
    """
    if not events:
        for m in moments:
            m.context = "[silent]"
        return moments

    for m in moments:
        overlapping = [
            e
            for e in events
            if e.start_s < m.end_s and e.start_s + e.duration_s > m.start_s
        ]
        m.events = overlapping
        words = [e.text for e in overlapping if e.type == "Word" and e.text]
        if words:
            spoken = " ".join(words)
            m.context = f'voiceover: "{spoken}"'
        elif any(e.type == "Audio" for e in overlapping):
            m.context = "[audio, no speech]"
        else:
            m.context = "[silent]"
    return moments


def serialize_events_dataframe(events_df) -> list[Event]:
    """Convert tribev2's events DataFrame into the JSON-friendly Event list.

    Defensive — only pulls known columns, defaults sensibly when fields
    are absent. Used by gpu_worker.handler before sending the response.
    """
    out: list[Event] = []
    if events_df is None:
        return out
    for _, row in events_df.iterrows():
        ev_type = str(row.get("type", "Unknown") or "Unknown")
        if ev_type not in ("Word", "Sentence", "Audio", "Video"):
            ev_type = "Unknown"
        text = row.get("text")
        text_str = str(text) if text and isinstance(text, str) else None
        out.append(
            Event(
                type=ev_type,  # type: ignore[arg-type]
                start_s=float(row.get("start", 0.0) or 0.0),
                duration_s=float(row.get("duration", 0.0) or 0.0),
                text=text_str,
            )
        )
    return out
