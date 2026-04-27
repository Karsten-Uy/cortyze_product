"""Templated LLM client for free local development.

Returns deterministic suggestions based on which region was flagged. Lets
the suggestion pipeline ship + the frontend render cards without spending
money or signing up for an API key.

# TODO(stage 2): replace with real LLM once prompt is dialed in.
"""

from __future__ import annotations

import re
from typing import Any

# Per-region templated suggestions. Two per region so the pipeline always
# returns 2-3 items even when only one region is flagged.
_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "visual_cortex": [
        {
            "title": "Lead with motion in the first second",
            "fix": "Replace the static opening shot with a fast push-in or whip pan. The first second has 2-3x the visual-cortex impact of the rest.",
            "why": "Visual cortex activates strongest on motion onset. Static openings register as background.",
        },
        {
            "title": "Cut twice as often in the dip window",
            "fix": "Shorten clips to 0.8-1.2 seconds during the flagged window. Add a punch-in mid-clip if you can't cut.",
            "why": "Sustained shots over ~2 seconds let visual attention drop off.",
        },
    ],
    "fusiform_face": [
        {
            "title": "Add a face close-up at the dip",
            "fix": "Insert a 2-3 second human reaction shot — eye contact with camera, clearly lit. Even a stock-footage face works.",
            "why": "Fusiform Face Area activates within 200ms of seeing a clear face. Without one, this region barely registers.",
        },
        {
            "title": "Reframe to bring the face larger",
            "fix": "If a face is in frame but small, push in. The face needs to occupy ~25%+ of the frame to drive activation.",
            "why": "Fusiform response scales with face size. Distant faces don't trigger personal-connection signals.",
        },
    ],
    "amygdala": [
        {
            "title": "Add a stakes moment",
            "fix": "Insert a 1-2 second beat showing what happens if the viewer does nothing — frustration, missed opportunity, or surprise. Pair with a sound design hit.",
            "why": "Emotional impact requires felt urgency. Without stakes, the brain treats your content as decorative.",
        },
        {
            "title": "Replace narration with reaction",
            "fix": "Cut the voiceover during the dip and replace with a person reacting — a smile, a flinch, a held breath.",
            "why": "Telling viewers what to feel is less effective than showing someone feeling it.",
        },
    ],
    "prefrontal": [
        {
            "title": "Make the use-case concrete",
            "fix": "Replace abstract benefits with one specific use-case: 'Use this when X happens.' Show the moment, not the category.",
            "why": "Prefrontal evaluation needs concrete scenarios to weigh. Generic benefits don't trigger consideration.",
        },
        {
            "title": "Surface the comparison",
            "fix": "Briefly show what the viewer is doing now vs. what they'd do with your product. Side-by-side or before/after.",
            "why": "The brain can't evaluate without alternatives. Frame the choice explicitly.",
        },
    ],
    "temporal_language": [
        {
            "title": "Slow down the voiceover or add captions",
            "fix": "If voiceover is faster than ~2.5 words/second during the dip, slow to 2.0-2.2 wps OR add burned-in captions with key words highlighted.",
            "why": "Language processing falls behind when speech is too dense. Captions give the brain a second chance.",
        },
        {
            "title": "Cut a redundant beat",
            "fix": "If the dip happens during a list ('available in three colors with eco packaging'), cut to one focused beat instead.",
            "why": "Lists fragment attention. One specific claim lands; three vague ones evaporate.",
        },
    ],
    "hippocampus": [
        {
            "title": "Add a sonic logo or repeated phrase",
            "fix": "Insert a 1-2 second audio signature in the dip — a chord, a tagline repeated, or a specific sound effect tied to your brand.",
            "why": "Memory encoding strengthens with auditory + visual co-activation. A repeated cue at peak moments is what gets remembered.",
        },
        {
            "title": "Reframe with a memorable visual hook",
            "fix": "Bring back a distinctive visual element from earlier in the content — same color, same shape, same person — at the dip moment.",
            "why": "Repetition with variation cues episodic memory. Reusing a visual signature anchors recall.",
        },
    ],
    "motor": [
        {
            "title": "Show the action explicitly",
            "fix": "Insert a 1-second shot of a hand or finger doing the next step — tapping, swiping, reaching. Big and centered.",
            "why": "Motor cortex mirrors observed actions. Without a visible 'action to take,' viewers don't mentally rehearse acting.",
        },
        {
            "title": "End with a single clear CTA",
            "fix": "Strip the closing to one verb + one direction: 'Tap below.' 'Click the link.' Cut anything else.",
            "why": "Decision motor planning requires a singular target. Multiple CTAs split the impulse.",
        },
    ],
    "reward": [
        {
            "title": "Add a payoff moment",
            "fix": "Insert a clear resolution — the satisfied face, the perfect fit, the music drop — right after the dip window.",
            "why": "Reward circuitry needs a release. Build-up without payoff feels unsatisfying and is forgotten.",
        },
        {
            "title": "Match music to the emotional arc",
            "fix": "If the dip happens during static visuals + flat music, swap to a track with a clear lift and time it to a visual change.",
            "why": "Music synchrony with visual beats drives reward activation. Mismatched tracks dampen it.",
        },
    ],
}


class MockLLMClient:
    """Returns templated suggestions based on the region in the user prompt.

    Parses just enough of the user prompt to know which region was
    flagged, then picks 2-3 pre-written suggestions for that region.
    Deterministic, free, and always succeeds — exactly what frontend
    development and tests need.
    """

    def chat_json(self, *, system: str, user: str) -> Any:
        region = _extract_region(user)
        templates = _TEMPLATES.get(region, [])[:3]
        if not templates:
            return [
                {
                    "title": "Region needs attention",
                    "fix": f"Specific advice for region {region!r} requires the live LLM.",
                    "why": "Mock client only carries templates for the 8 marketing regions.",
                }
            ]
        # Add timestamp suggestions when the user prompt mentions a window.
        ts_start, ts_end = _extract_timestamp(user)
        out = []
        for t in templates:
            row = dict(t)
            if ts_start is not None and ts_end is not None:
                row["timestamp_start_s"] = ts_start
                row["timestamp_end_s"] = ts_end
            out.append(row)
        return out


def _extract_region(user_prompt: str) -> str:
    match = re.search(r"\bregion[:\s]+([a-z_]+)", user_prompt, re.IGNORECASE)
    return match.group(1).lower() if match else ""


def _extract_timestamp(user_prompt: str) -> tuple[float | None, float | None]:
    """Pull the first 'M:SS-M:SS' or '<int>-<int>s' window from the prompt."""
    m = re.search(
        r"(\d+):(\d{2})\s*[-–]\s*(\d+):(\d{2})", user_prompt
    )
    if m:
        a = int(m.group(1)) * 60 + int(m.group(2))
        b = int(m.group(3)) * 60 + int(m.group(4))
        return float(a), float(b)
    # Matches both "14-19s" and "14s-19s" so the prompt format can vary.
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*s?\s*[-–]\s*(\d+(?:\.\d+)?)\s*s\b", user_prompt
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None
