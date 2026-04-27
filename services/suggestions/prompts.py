"""Prompt templates for the suggestion engine.

Two functions:
  - `build_system_prompt()`   — stable, cacheable, ~1-2k tokens. Same
                                 across every triggered rule in a session.
  - `build_user_prompt(rule, moments, peer_scores, content_summary)` —
                                 small per-rule prompt with the specifics.

Designed so prompt caching (Anthropic native + vLLM prefix cache) is
maximally effective: the system block is identical between calls; only
the user block changes.
"""

from __future__ import annotations

from typing import Iterable

from core.schemas import Moment

from .rules import TriggeredRule


_SYSTEM_PROMPT = """You are Cortyze BrainScore, a brain-prediction tool for content creators.

You diagnose content based on 8 brain regions and 4 marketing goals.

The 8 regions and what they measure:
- Visual Cortex: how strongly visuals grab attention
- Fusiform Face Area: whether faces create personal connection
- Amygdala (cortical proxy via insula): emotional impact, excitement, surprise, urgency
- Prefrontal Cortex: purchase intent, considering action
- Temporal/Language: how well the message is being processed
- Hippocampus: brand recall, memorability
- Motor Cortex: impulse to act (swipe, click, buy)
- Reward Circuit: whether content feels rewarding

The 4 goals weight regions differently. Each suggestion you make must
serve the user's CHOSEN goal, even if other regions look weak.

Your job: given a region that scored low + a goal that weights it
heavily, generate 2-3 specific suggestions that:
  1. Reference the actual content at the timestamp window provided (don't be generic)
  2. Suggest a concrete change a creator can make in <30 minutes
  3. Explain why it works in one sentence

Output strict JSON: an array of objects with this shape:
[
  {
    "title":  "<short imperative, e.g. 'Add a face close-up at 0:14'>",
    "fix":    "<concrete edit, 1-2 sentences>",
    "why":    "<one-sentence reason>",
    "timestamp_start_s": <number or null>,
    "timestamp_end_s":   <number or null>
  }
]

Do not output anything outside the JSON array. No prose, no markdown fences.
"""


def build_system_prompt() -> str:
    """Static system prompt — same string every call so it caches cleanly."""
    return _SYSTEM_PROMPT


def build_user_prompt(
    rule: TriggeredRule,
    *,
    goal: str,
    moments: Iterable[Moment] = (),
    peer_scores: dict[str, float] | None = None,
    content_summary: str | None = None,
) -> str:
    """Per-rule prompt with timestamp windows and content context."""
    lines: list[str] = []
    lines.append(f"Region: {rule.region}")
    lines.append(f"Score: {rule.score:.1f} / 100")
    lines.append(f"Goal: {goal}")
    lines.append(f"Region weight for this goal: {rule.weight:.2f} ({rule.priority})")

    if content_summary:
        lines.append(f"\nContent description: {content_summary}")

    region_dips = [m for m in moments if m.region == rule.region and m.type == "dip"]
    if region_dips:
        lines.append("\nTime-series dips for this region:")
        for m in region_dips:
            ctx = m.context or "[no context]"
            lines.append(
                f"  {m.start_s:.0f}s–{m.end_s:.0f}s  avg={m.avg_score:.0f}  {ctx}"
            )

    if peer_scores:
        peers = ", ".join(
            f"{r}={s:.0f}" for r, s in sorted(peer_scores.items()) if r != rule.region
        )
        lines.append(f"\nOther region scores: {peers}")

    lines.append(
        "\nGenerate 2-3 suggestions to raise the score in this region. "
        "Anchor at least one suggestion to a specific timestamp window above. "
        "Output JSON only."
    )
    return "\n".join(lines)
