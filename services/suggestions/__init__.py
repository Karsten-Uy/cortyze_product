"""Stage 2 suggestion engine — orchestrator.

`diagnose(report)` is the single public entry point: rules → prompt →
LLM → parsed Suggestion list. Designed to be wired into
`api/predict.py` once the user is ready to flip suggestions on (gated by
its own `ENABLE_SUGGESTIONS` env var to keep `/analyze` cheap during dev).

Today's flow:
  trigger_rules(region_scores, goal)  → list[TriggeredRule]
  for each rule: build_user_prompt(...) → llm.chat_json(...) → Suggestion list
  return aggregated suggestions
"""

from __future__ import annotations

import logging
import os

from core.schemas import BrainReport, Suggestion

from .llm import LLMClient, get_llm_client
from .prompts import build_system_prompt, build_user_prompt
from .rules import TriggeredRule, trigger_rules

_log = logging.getLogger(__name__)


def is_enabled() -> bool:
    """Suggestions are opt-in via ENABLE_SUGGESTIONS=true."""
    return os.environ.get("ENABLE_SUGGESTIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def diagnose(
    report: BrainReport,
    *,
    llm: LLMClient | None = None,
    content_summary: str | None = None,
    image_count: int = 0,
    seconds_per_image: float = 2.5,
    has_audio: bool = True,
    additional_context: str | None = None,
) -> list[Suggestion]:
    """Generate Suggestion objects for every triggered rule in the report.

    The system + user prompts adapt to the effective content shape:
      - video → timestamp-anchored cut/reframe/voiceover suggestions
      - single-image post → image / caption / audio lever suggestions
      - multi-image post (carousel) → reorder / swap / drop / add image
        suggestions, dip windows translated into image-range form

    `image_count` decides between the single-image and carousel post
    prompts; `seconds_per_image` is only consulted for carousels so dip
    windows can be translated into image-range form before being handed
    to the LLM.

    `has_audio` controls whether time-series moments are passed to the LLM.
    For single-image posts without audio, the time axis is synthetic (the
    image is held for 2.5s as a V-JEPA chunking convenience, not a real
    moment) and any "audio dip 0:00-0:05" framing the LLM picks up will be
    a hallucination — we observed both Sonnet and Haiku inventing voiceover
    fixes for content with no audio. Strip moments in that case so the LLM
    has no timestamps to anchor on. For videos and carousels and any post
    that does have audio, moments stay (the timeline is real).
    """
    rules = trigger_rules(report.region_scores, report.goal)
    if not rules:
        return []

    client = llm or get_llm_client()
    system_prompt = build_system_prompt(
        report.content_type, image_count=image_count
    )

    is_single_image_post = report.content_type == "post" and image_count <= 1
    moments_for_prompt = (
        () if is_single_image_post and not has_audio else report.moments
    )
    if is_single_image_post and not has_audio and report.moments:
        _log.info(
            "request_id=%s stripping %d moments from prompt "
            "(single-image post with no audio — time axis is synthetic)",
            report.request_id,
            len(report.moments),
        )

    suggestions: list[Suggestion] = []
    for rule in rules:
        try:
            raw = client.chat_json(
                system=system_prompt,
                user=build_user_prompt(
                    rule,
                    goal=report.goal.value,
                    content_type=report.content_type,
                    moments=moments_for_prompt,
                    peer_scores=report.region_scores,
                    content_summary=content_summary,
                    image_count=image_count,
                    seconds_per_image=seconds_per_image,
                    additional_context=additional_context,
                ),
            )
        except Exception as e:
            _log.warning(
                "request_id=%s suggestion failed for region=%s: %s",
                report.request_id,
                rule.region,
                e,
            )
            continue
        suggestions.extend(
            _to_suggestions(
                rule,
                raw,
                goal=report.goal,
                # Aggregate user text the relevance ranker should see:
                # caption + brand context. Helps surface topically
                # relevant reference ads when the library is large.
                user_text=" ".join(
                    [report.caption_text or "", additional_context or ""]
                ).strip()
                or None,
                user_content_type=report.content_type,
            )
        )
    return suggestions


def _example_names_for_rule(
    region: str,
    *,
    goal,
    user_text: str | None,
    user_content_type: str | None,
    n: int = 2,
) -> list[str]:
    """Pick the top-N reference-ad names for a given region + goal +
    content context.

    Best-effort: if the examples library isn't loadable (no data dir,
    JSON parse failure, etc.) we return [] rather than breaking
    suggestions. The frontend silently shows "No reference example
    registered" in that case.
    """
    try:
        from services.examples.library import best_examples

        return [
            ad["name"]
            for ad in best_examples(
                region=region,
                goal=goal,
                user_text=user_text,
                user_content_type=user_content_type,
                n=n,
            )
        ]
    except Exception:
        return []


def _to_suggestions(
    rule: TriggeredRule,
    raw,
    *,
    goal=None,
    user_text: str | None = None,
    user_content_type: str | None = None,
) -> list[Suggestion]:
    """Coerce an LLM JSON response into validated Suggestion objects.

    Each suggestion gets its `examples` field pre-populated with the
    top reference ads for the rule's region, ranked by region score ×
    goal relevance × user-content match. The frontend's SuggestionCard
    fetches the full ad details lazily on expand.
    """
    items = raw if isinstance(raw, list) else raw.get("suggestions", [])
    if goal is not None:
        example_names = _example_names_for_rule(
            rule.region,
            goal=goal,
            user_text=user_text,
            user_content_type=user_content_type,
        )
    else:
        # Backward compat for any callers not passing goal — fall back
        # to pure region-score ranking via the legacy helper.
        from services.examples.library import top_n_for_region

        try:
            example_names = [a["name"] for a in top_n_for_region(rule.region, n=2)]
        except Exception:
            example_names = []
    out: list[Suggestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                Suggestion(
                    region=rule.region,
                    priority=rule.priority,
                    title=str(item["title"]),
                    fix=str(item["fix"]),
                    why=str(item.get("why", "")),
                    timestamp_start_s=_maybe_float(item.get("timestamp_start_s")),
                    timestamp_end_s=_maybe_float(item.get("timestamp_end_s")),
                    image_index_start=_maybe_int(item.get("image_index_start")),
                    image_index_end=_maybe_int(item.get("image_index_end")),
                    examples=example_names,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["diagnose", "is_enabled", "Suggestion", "TriggeredRule"]
