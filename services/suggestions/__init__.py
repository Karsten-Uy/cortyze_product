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
) -> list[Suggestion]:
    """Generate Suggestion objects for every triggered rule in the report."""
    rules = trigger_rules(report.region_scores, report.goal)
    if not rules:
        return []

    client = llm or get_llm_client()
    system_prompt = build_system_prompt()

    suggestions: list[Suggestion] = []
    for rule in rules:
        try:
            raw = client.chat_json(
                system=system_prompt,
                user=build_user_prompt(
                    rule,
                    goal=report.goal.value,
                    moments=report.moments,
                    peer_scores=report.region_scores,
                    content_summary=content_summary,
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
        suggestions.extend(_to_suggestions(rule, raw))
    return suggestions


def _to_suggestions(rule: TriggeredRule, raw) -> list[Suggestion]:
    """Coerce an LLM JSON response into validated Suggestion objects."""
    items = raw if isinstance(raw, list) else raw.get("suggestions", [])
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


__all__ = ["diagnose", "is_enabled", "Suggestion", "TriggeredRule"]
