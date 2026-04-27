"""Tests for the Stage 2 suggestion engine.

Covers:
  - threshold rule engine (which regions fire, in what priority order)
  - mock LLM client (region-aware templated output)
  - factory dispatch via SUGGESTION_LLM_MODE
  - end-to-end diagnose() pipeline against the mock client
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.atlas.regions import REGIONS
from core.schemas import BrainReport, Moment, Suggestion
from core.scoring.goals import Goal
from services.suggestions import diagnose, is_enabled
from services.suggestions.llm import get_llm_client
from services.suggestions.llm.mock import MockLLMClient
from services.suggestions.rules import TriggeredRule, trigger_rules


# ---- Threshold rule engine ----

def _baseline_scores(value: float = 80.0) -> dict[str, float]:
    return {region: value for region in REGIONS}


def test_trigger_rules_no_low_scores_returns_empty():
    rules = trigger_rules(_baseline_scores(80.0), Goal.ENGAGEMENT)
    assert rules == []


def test_trigger_rules_critical_for_high_weight_low_score():
    """Engagement weights amygdala 0.25 (critical). A score of 30 there should fire."""
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    rules = trigger_rules(scores, Goal.ENGAGEMENT)
    assert len(rules) == 1
    assert rules[0].region == "amygdala"
    assert rules[0].priority == "critical"


def test_trigger_rules_ignores_below_5_percent_weight():
    """Engagement weights prefrontal at 0.02 — never fires regardless of score."""
    scores = _baseline_scores(80.0)
    scores["prefrontal"] = 5.0
    assert trigger_rules(scores, Goal.ENGAGEMENT) == []


def test_trigger_rules_priority_ordering():
    """Critical regions sort before important sort before minor."""
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0       # 0.25 weight → critical
    scores["fusiform_face"] = 30.0  # 0.15 weight → important
    scores["motor"] = 30.0          # 0.05 weight → minor
    rules = trigger_rules(scores, Goal.ENGAGEMENT)
    priorities = [r.priority for r in rules]
    assert priorities == ["critical", "important", "minor"]


def test_trigger_rules_high_score_short_circuits():
    """A region scoring at threshold (50) does NOT fire even with high weight."""
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 50.0
    assert trigger_rules(scores, Goal.ENGAGEMENT) == []


def test_trigger_rules_handles_all_four_goals():
    """No goal should crash; weights for each are well-formed."""
    scores = _baseline_scores(20.0)
    for goal in Goal:
        rules = trigger_rules(scores, goal)
        assert isinstance(rules, list)


# ---- Mock LLM client ----

def test_mock_llm_returns_region_specific_templates():
    client = MockLLMClient()
    result = client.chat_json(
        system="...",
        user="Region: amygdala\nScore: 30",
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    titles = [item["title"] for item in result]
    # Amygdala templates mention emotion/stakes
    assert any(
        "stakes" in t.lower() or "emotion" in t.lower() or "reaction" in t.lower()
        for t in titles
    ), titles


def test_mock_llm_attaches_timestamps_when_window_present():
    client = MockLLMClient()
    result = client.chat_json(
        system="...",
        user="Region: motor\nDip: 0:14–0:19",
    )
    assert any(
        item.get("timestamp_start_s") == 14.0 and item.get("timestamp_end_s") == 19.0
        for item in result
    )


def test_mock_llm_unknown_region_returns_fallback():
    client = MockLLMClient()
    result = client.chat_json(
        system="...",
        user="Region: cerebellum",  # not one of our 8
    )
    assert isinstance(result, list)
    assert len(result) >= 1


# ---- Factory dispatch ----

def test_factory_default_returns_mock(monkeypatch):
    monkeypatch.delenv("SUGGESTION_LLM_MODE", raising=False)
    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_factory_mock_explicit(monkeypatch):
    monkeypatch.setenv("SUGGESTION_LLM_MODE", "mock")
    assert isinstance(get_llm_client(), MockLLMClient)


def test_factory_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("SUGGESTION_LLM_MODE", "imaginary")
    with pytest.raises(ValueError):
        get_llm_client()


def test_is_enabled_default_false(monkeypatch):
    monkeypatch.delenv("ENABLE_SUGGESTIONS", raising=False)
    assert is_enabled() is False


def test_is_enabled_true_variants(monkeypatch):
    for v in ("true", "True", "1", "yes"):
        monkeypatch.setenv("ENABLE_SUGGESTIONS", v)
        assert is_enabled() is True


# ---- diagnose() end-to-end against mock ----

def _make_report(scores: dict[str, float], goal: Goal = Goal.ENGAGEMENT) -> BrainReport:
    return BrainReport(
        request_id="test-uuid",
        region_scores=scores,
        overall_score=sum(scores.values()) / len(scores),
        goal=goal,
        model_version="mock-test",
        elapsed_ms=0,
        moments=[
            Moment(
                region="amygdala",
                type="dip",
                start_s=14.0,
                end_s=19.0,
                avg_score=28.0,
                context='voiceover: "available three colors"',
            )
        ],
    )


def test_diagnose_returns_suggestions_for_weak_regions():
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    suggestions = diagnose(_make_report(scores), llm=MockLLMClient())
    assert len(suggestions) >= 1
    assert all(isinstance(s, Suggestion) for s in suggestions)
    assert all(s.region == "amygdala" for s in suggestions)
    assert all(s.priority == "critical" for s in suggestions)


def test_diagnose_returns_empty_when_no_regions_weak():
    scores = _baseline_scores(80.0)
    assert diagnose(_make_report(scores), llm=MockLLMClient()) == []


def test_diagnose_attaches_timestamps_from_moments():
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    suggestions = diagnose(_make_report(scores), llm=MockLLMClient())
    # Mock client extracts timestamps from the user prompt; our prompt includes
    # the moment "14s–19s" line, which the mock parses
    assert any(
        s.timestamp_start_s is not None and s.timestamp_end_s is not None
        for s in suggestions
    )


def test_diagnose_survives_llm_exception():
    """A flaky LLM client doesn't take the whole pipeline down."""

    class FlakyClient:
        def chat_json(self, **kwargs):
            raise RuntimeError("provider exploded")

    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    suggestions = diagnose(_make_report(scores), llm=FlakyClient())
    assert suggestions == []  # logged but didn't raise
