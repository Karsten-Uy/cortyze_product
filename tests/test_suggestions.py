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


def test_trigger_rules_ignores_below_min_weight():
    """Engagement weights prefrontal at 0.02 — below default 0.10 floor, never fires."""
    scores = _baseline_scores(80.0)
    scores["prefrontal"] = 5.0
    assert trigger_rules(scores, Goal.ENGAGEMENT) == []


def test_trigger_rules_priority_ordering():
    """Critical sorts before important sorts before minor.

    Pins min_weight=0.05 so the "minor" tier (motor at 0.05 weight) is reachable;
    the production default is 0.10 which excludes minor entirely.
    """
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0       # 0.25 weight → critical
    scores["fusiform_face"] = 30.0  # 0.15 weight → important
    scores["motor"] = 30.0          # 0.05 weight → minor
    rules = trigger_rules(scores, Goal.ENGAGEMENT, min_weight=0.05)
    priorities = [r.priority for r in rules]
    assert priorities == ["critical", "important", "minor"]


def test_trigger_rules_minor_tier_excluded_at_default_min_weight():
    """At the default min_weight=0.10, motor (0.05 weight) does NOT fire."""
    scores = _baseline_scores(80.0)
    scores["motor"] = 30.0
    assert trigger_rules(scores, Goal.ENGAGEMENT) == []


def test_trigger_rules_high_score_short_circuits():
    """A region at-or-above the score threshold does NOT fire even with high weight."""
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 75.0  # above default threshold of 70
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

def _make_report(
    scores: dict[str, float],
    goal: Goal = Goal.ENGAGEMENT,
    content_type: str = "video",
) -> BrainReport:
    return BrainReport(
        request_id="test-uuid",
        region_scores=scores,
        overall_score=sum(scores.values()) / len(scores),
        goal=goal,
        content_type=content_type,
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


def test_diagnose_strips_moments_for_single_image_post_without_audio():
    """Phantom audio fix: caption-only posts must not pass synthetic moments to the LLM.

    The mock fixture's brain scores carry baked-in audio dips even when the
    live request has no audio. Without this filter, both Sonnet and Haiku
    were observed inventing voiceover fixes for content with no audio.
    """

    class SpyClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat_json(self, *, system, user):
            self.prompts.append(user)
            return [{"title": "t", "fix": "f", "why": "w"}]

    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    report = _make_report(scores, content_type="post")

    # Single-image post + no audio → strip moments
    spy = SpyClient()
    diagnose(report, llm=spy, image_count=1, has_audio=False)
    assert spy.prompts, "spy never received a prompt"
    assert all("Time-series dips" not in p for p in spy.prompts)
    assert all("14s" not in p and "19s" not in p for p in spy.prompts)

    # Single-image post WITH audio → moments pass through
    spy = SpyClient()
    diagnose(report, llm=spy, image_count=1, has_audio=True)
    assert any("Time-series dips" in p for p in spy.prompts)


def test_diagnose_keeps_moments_for_carousel_without_audio():
    """Carousels have a real time axis (one image per N seconds) regardless of audio."""

    class SpyClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat_json(self, *, system, user):
            self.prompts.append(user)
            return [{"title": "t", "fix": "f", "why": "w"}]

    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    report = _make_report(scores, content_type="post")

    spy = SpyClient()
    diagnose(report, llm=spy, image_count=5, has_audio=False)
    assert any("Low-scoring image positions" in p for p in spy.prompts)


def test_diagnose_keeps_moments_for_video_without_audio():
    """Videos always have a real time axis — moments pass through regardless."""

    class SpyClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def chat_json(self, *, system, user):
            self.prompts.append(user)
            return [{"title": "t", "fix": "f", "why": "w"}]

    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    report = _make_report(scores, content_type="video")

    spy = SpyClient()
    diagnose(report, llm=spy, has_audio=False)
    assert any("Time-series dips" in p for p in spy.prompts)


def test_diagnose_survives_llm_exception():
    """A flaky LLM client doesn't take the whole pipeline down."""

    class FlakyClient:
        def chat_json(self, **kwargs):
            raise RuntimeError("provider exploded")

    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    suggestions = diagnose(_make_report(scores), llm=FlakyClient())
    assert suggestions == []  # logged but didn't raise


# ---- Content-type-aware prompts ----

def test_build_system_prompt_dispatches_by_image_count():
    from services.suggestions.prompts import build_system_prompt

    video_p = build_system_prompt("video")
    single_post_p = build_system_prompt("post", image_count=1)
    carousel_p = build_system_prompt("post", image_count=5)

    # Each prompt has distinguishing language tied to its content shape.
    assert "video" in video_p.lower()
    assert "image" in single_post_p.lower() and "caption" in single_post_p.lower()
    assert "image" in carousel_p.lower() and "carousel" in carousel_p.lower()
    # Three different prompts so cache stays correct.
    assert video_p != single_post_p != carousel_p


def test_build_user_prompt_carousel_uses_image_ranges():
    from services.suggestions.prompts import build_user_prompt

    rule = TriggeredRule(
        region="amygdala", score=30.0, weight=0.25, priority="critical"
    )
    moments = [
        Moment(
            region="amygdala",
            type="dip",
            start_s=2.5,
            end_s=7.5,
            avg_score=28.0,
            context="lead image",
        )
    ]
    text = build_user_prompt(
        rule,
        goal="engagement",
        content_type="post",
        moments=moments,
        image_count=5,
        seconds_per_image=2.5,
    )
    # Carousel prompt should reference image positions, not seconds, in
    # the dip context lines specifically.
    assert "Content shape: carousel" in text
    assert "Low-scoring image positions" in text
    # Dip lines should NOT include the synthetic-second range "2s-7s".
    assert "2s" not in text and "7s" not in text
    # And SHOULD include an image-range or single-image label.
    assert "image 2" in text or "images 2-3" in text


def test_build_user_prompt_single_post_no_image_ranges():
    """A 1-image post uses the post prompt with no image-range translation."""
    from services.suggestions.prompts import build_user_prompt

    rule = TriggeredRule(
        region="fusiform_face", score=35.0, weight=0.15, priority="important"
    )
    text = build_user_prompt(
        rule,
        goal="awareness",
        content_type="post",
        image_count=1,
        seconds_per_image=2.5,
    )
    assert "Content shape: post" in text
    assert "Low-scoring image positions" not in text


def test_build_user_prompt_uses_seconds_for_video():
    from services.suggestions.prompts import build_user_prompt

    rule = TriggeredRule(
        region="amygdala", score=30.0, weight=0.25, priority="critical"
    )
    moments = [
        Moment(
            region="amygdala",
            type="dip",
            start_s=14.0,
            end_s=19.0,
            avg_score=28.0,
            context="voiceover",
        )
    ]
    text = build_user_prompt(
        rule,
        goal="engagement",
        content_type="video",
        moments=moments,
    )
    assert "14s" in text and "19s" in text
    assert "Content shape: video" in text


def test_diagnose_carousel_attaches_image_indices():
    """A multi-image post → mock LLM populates image_index_* fields."""
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    report = _make_report(scores, content_type="post")
    suggestions = diagnose(
        report,
        llm=MockLLMClient(),
        image_count=5,
        seconds_per_image=2.5,
    )
    assert len(suggestions) >= 1
    # At least one suggestion should be image-anchored, none should be
    # timestamp-anchored (carousel prompt forbids timestamps).
    image_anchored = [
        s for s in suggestions if s.image_index_start is not None
    ]
    timestamp_anchored = [
        s for s in suggestions if s.timestamp_start_s is not None
    ]
    assert len(image_anchored) >= 1
    assert timestamp_anchored == []


def test_diagnose_video_attaches_timestamps_not_image_indices():
    scores = _baseline_scores(80.0)
    scores["amygdala"] = 30.0
    report = _make_report(scores, content_type="video")
    suggestions = diagnose(report, llm=MockLLMClient())
    assert len(suggestions) >= 1
    # Video moment is at 14s-19s — at least one suggestion picks it up.
    assert any(s.timestamp_start_s is not None for s in suggestions)
    # No image_index leakage on the video path.
    assert all(s.image_index_start is None for s in suggestions)


def test_diagnose_single_image_post_uses_post_templates():
    """A 1-image post pulls from the post template table — fixes target image / caption / audio."""
    scores = _baseline_scores(80.0)
    scores["fusiform_face"] = 30.0
    report = _make_report(scores, content_type="post")
    suggestions = diagnose(report, llm=MockLLMClient(), image_count=1)
    assert len(suggestions) >= 1
    fix_text = " ".join(s.fix.lower() for s in suggestions)
    # Post fusiform-face templates explicitly mention re-shoot/re-crop or
    # tighter portrait — language a video prompt wouldn't produce.
    assert any(
        keyword in fix_text
        for keyword in ("re-shoot", "re-crop", "portrait", "head-and-shoulders")
    )
