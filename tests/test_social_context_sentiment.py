"""Sentiment scoring — VADER polarity + sarcasm flag heuristic."""

from __future__ import annotations

import pytest

pytest.importorskip("vaderSentiment")

from services.social_context import sentiment as snt
from services.social_context.sentiment import score_sentiment


@pytest.fixture(autouse=True)
def _reset_sentiment_state():
    snt._reset_for_tests()
    yield
    snt._reset_for_tests()


def test_blank_text_returns_neutral():
    s = score_sentiment("")
    assert s.polarity == 0.0
    assert s.sarcasm_flag is False


def test_clearly_positive_text_scores_above_zero():
    s = score_sentiment("This is absolutely fantastic, I love it!")
    assert s.polarity > 0.5


def test_clearly_negative_text_scores_below_zero():
    s = score_sentiment("This is the worst, most disappointing ad ever.")
    assert s.polarity < -0.5


def test_sarcasm_tag_in_reddit_post():
    s = score_sentiment("Wow, what a brilliant ad campaign /s")
    assert s.sarcasm_flag is True


def test_sarcasm_eye_roll_emoji():
    s = score_sentiment("Sure, that totally worked 🙄")
    assert s.sarcasm_flag is True


def test_sarcasm_phrase_yeah_right():
    s = score_sentiment("Yeah right, like anyone believes that.")
    assert s.sarcasm_flag is True


def test_no_sarcasm_in_plain_positive():
    s = score_sentiment("That was a great commercial!")
    assert s.sarcasm_flag is False


def test_polarity_falls_back_to_neutral_when_vader_missing(monkeypatch):
    monkeypatch.setattr(snt, "_try_load_vader", lambda: None)
    s = score_sentiment("This is amazing!")
    assert s.polarity == 0.0
    # Sarcasm regex still runs even without VADER.
    assert s.sarcasm_flag is False
