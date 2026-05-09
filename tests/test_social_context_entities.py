"""Entity extraction: spaCy primary path + regex fallback.

The spaCy-with-model path is exercised when `en_core_web_sm` is
available locally; otherwise that test is skipped. The regex fallback
is always exercised by monkeypatching `_try_load_spacy` to return None.
"""

from __future__ import annotations

import pytest

from services.social_context import entities as ents
from services.social_context.entities import extract_entities


@pytest.fixture(autouse=True)
def _reset_entities_state():
    ents._reset_for_tests()
    yield
    ents._reset_for_tests()


def _has_spacy_model() -> bool:
    try:
        import spacy
    except ImportError:
        return False
    try:
        spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        return True
    except OSError:
        return False


def test_extract_returns_empty_for_blank_text():
    assert extract_entities("") == []
    assert extract_entities("   ") == []


def test_regex_fallback_extracts_capitalized_proper_nouns(monkeypatch):
    """Force the regex path even when spaCy is installed."""
    monkeypatch.setattr(ents, "_try_load_spacy", lambda: None)
    out = extract_entities(
        "Nike just dropped a campaign with Serena Williams during the Super Bowl."
    )
    names = {e.name for e in out}
    assert "Nike" in names
    # Multi-word proper nouns are preserved.
    assert any("Serena" in n for n in names)
    assert any("Super Bowl" in n for n in names)


def test_regex_fallback_skips_sentence_initial_articles(monkeypatch):
    monkeypatch.setattr(ents, "_try_load_spacy", lambda: None)
    out = extract_entities("The Quick Brown Fox.")
    # "The" is a stopword — the run should drop it from extraction.
    assert all(not e.name.startswith("The ") for e in out)


def test_dedup_collapses_repeated_mentions(monkeypatch):
    monkeypatch.setattr(ents, "_try_load_spacy", lambda: None)
    out = extract_entities(
        "Nike. Nike. Nike. And once: Adidas."
    )
    by_name = {e.name.lower(): e for e in out}
    assert "nike" in by_name
    assert "adidas" in by_name
    # Salience: Nike gets the higher score (more mentions).
    assert by_name["nike"].salience >= by_name["adidas"].salience


@pytest.mark.skipif(not _has_spacy_model(), reason="en_core_web_sm not installed")
def test_spacy_path_categorizes_orgs_as_brands():
    out = extract_entities(
        "Nike unveiled a new running shoe at the Paris Olympics."
    )
    assert out, "expected at least one entity"
    nike = next((e for e in out if e.name.lower() == "nike"), None)
    assert nike is not None
    assert nike.type == "BRAND"
