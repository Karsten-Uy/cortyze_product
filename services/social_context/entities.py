"""Entity extraction over arbitrary text — primary input for the
GraphRAG pipeline.

Production path uses spaCy's `en_core_web_sm` model (small enough to
ship in a Railway image, accurate enough for proper-noun extraction).
The model loads lazily on first call and is cached for the process'
lifetime.

If spaCy isn't installed (i.e. someone deploys without
`uv sync --extra social-context`) or the model isn't downloaded,
`extract_entities` returns an empty list and logs a one-shot warning.
The pipeline continues to work — the GraphRAG client just falls back
to the mock TrendContext in that case.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from services.trends.protocol import Entity

_log = logging.getLogger("cortyze.social_context.entities")

# Cached spaCy NLP instance. Populated on first successful call.
_nlp: Any | None = None
_nlp_load_attempted: bool = False
_nlp_load_warning_logged: bool = False

# Map spaCy entity labels onto our four `Entity.type` buckets. spaCy's
# fine-grained labels (ORG / PRODUCT / WORK_OF_ART / etc.) collapse to
# the four buckets the frontend / Phase 3 prompt understands.
_LABEL_MAP: dict[str, str] = {
    "ORG": "BRAND",
    "PRODUCT": "BRAND",
    "WORK_OF_ART": "BRAND",
    "PERSON": "PERSON",
    "EVENT": "EVENT",
    "GPE": "TOPIC",
    "LOC": "TOPIC",
    "NORP": "TOPIC",
    "FAC": "TOPIC",
    "LAW": "TOPIC",
}

# Regex fallback when spaCy is unavailable — picks up obvious capitalized
# multi-word names. This is intentionally low-recall: better to under-
# extract and trigger the GraphRAG fallback than to flood the graph with
# noise.
_PROPER_NOUN_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b"
)


def _try_load_spacy() -> Any | None:
    """Idempotent. Loads `en_core_web_sm` once and caches it.

    Logs the first failure at WARNING (so it surfaces in Railway logs
    once) and silently returns None on subsequent calls so a chatty
    pipeline doesn't spam the log.
    """
    global _nlp, _nlp_load_attempted, _nlp_load_warning_logged
    if _nlp is not None:
        return _nlp
    if _nlp_load_attempted:
        return None
    _nlp_load_attempted = True
    try:
        import spacy  # noqa: WPS433  (intentional lazy import)
    except ImportError:
        if not _nlp_load_warning_logged:
            _log.warning(
                "spaCy not installed — falling back to regex NER. "
                "Install via `uv sync --extra social-context` for full "
                "GraphRAG functionality."
            )
            _nlp_load_warning_logged = True
        return None
    try:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        if not _nlp_load_warning_logged:
            _log.warning(
                "spaCy installed but `en_core_web_sm` model is missing. "
                "Run `uv run python -m spacy download en_core_web_sm` "
                "to enable real NER. Falling back to regex extractor."
            )
            _nlp_load_warning_logged = True
        return None
    return _nlp


def extract_entities(text: str) -> list[Entity]:
    """Return deduplicated proper-noun entities from `text`.

    Salience is a crude per-text count normalized to [0, 1] — useful
    for ranking inside `query.get_trend_context` but not a confidence
    score for the NER itself. The fields `trend_velocity`,
    `sentiment_polarity`, `sarcasm_flag`, and `platform_peaks` are left
    at their defaults; the query layer populates them from graph edges.
    """
    if not text or not text.strip():
        return []

    nlp = _try_load_spacy()
    raw = (
        _extract_via_spacy(nlp, text)
        if nlp is not None
        else _extract_via_regex(text)
    )

    # Deduplicate on lowercased name; keep the highest-frequency
    # surface form. Cap salience at 1.0.
    by_key: dict[str, dict[str, Any]] = {}
    for name, etype in raw:
        key = name.lower().strip()
        if not key:
            continue
        slot = by_key.setdefault(key, {"name": name, "type": etype, "count": 0})
        slot["count"] += 1
        if etype == "BRAND" and slot["type"] != "BRAND":
            # Promote to BRAND if any extractor flagged it as one.
            slot["type"] = "BRAND"

    if not by_key:
        return []

    max_count = max(slot["count"] for slot in by_key.values())
    return [
        Entity(
            name=slot["name"],
            type=slot["type"],
            salience=min(slot["count"] / max_count, 1.0),
        )
        for slot in by_key.values()
    ]


def _extract_via_spacy(nlp: Any, text: str) -> list[tuple[str, str]]:
    doc = nlp(text)
    out: list[tuple[str, str]] = []
    for ent in doc.ents:
        mapped = _LABEL_MAP.get(ent.label_)
        if mapped is None:
            continue
        out.append((ent.text.strip(), mapped))
    return out


_REGEX_STOPWORDS = {
    "The",
    "An",
    "A",
    "This",
    "That",
    "These",
    "Those",
    "I",
    "We",
    "You",
    "They",
    "It",
}


def _extract_via_regex(text: str) -> list[tuple[str, str]]:
    """Cheap proper-noun extractor — only used when spaCy is unavailable.

    Treats every capitalized multi-word run as a TOPIC. We don't try to
    distinguish brands vs people without spaCy — the graph still
    benefits from co-occurrence edges even with coarse types.
    """
    out: list[tuple[str, str]] = []
    for match in _PROPER_NOUN_RE.finditer(text):
        candidate = match.group(0).strip()
        head = candidate.split(" ", 1)[0]
        if head in _REGEX_STOPWORDS:
            continue
        out.append((candidate, "TOPIC"))
    return out


def _reset_for_tests() -> None:
    """Test-only helper — re-loads spaCy on the next call.

    Pytest fixtures monkeypatch this when they want to exercise both
    the spaCy path and the regex fallback in the same run.
    """
    global _nlp, _nlp_load_attempted, _nlp_load_warning_logged
    _nlp = None
    _nlp_load_attempted = False
    _nlp_load_warning_logged = False
