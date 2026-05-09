"""Sentiment scoring for ingested social-context text.

VADER (Valence Aware Dictionary and Sentiment Reasoner) is purpose-built
for short social-media text — handles negations, intensifiers, emoji,
slang, and capitalization. Pure-Python, ~2 MB lexicon, no model load
overhead. Acceptable for in-process per-snapshot scoring at our volume.

The `sarcasm_flag` is a heuristic, not a learned classifier — Reddit's
`/s` convention plus a couple of well-known sarcasm signals. We err on
the side of false negatives: a falsely-flagged sarcasm bumps
`brand_risk_score` and would suppress legitimate suggestions, which is
worse than missing a sarcastic post.

If `vaderSentiment` isn't installed (deployment without the
`[social-context]` extras), `score_sentiment` returns a neutral
SentimentScore and the GraphRAG client's fallback behavior takes over.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .schemas import SentimentScore

_log = logging.getLogger("cortyze.social_context.sentiment")

# Cached VADER analyzer. Populated on first successful call.
_analyzer: Any | None = None
_analyzer_load_attempted: bool = False
_analyzer_load_warning_logged: bool = False

# Sarcasm cues — high-precision Reddit + general patterns. We only
# require ONE match because false negatives are cheaper than false
# positives in this pipeline.
_SARCASM_RE = re.compile(
    r"""
    (?:^|\s)/s\b              # Reddit's explicit "/s" tag
    | \byeah\s+right\b
    | \boh\s*sure\b
    | \bsuuure+\b
    | \btotally\s+(?:not|the\s+best)\b
    | 🙄                       # eye-roll emoji
    | 🤡                       # clown emoji
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _try_load_vader() -> Any | None:
    global _analyzer, _analyzer_load_attempted, _analyzer_load_warning_logged
    if _analyzer is not None:
        return _analyzer
    if _analyzer_load_attempted:
        return None
    _analyzer_load_attempted = True
    try:
        from vaderSentiment.vaderSentiment import (
            SentimentIntensityAnalyzer,
        )
    except ImportError:
        if not _analyzer_load_warning_logged:
            _log.warning(
                "vaderSentiment not installed — sentiment scores will be "
                "neutral (0.0) until `uv sync --extra social-context` runs."
            )
            _analyzer_load_warning_logged = True
        return None
    try:
        _analyzer = SentimentIntensityAnalyzer()
    except Exception:  # noqa: BLE001 — corrupt lexicon, etc.
        _log.exception("VADER analyzer failed to initialize")
        return None
    return _analyzer


def score_sentiment(text: str) -> SentimentScore:
    """Compute polarity + sarcasm flag for `text`.

    `polarity` is VADER's compound score, in [-1.0, 1.0]:
        compound > +0.05 → positive
        compound < -0.05 → negative
        else            → neutral

    Returns a SentimentScore with safe defaults (0.0 polarity, no
    sarcasm flag) if VADER is unavailable.
    """
    if not text or not text.strip():
        return SentimentScore(polarity=0.0, sarcasm_flag=False)

    analyzer = _try_load_vader()
    polarity = 0.0
    if analyzer is not None:
        try:
            scores = analyzer.polarity_scores(text)
            polarity = float(scores.get("compound", 0.0))
        except Exception:  # noqa: BLE001
            _log.exception("VADER scoring failed; defaulting to neutral")

    sarcasm = bool(_SARCASM_RE.search(text))
    return SentimentScore(polarity=polarity, sarcasm_flag=sarcasm)


def _reset_for_tests() -> None:
    """Test-only — re-loads VADER on the next call."""
    global _analyzer, _analyzer_load_attempted, _analyzer_load_warning_logged
    _analyzer = None
    _analyzer_load_attempted = False
    _analyzer_load_warning_logged = False
