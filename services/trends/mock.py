"""Templated TrendContext for free local development.

Returns a deterministic stub regardless of input. The point is to give
the synthesis layer something to consume so the rest of the pipeline
exercises end-to-end before any real GraphRAG infrastructure exists.

When the real implementation lands, this file stays — it's the
test-fixture-quality fallback that keeps unit tests hermetic.
"""

from __future__ import annotations

from .protocol import TrendContext, TrendReference


_DEFAULT_REFERENCES: list[TrendReference] = [
    TrendReference(
        brand="Aesop",
        campaign="'Othertopics' 2024",
        note="Story-arc opening, restrained product reveal",
        score_region=82,
        label_region="Memory",
        score_overall=91,
    ),
    TrendReference(
        brand="Apple",
        campaign="'Shot on iPhone' 2023",
        note="Pattern-break opening; viewer prediction violated in 0.6s",
        score_region=88,
        label_region="Emotion",
        score_overall=85,
    ),
    TrendReference(
        brand="Nike",
        campaign="'Never Done' 2024",
        note="High-contrast subject framing throughout",
        score_region=79,
        label_region="Attention",
        score_overall=83,
    ),
    TrendReference(
        brand="Dove",
        campaign="'Real Beauty' 2023",
        note="Face anchored within first second",
        score_region=91,
        label_region="Engagement",
        score_overall=88,
    ),
]


class MockTrendClient:
    """Returns a constant TrendContext. Deterministic, free, no I/O.

    Phase 2 v2 fields (entities, dominant_topic, etc.) are returned as
    safe defaults — empty/zero — so consumers that already iterate
    `references` keep working, while the GraphRAG client can fall back
    to this implementation and stamp a `fallback_reason` on top.
    """

    def fetch(
        self,
        *,
        brief: str,
        caption: str,
        goal: str,
        request_id: str | None = None,
    ) -> TrendContext:
        # We don't actually read the inputs; they're listed in the
        # signature so the protocol shape matches what a real
        # GraphRAG-backed client will need.
        del brief, caption, goal, request_id

        return TrendContext(
            summary=(
                "Recent comparable campaigns lean on story-arc structure, "
                "high-contrast framing, and a face anchored in the opening "
                "second. Sentiment around the category is neutral-positive "
                "with a mild trend toward calmer, less-saturated palettes."
            ),
            references=list(_DEFAULT_REFERENCES),
            # v2 fields — left empty for the mock. The GraphRAG client
            # will populate them; Phase 3 reads `references` either way.
            entities=[],
            dominant_topic=None,
            brand_risk_score=0.0,
            cultural_moment=None,
        )
