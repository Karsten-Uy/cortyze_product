"""Unit tests for the TrendContextV2 schema evolution (Phase 2 PR #1).

These verify that the new GraphRAG-shaped fields (entities, dominant_topic,
brand_risk_score, cultural_moment, snapshot_timestamp, fallback_reason)
ride alongside the legacy `summary` + `references` view without breaking
any existing consumer. The mock client is the canonical "safe defaults"
producer until the real GraphRAG implementation lands.
"""

from __future__ import annotations

from datetime import datetime, timezone

from services.trends.mock import MockTrendClient
from services.trends.protocol import Entity, TrendContext, TrendReference


def test_trend_context_legacy_fields_round_trip():
    ctx = TrendContext(
        summary="hi",
        references=[
            TrendReference(
                brand="Test",
                campaign="X",
                note="n",
                score_region=80,
                label_region="Memory",
                score_overall=85,
            )
        ],
    )
    assert ctx.summary == "hi"
    assert len(ctx.references) == 1
    # New v2 fields default to safe empty/zero values.
    assert ctx.entities == []
    assert ctx.dominant_topic is None
    assert ctx.brand_risk_score == 0.0
    assert ctx.cultural_moment is None
    assert ctx.fallback_reason is None
    # snapshot_timestamp defaults to now(UTC); confirm it's tz-aware.
    assert ctx.snapshot_timestamp.tzinfo is not None


def test_trend_context_accepts_v2_fields():
    ent = Entity(
        name="Nike",
        type="BRAND",
        salience=0.8,
        trend_velocity=0.62,
        sentiment_polarity=0.4,
        sarcasm_flag=False,
        platform_peaks={"reddit": 0.71, "news": 0.4},
    )
    ctx = TrendContext(
        summary="x",
        entities=[ent],
        dominant_topic="running shoes",
        brand_risk_score=0.15,
        cultural_moment="post-Super Bowl",
        snapshot_timestamp=datetime(2026, 5, 8, tzinfo=timezone.utc),
        fallback_reason=None,
    )
    assert ctx.entities[0].name == "Nike"
    assert ctx.dominant_topic == "running shoes"
    assert 0.0 <= ctx.brand_risk_score <= 1.0


def test_mock_client_returns_v2_shape_with_safe_defaults():
    client = MockTrendClient()
    ctx = client.fetch(brief="b", caption="c", goal="brand_recall")
    # Legacy view still populated for the synthesis layer.
    assert ctx.summary
    assert len(ctx.references) == 4
    assert ctx.references[0].brand == "Aesop"
    # v2 fields present but empty so consumers can pivot freely.
    assert ctx.entities == []
    assert ctx.dominant_topic is None
    assert ctx.brand_risk_score == 0.0
    assert ctx.fallback_reason is None
    # Timestamp set to a recent UTC moment.
    assert ctx.snapshot_timestamp.tzinfo is not None


def test_mock_client_accepts_request_id():
    """Protocol now allows a `request_id` kwarg for log correlation."""
    client = MockTrendClient()
    ctx = client.fetch(
        brief="b", caption="c", goal="brand_recall", request_id="r-test"
    )
    assert ctx.summary  # mock ignores request_id; just shouldn't error
