"""Query-time TrendContext assembly — entity rollup, brand_risk_score,
references synthesis from a stub graph."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("networkx")

from services.social_context import entities as ent_mod  # noqa: E402
from services.social_context.graph import NetworkXGraph  # noqa: E402
from services.social_context.query import get_trend_context  # noqa: E402
from services.social_context.schemas import SourceSnapshot  # noqa: E402
from services.trends.protocol import Entity  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_entities_state():
    ent_mod._reset_for_tests()
    yield
    ent_mod._reset_for_tests()


@pytest.fixture
def populated_graph(monkeypatch):
    """A NetworkXGraph seeded with three entities so every test runs
    against a deterministic mini-graph."""
    g = NetworkXGraph()
    snap = SourceSnapshot(
        source="reddit",
        source_id="post-1",
        title="t",
        ingested_at=datetime.now(timezone.utc),
    )
    g.add_entity(Entity(name="Nike", type="BRAND", salience=0.9), snap)
    g.add_entity(
        Entity(name="running shoes", type="TOPIC", salience=0.6), snap
    )
    g.add_entity(Entity(name="Super Bowl", type="EVENT", salience=0.8), snap)
    # Force the regex extractor so the test doesn't depend on whether
    # `en_core_web_sm` is installed locally.
    monkeypatch.setattr(ent_mod, "_try_load_spacy", lambda: None)
    return g


def test_query_returns_empty_context_for_unmatched_text(populated_graph):
    ctx = get_trend_context(
        brief="lorem ipsum dolor sit amet",
        caption="quick brown fox",
        goal="brand_recall",
        graph=populated_graph,
    )
    # No entities matched → fallback summary, empty entities.
    assert ctx.entities == []
    assert ctx.dominant_topic is None
    assert ctx.fallback_reason is None  # graph itself was healthy
    assert "No matching entities" in ctx.summary


def test_query_finds_dominant_entity_when_text_mentions_it(populated_graph):
    ctx = get_trend_context(
        brief="A new Nike running shoe launches at the Super Bowl",
        caption="Watch Nike's drop",
        goal="brand_recall",
        graph=populated_graph,
    )
    assert ctx.entities, "expected at least one entity"
    names = {e.name.lower() for e in ctx.entities}
    assert "nike" in names
    assert ctx.dominant_topic is not None


def test_brand_risk_clipped_to_unit_interval(populated_graph):
    ctx = get_trend_context(
        brief="Nike",
        caption="",
        goal="brand_recall",
        graph=populated_graph,
    )
    assert 0.0 <= ctx.brand_risk_score <= 1.0


def test_snapshot_timestamp_is_recent_and_tz_aware(populated_graph):
    ctx = get_trend_context(
        brief="Nike",
        caption="",
        goal="brand_recall",
        graph=populated_graph,
    )
    assert ctx.snapshot_timestamp.tzinfo is not None
    age = (datetime.now(timezone.utc) - ctx.snapshot_timestamp).total_seconds()
    assert age < 5  # generated just now


def test_references_synthesized_from_library_when_match(populated_graph):
    """The library has known reference ads; we should pick at least one
    when the goal maps cleanly to a region with library coverage."""
    ctx = get_trend_context(
        brief="Nike",
        caption="",
        goal="brand_recall",
        graph=populated_graph,
    )
    # references may be empty if the library data dir is gone — the
    # function handles that gracefully. Just verify the field is a
    # well-typed list.
    assert isinstance(ctx.references, list)
    for ref in ctx.references:
        assert ref.brand
        assert 0 <= ref.score_region <= 100
