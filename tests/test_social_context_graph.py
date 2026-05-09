"""NetworkXGraph backend — add/query/prune semantics.

Skipped automatically when networkx isn't installed (i.e. someone runs
the suite without `uv sync --extra social-context`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("networkx")

from services.social_context.graph import NetworkXGraph  # noqa: E402
from services.social_context.schemas import (  # noqa: E402
    Entity,
    EntityEdge,
    SourceSnapshot,
)


def _snap(source: str = "reddit", source_id: str = "abc") -> SourceSnapshot:
    return SourceSnapshot(
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title="t",
        body="b",
    )


def _ent(name: str, etype: str = "BRAND", salience: float = 0.5) -> Entity:
    return Entity(name=name, type=etype, salience=salience)  # type: ignore[arg-type]


def test_add_entity_creates_node_and_records_last_seen():
    g = NetworkXGraph()
    snap = _snap()
    g.add_entity(_ent("Nike"), snap)

    assert g.node_count() == 1
    assert g.last_ingest_at() == snap.ingested_at


def test_repeated_entity_increments_mention_count_and_platform_counts():
    g = NetworkXGraph()
    snap1 = _snap("reddit", "p1")
    snap2 = _snap("news", "n1")
    g.add_entity(_ent("Nike"), snap1)
    g.add_entity(_ent("Nike"), snap1)  # same source, same id — same node
    g.add_entity(_ent("Nike"), snap2)  # different source

    # Still one node — entity ids collapse on lowercased name.
    assert g.node_count() == 1
    matches = g.query_entities_for_text("nike")
    assert len(matches) == 1
    peaks = matches[0].platform_peaks
    # 2 reddit + 1 news = 3 total; reddit ratio 2/3.
    assert peaks["reddit"] > peaks["news"]


def test_query_entities_for_text_substring_match():
    g = NetworkXGraph()
    g.add_entity(_ent("Nike"), _snap())
    g.add_entity(_ent("Adidas"), _snap(source_id="other"))

    nike_only = g.query_entities_for_text("a new nike campaign")
    assert any(e.name.lower() == "nike" for e in nike_only)
    # Adidas shouldn't match a needle of "nike".
    assert all(e.name.lower() != "adidas" for e in nike_only)


def test_query_returns_empty_for_blank_or_short_text():
    g = NetworkXGraph()
    g.add_entity(_ent("Nike"), _snap())
    assert g.query_entities_for_text("") == []
    assert g.query_entities_for_text("a b c") == []  # no needle ≥ 3 chars


def test_neighbors_walks_edges_in_both_directions():
    g = NetworkXGraph()
    g.add_entity(_ent("Nike"), _snap())
    g.add_entity(_ent("running shoes", "TOPIC"), _snap(source_id="b"))
    g.add_entity(_ent("marathon", "TOPIC"), _snap(source_id="c"))
    g.add_edge(
        EntityEdge(
            src="Nike",
            dst="running shoes",
            kind="CO_OCCURS_WITH",
            weight=1.0,
        )
    )
    g.add_edge(
        EntityEdge(
            src="running shoes",
            dst="marathon",
            kind="CO_OCCURS_WITH",
            weight=1.0,
        )
    )
    direct = {e.name.lower() for e in g.neighbors("Nike", depth=1)}
    assert "running shoes" in direct
    assert "marathon" not in direct  # depth=1 cap
    twohop = {e.name.lower() for e in g.neighbors("Nike", depth=2)}
    assert "marathon" in twohop


def test_prune_removes_stale_nodes_and_keeps_fresh():
    g = NetworkXGraph()
    old = SourceSnapshot(
        source="reddit",
        source_id="old",
        ingested_at=datetime.now(timezone.utc) - timedelta(hours=72),
    )
    fresh = SourceSnapshot(
        source="reddit",
        source_id="fresh",
        ingested_at=datetime.now(timezone.utc),
    )
    g.add_entity(_ent("Stale"), old)
    g.add_entity(_ent("Fresh"), fresh)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    removed = g.prune_older_than(cutoff)
    assert removed == 1
    assert g.node_count() == 1


def test_healthcheck_succeeds_for_empty_graph():
    g = NetworkXGraph()
    assert g.healthcheck() is True
    assert g.last_ingest_at() is None
