"""GraphRAGTrendClient — fallback semantics + factory wiring."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("networkx")

from services.social_context import client as client_mod  # noqa: E402
from services.social_context import entities as ent_mod  # noqa: E402
from services.social_context.client import GraphRAGTrendClient  # noqa: E402
from services.social_context.graph import NetworkXGraph  # noqa: E402
from services.social_context.schemas import SourceSnapshot  # noqa: E402
from services.trends.protocol import Entity  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    ent_mod._reset_for_tests()
    yield
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    ent_mod._reset_for_tests()


def test_empty_graph_triggers_mock_fallback():
    """With no ingest yet, the client should hand back the mock context
    stamped with `fallback_reason='empty_graph'`."""
    graph = NetworkXGraph()
    c = GraphRAGTrendClient(graph=graph)
    ctx = c.fetch(
        brief="some brief",
        caption="some caption",
        goal="brand_recall",
        request_id="r-test",
    )
    assert ctx.fallback_reason == "empty_graph"
    # Mock content surfaces — Aesop is the first reference in the mock.
    assert any(r.brand == "Aesop" for r in ctx.references)


def test_stale_graph_triggers_fallback(monkeypatch):
    """An ingest older than `GRAPH_STALENESS_HOURS` should fall back."""
    graph = NetworkXGraph()
    snap = SourceSnapshot(
        source="reddit",
        source_id="old",
        ingested_at=datetime.now(timezone.utc) - timedelta(hours=72),
    )
    graph.add_entity(Entity(name="Nike", type="BRAND"), snap)

    c = GraphRAGTrendClient(graph=graph)
    ctx = c.fetch(
        brief="Nike",
        caption="",
        goal="brand_recall",
    )
    assert ctx.fallback_reason == "stale_graph"


def test_populated_graph_runs_real_path(monkeypatch):
    monkeypatch.setattr(ent_mod, "_try_load_spacy", lambda: None)
    graph = NetworkXGraph()
    graph.add_entity(
        Entity(name="Nike", type="BRAND", salience=0.9),
        SourceSnapshot(
            source="reddit",
            source_id="fresh",
            ingested_at=datetime.now(timezone.utc),
        ),
    )
    c = GraphRAGTrendClient(graph=graph)
    ctx = c.fetch(
        brief="Nike just launched a new ad",
        caption="",
        goal="brand_recall",
    )
    assert ctx.fallback_reason is None
    assert ctx.entities  # real path populated entities


def test_metrics_capture_fallback_reason():
    graph = NetworkXGraph()  # empty → triggers fallback
    c = GraphRAGTrendClient(graph=graph)
    c.fetch(brief="x", caption="y", goal="brand_recall")
    metrics = client_mod.get_metrics()
    assert metrics["fetch_total"] == 1
    assert metrics["fetch_fallback_total"] == 1
    by_reason: dict[str, int] = metrics["fetch_fallback_by_reason"]  # type: ignore[assignment]
    assert by_reason.get("empty_graph", 0) == 1


def test_factory_dispatches_to_graphrag_client(monkeypatch):
    """`get_client` returns a `GraphRAGTrendClient` for `TRENDS_MODE=graphrag`."""
    from services.trends import get_client

    monkeypatch.setenv("TRENDS_MODE", "graphrag")
    c = get_client()
    assert c.__class__.__name__ == "GraphRAGTrendClient"


def test_factory_unknown_mode_falls_back_to_mock(monkeypatch):
    from services.trends import get_client

    monkeypatch.setenv("TRENDS_MODE", "weirdo")
    c = get_client()
    assert c.__class__.__name__ == "MockTrendClient"


def test_neo4j_backend_requires_credentials(monkeypatch):
    """GRAPH_BACKEND=neo4j without NEO4J_URI/PASSWORD should raise loudly
    rather than silently downgrade to NetworkX."""
    monkeypatch.setenv("GRAPH_BACKEND", "neo4j")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    client_mod._reset_for_tests()
    # The error is either RuntimeError (missing env vars) or — if the
    # driver isn't installed — RuntimeError with the install hint.
    with pytest.raises(RuntimeError):
        client_mod.get_graph()
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    client_mod._reset_for_tests()


def test_unknown_backend_raises():
    os.environ["GRAPH_BACKEND"] = "redis"
    client_mod._reset_for_tests()
    try:
        with pytest.raises(RuntimeError):
            client_mod.get_graph()
    finally:
        del os.environ["GRAPH_BACKEND"]
        client_mod._reset_for_tests()
