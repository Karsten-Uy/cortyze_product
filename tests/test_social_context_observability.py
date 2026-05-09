"""PR #6 — Google Trends scraper + p95 latency tracking + health polish."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("networkx")

from api.main import app  # noqa: E402
from services.social_context import client as client_mod  # noqa: E402
from services.social_context import entities as ent_mod  # noqa: E402
from services.social_context import scraper as scraper_mod  # noqa: E402
from services.social_context.client import GraphRAGTrendClient  # noqa: E402
from services.social_context.graph import NetworkXGraph  # noqa: E402
from services.social_context.schemas import SourceSnapshot  # noqa: E402
from services.trends.protocol import Entity  # noqa: E402

_test_client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    ent_mod._reset_for_tests()
    monkeypatch.setattr(ent_mod, "_try_load_spacy", lambda: None)
    yield
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    ent_mod._reset_for_tests()


# -------------------------------------------------------------- Trends scraper


def test_trends_scraper_disabled_when_pytrends_missing(monkeypatch):
    if importlib.util.find_spec("pytrends") is not None:
        # pytrends is installed locally — exercise the explicit-disable env var.
        monkeypatch.setenv("TRENDS_SCRAPER_ENABLED", "false")
    sc = scraper_mod.GoogleTrendsScraper()
    assert sc.enabled is False
    assert sc.fetch() == []


def test_trends_scraper_in_all_scrapers_list():
    names = [s.source for s in scraper_mod.all_scrapers()]
    assert "trends" in names


# ------------------------------------------------------------ p95 latency


def test_percentile_helper_handles_empty_and_extremes():
    assert client_mod._percentile([], 0.5) == 0.0
    assert client_mod._percentile([5.0], 0.95) == 5.0
    assert client_mod._percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert client_mod._percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_metrics_record_latency_window():
    graph = NetworkXGraph()
    graph.add_entity(
        Entity(name="Nike", type="BRAND", salience=0.9),
        SourceSnapshot(
            source="reddit",
            source_id="x",
            ingested_at=datetime.now(timezone.utc),
        ),
    )
    c = GraphRAGTrendClient(graph=graph)
    for _ in range(5):
        c.fetch(brief="Nike", caption="", goal="brand_recall")
    metrics = client_mod.get_metrics()
    assert metrics["fetch_total"] == 5
    # All fields present and non-negative.
    for k in (
        "fetch_latency_p50_ms",
        "fetch_latency_p95_ms",
        "fetch_latency_p99_ms",
        "fetch_latency_max_ms",
    ):
        assert k in metrics
        assert float(metrics[k]) >= 0.0


# ------------------------------------------------------------- /health


def test_health_endpoint_exposes_latency_fields():
    body = _test_client.get("/health/social_context").json()
    for k in (
        "fetch_latency_p50_ms",
        "fetch_latency_p95_ms",
        "fetch_latency_p99_ms",
        "fetch_latency_max_ms",
        "last_ingest_stats",
    ):
        assert k in body, f"missing health field {k!r}"
