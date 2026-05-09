"""Scheduler + scrapers + ratelimit + /health/social_context endpoint.

External APIs (Reddit, NewsAPI) are mocked at the scraper boundary —
we never make a real HTTP call. The scheduler's cron triggers are
exercised via direct invocation of the job wrappers, not by waiting on
APScheduler timers.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("networkx")

from api.main import app  # noqa: E402
from services.social_context import client as client_mod  # noqa: E402
from services.social_context import entities as ent_mod  # noqa: E402
from services.social_context import scheduler as sched_mod  # noqa: E402
from services.social_context import scraper as scraper_mod  # noqa: E402
from services.social_context.ratelimit import TokenBucket  # noqa: E402
from services.social_context.schemas import SourceSnapshot  # noqa: E402

_test_client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    sched_mod._reset_counters_for_tests()
    ent_mod._reset_for_tests()
    # Force the regex NER fallback so spaCy model presence doesn't matter.
    monkeypatch.setattr(ent_mod, "_try_load_spacy", lambda: None)
    yield
    client_mod._reset_for_tests()
    client_mod._reset_metrics_for_tests()
    sched_mod._reset_counters_for_tests()
    ent_mod._reset_for_tests()


# ---------------------------------------------------------------- ratelimit


def test_token_bucket_consumes_when_available():
    b = TokenBucket(rate_per_min=60, burst=2)
    assert b.try_consume(1) is True
    assert b.try_consume(1) is True
    assert b.try_consume(1) is False  # burst exhausted


def test_token_bucket_refills_over_time(monkeypatch):
    import services.social_context.ratelimit as rl_mod

    fake_now = [0.0]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr(rl_mod.time, "monotonic", fake_monotonic)
    b = TokenBucket(rate_per_min=60, burst=1)
    assert b.try_consume(1) is True
    assert b.try_consume(1) is False
    fake_now[0] = 2.0  # 2s of refill at 1 token/s
    assert b.try_consume(1) is True


def test_token_bucket_rejects_invalid_rate():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_min=0)


# ---------------------------------------------------------------- scrapers


def test_reddit_scraper_disabled_without_env(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    sc = scraper_mod.RedditScraper()
    assert sc.enabled is False
    assert sc.fetch() == []


def test_news_scraper_disabled_without_key(monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    sc = scraper_mod.NewsScraper()
    assert sc.enabled is False
    assert sc.fetch() == []


def test_x_scraper_disabled_by_default(monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "fake")
    monkeypatch.delenv("X_SCRAPER_ENABLED", raising=False)
    sc = scraper_mod.XScraper()
    # Even with the token set, the explicit enable flag gates it off.
    assert sc.enabled is False


def test_ingest_one_returns_zero_for_disabled_scraper(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    sc = scraper_mod.RedditScraper()
    snaps, stats = scraper_mod.ingest_one(sc)
    assert snaps == []
    assert stats.snapshots_ingested == 0
    assert stats.errors == 0


# ---------------------------------------------------------------- scheduler


def _make_fake_scraper(name: str, snaps: list[SourceSnapshot]) -> Any:
    """Tiny stub that mimics the Scraper protocol."""

    class Fake:
        source = name

        @property
        def enabled(self) -> bool:
            return True

        def fetch(self, *, limit: int = 25) -> list[SourceSnapshot]:
            del limit
            return list(snaps)

    return Fake()


def test_run_ingest_pass_pushes_into_graph(monkeypatch):
    snap = SourceSnapshot(
        source="reddit",
        source_id="abc",
        title="Nike just unveiled a new sneaker at the Super Bowl",
        body="It's the biggest Nike drop of 2026.",
        ingested_at=datetime.now(timezone.utc),
    )
    fake = _make_fake_scraper("reddit", [snap])

    monkeypatch.setattr(scraper_mod, "all_scrapers", lambda: [fake])
    asyncio.run(sched_mod.run_ingest_pass())

    counters = sched_mod.get_counters()
    assert counters["ingest_runs_total"] == 1
    assert counters["last_ingest_at"] is not None
    assert counters["sources_healthy"] == {"reddit": True}

    # The graph should now contain at least one entity.
    graph = client_mod.get_graph()
    assert graph.node_count() >= 1
    assert graph.last_ingest_at() is not None


def test_run_prune_pass_uses_ttl(monkeypatch):
    """Prune is a thin wrapper; just ensure it runs and updates counters."""
    snap = SourceSnapshot(source="reddit", source_id="x", title="Nike")
    fake = _make_fake_scraper("reddit", [snap])
    monkeypatch.setattr(scraper_mod, "all_scrapers", lambda: [fake])
    asyncio.run(sched_mod.run_ingest_pass())
    asyncio.run(sched_mod.run_prune_pass())
    counters = sched_mod.get_counters()
    assert counters["last_prune_at"] is not None


def test_consecutive_failed_passes_increments_then_resets(monkeypatch):
    """A pass where every source erred should bump the failure counter;
    the next successful pass should reset it to zero."""

    class FailingScraper:
        source = "reddit"

        @property
        def enabled(self) -> bool:
            return True

        def fetch(self, *, limit: int = 25):
            raise RuntimeError("API down")

    monkeypatch.setattr(
        scraper_mod, "all_scrapers", lambda: [FailingScraper()]
    )
    asyncio.run(sched_mod.run_ingest_pass())
    assert (
        sched_mod.get_counters()["consecutive_failed_passes"] == 1
    )

    snap = SourceSnapshot(
        source="reddit", source_id="ok", title="Nike Super Bowl"
    )
    monkeypatch.setattr(
        scraper_mod,
        "all_scrapers",
        lambda: [_make_fake_scraper("reddit", [snap])],
    )
    asyncio.run(sched_mod.run_ingest_pass())
    assert (
        sched_mod.get_counters()["consecutive_failed_passes"] == 0
    )


# ---------------------------------------------------------------- /health


def test_health_endpoint_alive():
    r = _test_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_social_context_health_returns_structured_payload():
    r = _test_client.get("/health/social_context")
    assert r.status_code == 200
    body = r.json()
    # Required fields are present even on a fresh / empty graph.
    for key in (
        "graph_backend",
        "graph_healthy",
        "node_count",
        "ingest_runs_total",
        "fetch_total",
        "fetch_fallback_total",
        "fetch_fallback_rate",
        "sources_healthy",
        "checked_at",
    ):
        assert key in body, f"missing health field {key!r}"


def test_social_context_health_reflects_ingest_pass(monkeypatch):
    snap = SourceSnapshot(
        source="reddit",
        source_id="hl-1",
        title="Nike running shoes",
    )
    monkeypatch.setattr(
        scraper_mod,
        "all_scrapers",
        lambda: [_make_fake_scraper("reddit", [snap])],
    )
    asyncio.run(sched_mod.run_ingest_pass())

    r = _test_client.get("/health/social_context")
    body = r.json()
    assert body["ingest_runs_total"] == 1
    assert body["last_snapshot_at"] is not None
    assert body["node_count"] >= 1


# -------------------------------------------------------------- integration


def test_runs_with_graphrag_falls_back_when_graph_empty(monkeypatch):
    """End-to-end: TRENDS_MODE=graphrag with an empty graph should
    drive the fallback path and the run should still complete."""
    # Ensure clean global state, including a fresh graph singleton that
    # hasn't been populated by other tests in the same session.
    client_mod._reset_for_tests()
    monkeypatch.setenv("TRENDS_MODE", "graphrag")
    # The factory caches once; reset to pick up the env override.
    import services.trends as trends_pkg

    # The factory is called per request anyway, but reset for determinism.
    monkeypatch.setattr(
        trends_pkg, "get_client", trends_pkg.get_client
    )

    r = _test_client.post(
        "/runs",
        json={
            "name": "graphrag fallback test",
            "goal": "brand_recall",
            "brief": "a",
        },
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    # Poll until complete (5s ceiling, mock pipeline finishes much faster).
    import time

    deadline = time.monotonic() + 5.0
    final: dict[str, Any] = {}
    while time.monotonic() < deadline:
        gr = _test_client.get(f"/runs/{run_id}")
        assert gr.status_code == 200
        final = gr.json()
        if final["status"] in ("complete", "failed"):
            break
        time.sleep(0.05)
    assert final["status"] == "complete"

    # Fallback should have fired; metrics should reflect it.
    metrics = client_mod.get_metrics()
    assert metrics["fetch_total"] >= 1
    assert metrics["fetch_fallback_total"] >= 1


# --------------------------------------------------------------- start_scheduler


def test_start_scheduler_is_idempotent(monkeypatch):
    """AsyncIOScheduler needs a running event loop to actually start;
    patch `.start` to a no-op so we exercise the module's idempotency
    guard without spinning real threads in a sync test runner."""
    pytest.importorskip("apscheduler")
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    monkeypatch.setattr(AsyncIOScheduler, "start", lambda self, **_: None)
    sched_mod.stop_scheduler()
    s1 = sched_mod.start_scheduler()
    s2 = sched_mod.start_scheduler()
    assert s1 is not None
    assert s1 is s2
    sched_mod.stop_scheduler()


def test_start_scheduler_returns_none_when_apscheduler_missing(monkeypatch):
    """Simulate the 'no extras installed' deployment path."""
    sched_mod.stop_scheduler()
    # Pretend apscheduler isn't importable by injecting an ImportError
    # into the dynamic import inside start_scheduler.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("apscheduler"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    result = sched_mod.start_scheduler()
    assert result is None
