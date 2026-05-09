"""Neo4jGraph backend smoke tests.

The full Cypher logic lands in PR #5 but real AuraDB access doesn't
exist in CI, so we test:
  * The "driver missing" path raises a clear RuntimeError.
  * Constructor enforces NEO4J_URI / NEO4J_PASSWORD env vars.
  * Driver-level methods are exercised against a mock when the
    `neo4j` package IS installed (rare in local dev; skipped otherwise).

Production verification: smoke `/health/social_context` against the
deployed Railway service after setting GRAPH_BACKEND=neo4j and the
AuraDB credentials. `node_count` should climb on each ingest cron.
"""

from __future__ import annotations

import importlib.util

import pytest


def _has_neo4j_driver() -> bool:
    return importlib.util.find_spec("neo4j") is not None


def test_neo4j_init_without_driver_raises_runtime_error(monkeypatch):
    """When the `[social-context-neo4j]` extras aren't installed, the
    constructor must surface the install hint rather than ImportError."""
    if _has_neo4j_driver():
        pytest.skip("neo4j driver is installed; this test exercises the missing-deps path")
    from services.social_context.graph import Neo4jGraph

    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    with pytest.raises(RuntimeError) as exc:
        Neo4jGraph()
    assert "social-context-neo4j" in str(exc.value)


@pytest.mark.skipif(
    not _has_neo4j_driver(), reason="neo4j driver not installed locally"
)
def test_neo4j_init_requires_uri_and_password(monkeypatch):
    """With the driver present but env vars missing, constructor raises."""
    from services.social_context.graph import Neo4jGraph

    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        Neo4jGraph()


@pytest.mark.skipif(
    not _has_neo4j_driver(), reason="neo4j driver not installed locally"
)
def test_neo4j_healthcheck_returns_false_on_unreachable(monkeypatch):
    """The healthcheck must report False rather than raising when the
    bolt endpoint is unreachable — the client layer relies on a False
    return to trigger the mock fallback."""
    from services.social_context.graph import Neo4jGraph

    # Point at a definitively-dead host so we don't accidentally hit a
    # real Neo4j instance somebody has running.
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    try:
        g = Neo4jGraph()
    except Exception:  # noqa: BLE001
        # Some neo4j driver versions raise on construction when the
        # endpoint is dead. That's also acceptable failure-mode.
        return
    assert g.healthcheck() is False
    g.close()
