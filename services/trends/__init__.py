"""Phase 2 — social context (GraphRAG / trend firehose).

Barebones in this build: only the mock client is implemented. The
protocol + factory exist so a real GraphRAG-backed implementation can
drop in later by reading `TRENDS_MODE=graphrag` (matching the same
mock-vs-real env-toggle pattern used by `INFERENCE_MODE` and
`SUGGESTION_LLM_MODE`).
"""

from __future__ import annotations

import logging
import os

from .protocol import TrendClient, TrendContext  # noqa: F401  re-exported

_log = logging.getLogger("cortyze.trends")


def get_client() -> TrendClient:
    """Return a `TrendClient` based on `TRENDS_MODE`.

    Modes:
      * `mock` (default) — returns a stub TrendContext, no I/O.
      * `graphrag` — TODO: real Neo4j-backed implementation.
                     Raises NotImplementedError until built.

    The fall-through to mock is deliberate: every external dependency
    in this project is opt-in, so an unset env var should give you a
    working pipeline, not a crash.
    """
    mode = os.environ.get("TRENDS_MODE", "mock").strip().lower()

    if mode == "mock":
        from .mock import MockTrendClient

        return MockTrendClient()

    if mode == "graphrag":
        # Production GraphRAG client — queries the rolling 48h knowledge
        # graph, falls back to MockTrendClient (with `fallback_reason`
        # stamped) when the graph is empty / stale / unhealthy. See
        # services/social_context/client.py for the full state machine.
        from services.social_context.client import GraphRAGTrendClient

        return GraphRAGTrendClient()

    _log.warning("unknown TRENDS_MODE=%r; falling back to mock", mode)
    from .mock import MockTrendClient

    return MockTrendClient()
