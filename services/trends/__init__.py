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
        # Real implementation lives behind a NotImplementedError so the
        # operator gets a clear message (vs. a silent fall-through to
        # mock that would mask a misconfigured prod deployment).
        raise NotImplementedError(
            "TRENDS_MODE=graphrag is not implemented yet. "
            "Set TRENDS_MODE=mock or unset to use the mock client."
        )

    _log.warning("unknown TRENDS_MODE=%r; falling back to mock", mode)
    from .mock import MockTrendClient

    return MockTrendClient()
