"""`GraphRAGTrendClient` — production `TrendClient` implementation.

Wires the rolling 48-hour graph + the query layer into the contract
the orchestrator already calls. Owns the fallback-to-mock path that
prevents Phase 2 from blocking the pipeline when the graph is empty,
stale, or unhealthy.

Module-level singleton so the graph and metric counters survive across
requests within a single FastAPI worker.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Lock

from services.trends.mock import MockTrendClient
from services.trends.protocol import TrendContext

from .graph import KnowledgeGraph

_log = logging.getLogger("cortyze.social_context.client")

_DEFAULT_STALENESS_HOURS = 2.0


def _resolve_staleness_hours() -> float:
    raw = os.environ.get("GRAPH_STALENESS_HOURS")
    if not raw:
        return _DEFAULT_STALENESS_HOURS
    try:
        return float(raw)
    except ValueError:
        _log.warning(
            "GRAPH_STALENESS_HOURS=%r is not a float; using default %.1f",
            raw,
            _DEFAULT_STALENESS_HOURS,
        )
        return _DEFAULT_STALENESS_HOURS


# ---------------------------------------------------------------------------
# Process-wide graph singleton
# ---------------------------------------------------------------------------


_graph_lock = Lock()
_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    """Return the process-wide `KnowledgeGraph` singleton.

    Backend selection follows `GRAPH_BACKEND`:
      * `networkx` (default) — in-process, ephemeral.
      * `neo4j`              — wired in PR #5.

    Lazy-construction so a deployment that never flips
    `TRENDS_MODE=graphrag` doesn't pay the network/connection overhead.
    """
    global _graph
    with _graph_lock:
        if _graph is not None:
            return _graph
        backend = os.environ.get("GRAPH_BACKEND", "networkx").strip().lower()
        if backend == "networkx":
            from .graph import NetworkXGraph

            _graph = NetworkXGraph()
        elif backend == "neo4j":
            # AuraDB-backed graph. Constructor reads NEO4J_URI / USER /
            # PASSWORD from env and validates the connection by trying
            # to create the entity-id constraint. If the connection
            # fails the deployment will raise here at first use; the
            # client layer catches it via healthcheck() on subsequent
            # calls and triggers the mock fallback.
            from .graph import Neo4jGraph

            _graph = Neo4jGraph()
        else:
            raise RuntimeError(
                f"GRAPH_BACKEND={backend!r} not recognized. "
                "Use 'networkx' or 'neo4j'."
            )
        return _graph


def _reset_for_tests() -> None:
    """Test-only — drop the singleton so the next call rebuilds it."""
    global _graph
    with _graph_lock:
        _graph = None


# ---------------------------------------------------------------------------
# Fallback metrics (process-local; surfaced via /health/social_context)
# ---------------------------------------------------------------------------


_metrics_lock = Lock()
# Rolling window of the most recent N fetch durations (in milliseconds)
# so `/health/social_context` can report p95 without persisting a
# time-series. Cap at 1024 samples — small, fixed memory footprint.
_LATENCY_WINDOW = 1024
_metrics: dict[str, object] = {
    "fetch_total": 0,
    "fetch_fallback_total": 0,
    "fetch_fallback_by_reason": {},  # dict[str, int]
    "last_fetch_at": None,           # datetime | None
    "fetch_latencies_ms": deque(maxlen=_LATENCY_WINDOW),
}


def _reset_metrics_for_tests() -> None:
    """Test-only — reset counters so each test starts clean."""
    with _metrics_lock:
        _metrics["fetch_total"] = 0
        _metrics["fetch_fallback_total"] = 0
        _metrics["fetch_fallback_by_reason"] = {}
        _metrics["last_fetch_at"] = None
        _metrics["fetch_latencies_ms"] = deque(maxlen=_LATENCY_WINDOW)


def get_metrics() -> dict[str, object]:
    """Snapshot of the in-process counters. Safe to call concurrently.

    Latency window is summarized to p50 / p95 / p99 / max in ms; we
    never expose the raw deque so callers can't accidentally hold a
    reference to the live buffer.
    """
    with _metrics_lock:
        latencies: deque[float] = _metrics["fetch_latencies_ms"]  # type: ignore[assignment]
        snapshot_lat = sorted(latencies)
        # Shallow-copy so callers can't mutate the live state.
        return {
            "fetch_total": _metrics["fetch_total"],
            "fetch_fallback_total": _metrics["fetch_fallback_total"],
            "fetch_fallback_by_reason": dict(
                _metrics["fetch_fallback_by_reason"]  # type: ignore[arg-type]
            ),
            "last_fetch_at": _metrics["last_fetch_at"],
            "fetch_latency_p50_ms": _percentile(snapshot_lat, 0.5),
            "fetch_latency_p95_ms": _percentile(snapshot_lat, 0.95),
            "fetch_latency_p99_ms": _percentile(snapshot_lat, 0.99),
            "fetch_latency_max_ms": (
                snapshot_lat[-1] if snapshot_lat else 0.0
            ),
        }


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Nearest-rank percentile. Returns 0.0 on empty samples."""
    if not sorted_samples:
        return 0.0
    if q <= 0:
        return float(sorted_samples[0])
    if q >= 1:
        return float(sorted_samples[-1])
    # 1-based nearest-rank: ceil(q * n) → 0-indexed = ceil(q*n)-1.
    rank = max(1, int((q * len(sorted_samples)) + 0.999999))
    return float(sorted_samples[rank - 1])


def _record_fetch(
    *, fallback_reason: str | None, latency_ms: float
) -> None:
    with _metrics_lock:
        _metrics["fetch_total"] = int(_metrics["fetch_total"]) + 1  # type: ignore[arg-type]
        _metrics["last_fetch_at"] = datetime.now(timezone.utc)
        latencies: deque[float] = _metrics["fetch_latencies_ms"]  # type: ignore[assignment]
        latencies.append(latency_ms)
        if fallback_reason is not None:
            _metrics["fetch_fallback_total"] = (
                int(_metrics["fetch_fallback_total"]) + 1  # type: ignore[arg-type]
            )
            by_reason: dict[str, int] = _metrics[
                "fetch_fallback_by_reason"
            ]  # type: ignore[assignment]
            by_reason[fallback_reason] = by_reason.get(fallback_reason, 0) + 1


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GraphRAGTrendClient:
    """`TrendClient` impl backed by the social-context graph.

    Falls back to `MockTrendClient` (with a `fallback_reason` stamp) when:
      * the graph is empty (no scrape pass has run yet),
      * the graph's last ingest is older than `GRAPH_STALENESS_HOURS`,
      * the graph fails its healthcheck (e.g., Neo4j unreachable).

    The fallback path always returns a populated `TrendContext` so
    Phase 3 never sees `None` — the audit-trail `fallback_reason`
    surfaces the degradation honestly.
    """

    def __init__(self, *, graph: KnowledgeGraph | None = None) -> None:
        self._graph = graph
        self._mock = MockTrendClient()
        self._stale_after = timedelta(hours=_resolve_staleness_hours())

    def fetch(
        self,
        *,
        brief: str,
        caption: str,
        goal: str,
        request_id: str | None = None,
    ) -> TrendContext:
        started = time.monotonic()
        graph = self._graph or get_graph()
        reason = self._fallback_reason(graph)
        if reason is not None:
            _log.info(
                "graphrag fallback request_id=%s reason=%s",
                request_id,
                reason,
            )
            ctx = self._mock.fetch(
                brief=brief,
                caption=caption,
                goal=goal,
                request_id=request_id,
            )
            ctx.fallback_reason = reason
            _record_fetch(
                fallback_reason=reason,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
            return ctx

        # Real path — query the graph and assemble the TrendContext.
        from .query import get_trend_context

        try:
            ctx = get_trend_context(
                brief=brief,
                caption=caption,
                goal=goal,
                graph=graph,
                request_id=request_id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            _log.exception("graphrag query failed; falling back to mock")
            ctx = self._mock.fetch(
                brief=brief,
                caption=caption,
                goal=goal,
                request_id=request_id,
            )
            ctx.fallback_reason = f"query_error:{type(exc).__name__}"
            _record_fetch(
                fallback_reason=ctx.fallback_reason,
                latency_ms=(time.monotonic() - started) * 1000.0,
            )
            return ctx

        latency_ms = (time.monotonic() - started) * 1000.0
        _record_fetch(fallback_reason=None, latency_ms=latency_ms)
        if request_id:
            _log.info(
                "graphrag fetch request_id=%s latency_ms=%.1f entities=%d",
                request_id,
                latency_ms,
                len(ctx.entities),
            )
        return ctx

    # ---------------------------------------------------------- internals

    def _fallback_reason(self, graph: KnowledgeGraph) -> str | None:
        if not graph.healthcheck():
            return "graph_unhealthy"
        last = graph.last_ingest_at()
        if last is None:
            return "empty_graph"
        # `last_ingest_at` may be naive in older test fixtures — normalize.
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - last > self._stale_after:
            return "stale_graph"
        return None
