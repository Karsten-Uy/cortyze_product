"""Aggregate health snapshot for `GET /health/social_context`.

Returns a single dict the API route renders straight to JSON. Pulls
from the client metrics, the scheduler counters, and the graph itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def social_context_health() -> dict[str, Any]:
    """Snapshot the social-context subsystem.

    Always returns a 2xx-renderable dict — even if the GraphRAG client
    has never been instantiated. The frontend / external monitoring
    can compare `last_snapshot_at` to "now" to alarm on stalls.
    """
    from . import client as client_mod
    from . import scheduler as sched_mod

    client_metrics = client_mod.get_metrics()
    sched_counters = sched_mod.get_counters()

    # Probe the graph singleton without forcing construction.
    node_count = 0
    last_ingest_at: datetime | None = None
    graph_healthy = True
    try:
        graph = client_mod.get_graph()
        node_count = graph.node_count()
        last_ingest_at = graph.last_ingest_at()
        graph_healthy = graph.healthcheck()
    except NotImplementedError:
        # GRAPH_BACKEND=neo4j without PR #5 → report unhealthy.
        graph_healthy = False
    except Exception:  # noqa: BLE001
        graph_healthy = False

    # Compute fallback rate over the last hour from the running totals.
    # The GraphRAG client doesn't track per-window rates internally —
    # we surface raw counts and let the operator do the math.
    total = int(client_metrics["fetch_total"])  # type: ignore[arg-type]
    fbacks = int(client_metrics["fetch_fallback_total"])  # type: ignore[arg-type]
    fallback_rate = float(fbacks) / total if total else 0.0

    return {
        "graph_backend": _graph_backend_name(),
        "graph_healthy": graph_healthy,
        "node_count": node_count,
        "last_ingest_at": _iso(last_ingest_at),
        "last_snapshot_at": _iso(sched_counters["last_ingest_at"]),
        "last_prune_at": _iso(sched_counters["last_prune_at"]),
        "ingest_runs_total": sched_counters["ingest_runs_total"],
        "consecutive_failed_passes": sched_counters[
            "consecutive_failed_passes"
        ],
        "sources_healthy": sched_counters["sources_healthy"],
        "last_ingest_stats": sched_counters["last_ingest_stats"],
        "fetch_total": total,
        "fetch_fallback_total": fbacks,
        "fetch_fallback_rate": fallback_rate,
        "fetch_fallback_by_reason": client_metrics[
            "fetch_fallback_by_reason"
        ],
        "fetch_latency_p50_ms": client_metrics.get(
            "fetch_latency_p50_ms", 0.0
        ),
        "fetch_latency_p95_ms": client_metrics.get(
            "fetch_latency_p95_ms", 0.0
        ),
        "fetch_latency_p99_ms": client_metrics.get(
            "fetch_latency_p99_ms", 0.0
        ),
        "fetch_latency_max_ms": client_metrics.get(
            "fetch_latency_max_ms", 0.0
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _graph_backend_name() -> str:
    import os

    return os.environ.get("GRAPH_BACKEND", "networkx").strip().lower() or "networkx"


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()
