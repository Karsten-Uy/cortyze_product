"""APScheduler-driven ingest + prune loop for the social-context graph.

Two cron jobs:
  * `ingest` — every `INGEST_INTERVAL_MIN` (default 30): run all enabled
    scrapers concurrently, derive entities + sentiment, push into the
    graph, persist a row in `trend_snapshots`.
  * `prune`  — every `GRAPH_PRUNE_INTERVAL_MIN` (default 60): drop
    graph nodes whose `last_seen` is older than `GRAPH_TTL_HOURS`.

In-process AsyncIOScheduler so the FastAPI service stays a single
deployable. **Locks the service to `--workers 1`** — multiple workers
would each run their own scheduler. If/when the team needs multi-worker
the scheduler boundary already exists; pull it into a Railway worker
service with `python -m services.social_context.scheduler` as entrypoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

from .schemas import IngestStats, SourceSnapshot

_log = logging.getLogger("cortyze.social_context.scheduler")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("%s=%r not a float; using default %.1f", name, raw, default)
        return default


# ---------------------------------------------------------------------------
# Counters surfaced via /health/social_context
# ---------------------------------------------------------------------------


_counters_lock = Lock()
_counters: dict[str, Any] = {
    "ingest_runs_total": 0,
    "last_ingest_at": None,                  # datetime | None
    "last_prune_at": None,                   # datetime | None
    "last_ingest_stats": [],                 # list[dict] of IngestStats per source
    "consecutive_failed_passes": 0,
    "sources_healthy": {},                   # dict[str, bool]
}


def get_counters() -> dict[str, Any]:
    """Snapshot of scheduler counters. Safe to call concurrently."""
    with _counters_lock:
        return {
            "ingest_runs_total": _counters["ingest_runs_total"],
            "last_ingest_at": _counters["last_ingest_at"],
            "last_prune_at": _counters["last_prune_at"],
            "last_ingest_stats": list(_counters["last_ingest_stats"]),
            "consecutive_failed_passes": _counters["consecutive_failed_passes"],
            "sources_healthy": dict(_counters["sources_healthy"]),
        }


def _record_ingest(stats: list[IngestStats], *, all_failed: bool) -> None:
    with _counters_lock:
        _counters["ingest_runs_total"] = (
            int(_counters["ingest_runs_total"]) + 1
        )
        _counters["last_ingest_at"] = datetime.now(timezone.utc)
        _counters["last_ingest_stats"] = [s.model_dump(mode="json") for s in stats]
        _counters["sources_healthy"] = {
            s.source: s.errors == 0 and s.snapshots_ingested >= 0
            for s in stats
        }
        if all_failed:
            _counters["consecutive_failed_passes"] = (
                int(_counters["consecutive_failed_passes"]) + 1
            )
        else:
            _counters["consecutive_failed_passes"] = 0


def _record_prune(removed: int) -> None:
    del removed  # not surfaced; we just need the timestamp
    with _counters_lock:
        _counters["last_prune_at"] = datetime.now(timezone.utc)


def _reset_counters_for_tests() -> None:
    with _counters_lock:
        _counters["ingest_runs_total"] = 0
        _counters["last_ingest_at"] = None
        _counters["last_prune_at"] = None
        _counters["last_ingest_stats"] = []
        _counters["consecutive_failed_passes"] = 0
        _counters["sources_healthy"] = {}


# ---------------------------------------------------------------------------
# Ingest pipeline (snapshot → entities → graph)
# ---------------------------------------------------------------------------


async def run_ingest_pass() -> list[IngestStats]:
    """Run every enabled scraper, derive entities + sentiment, push into
    the graph. Returns per-source stats. Never raises out — failures
    are counted in `IngestStats.errors`.
    """
    from .client import get_graph
    from .entities import extract_entities
    from .scraper import all_scrapers, ingest_one
    from .sentiment import score_sentiment
    from .schemas import EntityEdge

    scrapers = all_scrapers()
    snaps_per_source: list[tuple[list[SourceSnapshot], IngestStats]] = []
    started = time.monotonic()
    # Run scrapers concurrently — `ingest_one` is sync (blocking I/O),
    # so wrap each in `asyncio.to_thread`.
    results = await asyncio.gather(
        *(asyncio.to_thread(ingest_one, sc) for sc in scrapers),
        return_exceptions=True,
    )
    for sc, res in zip(scrapers, results):
        if isinstance(res, Exception):
            _log.exception("scraper %s wrapper failed", sc.source)
            snaps_per_source.append(
                (
                    [],
                    IngestStats(
                        source=sc.source, snapshots_ingested=0, errors=1
                    ),
                )
            )
        else:
            snaps_per_source.append(res)

    graph = get_graph()
    stats: list[IngestStats] = []
    for snaps, base_stats in snaps_per_source:
        entities_added = 0
        edges_added = 0
        for snap in snaps:
            text = "\n".join(filter(None, [snap.title, snap.body]))
            if not text:
                continue
            ents = extract_entities(text)
            sentiment = score_sentiment(text)
            for ent in ents:
                graph.add_entity(ent, snap)
                entities_added += 1
                # Co-occurrence edges between every pair of entities in
                # the same snapshot. Cheap; the graph dedupes by id.
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    graph.add_edge(
                        EntityEdge(
                            src=ents[i].name,
                            dst=ents[j].name,
                            kind="CO_OCCURS_WITH",
                            weight=1.0,
                            ts=snap.ingested_at,
                        )
                    )
                    edges_added += 1
                # Sentiment edge: each entity → a synthetic "sentiment"
                # context node so the query layer can read polarity
                # off the edge weight without inventing a separate
                # sentiment column on the entity itself.
                graph.add_edge(
                    EntityEdge(
                        src=ents[i].name,
                        dst=f"_sentiment::{snap.source}::{snap.source_id}",
                        kind="SENTIMENT",
                        weight=sentiment.polarity,
                        ts=snap.ingested_at,
                    )
                )
                edges_added += 1
        stats.append(
            IngestStats(
                source=base_stats.source,
                snapshots_ingested=base_stats.snapshots_ingested,
                entities_added=entities_added,
                edges_added=edges_added,
                errors=base_stats.errors,
                latency_ms=base_stats.latency_ms,
                finished_at=base_stats.finished_at,
            )
        )

    elapsed = int((time.monotonic() - started) * 1000)
    all_failed = bool(stats) and all(
        s.snapshots_ingested == 0 and s.errors > 0 for s in stats
    )
    _record_ingest(stats, all_failed=all_failed)
    _log.info(
        "ingest_pass done elapsed_ms=%d sources=%s",
        elapsed,
        [s.source for s in stats],
    )
    # Persist a single audit row per pass — a roll-up of the counters
    # so the operator can answer "did the scheduler run at 14:30?".
    try:
        _persist_snapshot_audit(stats)
    except Exception:  # noqa: BLE001
        _log.exception("persist snapshot audit failed; continuing")
    return stats


def _persist_snapshot_audit(stats: list[IngestStats]) -> None:
    """Write a row to `trend_snapshots` per-source so an operator can
    audit when the scheduler last ran. Best-effort; missing DB is fine.
    """
    from services.persistence.runs_v2 import RUN_STORE

    backend_kind = type(RUN_STORE).__name__
    if backend_kind != "_PostgresRunStore":
        return  # in-memory dev store — skip
    # Reach into the same connection — RUN_STORE owns it. Acquiring its
    # lock keeps us thread-safe with concurrent inserts.
    try:
        conn = RUN_STORE._conn  # type: ignore[attr-defined]
        lock = RUN_STORE._lock  # type: ignore[attr-defined]
    except AttributeError:
        return
    ttl = datetime.now(timezone.utc) + timedelta(
        hours=_env_float("GRAPH_TTL_HOURS", 48.0)
    )
    with lock, conn.cursor() as cur:
        for s in stats:
            cur.execute(
                """
                INSERT INTO trend_snapshots (
                    snapshot_id, source, source_id, payload, ingested_at, ttl_at
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (source, source_id) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    s.source,
                    f"audit::{s.finished_at.isoformat()}",
                    s.model_dump_json(),
                    s.finished_at,
                    ttl,
                ),
            )


async def run_prune_pass() -> int:
    """Drop graph nodes whose `last_seen` is older than `GRAPH_TTL_HOURS`."""
    from .client import get_graph

    ttl_hours = _env_float("GRAPH_TTL_HOURS", 48.0)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
    graph = get_graph()
    removed = await asyncio.to_thread(graph.prune_older_than, cutoff)
    _record_prune(removed)
    _log.info("prune_pass removed=%d cutoff=%s", removed, cutoff.isoformat())
    return removed


# ---------------------------------------------------------------------------
# Boot hook
# ---------------------------------------------------------------------------


_scheduler: Any | None = None
_scheduler_lock = Lock()


def start_scheduler(app: Any | None = None) -> Any:
    """Boot the AsyncIOScheduler with our two cron jobs.

    Idempotent — safe to call from `create_app()`. If APScheduler isn't
    installed (deployment without `[social-context]` extras), logs a
    warning and returns None; the GraphRAG client falls back to mock.
    """
    del app  # reserved for future per-request lifecycle integration
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            return _scheduler
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError:
            _log.warning(
                "apscheduler not installed — social_context scheduler "
                "disabled. Install via `uv sync --extra social-context`."
            )
            return None

        ingest_min = _env_float("INGEST_INTERVAL_MIN", 30.0)
        prune_min = _env_float("GRAPH_PRUNE_INTERVAL_MIN", 60.0)

        sched = AsyncIOScheduler(timezone="UTC")
        sched.add_job(
            _ingest_job_wrapper,
            trigger=IntervalTrigger(minutes=ingest_min),
            id="social_context.ingest",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        sched.add_job(
            _prune_job_wrapper,
            trigger=IntervalTrigger(minutes=prune_min),
            id="social_context.prune",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        sched.start()
        _scheduler = sched
        _log.info(
            "social_context scheduler started ingest=%.1fmin prune=%.1fmin",
            ingest_min,
            prune_min,
        )
        # Kick off an immediate ingest so the graph starts filling
        # without waiting a full interval. Fire-and-forget; the wrapper
        # logs on failure.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_ingest_job_wrapper())
        except RuntimeError:
            pass
        return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            return
        try:
            _scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            _log.exception("scheduler shutdown raised; ignoring")
        _scheduler = None


async def _ingest_job_wrapper() -> None:
    try:
        await run_ingest_pass()
    except Exception:  # noqa: BLE001
        _log.exception("ingest pass crashed")


async def _prune_job_wrapper() -> None:
    try:
        await run_prune_pass()
    except Exception:  # noqa: BLE001
        _log.exception("prune pass crashed")
