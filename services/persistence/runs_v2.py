"""Persistence for the v2 (`/runs`) pipeline.

Two-mode storage:
  * `DATABASE_URL` set    — Postgres via psycopg.
  * `DATABASE_URL` unset  — in-memory dict (per-process).

The in-memory mode keeps local development frictionless: no DB needed
to exercise the full Lab Bench → Results flow against the mock
pipeline. Production must set DATABASE_URL.

The legacy `services.persistence.reports` module is unchanged — it
backs the old `/analyze` flow and shares no tables with this one.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from core.regions_v2 import REGION_KEYS
from core.schemas_v2 import (
    PastRun,
    Reference,
    RegionScore,
    RunRecord,
    RunStatus,
    Suggestion,
    SuggestionPlan,
)

_log = logging.getLogger("cortyze.persistence.runs_v2")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_date(dt: datetime) -> str:
    """'Apr 28' — server-side locale-stable formatting for the
    sidebar. Real implementations would localize this; the v2
    frontend hard-codes English so we do too."""
    return dt.strftime("%b %d").replace(" 0", " ")


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class _InMemoryRunStore:
    """Process-local run registry. Thread-safe (FastAPI worker-safe in
    single-process mode). NOT safe across uvicorn workers > 1."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def put(self, record: RunRecord) -> None:
        with self._lock:
            self._records[record.id] = record

    def update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            existing = self._records.get(run_id)
            if existing is None:
                _log.warning("update() for unknown run %s", run_id)
                return
            self._records[run_id] = existing.model_copy(update=fields)

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def list_for_user(
        self,
        user_id: str | None,
        limit: int = 20,
    ) -> list[PastRun]:
        with self._lock:
            records = [
                r for r in self._records.values() if r.user_id == user_id
            ]
        records.sort(key=lambda r: r.created_at, reverse=True)
        out: list[PastRun] = []
        for r in records[:limit]:
            score = r.result.score if r.result else 0.0
            try:
                date_label = _format_date(datetime.fromisoformat(r.created_at))
            except ValueError:
                date_label = r.created_at[:10]
            out.append(
                PastRun(
                    id=r.id,
                    name=r.name,
                    date=date_label,
                    kind=r.kind,
                    score=round(score, 0),
                )
            )
        return out

    def previous_score(
        self, user_id: str | None, exclude_id: str
    ) -> float | None:
        """Most recent completed run's composite score for this user.
        Used by Phase 3 to compute the `delta vs last run` field. None
        means this is the user's first run.
        """
        with self._lock:
            records = [
                r
                for r in self._records.values()
                if r.user_id == user_id
                and r.id != exclude_id
                and r.result is not None
            ]
        if not records:
            return None
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[0].result.score if records[0].result else None


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------


# Columns on the `runs` table that `update()` is allowed to write.
# Anything else (notably `id`, `created_at`, `user_id` post-creation) is
# silently dropped — the `result` field is handled separately because it
# spans three other tables.
_UPDATABLE_RUNS_COLUMNS = frozenset(
    {"name", "goal", "brief", "caption", "media_url", "media_object_key",
     "kind", "status", "completed_at", "error"}
)


class _PostgresRunStore:
    """psycopg 3-backed store. Schema lives in
    `services/persistence/migrations/002_runs_v2.sql`.

    The connection is opened eagerly at construction. `prepare_threshold=None`
    is required for Supabase pgbouncer (transaction-pooled) compatibility —
    pgbouncer doesn't share prepared-statement state across pooled
    backends, so caching them causes `DuplicatePreparedStatement` errors.
    Same pattern as the legacy `reports.py` store.

    Single shared connection is fine for FastAPI's single-event-loop
    model (all requests run on the same thread). For uvicorn `--workers
    > 1`, swap this for `psycopg_pool.ConnectionPool`.
    """

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._dsn = dsn
        self._lock = threading.Lock()
        self._conn = psycopg.connect(
            dsn,
            autocommit=True,
            prepare_threshold=None,
        )
        _log.info("PostgresRunStore connected: %s", self._safe_dsn())

    def _safe_dsn(self) -> str:
        return self._dsn.split("@")[-1] if "@" in self._dsn else self._dsn

    # ---------------------------------------------------------------- put

    def put(self, record: RunRecord) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    id, user_id, name, goal, brief, caption,
                    media_url, media_object_key, kind, status,
                    created_at, completed_at, error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    record.id,
                    record.user_id,
                    record.name,
                    record.goal,
                    record.brief,
                    record.caption,
                    record.media_url,
                    record.media_object_key,
                    record.kind,
                    record.status,
                    record.created_at,
                    record.completed_at,
                    record.error,
                ),
            )

    # ------------------------------------------------------------- update

    def update(self, run_id: str, **fields: Any) -> None:
        # `result` (SuggestionPlan) spans three child tables — composites,
        # region_scores, suggestions — so it gets its own transactional
        # write. Everything else is a single UPDATE on `runs`.
        result = fields.pop("result", None)

        try:
            if result is not None:
                self._write_result(run_id, result)
            if fields:
                self._update_run_row(run_id, fields)
        except Exception:
            _log.exception("update(%s) failed; fields=%s", run_id, list(fields))
            raise

    def _update_run_row(self, run_id: str, fields: dict[str, Any]) -> None:
        # Whitelist columns to avoid SQL injection via field names. (Values
        # are still parametrised — this is belt-and-braces.)
        cols = [c for c in fields if c in _UPDATABLE_RUNS_COLUMNS]
        if not cols:
            return
        set_clause = ", ".join(f"{c} = %s" for c in cols)
        params: list[Any] = [fields[c] for c in cols]
        params.append(run_id)
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE runs SET {set_clause} WHERE id = %s",
                params,
            )

    def _write_result(self, run_id: str, plan: SuggestionPlan) -> None:
        # Three writes — composites (one row), region_scores (six rows),
        # suggestions (N rows). Wrapped in a manual BEGIN/COMMIT because
        # the connection is autocommit; without the transaction, a
        # half-failed result write would leave the run with partial
        # data.
        with self._lock, self._conn.cursor() as cur:
            cur.execute("BEGIN")
            try:
                cur.execute(
                    """
                    INSERT INTO composites (run_id, score, benchmark, delta, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        score     = EXCLUDED.score,
                        benchmark = EXCLUDED.benchmark,
                        delta     = EXCLUDED.delta,
                        status    = EXCLUDED.status
                    """,
                    (run_id, plan.score, plan.benchmark, plan.delta, plan.status),
                )

                for r in plan.regions:
                    cur.execute(
                        """
                        INSERT INTO region_scores (run_id, region_key, score, benchmark)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (run_id, region_key) DO UPDATE SET
                            score     = EXCLUDED.score,
                            benchmark = EXCLUDED.benchmark
                        """,
                        (run_id, r.key, r.score, r.benchmark),
                    )

                # Suggestions are easier to delete-and-reinsert than to
                # diff-and-upsert: the rank may change between Phase 3
                # and Phase 4 if filtering kicks in, and `ord` is the PK.
                cur.execute(
                    "DELETE FROM suggestions WHERE run_id = %s", (run_id,)
                )
                for ord_idx, s in enumerate(plan.suggestions, start=1):
                    cur.execute(
                        """
                        INSERT INTO suggestions
                            (run_id, ord, priority, title, area, lift,
                             explanation, reference_json, examples_json,
                             peak_start_s, peak_end_s)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            ord_idx,
                            s.priority,
                            s.title,
                            s.area,
                            s.lift,
                            s.explanation,
                            json.dumps(s.reference.model_dump())
                            if s.reference else None,
                            json.dumps(s.examples) if s.examples else None,
                            s.peak_start_s,
                            s.peak_end_s,
                        ),
                    )
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    # ---------------------------------------------------------------- get

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, name, goal, brief, caption,
                       media_url, media_object_key, kind, status,
                       created_at, completed_at, error
                FROM runs WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None

        record_data: dict[str, Any] = {
            "id": row[0],
            "user_id": str(row[1]) if row[1] is not None else None,
            "name": row[2],
            "goal": row[3],
            "brief": row[4],
            "caption": row[5],
            "media_url": row[6],
            "media_object_key": row[7],
            "kind": row[8],
            "status": row[9],
            "created_at": _iso(row[10]),
            "completed_at": _iso(row[11]),
            "error": row[12],
            "result": None,
        }

        # Only fetch the result if the pipeline has reached a stage where
        # it's actually been written. Reading mid-pipeline avoids a
        # partial composites read.
        if record_data["status"] in ("plan_done", "validating", "complete"):
            record_data["result"] = self._fetch_result(run_id)

        return RunRecord(**record_data)

    def _fetch_result(self, run_id: str) -> SuggestionPlan | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT score, benchmark, delta, status "
                "FROM composites WHERE run_id = %s",
                (run_id,),
            )
            comp = cur.fetchone()
            if comp is None:
                return None
            cur.execute(
                "SELECT region_key, score, benchmark "
                "FROM region_scores WHERE run_id = %s",
                (run_id,),
            )
            region_rows = cur.fetchall()
            cur.execute(
                "SELECT ord, priority, title, area, lift, "
                "explanation, reference_json, examples_json, "
                "peak_start_s, peak_end_s "
                "FROM suggestions WHERE run_id = %s ORDER BY ord",
                (run_id,),
            )
            sug_rows = cur.fetchall()

        # Region rows come back in arbitrary order — re-emit in canonical
        # order (matches REGION_KEYS) so the frontend's render is stable.
        rows_by_key = {r[0]: r for r in region_rows}
        regions = [
            RegionScore(
                key=k,  # type: ignore[arg-type]
                score=float(rows_by_key[k][1]),
                benchmark=float(rows_by_key[k][2]),
            )
            for k in REGION_KEYS
            if k in rows_by_key
        ]

        suggestions: list[Suggestion] = []
        for row in sug_rows:
            ref = _decode_reference(row[6])
            examples_raw = row[7]
            # examples_json is a jsonb column; psycopg returns it already
            # parsed as a Python list. Older rows written before this
            # column existed come back as None — treat as empty list.
            examples = examples_raw if isinstance(examples_raw, list) else []
            suggestions.append(
                Suggestion(
                    id=int(row[0]),
                    priority=row[1],  # type: ignore[arg-type]
                    title=row[2],
                    area=row[3],  # type: ignore[arg-type]
                    lift=float(row[4]),
                    explanation=row[5],
                    reference=ref,
                    examples=examples,
                    peak_start_s=float(row[8]) if row[8] is not None else None,
                    peak_end_s=float(row[9]) if row[9] is not None else None,
                )
            )

        return SuggestionPlan(
            score=float(comp[0]),
            benchmark=float(comp[1]),
            delta=float(comp[2]),
            status=comp[3],  # type: ignore[arg-type]
            regions=regions,
            suggestions=suggestions,
        )

    # --------------------------------------------------------- list_for_user

    def list_for_user(
        self,
        user_id: str | None,
        limit: int = 20,
    ) -> list[PastRun]:
        # `IS NOT DISTINCT FROM` lets us match anonymous (NULL) runs
        # without a separate code path for the dev sentinel UUID.
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.name, r.kind, r.created_at, c.score
                FROM runs r
                LEFT JOIN composites c ON c.run_id = r.id
                WHERE r.user_id IS NOT DISTINCT FROM %s
                  AND r.status = 'complete'
                ORDER BY r.created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()

        out: list[PastRun] = []
        for row in rows:
            score = float(row[4]) if row[4] is not None else 0.0
            out.append(
                PastRun(
                    id=row[0],
                    name=row[1],
                    date=_format_date(row[3]),
                    kind=row[2],
                    score=round(score, 0),
                )
            )
        return out

    # ----------------------------------------------------------- previous_score

    def previous_score(
        self,
        user_id: str | None,
        exclude_id: str,
    ) -> float | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.score
                FROM runs r
                JOIN composites c ON c.run_id = r.id
                WHERE r.user_id IS NOT DISTINCT FROM %s
                  AND r.id != %s
                  AND r.status = 'complete'
                ORDER BY r.created_at DESC
                LIMIT 1
                """,
                (user_id, exclude_id),
            )
            row = cur.fetchone()
        return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Postgres `timestamptz` always returns aware datetimes, but
        # belt-and-braces in case the column is `timestamp` without tz.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _decode_reference(raw: Any) -> Reference | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return Reference(**raw)
    return Reference(**json.loads(raw))


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------


def _make_store():
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return _PostgresRunStore(dsn)
    return _InMemoryRunStore()


# Singleton — imported as `from services.persistence.runs_v2 import RUN_STORE`.
RUN_STORE = _make_store()


__all__ = ["RUN_STORE", "RunRecord", "RunStatus", "SuggestionPlan"]
