"""Supabase Postgres `reports` table CRUD.

Stage 1 has one table per IMPLEMENTATION_PLAN.md §5.2. Raw SQL with psycopg
3 — no ORM yet, per §6.7. Stage 2 adds bounded contexts for suggestions/
examples but keeps this layer untouched.

The store is constructed lazily so the API boots in mock mode without
DATABASE_URL present.
"""

import json
import os

from core.schemas import BrainReport
from core.scoring.goals import Goal


class ReportsStore:
    def __init__(self) -> None:
        import psycopg

        self._conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)

    def insert(self, report: BrainReport) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (request_id, user_id, goal, region_scores,
                                     overall_score, model_version, raw_predictions_uri)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_id) DO NOTHING
                """,
                (
                    report.request_id,
                    report.user_id,
                    report.goal.value,
                    json.dumps(report.region_scores),
                    report.overall_score,
                    report.model_version,
                    report.raw_predictions_uri,
                ),
            )

    def get(self, request_id: str) -> BrainReport | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, user_id, goal, region_scores,
                       overall_score, model_version, raw_predictions_uri
                FROM reports WHERE request_id = %s
                """,
                (request_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        region_scores = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        return BrainReport(
            request_id=row[0],
            user_id=row[1],
            goal=Goal(row[2]),
            region_scores=region_scores,
            overall_score=row[4],
            model_version=row[5],
            raw_predictions_uri=row[6],
            elapsed_ms=0,  # not persisted — historical reports have no fresh elapsed time
        )


_store: ReportsStore | None = None


def get_store() -> ReportsStore | None:
    """Return ReportsStore if DATABASE_URL set; None otherwise (mock mode)."""
    global _store
    if _store is not None:
        return _store
    if not os.environ.get("DATABASE_URL"):
        return None
    _store = ReportsStore()
    return _store
