"""Supabase Postgres `reports` table CRUD.

Stage 1 had one table; Stage 3 adds `ad_campaigns` (see campaigns.py) and
extends `reports` with campaign / context / sidebar columns per migration
002. Raw SQL with psycopg 3 — no ORM yet.

The store is constructed lazily so the API boots in mock mode without
DATABASE_URL present.
"""

from __future__ import annotations

import json
import os

from core.schemas import BrainReport, ReportSummary
from core.scoring.goals import Goal


class ReportsStore:
    def __init__(self) -> None:
        import psycopg

        # `prepare_threshold=None` disables psycopg 3's automatic prepared-
        # statement caching. Required when DATABASE_URL points at Supabase's
        # pgbouncer (port 6543, transaction-pooling mode) — bouncer doesn't
        # share prepared-statement state across pooled backends, so statement
        # names collide on subsequent calls with `DuplicatePreparedStatement`.
        self._conn = psycopg.connect(
            os.environ["DATABASE_URL"],
            autocommit=True,
            prepare_threshold=None,
        )

    def insert(self, report: BrainReport) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (
                    request_id, user_id, goal, region_scores, overall_score,
                    model_version, raw_predictions_uri,
                    campaign_id, additional_context, caption_text,
                    thumbnail_url, title, content_type,
                    overall_by_goal, audio_url, image_count, seconds_per_image,
                    brain_image_uri
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    report.campaign_id,
                    report.additional_context,
                    report.caption_text,
                    report.thumbnail_url,
                    report.title,
                    report.content_type,
                    json.dumps(report.overall_by_goal) if report.overall_by_goal else None,
                    report.audio_url,
                    report.image_count,
                    report.seconds_per_image,
                    report.brain_image_uri,
                ),
            )

    def get(self, request_id: str) -> BrainReport | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT request_id, user_id, goal, region_scores,
                       overall_score, model_version, raw_predictions_uri,
                       campaign_id, additional_context, caption_text,
                       thumbnail_url, title, created_at, content_type,
                       overall_by_goal, audio_url, image_count, seconds_per_image,
                       brain_image_uri
                FROM reports WHERE request_id = %s
                """,
                (request_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        region_scores = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        overall_by_goal = row[14]
        if isinstance(overall_by_goal, str):
            overall_by_goal = json.loads(overall_by_goal)
        return BrainReport(
            # Postgres `uuid` columns surface as `UUID` objects via psycopg;
            # str-cast so Pydantic's `request_id: str` field validates.
            # Same treatment for any other UUID-typed columns below.
            request_id=str(row[0]),
            user_id=str(row[1]) if row[1] is not None else None,
            goal=Goal(row[2]),
            region_scores=region_scores,
            overall_score=row[4],
            model_version=row[5],
            raw_predictions_uri=row[6],
            campaign_id=str(row[7]) if row[7] else None,
            additional_context=row[8],
            caption_text=row[9],
            thumbnail_url=row[10],
            title=row[11],
            created_at=row[12].isoformat() if row[12] else None,
            content_type=row[13] or "video",
            overall_by_goal=overall_by_goal,
            audio_url=row[15],
            image_count=row[16],
            seconds_per_image=float(row[17]) if row[17] is not None else None,
            brain_image_uri=row[18],
            elapsed_ms=0,  # not persisted — historical reports have no fresh elapsed time
        )

    def list_for_user(
        self,
        user_id: str,
        *,
        campaign_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[ReportSummary], str | None]:
        """Return paginated sidebar-friendly summaries.

        Cursor is the ISO-8601 created_at of the last item returned;
        fetch the next page by passing it as `cursor`. Returns
        (items, next_cursor) where next_cursor is None when exhausted.
        """
        params: list = [user_id]
        where = ["user_id = %s"]
        if campaign_id:
            where.append("campaign_id = %s")
            params.append(campaign_id)
        if cursor:
            where.append("created_at < %s")
            params.append(cursor)
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT request_id, title, thumbnail_url, overall_score, goal,
                       coalesce(content_type, 'video') as content_type, campaign_id, created_at
                FROM reports
                WHERE {" AND ".join(where)}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        # `request_id` and `campaign_id` are `uuid` columns — cast to str
        # so Pydantic's str-typed fields validate.
        items = [
            ReportSummary(
                request_id=str(r[0]),
                title=r[1],
                thumbnail_url=r[2],
                overall_score=float(r[3]),
                goal=Goal(r[4]),
                content_type=r[5] or "video",
                campaign_id=str(r[6]) if r[6] is not None else None,
                created_at=r[7].isoformat() if r[7] else "",
            )
            for r in rows
        ]
        next_cursor = items[-1].created_at if len(items) == limit else None
        return items, next_cursor


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
