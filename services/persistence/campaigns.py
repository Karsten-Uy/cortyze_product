"""Supabase Postgres `ad_campaigns` table CRUD.

Stage 3 addition. A campaign is a folder grouping runs by ad-brief /
product-launch / experiment. The frontend sidebar collapses runs under
their campaigns; if no campaign is set, runs appear under "Uncategorized".

Same lazy-init pattern as reports.py — boots in mock mode without
DATABASE_URL, returns None from get_store() in that case.
"""

from __future__ import annotations

import os
from datetime import datetime
from uuid import UUID, uuid4

from core.schemas import Campaign, CampaignSummary


class CampaignsStore:
    def __init__(self) -> None:
        import psycopg

        # See reports.py for why prepare_threshold=None — Supabase pgbouncer
        # in transaction-pooling mode breaks psycopg's prepared-statement
        # cache. Disable it so every call sends the SQL as a simple query.
        self._conn = psycopg.connect(
            os.environ["DATABASE_URL"],
            autocommit=True,
            prepare_threshold=None,
        )

    def create(
        self, *, user_id: str, name: str, description: str | None = None
    ) -> Campaign:
        campaign_id = str(uuid4())
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ad_campaigns (id, user_id, name, description)
                VALUES (%s, %s, %s, %s)
                RETURNING created_at
                """,
                (campaign_id, user_id, name, description),
            )
            (created_at,) = cur.fetchone()
        return Campaign(
            id=campaign_id,
            user_id=user_id,
            name=name,
            description=description,
            created_at=created_at.isoformat(),
        )

    def get(self, campaign_id: str, *, user_id: str) -> Campaign | None:
        """Get a single campaign, scoped to the user. Returns None if not
        found or if it belongs to another user (avoids leaking existence)."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, name, description, created_at
                FROM ad_campaigns
                WHERE id = %s AND user_id = %s
                """,
                (campaign_id, user_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Campaign(
            id=str(row[0]),
            user_id=str(row[1]),
            name=row[2],
            description=row[3],
            created_at=row[4].isoformat(),
        )

    def list_for_user(self, user_id: str) -> list[CampaignSummary]:
        """List campaigns with run-count aggregation for the sidebar.

        Joins against `reports` to give each campaign a run_count and
        last_run_at — without these the sidebar can't sort campaigns
        sensibly or show "no runs yet" empty states.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id, c.name, c.description,
                    count(r.request_id) as run_count,
                    max(r.created_at) as last_run_at
                FROM ad_campaigns c
                LEFT JOIN reports r ON r.campaign_id = c.id
                WHERE c.user_id = %s
                GROUP BY c.id, c.name, c.description, c.created_at
                ORDER BY c.created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            CampaignSummary(
                id=str(r[0]),
                name=r[1],
                description=r[2],
                run_count=int(r[3]),
                last_run_at=r[4].isoformat() if r[4] else None,
            )
            for r in rows
        ]

    def update(
        self,
        campaign_id: str,
        *,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Campaign | None:
        """Update name / description. Returns None if not found or not owned."""
        # Build a minimal SET clause so we don't clobber unset fields.
        sets: list[str] = []
        params: list = []
        if name is not None:
            sets.append("name = %s")
            params.append(name)
        if description is not None:
            sets.append("description = %s")
            params.append(description)
        if not sets:
            return self.get(campaign_id, user_id=user_id)
        params.extend([campaign_id, user_id])
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE ad_campaigns SET {', '.join(sets)}
                WHERE id = %s AND user_id = %s
                """,
                tuple(params),
            )
        return self.get(campaign_id, user_id=user_id)

    def delete(self, campaign_id: str, *, user_id: str) -> bool:
        """Hard-delete the campaign. `reports.campaign_id` is set null via
        the FK constraint (ON DELETE SET NULL) so historical runs survive,
        just lose their grouping. Returns True iff a row was deleted.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM ad_campaigns
                WHERE id = %s AND user_id = %s
                """,
                (campaign_id, user_id),
            )
            return cur.rowcount > 0


_store: CampaignsStore | None = None


def get_store() -> CampaignsStore | None:
    """Return CampaignsStore if DATABASE_URL set; None otherwise (mock mode)."""
    global _store
    if _store is not None:
        return _store
    if not os.environ.get("DATABASE_URL"):
        return None
    _store = CampaignsStore()
    return _store
