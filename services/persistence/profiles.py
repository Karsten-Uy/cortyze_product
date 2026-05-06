"""User profile CRUD against the `profiles` table.

Lazy-create on first read — the auth flow doesn't need a database
trigger on `auth.users`, which keeps the schema portable. Display name
defaults to the email username (`alice@x.com` → `alice`) so the UI has
something to render before the user explicitly sets it.

Falls back to in-memory when DATABASE_URL is unset, matching the
runs_v2 pattern. The legacy `/analyze` flow doesn't read profiles, so
this module is independent.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

_log = logging.getLogger("cortyze.persistence.profiles")


class Profile(BaseModel):
    user_id: str
    display_name: str | None = None
    avatar_url: str | None = None
    created_at: str | None = None  # ISO-8601
    updated_at: str | None = None


def _email_default(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    return email.split("@", 1)[0]


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class _InMemoryProfilesStore:
    def __init__(self) -> None:
        self._records: dict[str, Profile] = {}
        self._lock = threading.Lock()

    def get_or_create(self, user_id: str, *, email: str | None = None) -> Profile:
        with self._lock:
            existing = self._records.get(user_id)
            if existing is not None:
                return existing
            now = datetime.now(timezone.utc).isoformat()
            profile = Profile(
                user_id=user_id,
                display_name=_email_default(email),
                avatar_url=None,
                created_at=now,
                updated_at=now,
            )
            self._records[user_id] = profile
            return profile

    def update(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> Profile:
        with self._lock:
            existing = self._records.get(user_id)
            if existing is None:
                # Caller should have called get_or_create first; we
                # treat this as create-with-fields rather than erroring,
                # matching how Postgres ON CONFLICT DO UPDATE behaves.
                now = datetime.now(timezone.utc).isoformat()
                self._records[user_id] = Profile(
                    user_id=user_id,
                    display_name=display_name,
                    avatar_url=avatar_url,
                    created_at=now,
                    updated_at=now,
                )
                return self._records[user_id]
            updates: dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if display_name is not None:
                updates["display_name"] = display_name
            if avatar_url is not None:
                updates["avatar_url"] = avatar_url
            self._records[user_id] = existing.model_copy(update=updates)
            return self._records[user_id]


# ---------------------------------------------------------------------------
# Postgres backend
# ---------------------------------------------------------------------------


class _PostgresProfilesStore:
    def __init__(self, dsn: str) -> None:
        import psycopg

        self._dsn = dsn
        self._lock = threading.Lock()
        self._conn = psycopg.connect(
            dsn,
            autocommit=True,
            prepare_threshold=None,
        )
        _log.info(
            "PostgresProfilesStore connected: %s",
            self._dsn.split("@")[-1] if "@" in self._dsn else self._dsn,
        )

    def get_or_create(self, user_id: str, *, email: str | None = None) -> Profile:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profiles (user_id, display_name)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (user_id, _email_default(email)),
            )
            cur.execute(
                "SELECT user_id, display_name, avatar_url, "
                "       created_at, updated_at "
                "FROM profiles WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        if row is None:
            # Race that "shouldn't happen" — fall back to a synthetic
            # profile so the route doesn't 500.
            return Profile(user_id=user_id, display_name=_email_default(email))
        return _row_to_profile(row)

    def update(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        avatar_url: str | None = None,
    ) -> Profile:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profiles (user_id, display_name, avatar_url)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    display_name = COALESCE(EXCLUDED.display_name, profiles.display_name),
                    avatar_url   = COALESCE(EXCLUDED.avatar_url,   profiles.avatar_url)
                """,
                (user_id, display_name, avatar_url),
            )
            cur.execute(
                "SELECT user_id, display_name, avatar_url, "
                "       created_at, updated_at "
                "FROM profiles WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        if row is None:
            return Profile(user_id=user_id, display_name=display_name, avatar_url=avatar_url)
        return _row_to_profile(row)


def _row_to_profile(row: tuple) -> Profile:
    return Profile(
        user_id=str(row[0]),
        display_name=row[1],
        avatar_url=row[2],
        created_at=row[3].isoformat() if row[3] else None,
        updated_at=row[4].isoformat() if row[4] else None,
    )


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------


def _make_store():
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return _PostgresProfilesStore(dsn)
    return _InMemoryProfilesStore()


PROFILES_STORE = _make_store()


__all__ = ["Profile", "PROFILES_STORE"]
