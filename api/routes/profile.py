"""GET /me, PATCH /me — user profile metadata.

The Supabase JWT carries the user_id (`sub`) and email — we don't store
either of those on our side. The `profiles` table holds everything
else: display_name, avatar_url, timestamps. Frontend uses /me to
populate the top-nav avatar + dropdown.

Profile rows are lazy-created on first GET so signup doesn't need a
database trigger touching auth.users (which requires elevated perms
and complicates portable migrations).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from services.persistence.profiles import PROFILES_STORE, Profile

from ..auth import require_user
from ..limiter import limiter

router = APIRouter()


class MeResponse(BaseModel):
    """Outbound shape for GET /me — the profile row plus the email
    pulled straight from the JWT (we don't store email; it's the
    source of truth in Supabase Auth)."""

    user_id: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ProfilePatch(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)
    avatar_url: str | None = Field(default=None, max_length=2048)


def _to_response(profile: Profile, email: str | None) -> MeResponse:
    return MeResponse(
        user_id=profile.user_id,
        email=email,
        display_name=profile.display_name,
        avatar_url=profile.avatar_url,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.get("/me", response_model=MeResponse)
@limiter.limit("60/minute")
async def get_me(
    request: Request,
    user_id: str = Depends(require_user),
) -> MeResponse:
    email: str | None = getattr(request.state, "user_email", None)
    profile = PROFILES_STORE.get_or_create(user_id, email=email)
    return _to_response(profile, email)


@router.patch("/me", response_model=MeResponse)
@limiter.limit("30/minute")
async def patch_me(
    request: Request,
    body: ProfilePatch,
    user_id: str = Depends(require_user),
) -> MeResponse:
    email: str | None = getattr(request.state, "user_email", None)
    # Touch the row so display_name/avatar_url updates work even if the
    # caller never hit GET /me first.
    PROFILES_STORE.get_or_create(user_id, email=email)
    profile = PROFILES_STORE.update(
        user_id,
        display_name=body.display_name,
        avatar_url=body.avatar_url,
    )
    return _to_response(profile, email)
