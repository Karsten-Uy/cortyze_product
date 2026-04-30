"""Supabase JWT verification for FastAPI.

Stage 3 — gates `/analyze`, `/report/{id}`, `/reports`, `/campaigns`, and
`/compare` behind a valid Supabase Auth bearer token.

Supabase has two JWT-signing modes; we support both transparently:

  1. **Legacy HS256** — single shared HMAC secret (`SUPABASE_JWT_SECRET`).
     Used by older projects and by the `anon`/`service_role` API keys.

  2. **JWT Signing Keys** (ES256/RS256, asymmetric) — the modern path.
     User-session tokens for projects that have switched to "JWT Signing
     Keys" are signed with a private key held by Supabase; we fetch the
     public-key JWKS at `<project>/auth/v1/.well-known/jwks.json` and
     verify against it.

The dispatch is automatic: we peek at the `alg` claim in the token
header and pick the right verifier. Tokens with `alg=HS256` go through
the legacy path; everything else is verified via JWKS.

Env vars:
  SUPABASE_JWT_SECRET            HMAC secret (legacy / HS256 path).
                                 Settings → JWT Keys → Legacy JWT Secret.
  NEXT_PUBLIC_SUPABASE_URL       Project URL (used by the JWKS fetcher
                                 for ES256/RS256 verification). Same
                                 value the frontend uses.
  AUTH_DISABLED                  (optional dev escape hatch) when "true",
                                 auth is bypassed and a synthetic
                                 user_id is returned. NEVER in production.

Returns the user's UUID (`sub` claim) as a string.

Usage:
    from api.auth import require_user

    @router.post("/protected")
    async def handler(user_id: str = Depends(require_user)):
        ...
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Auto-error=False so we can return our own 401 with a more useful message
# (FastAPI's default is just "Not authenticated").
_bearer = HTTPBearer(auto_error=False)


def _is_dev_bypass() -> bool:
    return os.environ.get("AUTH_DISABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


_DEV_BYPASS_USER = "00000000-0000-0000-0000-000000000000"


_jwks_clients: dict[str, "object"] = {}


def _get_jwks_client(jwks_uri: str):
    """Cache PyJWKClient per URI. PyJWKClient handles HTTP fetching +
    in-memory key caching (default 5 min TTL) — we just memoize the
    client itself so we don't rebuild it on every request."""
    import jwt

    if jwks_uri not in _jwks_clients:
        _jwks_clients[jwks_uri] = jwt.PyJWKClient(jwks_uri, cache_keys=True)
    return _jwks_clients[jwks_uri]


def _verify_jwt(token: str) -> dict:
    """Dispatch HS256 vs JWKS-based verification by sniffing the alg
    header. Raises jwt.InvalidTokenError on any failure; caller maps to
    a 401."""
    import jwt

    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"malformed token header: {e}") from e

    alg = (unverified_header.get("alg") or "").upper()

    # Legacy HS256 path — uses the static project secret.
    if alg == "HS256":
        secret = os.environ.get("SUPABASE_JWT_SECRET")
        if not secret:
            raise jwt.InvalidTokenError(
                "server missing SUPABASE_JWT_SECRET; cannot verify HS256 token"
            )
        return jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
        )

    # Modern JWT Signing Keys path — fetch JWKS from the Supabase project.
    if alg in {"ES256", "RS256", "EdDSA"}:
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or os.environ.get(
            "SUPABASE_URL"
        )
        if not supabase_url:
            raise jwt.InvalidTokenError(
                f"server missing NEXT_PUBLIC_SUPABASE_URL; cannot fetch JWKS for {alg}"
            )
        jwks_uri = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        client = _get_jwks_client(jwks_uri)
        signing_key = client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=[alg],
            audience="authenticated",
        )

    raise jwt.InvalidTokenError(f"unsupported alg: {alg or '(missing)'}")


def require_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency that returns the caller's Supabase user_id.

    Order of resolution:
      1. AUTH_DISABLED=true → return the dev sentinel UUID (only for local
         development; never enable in prod).
      2. Authorization: Bearer <jwt> header → verify against either the
         legacy HS256 secret or the project's JWKS (auto-detected from
         the token's `alg` header). Return the `sub` claim.
      3. Otherwise → 401.

    The token is also stashed on `request.state.access_token` so downstream
    code can forward it to Supabase if it ever wants to call PostgREST
    with the user's RLS context (currently we use the service-role key,
    so this is unused but cheap to keep).
    """
    if _is_dev_bypass():
        request.state.user_id = _DEV_BYPASS_USER
        return _DEV_BYPASS_USER

    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <token> header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        import jwt  # PyJWT, pulled in via pyproject extra
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT not installed. Run: uv sync --extra auth",
        ) from e

    try:
        payload = _verify_jwt(creds.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token missing 'sub' claim",
        )

    request.state.user_id = user_id
    request.state.access_token = creds.credentials
    return user_id


def optional_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str | None:
    """Like require_user but returns None instead of raising on no token.

    Useful for routes that work both authenticated and anonymously
    (notably `/analyze` during the migration window — anonymous calls
    can still succeed, they just won't be persisted to the user's
    history). Once auth is fully rolled out, swap callers to require_user.
    """
    if _is_dev_bypass():
        return _DEV_BYPASS_USER
    if creds is None or not creds.credentials:
        return None
    try:
        return require_user(request, creds)
    except HTTPException:
        return None
