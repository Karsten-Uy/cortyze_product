"""Verify all configured secrets are present + functional.

To set a new variable do something like the following:

# Replace 'ghp_xxxxx' with the actual value (you'll paste once, never again)
security add-generic-password -s 'cortyze-github-pat' -a $USER -w 'ghp_xxxxx'
security add-generic-password -s 'cortyze-hf-token' -a $USER -w 'hf_xxxxx'
security add-generic-password -s 'cortyze-runpod-api' -a $USER -w 'YOUR_RUNPOD_KEY'
security add-generic-password -s 'cortyze-r2-secret' -a $USER -w 'YOUR_R2_SECRET'
security add-generic-password -s 'cortyze-supabase-db' -a $USER -w 'postgresql://...'

Run after sourcing your secrets (e.g. `cortyze-secrets` shell function) to
confirm everything's wired without exposing values:

    cortyze-secrets
    uv run python scripts/check_secrets.py

Each check is independent and tolerant of missing config — unset secrets
are reported as "skipping" rather than failures, so you can run this
during early setup when only some services are configured.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Load non-secret config from .env so vars like R2_ACCOUNT_ID and
# R2_ACCESS_KEY (which live in .env, not Keychain) are visible. Keychain-
# loaded secrets (R2_SECRET_KEY, etc.) take precedence because override=False.
#
# Also loads the FRONTEND .env.local — the NEXT_PUBLIC_SUPABASE_URL +
# NEXT_PUBLIC_SUPABASE_ANON_KEY live there (they're public values, safe to
# read into the same process for the purposes of this check).
try:
    from dotenv import load_dotenv

    _BACKEND_ENV = Path(__file__).resolve().parent.parent / ".env"
    if _BACKEND_ENV.exists():
        load_dotenv(_BACKEND_ENV, override=False)

    _FRONTEND_ENV = (
        Path(__file__).resolve().parent.parent.parent
        / "cortyze_frontend"
        / ".env.local"
    )
    if _FRONTEND_ENV.exists():
        load_dotenv(_FRONTEND_ENV, override=False)
except ImportError:
    # python-dotenv is in pyproject.toml deps; only an issue in stripped envs
    pass


# ANSI color codes; auto-disable when stdout isn't a terminal
_ISATTY = sys.stdout.isatty()


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ISATTY else text


def _green(s: str) -> str:
    return _color("32", s)


def _red(s: str) -> str:
    return _color("31", s)


def _yellow(s: str) -> str:
    return _color("33", s)


def _dim(s: str) -> str:
    return _color("2", s)


_PRESENT = "present"
_WORKING = "working"
_NOT_SET = "not_set"
_FAILED = "failed"


def _print(name: str, status: str, detail: str = "") -> None:
    if status == _NOT_SET:
        print(f"  {_yellow('○')} {name:24} {_dim('not set, skipping')}")
    elif status == _WORKING:
        print(f"  {_green('✓')} {name:24} {detail}")
    elif status == _PRESENT:
        print(f"  {_yellow('~')} {name:24} {_dim('present but not tested')}  {detail}")
    elif status == _FAILED:
        print(f"  {_red('✗')} {name:24} {detail}")


def _has(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


_USER_AGENT = "cortyze-secret-check/1.0"


def _http_json(url: str, headers: dict, timeout: int = 10) -> dict | list:
    # Default urllib User-Agent ("Python-urllib/3.x") trips Cloudflare bot
    # detection — RunPod, GitHub, and HF all sit behind WAFs that 403 it.
    full_headers = {"User-Agent": _USER_AGENT, "Accept": "application/json", **headers}
    req = urllib.request.Request(url, headers=full_headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def check_github_pat() -> tuple[str, str]:
    if not _has("GITHUB_PAT"):
        return _NOT_SET, ""
    try:
        data = _http_json(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {os.environ['GITHUB_PAT']}"},
        )
        login = data.get("login", "?")
        return _WORKING, f"authenticated as @{login}"
    except urllib.error.HTTPError as e:
        return _FAILED, f"HTTP {e.code} from github.com/user"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_hf_token() -> tuple[str, str]:
    if not _has("HF_TOKEN"):
        return _NOT_SET, ""
    try:
        data = _http_json(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
        )
        name = data.get("name") or data.get("fullname") or "?"
        return _WORKING, f"authenticated as {name}"
    except urllib.error.HTTPError as e:
        return _FAILED, f"HTTP {e.code} from huggingface.co"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_anthropic() -> tuple[str, str]:
    if not _has("ANTHROPIC_API_KEY"):
        return _NOT_SET, ""
    try:
        # GET /v1/models is auth-required but free — no tokens consumed.
        data = _http_json(
            "https://api.anthropic.com/v1/models?limit=1",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
        )
        count = len(data.get("data", []))
        return _WORKING, f"auth ok ({count} model(s) visible)"
    except urllib.error.HTTPError as e:
        return _FAILED, f"HTTP {e.code} from api.anthropic.com"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_runpod() -> tuple[str, str]:
    if not _has("RUNPOD_API_KEY"):
        return _NOT_SET, ""
    try:
        body = json.dumps({"query": "{ myself { id } }"}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("errors"):
            return _FAILED, f"GraphQL: {data['errors'][0].get('message', '?')}"
        if not data.get("data", {}).get("myself"):
            return _FAILED, "no `myself` in response — auth likely rejected"
        return _WORKING, "GraphQL myself OK"
    except urllib.error.HTTPError as e:
        return _FAILED, f"HTTP {e.code} from api.runpod.io"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_r2() -> tuple[str, str]:
    needed = ["R2_ACCESS_KEY", "R2_SECRET_KEY"]
    if not all(_has(n) for n in needed):
        return _NOT_SET, ""
    if not _has("R2_ACCOUNT_ID") and not _has("S3_ENDPOINT_URL"):
        return _FAILED, "need R2_ACCOUNT_ID or S3_ENDPOINT_URL (for MinIO)"
    try:
        import boto3
        from botocore.config import Config

        endpoint = os.environ.get("S3_ENDPOINT_URL") or (
            f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            config=Config(signature_version="s3v4", connect_timeout=10),
            region_name="auto",
        )
        flavor = "MinIO" if "localhost" in endpoint else "R2"
        # Prefer head_bucket on a configured bucket — scoped R2 tokens deny
        # account-wide ListBuckets but allow per-bucket operations. Fall
        # back to list_buckets only if no bucket is configured (e.g. MinIO
        # admin or a not-yet-provisioned setup).
        probe_bucket = os.environ.get("R2_BUCKET_UPLOADS")
        if probe_bucket:
            client.head_bucket(Bucket=probe_bucket)
            return _WORKING, f"{flavor} auth ok (verified via {probe_bucket})"
        result = client.list_buckets()
        names = [b["Name"] for b in result.get("Buckets", [])]
        bucket_str = ", ".join(names) if names else "(none)"
        return _WORKING, f"{flavor} reachable, {len(names)} bucket(s): {bucket_str}"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:120]


def check_r2_buckets_exist() -> tuple[str, str]:
    """Verify the two named buckets actually exist (separate check)."""
    if not _has("R2_BUCKET_UPLOADS") or not _has("R2_BUCKET_PREDICTIONS"):
        return _NOT_SET, ""
    if not _has("R2_ACCESS_KEY"):
        return _NOT_SET, ""
    try:
        import boto3
        from botocore.config import Config

        endpoint = os.environ.get("S3_ENDPOINT_URL") or (
            f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com"
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            config=Config(signature_version="s3v4", connect_timeout=10),
            region_name="auto",
        )
        for bucket in (os.environ["R2_BUCKET_UPLOADS"], os.environ["R2_BUCKET_PREDICTIONS"]):
            client.head_bucket(Bucket=bucket)
        return _WORKING, "both named buckets reachable"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:120]


def check_supabase_url() -> tuple[str, str]:
    """Verify NEXT_PUBLIC_SUPABASE_URL points at a real Supabase project."""
    if not _has("NEXT_PUBLIC_SUPABASE_URL"):
        return _NOT_SET, ""
    url = os.environ["NEXT_PUBLIC_SUPABASE_URL"].rstrip("/")
    # Basic URL shape sanity — Supabase project URLs are always
    # https://<ref>.supabase.co (or self-hosted). Catch typos early.
    if not url.startswith("http"):
        return _FAILED, f"URL must start with https:// (got: {url[:30]})"
    # Supabase's /auth/v1/settings requires the anon key as the apikey
    # header. Without it we get 401, which is correct behavior but not
    # what we want for a connectivity check.
    headers: dict = {}
    if _has("NEXT_PUBLIC_SUPABASE_ANON_KEY"):
        headers["apikey"] = os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"]
    try:
        data = _http_json(f"{url}/auth/v1/settings", headers=headers)
        if isinstance(data, dict):
            providers = [
                k.replace("external_", "")
                for k, v in data.items()
                if k.startswith("external_") and v is True
            ]
            providers_msg = (
                f"{len(providers)} OAuth provider(s)"
                if providers
                else "email-only"
            )
            return _WORKING, f"reachable · {providers_msg}"
        return _WORKING, "reachable"
    except urllib.error.HTTPError as e:
        if e.code == 401 and not headers:
            return _FAILED, "401 — set NEXT_PUBLIC_SUPABASE_ANON_KEY too"
        return _FAILED, f"HTTP {e.code} from {url}/auth/v1/settings"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_supabase_anon_key() -> tuple[str, str]:
    """Verify NEXT_PUBLIC_SUPABASE_ANON_KEY is a valid JWT signed by this project.

    The anon key IS a JWT (issued by Supabase, signed with the project's
    JWT secret). Decoding without verification confirms shape + project
    ref; verifying with the secret confirms the key really belongs to
    this project (not a copy-paste from another project).
    """
    if not _has("NEXT_PUBLIC_SUPABASE_ANON_KEY"):
        return _NOT_SET, ""
    try:
        import jwt
    except ImportError:
        return _FAILED, "PyJWT not installed (uv sync --extra auth)"

    key = os.environ["NEXT_PUBLIC_SUPABASE_ANON_KEY"]
    try:
        payload = jwt.decode(key, options={"verify_signature": False})
    except Exception as e:
        return _FAILED, f"not a valid JWT: {type(e).__name__}: {e}"[:80]
    role = payload.get("role")
    iss = payload.get("iss", "")
    if role != "anon":
        return _FAILED, f"expected role='anon' got role='{role}'"

    # Cross-check: project ref in the iss claim should match the URL.
    if _has("NEXT_PUBLIC_SUPABASE_URL"):
        url = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
        if iss and iss not in url and "/auth/v1" not in iss:
            # iss looks like https://<ref>.supabase.co/auth/v1
            ref_in_iss = iss.replace("https://", "").split(".")[0]
            ref_in_url = url.replace("https://", "").split(".")[0]
            if ref_in_iss and ref_in_url and ref_in_iss != ref_in_url:
                return (
                    _FAILED,
                    f"anon key is for project '{ref_in_iss}' but URL is '{ref_in_url}'",
                )

    # Cross-check: if SUPABASE_JWT_SECRET is set, the key should verify
    # against it. This is the strongest test — proves both values come
    # from the same project.
    if _has("SUPABASE_JWT_SECRET"):
        try:
            jwt.decode(
                key,
                os.environ["SUPABASE_JWT_SECRET"],
                algorithms=["HS256"],
                # The anon key uses aud='authenticated' or none; tolerate both.
                options={"verify_aud": False},
            )
            return _WORKING, "anon JWT verified against JWT secret"
        except jwt.InvalidSignatureError:
            return (
                _FAILED,
                "JWT signature mismatch — SUPABASE_JWT_SECRET and anon key are for different projects",
            )
        except Exception as e:
            return _PRESENT, f"shape ok; signature check failed: {e}"[:80]

    return _PRESENT, f"role={role}, iss={iss[:40]}"


def check_supabase_jwt_secret() -> tuple[str, str]:
    """Verify SUPABASE_JWT_SECRET is set and usable for signing/verification."""
    if not _has("SUPABASE_JWT_SECRET"):
        return _NOT_SET, ""
    try:
        import jwt
    except ImportError:
        return _FAILED, "PyJWT not installed (uv sync --extra auth)"
    secret = os.environ["SUPABASE_JWT_SECRET"]
    if len(secret) < 20:
        return _FAILED, f"secret looks too short ({len(secret)} chars)"
    try:
        # Round-trip a synthetic token through the secret. If both sides
        # work, the secret is viable for HS256 verification.
        token = jwt.encode(
            {"sub": "test", "aud": "authenticated"},
            secret,
            algorithm="HS256",
        )
        jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        return _WORKING, "HS256 sign+verify round-trip ok"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:80]


def check_database() -> tuple[str, str]:
    if not _has("DATABASE_URL"):
        return _NOT_SET, ""
    try:
        import psycopg

        with psycopg.connect(
            os.environ["DATABASE_URL"], connect_timeout=10
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0].split(",")[0]
                # Validate every table our migrations should have created.
                expected = {
                    "reports": "001_reports.sql",
                    "ad_campaigns": "002_campaigns_and_context.sql",
                }
                missing: list[str] = []
                for tbl, mig in expected.items():
                    cur.execute(
                        "SELECT to_regclass(%s)::text", (f"public.{tbl}",)
                    )
                    if not cur.fetchone()[0]:
                        missing.append(f"{tbl} (run {mig})")

                # Stage 3 columns added by 002. If reports exists but
                # campaign_id doesn't, migration 002 hasn't been applied.
                if "reports" not in [m.split(" ")[0] for m in missing]:
                    cur.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='reports'
                          AND column_name IN ('campaign_id', 'additional_context', 'title')
                        """
                    )
                    have = {r[0] for r in cur.fetchall()}
                    needed = {"campaign_id", "additional_context", "title"}
                    if not needed.issubset(have):
                        missing.append(
                            f"reports.{','.join(sorted(needed - have))} cols (run 002_campaigns_and_context.sql)"
                        )

        if missing:
            return _FAILED, f"{version} · MISSING: {'; '.join(missing)}"
        return _WORKING, f"{version} · all tables + Stage 3 columns present"
    except Exception as e:
        return _FAILED, f"{type(e).__name__}: {e}"[:120]


def main() -> int:
    print()
    print(_dim("Cortyze secret check — verifies presence + functionality."))
    print(_dim("No secret values are printed; each check shows only pass/fail."))
    print()

    inference = os.environ.get("INFERENCE_MODE", "mock").strip().lower()
    suggestions = os.environ.get("ENABLE_SUGGESTIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    llm_mode = os.environ.get("SUGGESTION_LLM_MODE", "mock").strip().lower()
    auth_disabled = os.environ.get("AUTH_DISABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    auth_label = "DISABLED (dev only)" if auth_disabled else (
        "configured" if os.environ.get("SUPABASE_JWT_SECRET") else "off"
    )

    print("Configuration (non-secret):")
    print(f"  inference mode:      {inference}")
    print(f"  suggestions:         {'enabled' if suggestions else 'disabled'} (mode={llm_mode})")
    print(f"  S3 endpoint:         {os.environ.get('S3_ENDPOINT_URL') or '(R2 production)'}")
    print(f"  auth:                {auth_label}")
    print()

    checks = [
        ("GITHUB_PAT", check_github_pat),
        ("HF_TOKEN", check_hf_token),
        ("ANTHROPIC_API_KEY", check_anthropic),
        ("RUNPOD_API_KEY", check_runpod),
        ("R2/MinIO auth", check_r2),
        ("R2 named buckets", check_r2_buckets_exist),
        ("DATABASE_URL", check_database),
        ("SUPABASE_JWT_SECRET", check_supabase_jwt_secret),
        ("SUPABASE_URL (frontend)", check_supabase_url),
        ("SUPABASE_ANON_KEY", check_supabase_anon_key),
    ]

    print("Secrets:")
    failures = 0
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = _FAILED, f"check itself crashed: {e}"
        _print(name, status, detail)
        if status == _FAILED:
            failures += 1

    print()
    if failures == 0:
        print(_green("All configured secrets verified."))
    else:
        print(
            _red(
                f"{failures} secret(s) failed. Fix before running the RunPod session."
            )
        )
    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
