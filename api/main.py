"""FastAPI app entry point."""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root before any module reads os.environ. Skipped when
# running under pytest so tests stay hermetic (no real DATABASE_URL leakage).
# `override=False` so real shell env vars still take precedence.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if "pytest" not in sys.modules:
    load_dotenv(_REPO_ROOT / ".env", override=False)

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from .limiter import limiter  # noqa: E402
from .routes import (  # noqa: E402
    analyze,
    campaigns,
    compare,
    examples,
    health,
    profile,
    regoal,
    runs,
    upload,
)

_log = logging.getLogger("cortyze.startup")


def _log_feature_flags() -> None:
    """Print which optional features picked up env vars. No secrets — just yes/no."""
    inference_mode = os.environ.get("INFERENCE_MODE", "mock").strip().lower()
    storage_mode = os.environ.get("STORAGE_MODE", "").strip().lower()
    if not storage_mode:
        # Legacy auto-detect for .env files that pre-date STORAGE_MODE.
        if not os.environ.get("R2_ACCESS_KEY"):
            storage_mode = "off"
        else:
            storage_mode = "minio" if os.environ.get("S3_ENDPOINT_URL") else "r2"
    storage = {"off": "off", "minio": "minio (local)", "r2": "r2"}.get(
        storage_mode, f"unknown ({storage_mode})"
    )
    persistence = "configured" if os.environ.get("DATABASE_URL") else "off"
    suggestions = "enabled" if (
        os.environ.get("ENABLE_SUGGESTIONS", "").strip().lower() in ("1", "true", "yes")
    ) else "disabled"
    suggestion_llm = os.environ.get("SUGGESTION_LLM_MODE", "mock").strip().lower()

    # v2 (`/runs`) modes — separate from the legacy /analyze suggestion engine.
    trends_mode = os.environ.get("TRENDS_MODE", "mock").strip().lower()
    synthesis_mode = os.environ.get("SYNTHESIS_MODE", "mock").strip().lower()
    validation_mode = os.environ.get("VALIDATION_MODE", "mock").strip().lower()

    auth_disabled = os.environ.get("AUTH_DISABLED", "").strip().lower() in (
        "1", "true", "yes",
    )
    auth = "DISABLED (dev only)" if auth_disabled else (
        "configured" if os.environ.get("SUPABASE_JWT_SECRET") else "off"
    )

    cors_origins = os.environ.get("FRONTEND_ORIGINS", "http://localhost:3000")
    _log.info("startup feature flags:")
    _log.info("  inference:      %s", inference_mode)
    _log.info("  object storage: %s", storage)
    _log.info("  persistence:    %s", persistence)
    _log.info("  auth:           %s", auth)
    _log.info("  suggestions:    %s (mode=%s)", suggestions, suggestion_llm)
    _log.info("  v2 trends:      %s", trends_mode)
    _log.info("  v2 synthesis:   %s", synthesis_mode)
    _log.info("  v2 validation:  %s", validation_mode)
    _log.info("  cors origins:   %s", cors_origins)


def create_app() -> FastAPI:
    app = FastAPI(title="Cortyze BrainScore", version="0.0.1")

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    origins = [
        o.strip()
        for o in os.environ.get("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(campaigns.router)
    app.include_router(compare.router)
    app.include_router(examples.router)
    app.include_router(regoal.router)
    app.include_router(runs.router)  # v2 pipeline (POST/GET /runs, SSE)
    app.include_router(upload.router)  # POST /upload-url (presigned R2 PUT)
    app.include_router(profile.router)  # GET/PATCH /me
    _log_feature_flags()
    return app


app = create_app()
