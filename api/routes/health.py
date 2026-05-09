"""GET /health — liveness probes.

  * `/health`                  — Railway healthcheck target. Always 2xx.
  * `/health/social_context`   — GraphRAG subsystem snapshot. Always 2xx
                                 with a structured payload; external
                                 monitoring fires on field thresholds
                                 (stale `last_snapshot_at`, high
                                 `fetch_fallback_rate`, etc.).
"""

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/social_context")
def social_context_health() -> dict[str, Any]:
    """Always-2xx snapshot of the Phase 2 GraphRAG subsystem.

    Returns the same shape regardless of `TRENDS_MODE` so monitoring
    doesn't have to special-case mock-mode deployments — counts will
    just be zero and `last_snapshot_at` will be null.
    """
    from services.social_context.health import (
        social_context_health as build_snapshot,
    )

    return build_snapshot()
