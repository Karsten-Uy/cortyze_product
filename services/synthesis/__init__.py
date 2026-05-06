"""Phase 3 — suggestion plan synthesis (Claude).

Produces a `SuggestionPlan` matching the v2 frontend contract. Two
implementations:

  * `mock`     — deterministic, free, used in dev + tests.
  * `claude`   — Anthropic API (TODO: real client; falls through to
                 mock with a warning until the live path is wired in).

The legacy `services.suggestions` module is unchanged — that's what
the `/analyze` flow uses. This module is only consumed by the new
`/runs` flow.
"""

from __future__ import annotations

import logging
import os

from .protocol import SynthesisClient, SynthesisInput  # noqa: F401  re-exported

_log = logging.getLogger("cortyze.synthesis")


def get_client() -> SynthesisClient:
    """Return a `SynthesisClient` based on `SYNTHESIS_MODE`.

    Modes:
      * `mock` (default) — templated SuggestionPlan, no I/O.
      * `claude` — Anthropic-backed synthesis (real implementation).

    Note: `SYNTHESIS_MODE` is independent of the legacy
    `SUGGESTION_LLM_MODE` (used by /analyze). Both can be set
    differently if you want the v2 flow on Claude while keeping
    /analyze on a mock during development.
    """
    mode = os.environ.get("SYNTHESIS_MODE", "mock").strip().lower()

    if mode == "mock":
        from .mock import MockSynthesisClient

        return MockSynthesisClient()

    if mode == "claude":
        from .claude import ClaudeSynthesisClient

        return ClaudeSynthesisClient()

    _log.warning("unknown SYNTHESIS_MODE=%r; falling back to mock", mode)
    from .mock import MockSynthesisClient

    return MockSynthesisClient()
