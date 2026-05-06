"""Phase 4 — validation swarm (MiroFish OASIS engine).

Mock-only in this build: a deterministic perturbation client that
nudges Phase 3's heuristic `lift` values so the output looks like it
was simulated. The protocol + factory exist so a real MiroFish-backed
implementation can drop in later under `VALIDATION_MODE=mirofish`.
"""

from __future__ import annotations

import logging
import os

from .protocol import ValidationClient  # noqa: F401  re-exported

_log = logging.getLogger("cortyze.validation")


def get_client() -> ValidationClient:
    """Return a `ValidationClient` based on `VALIDATION_MODE`.

    Modes:
      * `mock` (default) — deterministic perturbation, no I/O.
      * `mirofish` — TODO: real swarm-simulation client.
                     Raises NotImplementedError until built.
      * `passthrough` — no-op; returns the plan unchanged.
                       Useful for v1 deploys that ship without Phase 4.
    """
    mode = os.environ.get("VALIDATION_MODE", "mock").strip().lower()

    if mode == "mock":
        from .mock import MockValidationClient

        return MockValidationClient()

    if mode == "passthrough":
        from .mock import PassthroughValidationClient

        return PassthroughValidationClient()

    if mode == "mirofish":
        raise NotImplementedError(
            "VALIDATION_MODE=mirofish is not implemented yet. "
            "Set VALIDATION_MODE=mock or VALIDATION_MODE=passthrough."
        )

    _log.warning("unknown VALIDATION_MODE=%r; falling back to mock", mode)
    from .mock import MockValidationClient

    return MockValidationClient()
