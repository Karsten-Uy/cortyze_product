"""Claude-backed SuggestionPlan synthesis.

Stub implementation. The mock client at `services.synthesis.mock` is the
canonical reference shape; this file is where the real Anthropic SDK
call goes once the prompt is dialed in.

For now, this falls back to the mock and logs a one-line warning so
operators get a clear signal that real synthesis isn't wired up yet.
The intent is that the file structure + factory wiring is in place so
swapping in the real implementation is a one-PR change.
"""

from __future__ import annotations

import logging

from core.schemas_v2 import SuggestionPlan

from .mock import MockSynthesisClient
from .protocol import SynthesisInput


_log = logging.getLogger("cortyze.synthesis.claude")
_warned_once = False


class ClaudeSynthesisClient:
    """Anthropic-backed synthesis. Currently delegates to mock.

    TODO: implement the real call.

      1. Build a system prompt + JSON schema (cacheable). Use
         `anthropic` SDK (already in pyproject `[suggestions]` extra).
      2. Construct the user prompt from `SynthesisInput`: region scores,
         their gaps to benchmark, the trend summary, the user's brief
         and caption, and the selected goal.
      3. Use `cache_control={"type": "ephemeral"}` on the system
         message so the cacheable bits don't pay full input cost on
         every call.
      4. Validate the response against `SuggestionPlan` (Pydantic).
         Retry once on `ValidationError`; on second failure, fall back
         to MockSynthesisClient and surface the failure in logs.
      5. The composite score / benchmark / delta come from
         `core.goals_v2`, NOT from Claude — Claude only decides
         suggestions.
    """

    def __init__(self) -> None:
        self._mock = MockSynthesisClient()

    def synthesize(self, payload: SynthesisInput) -> SuggestionPlan:
        global _warned_once
        if not _warned_once:
            _log.warning(
                "ClaudeSynthesisClient is a stub — falling back to mock. "
                "Set SYNTHESIS_MODE=mock to silence this warning, or "
                "implement the real call (see TODO in services/synthesis/claude.py)."
            )
            _warned_once = True
        return self._mock.synthesize(payload)
