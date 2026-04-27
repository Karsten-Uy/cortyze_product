"""LLM client protocol — what every implementation must satisfy."""

from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    """Single-method LLM interface for the suggestion engine.

    `chat_json` sends a system + user prompt and demands a JSON object or
    array as response. Implementations are responsible for parsing,
    retry-on-parse-failure, and prompt-cache headers. The contract:

    - `system`: the cacheable rules + context block
    - `user`:   the per-rule, per-content question
    - returns: parsed JSON (dict or list); raises on unrecoverable error
    """

    def chat_json(self, *, system: str, user: str) -> Any:
        ...
