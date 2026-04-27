"""Native Anthropic SDK client with prompt caching.

Use this when you want Claude-specific features the OpenAI-compat layer
doesn't expose — primarily prompt caching, which makes Claude the
cheapest-per-call option once the system prompt warms up (90% discount on
cached tokens).

Lazy-imports `anthropic` so the package only needs to be installed when
SUGGESTION_LLM_MODE=anthropic.
"""

from __future__ import annotations

import json
import os
from typing import Any


class AnthropicLLMClient:
    def __init__(self) -> None:
        try:
            import anthropic  # lazy
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. Run: uv sync --extra suggestions"
            ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._max_tokens = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))
        self._temperature = float(os.environ.get("LLM_TEMPERATURE", "0.5"))

    def chat_json(self, *, system: str, user: str) -> Any:
        # Cache the system block — it's reused across every triggered rule
        # in a session, so the second-and-onward calls cost ~10% of the first.
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": user},
            ],
        )
        # Concatenate any text blocks in the response
        chunks = [
            block.text for block in message.content if hasattr(block, "text")
        ]
        text = "".join(chunks).strip()
        return _parse_json(text)


def _parse_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        stripped = content.strip()
        for prefix in ("```json", "```"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        return json.loads(stripped)
