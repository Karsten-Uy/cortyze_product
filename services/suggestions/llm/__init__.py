"""LLM client layer — the swap surface for the Stage 2 suggestion engine.

Three implementations, one Protocol. Pick which one runs via env var:

    SUGGESTION_LLM_MODE=mock                (default)
    SUGGESTION_LLM_MODE=openai_compatible   (OpenAI, vLLM, Ollama, Groq, Together, …)
    SUGGESTION_LLM_MODE=anthropic           (Claude, with prompt-caching support)

The `LLMClient` Protocol is the only thing higher-level code (`diagnose()`,
prompt builder) depends on. Adding a new provider = drop a new file in this
directory implementing the same `chat_json()` method, then register it in
`get_llm_client()` below. No callers change.
"""

from __future__ import annotations

import os

from .protocol import LLMClient


def get_llm_client() -> LLMClient:
    """Pick the LLM client based on SUGGESTION_LLM_MODE env var."""
    mode = os.environ.get("SUGGESTION_LLM_MODE", "mock").strip().lower()

    if mode == "mock":
        from .mock import MockLLMClient

        return MockLLMClient()

    if mode == "openai_compatible":
        from .openai_compatible import OpenAICompatibleLLMClient

        return OpenAICompatibleLLMClient()

    if mode == "anthropic":
        from .anthropic_native import AnthropicLLMClient

        return AnthropicLLMClient()

    raise ValueError(
        f"SUGGESTION_LLM_MODE={mode!r} not recognized. "
        "Use 'mock', 'openai_compatible', or 'anthropic'."
    )


__all__ = ["LLMClient", "get_llm_client"]
