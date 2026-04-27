"""OpenAI-compatible LLM client.

Works against any service that speaks the OpenAI Chat Completions API:
  - OpenAI itself           (LLM_BASE_URL=https://api.openai.com/v1)
  - vLLM (self-hosted)      (LLM_BASE_URL=http://your-pod:8001/v1)
  - Ollama (local)          (LLM_BASE_URL=http://localhost:11434/v1)
  - Groq                    (LLM_BASE_URL=https://api.groq.com/openai/v1)
  - Together AI / Anyscale  (LLM_BASE_URL=https://api.together.xyz/v1)
  - Anthropic OpenAI-compat (LLM_BASE_URL=https://api.anthropic.com/v1, beta)

Same env vars across all of them — change `LLM_BASE_URL` + `LLM_MODEL` +
`LLM_API_KEY` and the suggestion engine retargets without code changes.

Lazy-imports `openai` so the package doesn't need to be installed unless
the user actually flips SUGGESTION_LLM_MODE to openai_compatible.
"""

from __future__ import annotations

import json
import os
from typing import Any


class OpenAICompatibleLLMClient:
    def __init__(self) -> None:
        try:
            from openai import OpenAI  # lazy
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run: uv sync --extra suggestions"
            ) from e

        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "LLM_BASE_URL not set (e.g. http://localhost:11434/v1 for Ollama)"
            )
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("LLM_API_KEY", "unused"),
        )
        self._model = os.environ["LLM_MODEL"]
        self._temperature = float(os.environ.get("LLM_TEMPERATURE", "0.5"))

    def chat_json(self, *, system: str, user: str) -> Any:
        # `response_format={"type": "json_object"}` is supported by OpenAI,
        # vLLM, Ollama (recent), and most others. If your provider doesn't,
        # the call still usually returns valid JSON because the system
        # prompt asks for it.
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback for providers that reject response_format
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
            )

        content = resp.choices[0].message.content or "{}"
        return _parse_json(content)


def _parse_json(content: str) -> Any:
    """Try strict JSON; fall back to extracting the first {...} or [...] block."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some models wrap output in markdown fences; strip and retry
        stripped = content.strip()
        for prefix in ("```json", "```"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
        return json.loads(stripped)
