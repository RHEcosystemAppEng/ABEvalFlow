"""Unified LLM client using the OpenAI-compatible API.

Supports all three pipeline LLM modes through a single interface:
  - LiteLLM proxy (default): exposes /v1/chat/completions
  - Direct API: OpenAI / Anthropic keys via OpenAI-compatible SDK
  - Self-hosted: vLLM / ollama expose OpenAI-compatible endpoints

Configuration is resolved from environment variables so that Tekton tasks
can inject values from Secrets and ConfigMaps.
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://litellm.ab-eval-flow.svc:4000/v1"
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _resolve_config() -> dict:
    """Build client kwargs from environment variables."""
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("LLM_API_KEY", "not-set")
    return {"base_url": base_url, "api_key": api_key}


def get_client() -> OpenAI:
    """Return a configured OpenAI client."""
    cfg = _resolve_config()
    logger.info("LLM client → base_url=%s", cfg["base_url"])
    return OpenAI(**cfg)


def get_model() -> str:
    """Return the model identifier from env or default."""
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Raises on API errors so callers can handle retries at a higher level.
    """
    client = get_client()
    resolved_model = model or get_model()

    logger.info(
        "chat_completion → model=%s, messages=%d, temperature=%.1f",
        resolved_model,
        len(messages),
        temperature,
    )

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    content = response.choices[0].message.content
    logger.info("chat_completion ← %d chars", len(content) if content else 0)
    return content or ""
