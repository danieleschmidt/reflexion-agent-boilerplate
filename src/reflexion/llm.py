"""Thin wrapper around the Anthropic API.

Swap this out for OpenAI, Ollama, or any other backend by implementing
the same `complete(messages) -> str` interface.
"""

from __future__ import annotations

import os
from typing import Protocol

import anthropic


class LLMBackend(Protocol):
    """Any object that can complete a list of messages."""

    def complete(self, messages: list[dict], system: str | None = None) -> str:
        ...


class AnthropicBackend:
    """Claude via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        max_tokens: int = 2048,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, messages: list[dict], system: str | None = None) -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()
