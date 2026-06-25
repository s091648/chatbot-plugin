from __future__ import annotations

import logging

import anthropic

logger = logging.getLogger(__name__)

_THINKING_BUDGET = 1024


class ClaudeProvider:
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6-20250514") -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> tuple[str | None, str]:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
            messages=messages,
        )
        logger.info(
            "claude_api_called",
            extra={
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

        thinking: str | None = None
        reply = ""
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                reply += block.text

        return (thinking, reply)
