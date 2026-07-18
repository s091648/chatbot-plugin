from __future__ import annotations

import logging

import anthropic

from chatbot_plugin.llm.base import LLMResult, ToolCallRequest, ToolSpec

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
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system = "\n\n".join(system_parts) or None
        claude_messages = [self._to_claude_message(m) for m in messages if m.get("role") != "system"]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET},
            "messages": claude_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        response = await self._client.messages.create(**kwargs)
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
        tool_calls: list[ToolCallRequest] = []
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                reply += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, arguments=block.input))

        return LLMResult(thinking=thinking, text=reply, tool_calls=tool_calls)

    @staticmethod
    def _to_claude_message(message: dict) -> dict:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            return {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                    for tc in message["tool_calls"]
                ],
            }
        if role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                        "is_error": message.get("is_error", False),
                    }
                ],
            }
        return {"role": role, "content": message["content"]}
