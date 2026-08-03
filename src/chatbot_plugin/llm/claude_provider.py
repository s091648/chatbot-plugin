from __future__ import annotations

import json
import logging

import anthropic

from chatbot_plugin.llm.base import LLMResult, TextDelta, ThinkingDelta, ToolCallRequest, ToolSpec

logger = logging.getLogger(__name__)

_THINKING_BUDGET = 1024


class ClaudeProvider:
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6-20250514") -> None:
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def _build_kwargs(self, messages: list[dict], max_tokens: int, tools: list[ToolSpec] | None) -> dict:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system = "\n\n".join(system_parts) or None
        claude_messages = [self._to_claude_message(m) for m in messages if m.get("role") != "system"]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
        else:
            # Extended thinking requires the original signed thinking block(s) to be
            # carried forward on any tool-call round-trip continuation. This provider's
            # round-trip reconstruction (_to_claude_message) does not thread those
            # through, so thinking is disabled whenever tools are offered to avoid the
            # API rejecting the request with a 400.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": _THINKING_BUDGET}
        return kwargs

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        kwargs = self._build_kwargs(messages, max_tokens, tools)
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

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ):
        kwargs = self._build_kwargs(messages, max_tokens, tools)

        # index -> in-progress tool_use block being assembled from input_json_delta fragments
        tool_use_blocks: dict[int, dict] = {}
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        tool_use_blocks[event.index] = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "json": "",
                        }
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif delta.type == "thinking_delta":
                        yield ThinkingDelta(text=delta.thinking)
                    elif delta.type == "input_json_delta":
                        block = tool_use_blocks.get(event.index)
                        if block is not None:
                            block["json"] += delta.partial_json
                elif event.type == "content_block_stop":
                    block = tool_use_blocks.pop(event.index, None)
                    if block is not None:
                        try:
                            arguments = json.loads(block["json"]) if block["json"] else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        yield ToolCallRequest(id=block["id"], name=block["name"], arguments=arguments)

        logger.info("claude_stream_completed", extra={"model": self.model})

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
