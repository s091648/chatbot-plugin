from __future__ import annotations

import json
import logging

import httpx

from chatbot_plugin.llm.base import LLMResult, ToolCallRequest, ToolSpec

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    """OpenRouter LLM provider via OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._api_key = api_key

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        payload: dict = {
            "model": self.model,
            "messages": [self._to_openrouter_message(m) for m in messages],
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                _API_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        logger.info("openrouter_api_called", extra={"model": self.model})
        message = data["choices"][0]["message"]
        tool_calls = [
            ToolCallRequest(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {},
            )
            for tc in message.get("tool_calls") or []
        ]
        return LLMResult(thinking=None, text=message.get("content") or "", tool_calls=tool_calls)

    @staticmethod
    def _to_openrouter_message(message: dict) -> dict:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                    for tc in message["tool_calls"]
                ],
            }
        if role == "tool":
            return {"role": "tool", "tool_call_id": message["tool_call_id"], "content": message["content"]}
        return {"role": role, "content": message["content"]}
