from __future__ import annotations

import json
import logging

import httpx

from chatbot_plugin.llm.base import LLMResult, TextDelta, ThinkingDelta, ToolCallRequest, ToolSpec

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
            # No-op for models that don't support reasoning — OpenRouter's unified API just
            # drops unsupported params rather than erroring — but surfaces it for the ones
            # that do (routed provider must actually support extended thinking).
            "reasoning": {"enabled": True},
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
                arguments=self._parse_tool_arguments(tc["function"].get("arguments")),
            )
            for tc in message.get("tool_calls") or []
        ]
        return LLMResult(thinking=message.get("reasoning") or None, text=message.get("content") or "", tool_calls=tool_calls)

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ):
        payload: dict = {
            "model": self.model,
            "messages": [self._to_openrouter_message(m) for m in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "reasoning": {"enabled": True},
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ]

        # index -> in-progress tool call being assembled from OpenAI-style streamed fragments
        tool_calls: dict[int, dict] = {}
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                _API_URL,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield TextDelta(text=content)
                    reasoning = delta.get("reasoning")
                    if reasoning:
                        yield ThinkingDelta(text=reasoning)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        entry = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["name"] = fn["name"]
                        if fn.get("arguments"):
                            entry["arguments"] += fn["arguments"]

        for idx, entry in tool_calls.items():
            yield ToolCallRequest(
                id=entry["id"] or f"call_{idx}",
                name=entry["name"] or "",
                arguments=self._parse_tool_arguments(entry["arguments"]),
            )

        logger.info("openrouter_stream_completed", extra={"model": self.model})

    @staticmethod
    def _parse_tool_arguments(raw_arguments: str | None) -> dict:
        """Parse a tool call's JSON arguments string, degrading gracefully on malformed JSON.

        Claude/Gemini SDKs hand back pre-parsed dicts, so only OpenRouter's raw JSON
        string is exposed to this failure mode. Falling back to {} here routes into
        ChatService._execute_tool_calls' existing "missing 'query' argument" handling
        instead of crashing the call.
        """
        if not raw_arguments:
            return {}
        try:
            return json.loads(raw_arguments)
        except json.JSONDecodeError:
            logger.warning("openrouter_malformed_tool_arguments", extra={"raw_arguments": raw_arguments})
            return {}

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
