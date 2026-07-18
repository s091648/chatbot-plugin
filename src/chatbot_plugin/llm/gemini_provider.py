from __future__ import annotations

import asyncio
import logging

from google import genai

from chatbot_plugin_sdk import RateLimitExhausted
from chatbot_plugin.llm.base import LLMResult, ToolCallRequest, ToolSpec

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_instruction = "\n\n".join(system_parts) or None
        contents = [self._to_gemini_content(m) for m in messages if m.get("role") != "system"]

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": genai.types.AutomaticFunctionCallingConfig(disable=True),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [
                genai.types.Tool(function_declarations=[
                    genai.types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.input_schema)
                    for t in tools
                ])
            ]

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**config_kwargs),
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
            raise

        if not response.candidates:
            return LLMResult(thinking=None, text="")

        candidate = response.candidates[0]
        fr = candidate.finish_reason
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name not in ("STOP", "1"):
            logger.warning("gemini_blocked", extra={"model": self.model, "finish_reason": fr_name})
            if fr_name != "MAX_TOKENS":
                return LLMResult(thinking=None, text="")

        content_parts = candidate.content.parts if candidate.content else []
        thinking_chunks: list[str] = []
        reply_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for i, p in enumerate(content_parts):
            fc = getattr(p, "function_call", None)
            if fc is not None:
                tool_calls.append(ToolCallRequest(id=f"call_{i}", name=fc.name, arguments=dict(fc.args or {})))
                continue
            text = getattr(p, "text", None)
            if not text:
                continue
            if getattr(p, "thought", False):
                thinking_chunks.append(text)
            else:
                reply_chunks.append(text)

        if not reply_chunks and not thinking_chunks and not tool_calls:
            reply_chunks.append(response.text or "")

        thinking = "".join(thinking_chunks).strip() or None
        reply = "".join(reply_chunks).strip()

        logger.info(
            "gemini_api_called",
            extra={
                "model": self.model,
                "finish_reason": fr_name,
                "reply_len": len(reply),
                "has_thinking": thinking is not None,
                "tool_call_count": len(tool_calls),
            },
        )
        return LLMResult(thinking=thinking, text=reply, tool_calls=tool_calls)

    @staticmethod
    def _to_gemini_content(message: dict):
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            parts = [
                genai.types.Part(function_call=genai.types.FunctionCall(name=tc["name"], args=tc["arguments"]))
                for tc in message["tool_calls"]
            ]
            return genai.types.Content(role="model", parts=parts)
        if role == "tool":
            payload = {"error": message["content"]} if message.get("is_error") else {"result": message["content"]}
            return genai.types.Content(
                role="user",
                parts=[genai.types.Part(function_response=genai.types.FunctionResponse(
                    name=message["name"], response=payload,
                ))],
            )
        gemini_role = "model" if role == "assistant" else "user"
        return genai.types.Content(role=gemini_role, parts=[genai.types.Part(text=message["content"])])
