from __future__ import annotations

import asyncio
import logging

from google import genai

from chatbot_plugin_sdk import RateLimitExhausted

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
    ) -> tuple[str | None, str]:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System Instructions]\n{content}")
            else:
                parts.append(content)
        contents = "\n\n".join(parts)

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                            disable=True,
                        ),
                    ),
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
            raise

        if not response.candidates:
            return (None, "")

        candidate = response.candidates[0]
        fr = candidate.finish_reason
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name not in ("STOP", "1"):
            logger.warning("gemini_blocked", extra={"model": self.model, "finish_reason": fr_name})
            if fr_name != "MAX_TOKENS":
                return (None, "")

        # Separate thinking parts (thought=True) from reply parts (thought=False).
        # Non-thinking models only have reply parts; response.text is a safe fallback.
        content_parts = candidate.content.parts if candidate.content else []
        thinking_chunks: list[str] = []
        reply_chunks: list[str] = []

        for p in content_parts:
            text = getattr(p, "text", None)
            if not text:
                continue
            if getattr(p, "thought", False):
                thinking_chunks.append(text)
            else:
                reply_chunks.append(text)

        # Fallback for non-thinking models where parts may be empty
        if not reply_chunks and not thinking_chunks:
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
            },
        )
        return (thinking, reply)
