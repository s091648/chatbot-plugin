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
    ) -> str:
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
                    config=genai.types.GenerateContentConfig(max_output_tokens=max_tokens),
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
            raise

        if not response.candidates:
            return ""

        candidate = response.candidates[0]
        fr = candidate.finish_reason
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name not in ("STOP", "1"):
            logger.warning("gemini_blocked", extra={"model": self.model, "finish_reason": fr_name})
            if fr_name != "MAX_TOKENS":
                return ""
            # MAX_TOKENS: response is truncated but still usable — fall through

        logger.info("gemini_api_called", extra={"model": self.model, "finish_reason": fr_name})
        return (response.text or "").strip()
