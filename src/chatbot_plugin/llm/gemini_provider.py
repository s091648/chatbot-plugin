"""Gemini provider using google.genai Client with asyncio.to_thread."""

import asyncio

import structlog
from google import genai

from chatbot_plugin.llm.base_provider import BaseProvider
from chatbot_plugin.llm.rate_limit.quota_strategy import RateLimitExhausted

logger = structlog.get_logger()


class GeminiProvider(BaseProvider):
    """LLM provider for Google Gemini via the genai SDK.

    The google.genai.Client is synchronous, so calls are wrapped
    with asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model=model)
        self._client = genai.Client(api_key=api_key)

    async def _call_api(self, system_prompt: str, human_prompt: str) -> str:
        return await asyncio.to_thread(
            self._sync_generate, system_prompt, human_prompt
        )

    def _sync_generate(self, system_prompt: str, human_prompt: str) -> str:
        """Synchronous Gemini API call — runs in a thread pool."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=human_prompt,
                config=genai.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2048,
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Gemini daily quota exhausted: {error_str}") from e
            raise

        # Check for blocked/safety-filtered responses
        if not response.candidates:
            logger.warning("gemini_no_candidates", model=self._model)
            return ""

        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason") and candidate.finish_reason not in (1, "STOP"):
            logger.warning(
                "gemini_response_blocked",
                model=self._model,
                finish_reason=str(candidate.finish_reason),
            )
            return ""

        token_counts = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_counts = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
            }

        logger.info("gemini_api_called", model=self._model, **token_counts)
        return response.text or ""
