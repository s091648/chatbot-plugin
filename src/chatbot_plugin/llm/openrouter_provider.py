from __future__ import annotations

import logging

import httpx

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
    ) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        logger.info("openrouter_api_called", model=self.model)
        return data["choices"][0]["message"]["content"]
