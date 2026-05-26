"""OpenRouter provider using httpx AsyncClient (OpenAI-compatible API)."""

import httpx
import structlog

from chatbot_plugin.llm.base_provider import BaseProvider

logger = structlog.get_logger()

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(BaseProvider):
    """LLM provider for OpenRouter via OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model=model)
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=60.0)

    async def _call_api(self, system_prompt: str, human_prompt: str) -> str:
        response = await self._client.post(
            _API_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        logger.info(
            "openrouter_api_called",
            model=self._model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        return content
