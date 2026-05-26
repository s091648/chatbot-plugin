"""Claude provider using anthropic AsyncAnthropic SDK."""

import anthropic
import structlog

from chatbot_plugin.llm.base_provider import BaseProvider

logger = structlog.get_logger()


class ClaudeProvider(BaseProvider):
    """LLM provider for Anthropic Claude via the official async SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model=model)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _call_api(self, system_prompt: str, human_prompt: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": human_prompt}],
        )
        logger.info(
            "claude_api_called",
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response.content[0].text
