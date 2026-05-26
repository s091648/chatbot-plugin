"""Resilient LLM service with provider fallback chain."""

from dataclasses import dataclass

import structlog

from chatbot_plugin.llm.base_provider import BaseProvider
from chatbot_plugin.llm.rate_limit.quota_strategy import QuotaStrategy, RateLimitExhausted

logger = structlog.get_logger()


@dataclass
class ProviderHandler:
    """Pairs a provider with its rate limiting strategy."""

    provider: BaseProvider
    strategy: QuotaStrategy
    priority: int
    name: str

    async def generate(self, system_prompt: str, human_prompt: str) -> str | None:
        """Acquire a rate limit slot, call the provider, record usage."""
        await self.strategy.acquire(estimated_tokens=len(human_prompt) // 4)
        result = await self.provider.generate(system_prompt, human_prompt)
        if result is not None:
            await self.strategy.record_usage(
                len(human_prompt) // 4 + len(result) // 4
            )
        return result


class ResilientLLMService:
    """LLM service that tries providers in priority order, falling back on failure.

    If a provider raises RateLimitExhausted, the next provider is tried.
    If a provider returns None (retries exhausted), the next provider is tried.
    If all providers fail, returns None.
    """

    def __init__(self, handlers: list[ProviderHandler]) -> None:
        self._handlers = sorted(handlers, key=lambda h: h.priority)

    async def generate(self, system_prompt: str, human_prompt: str) -> str | None:
        """Generate a response using the fallback chain.

        Args:
            system_prompt: System instructions.
            human_prompt: User context + query.

        Returns:
            LLM response text, or None if all providers fail.
        """
        for handler in self._handlers:
            try:
                result = await handler.generate(system_prompt, human_prompt)
                if result is not None:
                    return result
                logger.warning(
                    "provider_returned_none",
                    provider=handler.name,
                    model=handler.provider._model,
                )
            except RateLimitExhausted:
                logger.warning(
                    "provider_daily_limit_reached",
                    provider=handler.name,
                )
            except Exception as e:
                logger.error(
                    "provider_failed",
                    provider=handler.name,
                    error=str(e),
                )

        logger.error("all_providers_exhausted")
        return None
