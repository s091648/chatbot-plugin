from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk import SlidingWindowStrategy, RateLimitExhausted

logger = logging.getLogger(__name__)


class AllProvidersExhausted(Exception):
    """Raised when every LLM provider has failed or hit its rate limit."""


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        """Send messages array, return assistant text reply.

        messages format: [{"role": "system"|"user", "content": "..."}]
        """
        ...


@dataclass
class ProviderHandler:
    provider: LLMProvider
    strategy: SlidingWindowStrategy
    priority: int
    name: str

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await self.strategy.acquire(estimated_tokens=estimated_tokens)
        result = await self.provider.complete(messages, max_tokens)
        self.strategy.record_usage(estimated_tokens)
        return result


class ResilientLLMService:
    """Walk an ordered list of ProviderHandlers. Fall back on rate-limit or failure."""

    def __init__(self, handlers: list[ProviderHandler]) -> None:
        self._handlers = sorted(handlers, key=lambda h: h.priority)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> str:
        if not self._handlers:
            raise AllProvidersExhausted()

        handlers_snapshot = list(self._handlers)

        for handler in handlers_snapshot:
            try:
                result = await handler.complete(messages, max_tokens)
                if result is not None:
                    return result
                logger.warning("provider_returned_none", extra={"provider": handler.name})
            except RateLimitExhausted:
                logger.warning("provider_daily_limit_reached", extra={"provider": handler.name})
                self._handlers.remove(handler)
                self._handlers.append(handler)
                logger.warning("provider_moved_to_end", extra={"provider": handler.name})
            except Exception as e:
                logger.error("provider_failed", extra={"provider": handler.name, "error": str(e)})

        logger.error("all_providers_exhausted")
        raise AllProvidersExhausted()
