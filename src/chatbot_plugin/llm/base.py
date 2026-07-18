from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from chatbot_plugin_sdk import SlidingWindowStrategy, RateLimitExhausted

logger = logging.getLogger(__name__)


class AllProvidersExhausted(Exception):
    """Raised when every LLM provider has failed or hit its rate limit."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    thinking: str | None
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        """Send messages array (+ optional tool definitions), return an LLMResult.

        messages format: [{"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}]
        Tool-call turn: {"role": "assistant", "content": "", "tool_calls": [{"id","name","arguments"}]}
        Tool-result turn: {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
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
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await self.strategy.acquire(estimated_tokens=estimated_tokens)
        result = await self.provider.complete(messages, max_tokens, tools)
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
        tools: list["ToolSpec"] | None = None,
        pinned_handler: ProviderHandler | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[LLMResult, ProviderHandler]:
        if pinned_handler is not None:
            try:
                result = await pinned_handler.complete(messages, max_tokens, tools)
            except Exception as e:
                logger.error(
                    "pinned_provider_failed",
                    extra={"provider": pinned_handler.name, "error": str(e)},
                )
                raise AllProvidersExhausted() from e
            return (result, pinned_handler)

        excluded = exclude or set()
        candidates = [h for h in self._handlers if h.name not in excluded]
        if not candidates:
            raise AllProvidersExhausted()

        for handler in list(candidates):
            try:
                result = await handler.complete(messages, max_tokens, tools)
                return (result, handler)
            except RateLimitExhausted:
                logger.warning("provider_daily_limit_reached", extra={"provider": handler.name})
                self._handlers.remove(handler)
                self._handlers.append(handler)
                logger.warning("provider_moved_to_end", extra={"provider": handler.name})
            except Exception as e:
                logger.error("provider_failed", extra={"provider": handler.name, "error": str(e)})

        logger.error("all_providers_exhausted")
        raise AllProvidersExhausted()
