from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

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
    # Gemini 3+ only: opaque signature the model attaches to a function-call part when
    # thinking is enabled. Must be echoed back verbatim on the follow-up turn or the API
    # rejects the request with 400 INVALID_ARGUMENT. None for Claude/OpenRouter and for
    # Gemini models that don't emit one (see GeminiProvider._to_gemini_content).
    thought_signature: bytes | None = None


@dataclass
class LLMResult:
    thinking: str | None
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class StreamError:
    """Yielded by ResilientLLMService.stream_complete (never raised by a provider directly)
    when a provider fails partway through a turn — i.e. after it already yielded at least one
    event. At that point some output has already reached the client, so silently retrying with
    a different provider would produce a visibly duplicated/inconsistent turn; the caller must
    surface this and stop instead."""
    message: str


# What a provider's stream() emits, in the order the model produced it. A ToolCallRequest is
# only emitted once the tool call is fully assembled (providers that deliver it in fragments,
# e.g. OpenRouter's index-keyed argument deltas, buffer internally and yield it whole).
LLMStreamEvent = ThinkingDelta | TextDelta | ToolCallRequest


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

    def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Same request shape as complete(), but yields deltas as the model produces them
        instead of waiting for the full response. Raises the same exceptions as complete()
        (including RateLimitExhausted), at whatever point in the stream they occur."""
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

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await self.strategy.acquire(estimated_tokens=estimated_tokens)
        async for event in self.provider.stream(messages, max_tokens, tools):
            yield event
        self.strategy.record_usage(estimated_tokens)


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

    async def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
        pinned_handler: ProviderHandler | None = None,
        exclude: set[str] | None = None,
    ) -> AsyncIterator[tuple[ProviderHandler, LLMStreamEvent | StreamError]]:
        """Streaming counterpart to complete(). Fallback across providers only happens in the
        safe window before a candidate has yielded anything — once a provider has produced at
        least one event, that output is already on its way to the client, so a failure past
        that point is surfaced as a StreamError event (not a silent retry with a different
        provider, which would produce a visibly duplicated/inconsistent turn) and the generator
        ends. AllProvidersExhausted is only ever raised before any event has been yielded.
        """
        if pinned_handler is not None:
            yielded_any = False
            try:
                async for event in pinned_handler.stream(messages, max_tokens, tools):
                    yielded_any = True
                    yield (pinned_handler, event)
            except Exception as e:
                if yielded_any:
                    logger.error(
                        "pinned_provider_stream_failed_mid_response",
                        extra={"provider": pinned_handler.name, "error": str(e)},
                    )
                    yield (pinned_handler, StreamError(str(e)))
                    return
                logger.error(
                    "pinned_provider_stream_failed",
                    extra={"provider": pinned_handler.name, "error": str(e)},
                )
                raise AllProvidersExhausted() from e
            return

        excluded = exclude or set()
        candidates = [h for h in self._handlers if h.name not in excluded]
        if not candidates:
            raise AllProvidersExhausted()

        for handler in list(candidates):
            yielded_any = False
            try:
                async for event in handler.stream(messages, max_tokens, tools):
                    yielded_any = True
                    yield (handler, event)
                return
            except RateLimitExhausted:
                if yielded_any:
                    yield (handler, StreamError("rate limit reached mid-response"))
                    return
                logger.warning("provider_daily_limit_reached", extra={"provider": handler.name})
                self._handlers.remove(handler)
                self._handlers.append(handler)
                logger.warning("provider_moved_to_end", extra={"provider": handler.name})
            except Exception as e:
                if yielded_any:
                    logger.error(
                        "provider_stream_failed_mid_response",
                        extra={"provider": handler.name, "error": str(e)},
                    )
                    yield (handler, StreamError(str(e)))
                    return
                logger.error("provider_stream_failed", extra={"provider": handler.name, "error": str(e)})

        logger.error("all_providers_exhausted")
        raise AllProvidersExhausted()
