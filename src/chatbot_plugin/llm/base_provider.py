"""Base LLM provider with async tenacity retry."""

from abc import ABC, abstractmethod

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from chatbot_plugin.llm.rate_limit.quota_strategy import RateLimitExhausted

logger = structlog.get_logger()


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient errors that should be retried."""
    if isinstance(exc, RateLimitExhausted):
        return False
    if isinstance(exc, (ValueError, KeyError)):
        return False
    return True


class BaseProvider(ABC):
    """Abstract base for LLM providers with async tenacity retry.

    Subclasses implement `_call_api(system_prompt, human_prompt)` which
    returns the raw text response from the LLM.

    The public `generate()` method wraps `_call_api` with retry logic
    and error handling.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._retry = AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
        )

    @abstractmethod
    async def _call_api(self, system_prompt: str, human_prompt: str) -> str:
        """Call the provider API and return the raw text response.

        Args:
            system_prompt: System instructions (RAG behavior, citation rules).
            human_prompt: User context + query.

        Returns:
            Raw text response from the LLM.

        Raises:
            RateLimitExhausted: If daily quota is reached.
            Exception: For transient errors (will be retried).
        """
        ...

    async def generate(self, system_prompt: str, human_prompt: str) -> str | None:
        """Generate a response with retry logic.

        Args:
            system_prompt: System instructions.
            human_prompt: User context + query.

        Returns:
            LLM response text, or None if all retries are exhausted.
        """
        try:
            async for attempt in self._retry:
                with attempt:
                    return await self._call_api(system_prompt, human_prompt)
        except RateLimitExhausted:
            raise
        except Exception as e:
            logger.warning("provider_generate_failed", model=self._model, error=str(e))
            return None
