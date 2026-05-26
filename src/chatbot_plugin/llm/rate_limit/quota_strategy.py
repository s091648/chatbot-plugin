"""Quota strategy ABC for async rate limiting."""

from abc import ABC, abstractmethod


class QuotaStrategy(ABC):
    """Abstract base for rate limiting strategies."""

    @abstractmethod
    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until a request slot is available.

        Raises:
            RateLimitExhausted: If the daily quota is reached.
        """
        ...

    @abstractmethod
    async def record_usage(self, actual_tokens: int) -> None:
        """Update usage counters after a successful API call."""
        ...


class RateLimitExhausted(Exception):
    """Raised when the daily request quota is exhausted."""
