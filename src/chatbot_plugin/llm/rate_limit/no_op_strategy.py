"""No-op rate limiting strategy — passes all requests through."""

from chatbot_plugin.llm.rate_limit.quota_strategy import QuotaStrategy


class NoOpStrategy(QuotaStrategy):
    """Rate limiter that does nothing. Used when no rate limiting is configured."""

    async def acquire(self, estimated_tokens: int = 0) -> None:
        pass

    async def record_usage(self, actual_tokens: int) -> None:
        pass
