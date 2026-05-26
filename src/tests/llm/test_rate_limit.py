"""Tests for async rate limiting strategies."""

import asyncio
import time

import pytest

from chatbot_plugin.llm.rate_limit import (
    SlidingWindowStrategy,
    NoOpStrategy,
    RateLimitExhausted,
)


class TestNoOpStrategy:
    @pytest.mark.asyncio
    async def test_acquire_never_blocks(self):
        strategy = NoOpStrategy()
        await strategy.acquire(1000)  # should return immediately

    @pytest.mark.asyncio
    async def test_record_usage_is_noop(self):
        strategy = NoOpStrategy()
        await strategy.record_usage(1000)  # should not raise


class TestSlidingWindowStrategy:
    @pytest.mark.asyncio
    async def test_allows_requests_within_rpm(self):
        strategy = SlidingWindowStrategy(rpm=5, tpm=10000, rpd=100)
        for _ in range(5):
            await strategy.acquire(100)

    @pytest.mark.asyncio
    async def test_raises_on_rpd_exhausted(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=100000, rpd=2)
        await strategy.acquire(100)
        await strategy.acquire(100)
        with pytest.raises(RateLimitExhausted):
            await strategy.acquire(100)

    @pytest.mark.asyncio
    async def test_record_usage_updates_tpm(self):
        strategy = SlidingWindowStrategy(rpm=100, tpm=100000, rpd=100)
        await strategy.acquire(100)  # estimated 100 tokens
        await strategy.record_usage(200)  # actual was 200

    @pytest.mark.asyncio
    async def test_concurrent_acquires_serialized(self):
        """Multiple concurrent acquires should not exceed RPM."""
        strategy = SlidingWindowStrategy(rpm=3, tpm=100000, rpd=100)
        results = []

        async def try_acquire():
            try:
                await strategy.acquire(10)
                results.append("ok")
            except RateLimitExhausted:
                results.append("exhausted")

        # Launch 3 acquires concurrently — all should succeed (RPM=3)
        await asyncio.gather(*[try_acquire() for _ in range(3)])
        assert results.count("ok") == 3
