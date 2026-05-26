"""Async sliding window rate limiter using asyncio.Lock."""

import asyncio
import time
from collections import deque

from chatbot_plugin.llm.rate_limit.quota_strategy import QuotaStrategy, RateLimitExhausted


class SlidingWindowStrategy(QuotaStrategy):
    """Sliding window rate limiter with RPM, TPM, and RPD limits.

    Uses asyncio.Lock for thread safety in async contexts.
    Two 60-second rolling windows track requests and tokens.
    A daily counter tracks total requests.
    """

    def __init__(self, rpm: int, tpm: int, rpd: int) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._rpd = rpd
        self._lock = asyncio.Lock()
        self._rpm_window: deque[float] = deque()
        self._tpm_window: deque[tuple[float, int]] = deque()
        self._daily_count = 0

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until a request slot is available.

        Raises:
            RateLimitExhausted: If the daily quota is reached.
        """
        while True:
            wait = await self._compute_wait(estimated_tokens)
            if wait == 0:
                return
            await asyncio.sleep(wait)

    async def record_usage(self, actual_tokens: int) -> None:
        """Replace the last estimated token entry with actual usage."""
        async with self._lock:
            if self._tpm_window:
                self._tpm_window.pop()
            now = time.monotonic()
            self._tpm_window.append((now, actual_tokens))

    async def _compute_wait(self, estimated_tokens: int) -> float:
        """Compute seconds to wait. Returns 0 if the request can proceed."""
        async with self._lock:
            now = time.monotonic()

            # Check daily quota
            if self._rpd > 0 and self._daily_count >= self._rpd:
                raise RateLimitExhausted(
                    f"Daily quota reached: {self._daily_count}/{self._rpd}"
                )

            # Evict stale entries (older than 60s)
            cutoff = now - 60.0
            while self._rpm_window and self._rpm_window[0] < cutoff:
                self._rpm_window.popleft()
            while self._tpm_window and self._tpm_window[0][0] < cutoff:
                self._tpm_window.popleft()

            # Compute wait times
            rpm_wait = self._rpm_wait(now)
            tpm_wait = self._tpm_wait(now, estimated_tokens)
            wait = max(rpm_wait, tpm_wait)

            if wait == 0:
                # Record the request
                self._rpm_window.append(now)
                self._tpm_window.append((now, estimated_tokens))
                self._daily_count += 1

            return wait

    def _rpm_wait(self, now: float) -> float:
        """Seconds until an RPM slot opens."""
        if len(self._rpm_window) < self._rpm:
            return 0
        return self._rpm_window[0] + 60.0 - now

    def _tpm_wait(self, now: float, estimated_tokens: int) -> float:
        """Seconds until enough TPM capacity opens."""
        current_tokens = sum(t for _, t in self._tpm_window)
        if current_tokens + estimated_tokens <= self._tpm:
            return 0
        # Wait until the oldest entry exits the window
        if not self._tpm_window:
            # Single request exceeds TPM — no window to wait on, proceed anyway
            return 0
        return self._tpm_window[0][0] + 60.0 - now
