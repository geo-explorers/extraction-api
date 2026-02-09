"""Async rate limiter using a sliding window for Gemini API call throttling."""

import asyncio
import time
from collections import deque

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Sliding-window rate limiter that ensures at most `max_calls` happen
    within any rolling `window_seconds` period.

    Usage:
        limiter = RateLimiter(max_calls=100, window_seconds=60)

        async def call_llm():
            await limiter.acquire()
            return client.models.generate_content(...)
    """

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a call slot is available within the rate limit window."""
        async with self._lock:
            while True:
                now = time.monotonic()

                # Evict timestamps outside the sliding window
                while (
                    self._timestamps
                    and self._timestamps[0] <= now - self.window_seconds
                ):
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                # Calculate how long to sleep until the oldest call exits the window
                sleep_for = self._timestamps[0] - (now - self.window_seconds)
                logger.info(
                    f"Rate limit reached ({self.max_calls}/{self.window_seconds}s). "
                    f"Waiting {sleep_for:.2f}s for next slot."
                )

                # Release the lock while sleeping so other coroutines aren't blocked
                # unnecessarily — they'll queue up on the lock and re-check.
                self._lock.release()
                try:
                    await asyncio.sleep(sleep_for)
                finally:
                    await self._lock.acquire()


# Global Gemini rate limiter — initialized once at server startup via init_gemini_rate_limiter()
_gemini_rate_limiter: RateLimiter | None = None


def init_gemini_rate_limiter(
    max_calls: int = 100, window_seconds: float = 60
) -> RateLimiter:
    """Initialize the global Gemini rate limiter. Should be called once at server startup."""
    global _gemini_rate_limiter
    _gemini_rate_limiter = RateLimiter(
        max_calls=max_calls, window_seconds=window_seconds
    )
    logger.info(
        f"Gemini rate limiter initialized: {max_calls} calls per {window_seconds}s"
    )
    return _gemini_rate_limiter


def get_gemini_rate_limiter() -> RateLimiter:
    """Get the global Gemini rate limiter instance. Raises if not initialized."""
    if _gemini_rate_limiter is None:
        raise RuntimeError(
            "Gemini rate limiter not initialized. Call init_gemini_rate_limiter() at server startup."
        )
    return _gemini_rate_limiter
