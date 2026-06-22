"""In-process LLM spend circuit breaker.

A rolling-hour ceiling on LLM calls per provider. When exceeded, `check_and_record`
raises `SpendLimitExceeded` so a task fails fast instead of burning provider quota
and money in a runaway loop (e.g. a bad prompt that always errors and retries).

This is DISTINCT from the Hatchet rate limiter: the rate limiter smooths the rate
(calls/sec); this caps total volume (calls/hour) as a hard budget backstop.

NOTE: state is per-worker-process. Global enforcement across multiple worker
replicas needs shared state (the Hatchet DB or a small KV). At single-worker
scale (today's volume) this is sufficient; revisit when scaling workers out.
Set LLM_MAX_CALLS_PER_HOUR=0 (default) to disable.
"""

import threading
import time
from collections import deque

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS = 3600.0


class SpendLimitExceeded(Exception):
    """Raised when a provider's hourly call budget is exhausted."""


class SpendGuard:
    def __init__(self, max_calls_per_hour: int):
        self._max = max_calls_per_hour
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check_and_record(self, provider: str) -> None:
        """Record one call against `provider`; raise if over the hourly budget.

        A no-op when the budget is <= 0 (disabled).
        """
        if self._max <= 0:
            return
        now = time.monotonic()
        with self._lock:
            dq = self._calls.setdefault(provider, deque())
            while dq and now - dq[0] > _WINDOW_SECONDS:
                dq.popleft()
            if len(dq) >= self._max:
                logger.error(
                    f"Spend circuit breaker tripped for {provider}: "
                    f"{len(dq)} calls in the last hour (limit {self._max})"
                )
                raise SpendLimitExceeded(
                    f"{provider} call budget exceeded ({self._max}/hour)"
                )
            dq.append(now)


# Shared instance. Reads the budget once at import; tasks call check_and_record().
spend_guard = SpendGuard(settings.llm_max_calls_per_hour)
