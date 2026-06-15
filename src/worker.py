"""Hatchet worker entrypoint for the extraction-worker Railway service.

Run with:  uv run python -m src.worker

This process registers every task type (from the registry) and starts the
worker, which connects to the Hatchet engine over gRPC (HATCHET_CLIENT_* env).
It is the SAME image as the API service — only the start command differs. It
needs NO DATABASE_URL.

`worker.start()` is synchronous and creates its own event loop, so it must run
from a plain __main__ (never inside asyncio.run()).
"""

import os

from hatchet_sdk import RateLimitDuration

from src.hatchet_client import hatchet
from src.config.settings import settings
from src.tasks.registry import all_tasks, task_names
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def _declare_rate_limits() -> None:
    """Declare global provider rate limits once at startup (idempotent put)."""
    hatchet.rate_limits.put(
        "gemini_global", settings.gemini_global_rate_per_min, RateLimitDuration.MINUTE
    )
    hatchet.rate_limits.put(
        "claude_global", settings.claude_global_rate_per_min, RateLimitDuration.MINUTE
    )
    logger.info(
        f"Declared rate limits: gemini_global={settings.gemini_global_rate_per_min}/min, "
        f"claude_global={settings.claude_global_rate_per_min}/min"
    )


def main() -> None:
    logger.info("=" * 80)
    logger.info("Extraction Worker starting")
    logger.info(f"Hatchet host: {os.getenv('HATCHET_CLIENT_HOST_PORT', 'unset')}")
    logger.info(f"Worker slots: {settings.hatchet_worker_slots}")
    logger.info(f"Registered task types: {task_names()}")
    logger.info("=" * 80)

    _declare_rate_limits()

    worker = hatchet.worker(
        "extraction-worker",
        slots=settings.hatchet_worker_slots,
        workflows=all_tasks(),
    )
    worker.start()  # blocking; runs its own event loop


if __name__ == "__main__":
    main()
