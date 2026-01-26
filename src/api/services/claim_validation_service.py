"""Claim validation service for concurrent context-independence validation."""

import asyncio
import time
from collections import deque
from typing import Dict, List, Tuple, Deque

from src.infrastructure.gemini_service import GeminiService, SingleClaimValidationResult
from src.config.prompts.claim_validation_prompt import CLAIM_VALIDATION_PROMPT
from src.config.settings import settings
from src.database.models import Claim
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class AsyncSlidingWindowRateLimiter:
    """
    Async-safe sliding window rate limiter.

    Limits API calls to a maximum number of tokens within a sliding time window.
    Thread-safe for concurrent async operations.
    """

    def __init__(self, max_tokens: int, window_seconds: float = 60.0) -> None:
        """
        Initialize the rate limiter.

        Args:
            max_tokens: Maximum number of tokens allowed in the window
            window_seconds: Time window in seconds (default: 60.0)
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens
        self._window_seconds = window_seconds
        self._lock = asyncio.Lock()
        self._timestamps: Deque[float] = deque()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the rate limiter.

        Blocks until tokens are available within the sliding window.

        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        if tokens <= 0:
            return
        if tokens > self._max_tokens:
            raise ValueError("tokens must be <= max_tokens")

        while True:
            async with self._lock:
                now = time.monotonic()
                # Clean up expired timestamps
                while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
                    self._timestamps.popleft()

                available = self._max_tokens - len(self._timestamps)
                if tokens <= available:
                    # Acquire tokens by recording timestamps
                    self._timestamps.extend([now] * tokens)
                    return

                # Calculate wait time based on oldest timestamp that needs to expire
                tokens_needed = tokens - available
                oldest_needed = self._timestamps[tokens_needed - 1]
                wait_time = self._window_seconds - (now - oldest_needed)

            # Wait outside the lock
            await asyncio.sleep(max(wait_time, 0.01))


class ClaimValidationService:
    """
    Service for validating claims for context independence.

    Uses concurrent Gemini API calls with rate limiting to validate
    multiple claims efficiently.
    """

    def __init__(self):
        """Initialize the claim validation service."""
        self.gemini_service: GeminiService | None = None
        self._rate_limiter = AsyncSlidingWindowRateLimiter(
            max_tokens=settings.validation_rate_limit_tokens,
            window_seconds=settings.validation_rate_limit_window
        )
        logger.info(
            f"ClaimValidationService initialized with rate_limit={settings.validation_rate_limit_tokens}/"
            f"{settings.validation_rate_limit_window}s, max_concurrency={settings.validation_max_concurrency}"
        )

    def _get_gemini_service(self) -> GeminiService:
        """Get or create the Gemini service singleton."""
        if self.gemini_service is None:
            logger.info("Initializing GeminiService for claim validation...")
            self.gemini_service = GeminiService()
        return self.gemini_service

    async def validate_claims_concurrent(
        self,
        claims: List[Claim],
        max_concurrency: int | None = None
    ) -> Dict[int, Tuple[bool, str]]:
        """
        Validate claims concurrently with rate limiting.

        Args:
            claims: List of Claim objects to validate
            max_concurrency: Maximum concurrent LLM calls (default from settings)

        Returns:
            Dict mapping claim_id -> (is_valid, explanation)
        """
        if not claims:
            logger.info("No claims to validate")
            return {}

        if max_concurrency is None:
            max_concurrency = settings.validation_max_concurrency

        logger.info(
            f"Validating {len(claims)} claims concurrently "
            f"(max_concurrency={max_concurrency}, rate_limit={settings.validation_rate_limit_tokens}/"
            f"{settings.validation_rate_limit_window}s)"
        )

        gemini_service = self._get_gemini_service()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def validate_one(claim: Claim) -> Tuple[int, bool, str]:
            """Validate a single claim with rate limiting."""
            async with semaphore:
                await self._rate_limiter.acquire(tokens=1)
                try:
                    result = await gemini_service.validate_single_claim(
                        claim.claim_text,
                        CLAIM_VALIDATION_PROMPT
                    )
                    status = "✓ VALID" if result.is_valid else "✗ INVALID"
                    logger.info(
                        f"[{status}] Claim {claim.id}: \"{claim.claim_text[:80]}{'...' if len(claim.claim_text) > 80 else ''}\" "
                        f"| Reason: {result.explanation}"
                    )
                    return (claim.id, result.is_valid, result.explanation)
                except Exception as e:
                    logger.error(f"Error validating claim {claim.id}: {e}")
                    # On error, mark as invalid (conservative approach)
                    return (claim.id, False, f"Validation error: {str(e)}")

        # Run all validations concurrently
        tasks = [validate_one(claim) for claim in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        validation_map: Dict[int, Tuple[bool, str]] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected validation error: {result}")
            else:
                claim_id, is_valid, explanation = result
                validation_map[claim_id] = (is_valid, explanation)

        valid_count = sum(1 for is_valid, _ in validation_map.values() if is_valid)
        invalid_count = len(validation_map) - valid_count
        logger.info(
            f"Validation complete: {valid_count} valid, {invalid_count} invalid "
            f"out of {len(claims)} claims"
        )

        return validation_map
