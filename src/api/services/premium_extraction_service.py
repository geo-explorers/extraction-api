"""Premium extraction service for API endpoints."""

import asyncio
import time
from collections import deque
from typing import Deque, List
from sqlalchemy.orm import Session

from src.pipeline.premium_extraction_pipeline import PremiumExtractionPipeline, PremiumPipelineResult
from src.cli.episode_query import EpisodeQueryService
from src.api.schemas.responses import (
    SimplifiedExtractionResponse,
    SimplifiedBatchExtractionResponse,
    BatchExtractionSummary,
)
from src.api.exceptions import (
    EpisodeNotFoundError,
    PodcastNotFoundError,
    ProcessingError,
    ProcessingTimeoutError,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger


logger = get_logger(__name__)


class _AsyncSlidingWindowRateLimiter:
    def __init__(self, max_tokens: int, window_seconds: float = 60.0) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._max_tokens = max_tokens
        self._window_seconds = window_seconds
        self._lock = asyncio.Lock()
        self._timestamps: Deque[float] = deque()

    async def acquire(self, tokens: int = 1) -> None:
        if tokens <= 0:
            return
        if tokens > self._max_tokens:
            raise ValueError("tokens must be <= max_tokens")

        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
                    self._timestamps.popleft()

                available = self._max_tokens - len(self._timestamps)
                if tokens <= available:
                    self._timestamps.extend([now] * tokens)
                    return

                tokens_needed = tokens - available
                oldest_needed = self._timestamps[tokens_needed - 1]
                wait_time = self._window_seconds - (now - oldest_needed)

            await asyncio.sleep(max(wait_time, 0.01))


class PremiumExtractionService:
    """
    Service for handling premium extraction requests using Gemini 3 Pro.

    Wraps the PremiumExtractionPipeline and converts domain models to API schemas.
    """

    def __init__(self):
        """Initialize the premium extraction service."""
        self.pipeline: PremiumExtractionPipeline | None = None
        self._gemini_rate_limiter = _AsyncSlidingWindowRateLimiter(
            max_tokens=settings.premium_extraction_rate_limit_max_tokens,
            window_seconds=settings.premium_extraction_rate_limit_window_seconds,
        )

    def _get_pipeline(self) -> PremiumExtractionPipeline:
        """Get or create the premium extraction pipeline singleton."""
        if self.pipeline is None:
            logger.info("Initializing PremiumExtractionPipeline...")
            self.pipeline = PremiumExtractionPipeline()
        return self.pipeline

    async def _extract_single_episode(
        self, episode_id: int, force: bool = False, db_session: Session | None = None
    ) -> SimplifiedExtractionResponse:
        """
        Extract claims from a single episode using Gemini 3 Pro (internal use only).

        Args:
            episode_id: Episode ID to process
            force: Force reprocessing even if claims exist
            db_session: Optional database session for queries

        Returns:
            SimplifiedExtractionResponse with statistics only

        Raises:
            EpisodeNotFoundError: If episode doesn't exist
            ProcessingError: If extraction fails
            ProcessingTimeoutError: If processing exceeds timeout
        """
        logger.info(f"Processing episode {episode_id} with PREMIUM pipeline (force={force})")

        # Verify episode exists
        query_service = EpisodeQueryService(db_session)
        episode = query_service.get_episode_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(episode_id)

        # Check if already processed (unless force=True)
        if not force and query_service.is_episode_processed(episode_id):
            logger.info(f"Episode {episode_id} already processed (use force=True to reprocess)")

        # Process with timeout (if configured)
        try:
            pipeline = self._get_pipeline()

            if settings.api_timeout > 0:
                # Process with timeout
                result = await asyncio.wait_for(
                    pipeline.process_episode(episode_id, save_to_db=True),
                    timeout=settings.api_timeout
                )
            else:
                # No timeout - process indefinitely
                result = await pipeline.process_episode(episode_id, save_to_db=True)

        except asyncio.TimeoutError:
            logger.error(f"Processing episode {episode_id} timed out after {settings.api_timeout}s")
            raise ProcessingTimeoutError(settings.api_timeout)
        except Exception as e:
            logger.error(f"Error processing episode {episode_id}: {e}", exc_info=True)
            raise ProcessingError(str(e))

        # Convert to simplified API response
        return self._convert_to_simplified_response(result)

    async def extract_batch_episodes(
        self,
        podcast_ids: List[int],
        target: int | None = None,
        force: bool = False,
        continue_on_error: bool = False,
        db_session: Session | None = None,
    ) -> SimplifiedBatchExtractionResponse:
        """
        Extract claims from multiple episodes using Gemini 3 Pro.

        Args:
            podcast_ids: List of podcast IDs to process
            target: Maintain claims for latest N episodes per podcast
            force: Force reprocessing even if claims exist
            continue_on_error: Continue processing if an episode fails
            db_session: Optional database session for queries

        Returns:
            SimplifiedBatchExtractionResponse with statistics for all episodes

        Raises:
            PodcastNotFoundError: If no episodes found for podcasts
            ProcessingError: If extraction fails (only when continue_on_error=False)
        """
        logger.info(
            f"Processing PREMIUM batch: podcasts={podcast_ids}, target={target}, "
            f"force={force}, continue_on_error={continue_on_error}"
        )

        # Query episodes to process
        query_service = EpisodeQueryService(db_session)
        episodes_to_process = query_service.get_episodes_to_process(
            podcast_ids=podcast_ids,
            target=target or 0,
            force=force
        )

        if not episodes_to_process:
            logger.info(f"No new episodes to process for podcasts {podcast_ids} (all up-to-date)")
            # Return successful empty response (nothing to process is success, not error)
            summary = BatchExtractionSummary(
                total_episodes=0,
                successful_episodes=0,
                failed_episodes=0,
                total_claims_extracted=0,
                total_processing_time_seconds=0.0,
            )
            return SimplifiedBatchExtractionResponse(
                results=[],
                summary=summary,
                errors={},
            )

        logger.info(f"Found {len(episodes_to_process)} episodes to process with PREMIUM pipeline")

        # Process each episode
        responses_by_id: dict[int, SimplifiedExtractionResponse] = {}
        errors: dict[int, str] = {}
        total_claims = 0
        total_time = 0.0

        max_parallel_episodes = max(1, settings.premium_extraction_max_parallel_episodes)
        gemini_calls_per_episode = settings.premium_extraction_gemini_calls_per_episode
        semaphore = asyncio.Semaphore(max_parallel_episodes)
        rate_limiter = self._gemini_rate_limiter

        async def process_episode(episode):
            async with semaphore:
                await rate_limiter.acquire(tokens=gemini_calls_per_episode)
                return await self._extract_single_episode(
                    episode.id, force=force, db_session=db_session
                )

        tasks = {
            asyncio.create_task(process_episode(episode)): episode.id
            for episode in episodes_to_process
        }

        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                episode_id = tasks[task]
                try:
                    response = await task
                except Exception as e:
                    error_msg = str(e)
                    errors[episode_id] = error_msg
                    logger.error(f"✗ Episode {episode_id} failed: {error_msg}")

                    if not continue_on_error:
                        for pending_task in pending:
                            if not pending_task.done():
                                pending_task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        raise ProcessingError(
                            f"Episode {episode_id} failed: {error_msg}"
                        )
                else:
                    responses_by_id[episode_id] = response
                    total_claims += response.claims_count
                    total_time += response.processing_time_seconds
                    logger.info(
                        f"✓ Episode {episode_id}: {response.claims_count} claims "
                        f"in {response.processing_time_seconds:.1f}s (PREMIUM)"
                    )

        results = [
            responses_by_id[episode.id]
            for episode in episodes_to_process
            if episode.id in responses_by_id
        ]

        # Create summary
        summary = BatchExtractionSummary(
            total_episodes=len(episodes_to_process),
            successful_episodes=len(results),
            failed_episodes=len(errors),
            total_claims_extracted=total_claims,
            total_processing_time_seconds=total_time,
        )

        return SimplifiedBatchExtractionResponse(
            results=results,
            summary=summary,
            errors=errors,
        )

    def _convert_to_simplified_response(
        self, result: PremiumPipelineResult
    ) -> SimplifiedExtractionResponse:
        """
        Convert PremiumPipelineResult to SimplifiedExtractionResponse.

        Args:
            result: Pipeline result domain object

        Returns:
            SimplifiedExtractionResponse for API
        """
        return SimplifiedExtractionResponse(
            episode_id=result.episode_id,
            processing_time_seconds=result.processing_time_seconds,
            claims_count=result.claims_extracted,
            quotes_count=0,  # Premium pipeline doesn't process quotes
        )
