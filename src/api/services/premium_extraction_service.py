"""Premium extraction service for API endpoints."""

import asyncio
from typing import List, Tuple
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
from src.api.services.claim_validation_service import ClaimValidationService
from src.database.claim_repository import ClaimRepository
from src.database.connection import get_db_session
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class PremiumExtractionService:
    """
    Service for handling premium extraction requests using Gemini 3 Pro.

    Wraps the PremiumExtractionPipeline and converts domain models to API schemas.
    """

    def __init__(self):
        """Initialize the premium extraction service."""
        self.pipeline: PremiumExtractionPipeline | None = None
        self.validation_service: ClaimValidationService | None = None

    def _get_validation_service(self) -> ClaimValidationService:
        """Get or create the claim validation service singleton."""
        if self.validation_service is None:
            logger.info("Initializing ClaimValidationService...")
            self.validation_service = ClaimValidationService()
        return self.validation_service

    def _get_pipeline(self) -> PremiumExtractionPipeline:
        """Get or create the premium extraction pipeline singleton."""
        if self.pipeline is None:
            logger.info("Initializing PremiumExtractionPipeline...")
            self.pipeline = PremiumExtractionPipeline()
        return self.pipeline

    async def _extract_single_episode(
        self,
        episode_id: int,
        force: bool = False,
        should_validate: bool = False,
        db_session: Session | None = None
    ) -> SimplifiedExtractionResponse:
        """
        Extract claims from a single episode using Gemini 3 Pro (internal use only).

        Args:
            episode_id: Episode ID to process
            force: Force reprocessing even if claims exist
            should_validate: If true, validate claims for context independence after extraction
            db_session: Optional database session for queries

        Returns:
            SimplifiedExtractionResponse with statistics only

        Raises:
            EpisodeNotFoundError: If episode doesn't exist
            ProcessingError: If extraction fails
            ProcessingTimeoutError: If processing exceeds timeout
        """
        logger.info(
            f"Processing episode {episode_id} with PREMIUM pipeline "
            f"(force={force}, should_validate={should_validate})"
        )

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

        # Validate claims if requested
        validated_count = None
        invalid_count = None

        if should_validate and result.claims_extracted > 0:
            validated_count, invalid_count = await self._validate_episode_claims(
                episode_id, db_session
            )

        # Convert to simplified API response
        return self._convert_to_simplified_response(
            result, validated_count, invalid_count
        )

    async def _validate_episode_claims(
        self,
        episode_id: int,
        db_session: Session | None = None
    ) -> Tuple[int, int]:
        """
        Validate claims for an episode for context independence.

        Args:
            episode_id: Episode ID whose claims to validate
            db_session: Optional database session

        Returns:
            Tuple of (validated_count, invalid_count)
        """
        logger.info(f"Validating claims for episode {episode_id}...")

        # Get or create database session
        session = db_session or get_db_session()
        claim_repo = ClaimRepository(session)

        # Get claims that were just saved (include all, even verified ones for fresh validation)
        claims = claim_repo.get_claims_for_episode(
            episode_id,
            include_flagged=False,
            include_verified=True
        )

        if not claims:
            logger.info(f"No claims to validate for episode {episode_id}")
            return (0, 0)

        # Validate claims concurrently
        validation_service = self._get_validation_service()
        validation_results = await validation_service.validate_claims_concurrent(claims)

        # Separate valid and invalid claims
        valid_claim_ids = [
            claim_id for claim_id, (is_valid, _) in validation_results.items()
            if is_valid
        ]
        invalid_claim_ids = [
            claim_id for claim_id, (is_valid, _) in validation_results.items()
            if not is_valid
        ]

        # Update database: mark valid claims as verified
        if valid_claim_ids:
            claim_repo.mark_claims_verified(valid_claim_ids)
            logger.info(f"Marked {len(valid_claim_ids)} claims as verified for episode {episode_id}")

        # Note: invalid claims remain is_verified=False (the default)
        # They are NOT flagged - is_flagged is for different purpose (quality issues)

        # Commit the validation results (always commit since this is the final step)
        logger.info(f"Committing validation results for episode {episode_id}...")
        session.commit()
        logger.info(f"✓ Validation results committed for episode {episode_id}")

        if db_session is None:
            session.close()

        return (len(valid_claim_ids), len(invalid_claim_ids))

    async def extract_batch_episodes(
        self,
        podcast_ids: List[int],
        target: int | None = None,
        force: bool = False,
        continue_on_error: bool = False,
        should_validate: bool = False,
        db_session: Session | None = None,
    ) -> SimplifiedBatchExtractionResponse:
        """
        Extract claims from multiple episodes using Gemini 3 Pro.

        Args:
            podcast_ids: List of podcast IDs to process
            target: Maintain claims for latest N episodes per podcast
            force: Force reprocessing even if claims exist
            continue_on_error: Continue processing if an episode fails
            should_validate: If true, validate claims for context independence after extraction
            db_session: Optional database session for queries

        Returns:
            SimplifiedBatchExtractionResponse with statistics for all episodes

        Raises:
            PodcastNotFoundError: If no episodes found for podcasts
            ProcessingError: If extraction fails (only when continue_on_error=False)
        """
        logger.info(
            f"Processing PREMIUM batch: podcasts={podcast_ids}, target={target}, "
            f"force={force}, continue_on_error={continue_on_error}, "
            f"should_validate={should_validate}"
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
        results: List[SimplifiedExtractionResponse] = []
        errors: dict[int, str] = {}
        total_claims = 0
        total_time = 0.0

        for episode in episodes_to_process:
            try:
                response = await self._extract_single_episode(
                    episode.id,
                    force=force,
                    should_validate=should_validate,
                    db_session=db_session
                )
                results.append(response)
                total_claims += response.claims_count
                total_time += response.processing_time_seconds

                # Build log message with optional validation stats
                log_msg = (
                    f"Episode {episode.id}: {response.claims_count} claims "
                    f"in {response.processing_time_seconds:.1f}s (PREMIUM)"
                )
                if response.validated_count is not None:
                    log_msg += (
                        f" [validated: {response.validated_count}, "
                        f"invalid: {response.invalid_count}]"
                    )
                logger.info(f"✓ {log_msg}")

            except Exception as e:
                error_msg = str(e)
                errors[episode.id] = error_msg
                logger.error(f"✗ Episode {episode.id} failed: {error_msg}")

                if not continue_on_error:
                    raise ProcessingError(f"Episode {episode.id} failed: {error_msg}")

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
        self,
        result: PremiumPipelineResult,
        validated_count: int | None = None,
        invalid_count: int | None = None
    ) -> SimplifiedExtractionResponse:
        """
        Convert PremiumPipelineResult to SimplifiedExtractionResponse.

        Args:
            result: Pipeline result domain object
            validated_count: Number of claims that passed validation (if validation was run)
            invalid_count: Number of claims that failed validation (if validation was run)

        Returns:
            SimplifiedExtractionResponse for API
        """
        return SimplifiedExtractionResponse(
            episode_id=result.episode_id,
            processing_time_seconds=result.processing_time_seconds,
            claims_count=result.claims_extracted,
            quotes_count=0,  # Premium pipeline doesn't process quotes
            validated_count=validated_count,
            invalid_count=invalid_count,
        )
