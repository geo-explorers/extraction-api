"""Extraction API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.services.extraction_service import ExtractionService
from src.api.services.premium_extraction_service import PremiumExtractionService
from src.api.schemas.requests import BatchExtractionRequest
from src.api.schemas.responses import (
    SimplifiedBatchExtractionResponse,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["extraction"])


@router.post(
    "/claims",
    response_model=SimplifiedBatchExtractionResponse,
    summary="Extract claims from multiple episodes",
    description="""
    Extract claims from multiple episodes based on podcast IDs and target count.

    This endpoint mimics the CLI batch processing behavior:
    - Process multiple episodes across one or more podcasts
    - Maintain claims for the latest N episodes per podcast (when target is set)
    - Continue processing on errors (when continue_on_error=true)

    Processing time scales with episode count (~6 minutes per episode).
    The API processes episodes sequentially and returns all results when complete.

    The API saves all results to the database and returns statistics only.
    Full claims and quotes data is stored in the database for later retrieval.
    """,
    responses={
        200: {
            "description": "Batch extraction completed (may include partial failures)",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "episode_id": 123,
                                "processing_time_seconds": 342.5,
                                "claims_count": 28,
                                "quotes_count": 85
                            },
                            {
                                "episode_id": 124,
                                "processing_time_seconds": 351.2,
                                "claims_count": 32,
                                "quotes_count": 96
                            }
                        ],
                        "summary": {
                            "total_episodes": 2,
                            "successful_episodes": 2,
                            "failed_episodes": 0,
                            "total_claims_extracted": 60,
                            "total_processing_time_seconds": 693.7
                        },
                        "errors": {}
                    }
                }
            },
        },
        404: {"description": "No episodes found for specified podcasts"},
        500: {"description": "Processing error (when continue_on_error=false)"},
    },
)
async def extract_episodes_batch(
    request: BatchExtractionRequest, db: Session = Depends(get_db)
) -> SimplifiedBatchExtractionResponse:
    """
    Extract claims from multiple episodes in batch.

    Args:
        request: Batch extraction request with podcast IDs and settings
        db: Database session (injected)

    Returns:
        SimplifiedBatchExtractionResponse with statistics for all episodes (data saved to database)
    """
    logger.info(
        f"API request: batch extract podcasts={request.podcast_ids}, "
        f"target={request.target}, force={request.force}, "
        f"continue_on_error={request.continue_on_error}, "
        f"should_validate={request.should_validate}"
    )

    service = ExtractionService()
    result = await service.extract_batch_episodes(
        podcast_ids=request.podcast_ids,
        target=request.target,
        force=request.force,
        continue_on_error=request.continue_on_error,
        should_validate=request.should_validate,
        db_session=db,
    )

    logger.info(
        f"API response: batch completed - {result.summary.successful_episodes}/"
        f"{result.summary.total_episodes} successful, "
        f"{result.summary.total_claims_extracted} total claims"
    )
    return result


@router.post(
    "/claims/premium",
    response_model=SimplifiedBatchExtractionResponse,
    summary="Extract claims using Gemini 3 Pro (10x faster, full-context)",
    description="""
    PREMIUM extraction using Gemini 3 Pro with full transcript context.

    Key differences from standard /claims endpoint:
    - Uses Gemini 3 Pro (1M context window)
    - NO chunking (processes full transcript)
    - Zero-shot extraction
    - ~10x faster (~30-60s vs 5-6 minutes per episode)
    - No quote processing (claims only)

    Cost: ~$0.10-0.20 per episode
    Processing time: ~30-60 seconds per episode

    Requires GEMINI_API_KEY in environment.
    """,
    responses={
        200: {
            "description": "Premium extraction completed (may include partial failures)",
            "content": {
                "application/json": {
                    "example": {
                        "results": [
                            {
                                "episode_id": 123,
                                "processing_time_seconds": 45.2,
                                "claims_count": 32,
                                "quotes_count": 0
                            },
                            {
                                "episode_id": 124,
                                "processing_time_seconds": 52.8,
                                "claims_count": 41,
                                "quotes_count": 0
                            }
                        ],
                        "summary": {
                            "total_episodes": 2,
                            "successful_episodes": 2,
                            "failed_episodes": 0,
                            "total_claims_extracted": 73,
                            "total_processing_time_seconds": 98.0
                        },
                        "errors": {}
                    }
                }
            },
        },
        404: {"description": "No episodes found for specified podcasts"},
        500: {"description": "Processing error (when continue_on_error=false)"},
    },
)
async def extract_episodes_batch_premium(
    request: BatchExtractionRequest, db: Session = Depends(get_db)
) -> SimplifiedBatchExtractionResponse:
    """
    Extract claims from multiple episodes using PREMIUM Gemini 3 Pro pipeline.

    Args:
        request: Batch extraction request with podcast IDs and settings
        db: Database session (injected)

    Returns:
        SimplifiedBatchExtractionResponse with statistics (data saved to database)
    """
    logger.info(
        f"API request: PREMIUM batch extract podcasts={request.podcast_ids}, "
        f"target={request.target}, force={request.force}, "
        f"continue_on_error={request.continue_on_error}, "
        f"should_validate={request.should_validate}"
    )

    service = PremiumExtractionService()
    result = await service.extract_batch_episodes(
        podcast_ids=request.podcast_ids,
        target=request.target,
        force=request.force,
        continue_on_error=request.continue_on_error,
        should_validate=request.should_validate,
        db_session=db,
    )

    logger.info(
        f"API response: PREMIUM batch completed - {result.summary.successful_episodes}/"
        f"{result.summary.total_episodes} successful, "
        f"{result.summary.total_claims_extracted} total claims"
    )
    return result
