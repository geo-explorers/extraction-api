"""Single claim extraction API endpoint."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from starlette import status

from src.api.schemas.single_claim_extraction_schema import SingleClaimExtractionRequest
from src.api.services.premium_extraction_service import PremiumExtractionService
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/extract", tags=["single-claim-extraction"])


@router.post(
    "/claim/single",
    summary="Extract claims from a single episode",
    description="Extract claims from a single episode by ID with optional validation.",
)
async def extract_single_claim(request: SingleClaimExtractionRequest) -> JSONResponse:
    """
    Extract claims from a single episode.

    Args:
        request: Single claim extraction request with episode ID and settings

    Returns:
        JSONResponse with success status
    """
    logger.info(
        f"API request: single claim extract episode_id={request.episode_id}, "
        f"force={request.force}, should_validate={request.should_validate}"
    )
    premium_extraction_service = PremiumExtractionService()
    result = await premium_extraction_service._extract_single_episode(
        episode_id=request.episode_id,
        force=request.force,
    )

    return JSONResponse(
        content={"success": True, "result": result},
        status_code=status.HTTP_200_OK,
    )
