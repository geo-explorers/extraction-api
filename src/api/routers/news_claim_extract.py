from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.schemas.news_claim_extract_schema import NewsClaimExtractRequest
from src.api.services.news_claim_extract_service import extract_news_claims
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/news/claims")
def news_claim_extract(request: NewsClaimExtractRequest) -> JSONResponse:
  logger.info(
    f"News claim extract request - Headline: '{request.headline[:80]}', "
    f"Sources: {len(request.sources)}, "
    f"Topics: {len(request.topics)}"
  )

  try:
    result = extract_news_claims(
      headline=request.headline,
      sources=request.sources,
      topics=request.topics,
    )

    response_data = result.model_dump()
    response_data["error"] = None

    logger.info(
      f"News claim extract response - "
      f"Claims: {len(result.claims)}, "
      f"Quotes: {len(result.quotes)}, "
      f"Collections: {len(result.collections)}, "
      f"Summary chars: {len(result.summary)}"
    )
    logger.debug(f"Response body: {json.dumps(response_data)[:2000]}")

    return JSONResponse(
      content=response_data,
      status_code=status.HTTP_200_OK,
    )
  except Exception as e:
    error_msg = f"An internal error occurred: {str(e)}"
    logger.error(f"News claim extract failed - {type(e).__name__}: {str(e)}")

    return JSONResponse(
      content={
        "error": error_msg,
        "claims": None,
        "quotes": None,
        "collections": None,
        "collection_order": None,
        "summary": None,
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
