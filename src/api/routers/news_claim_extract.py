from typing import Callable, List
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.schemas.news_claim_extract_schema import (
  NewsArticleSource,
  NewsClaimExtractRequest,
  NewsClaimExtractResponse,
)
from src.api.services.news_claim_extract_service import (
  extract_news_claims,
  extract_news_claims_claude,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Type of the two interchangeable extractor functions (Gemini / Claude).
Extractor = Callable[[str, List[NewsArticleSource], List[str]], NewsClaimExtractResponse]


def _extract_and_respond(
  extractor: Extractor,
  request: NewsClaimExtractRequest,
  provider: str,
) -> JSONResponse:
  """Shared request/response handling for both the Gemini and Claude
  news-claim endpoints. Only the injected `extractor` differs — they run the
  identical NEWS_CLAIM_EXTRACT_PROMPT and return the same schema."""
  logger.info(
    f"News claim extract ({provider}) - Headline: '{request.headline[:80]}', "
    f"Sources: {len(request.sources)}, "
    f"Topics: {len(request.topics)}"
  )

  try:
    result = extractor(request.headline, request.sources, request.topics)

    response_data = result.model_dump()
    response_data["error"] = None

    logger.info(
      f"News claim extract ({provider}) response - "
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
    logger.error(
      f"News claim extract ({provider}) failed - {type(e).__name__}: {str(e)}"
    )

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


@router.post("/news/claims")
def news_claim_extract(request: NewsClaimExtractRequest) -> JSONResponse:
  """Primary news-claim extraction via Gemini."""
  return _extract_and_respond(extract_news_claims, request, "gemini")


@router.post("/news/claims/claude")
def news_claim_extract_claude(request: NewsClaimExtractRequest) -> JSONResponse:
  """Fallback news-claim extraction via Claude on the SAME strong prompt.

  Separate endpoint (not a flag on /news/claims) so the proven Gemini path is
  untouched. news-worker calls this only when the Gemini path errors out."""
  return _extract_and_respond(extract_news_claims_claude, request, "claude")
