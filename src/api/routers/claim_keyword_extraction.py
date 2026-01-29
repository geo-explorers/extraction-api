from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.services.claim_keyword_extraction_service import extract_claim_keywords_and_topics
from src.api.schemas.claim_keyword_extraction_schema import (
    ClaimKeywordExtractionRequest,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/claim-keywords")
def claim_keyword_extraction(request: ClaimKeywordExtractionRequest) -> JSONResponse:
    # Log incoming request payload
    logger.info(f"Claim keyword extraction request - Episode: '{request.title}', Claims: {len(request.claims)}, Topics: {len(request.topics_list)}")

    try:
        # Convert Pydantic models to dicts for the service
        claims_data = [{"id": c.id, "text": c.text} for c in request.claims]

        claim_keywords, claim_topics, topic_keywords = extract_claim_keywords_and_topics(
            claims=claims_data,
            title=request.title,
            description=request.description,
            topics_list=request.topics_list,
            min_keywords=request.min_keywords,
            max_keywords=request.max_keywords,
            min_topics=request.min_topics,
            max_topics=request.max_topics,
        )

        response_data = {
            "error": None,
            "claim_keywords": claim_keywords,
            "claim_topics": claim_topics,
            "topic_keywords": topic_keywords
        }

        # Log response body
        logger.info(f"Claim keyword extraction response - Claims processed: {len(claim_keywords)}, Topic Keywords: {len(topic_keywords)}")
        logger.debug(f"Response body: {json.dumps(response_data)}")

        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        error_msg = f"An internal error occurred: {str(e)}"
        logger.error(f"Claim keyword extraction failed - {type(e).__name__}: {str(e)}")

        return JSONResponse(
            content={
                "error": error_msg,
                "claim_keywords": None,
                "claim_topics": None,
                "topic_keywords": None
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
