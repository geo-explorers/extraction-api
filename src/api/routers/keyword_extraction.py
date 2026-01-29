from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.services.keyword_extraction_service import extract_keyword_and_topics
from src.api.schemas.keyword_extraction_schema import (
  KeywordExtractionRequest,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/keywords")
def keyword_extraction(request: KeywordExtractionRequest) -> JSONResponse:
  # Log incoming request payload
  logger.info(f"Keyword extraction request - Episode: '{request.episode.get('title', 'N/A')}', Claims: {len(request.episode.get('claims', []))}, Topics: {len(request.topics_list)}")

  try:
    keywords, topics, topic_keywords = extract_keyword_and_topics(
      episode=request.episode,
      topics_list=request.topics_list,
      min_keywords=request.min_keywords,
      max_keywords=request.max_keywords,
      min_topics=request.min_topics,
      max_topics=request.max_topics,
    )

    response_data = {
      "error": None,
      "keywords": keywords,
      "topics": topics,
      "topic_keywords": topic_keywords
    }

    # Log response body
    logger.info(f"Keyword extraction response - Keywords: {len(keywords) if keywords else 0}, Topics: {len(topics) if topics else 0}, Topic Keywords: {len(topic_keywords) if topic_keywords else 0}")
    logger.debug(f"Response body: {json.dumps(response_data)}")

    return JSONResponse(
      content=response_data,
      status_code=status.HTTP_200_OK
    )
  except Exception as e:
    error_msg = f"An internal error occurred: {str(e)}"
    logger.error(f"Keyword extraction failed - {type(e).__name__}: {str(e)}")

    return JSONResponse(
      content={
        "error": error_msg,
        "keywords": None,
        "topics": None,
        "topic_keywords": None
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )