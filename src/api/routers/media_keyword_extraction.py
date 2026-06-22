from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
import json

from src.api.services.media_keyword_extraction_service import extract_media_keywords_and_topics
from src.api.schemas.media_keyword_extraction_schema import (
  MediaKeywordExtractionRequest,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/media/keywords")
def media_keyword_extraction(request: MediaKeywordExtractionRequest) -> JSONResponse:
  logger.info(
    f"Media keyword extraction request - Type: '{request.media_type or 'content'}', "
    f"Title: '{request.media.get('title', 'N/A')}', Topics: {len(request.topics_list)}"
  )

  try:
    keywords, topics, topic_keywords = extract_media_keywords_and_topics(
      media=request.media,
      media_type=request.media_type,
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

    logger.info(
      f"Media keyword extraction response - Keywords: {len(keywords) if keywords else 0}, "
      f"Topics: {len(topics) if topics else 0}, Topic Keywords: {len(topic_keywords) if topic_keywords else 0}"
    )
    logger.debug(f"Response body: {json.dumps(response_data)}")

    return JSONResponse(
      content=response_data,
      status_code=status.HTTP_200_OK
    )
  except Exception as e:
    error_msg = f"An internal error occurred: {str(e)}"
    logger.error(f"Media keyword extraction failed - {type(e).__name__}: {str(e)}")

    return JSONResponse(
      content={
        "error": error_msg,
        "keywords": None,
        "topics": None,
        "topic_keywords": None
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
