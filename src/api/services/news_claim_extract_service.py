import json
import re
from typing import Any, Dict, List

from src.api.schemas.news_claim_extract_schema import (
  NewsArticleSource,
  NewsClaimExtractResponse,
)
from src.api.utils import llm_model
from src.config.prompts.news_claim_extract_prompt import NEWS_CLAIM_EXTRACT_PROMPT
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def extract_news_claims(
  headline: str,
  sources: List[NewsArticleSource],
  topics: List[str],
) -> NewsClaimExtractResponse:
  """Fresh news-claim extraction via a single Gemini Pro call.

  Combines what news-worker currently does as Pass 2 (claims by topic) +
  Pass 3 (cross-source verification) + Pass 4 (perspectives + summary)
  into one coordinated structured-output Gemini call. Pass 1 (topics) is
  provided by the caller; Pass 5 (entities) remains on news-worker.

  Uses the premium Gemini model (gemini_premium_model) for large-context
  reasoning over full source bodies plus topics.
  """
  try:
    chain = llm_model.build_chain(
      prompt=NEWS_CLAIM_EXTRACT_PROMPT,
      model_name=settings.gemini_premium_model,
      temperature=settings.gemini_premium_temperature,
    )
  except Exception as e:
    logger.error(f"Failed to build extraction chain: {e}")
    raise Exception("Error building chain") from e

  invoke_params: Dict[str, Any] = {
    "headline": headline,
    "sources": [s.model_dump() for s in sources],
    "topics": topics,
  }

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      raw_response = chain.invoke(invoke_params)
      parsed = _parse_llm_response(raw_response)
      return NewsClaimExtractResponse.model_validate(parsed)
    except Exception as e:
      last_error = e
      logger.warning(
        f"News claim extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"News claim extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  # Unreachable — loop above either returns or raises
  raise Exception("News claim extraction: unreachable code path")


def _parse_llm_response(raw_response: str) -> dict:
  """Parse a JSON response from the LLM, handling markdown fences."""
  try:
    return json.loads(raw_response)
  except (json.JSONDecodeError, TypeError):
    pass

  response_text = raw_response.strip()
  fenced_match = re.search(
    r"```(?:json)?\s*(.*?)\s*```",
    response_text,
    re.DOTALL | re.IGNORECASE,
  )

  if fenced_match:
    response_text = fenced_match.group(1)

  return json.loads(response_text)
