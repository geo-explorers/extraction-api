import json
import re
from typing import List

from google import genai
from google.genai import types

from src.api.schemas.news_claim_extract_schema import (
  NewsArticleSource,
  NewsClaimExtractResponse,
)
from src.config.prompts.news_claim_extract_prompt import NEWS_CLAIM_EXTRACT_PROMPT
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3

# Request timeout (ms). Generous — a dense multi-source extraction can take a
# while, and the news-worker caller has its own 240s budget + retries on top.
_REQUEST_TIMEOUT_MS = 180_000


def extract_news_claims(
  headline: str,
  sources: List[NewsArticleSource],
  topics: List[str],
) -> NewsClaimExtractResponse:
  """Fresh news-claim extraction via a single Gemini call.

  Combines what news-worker currently does as Pass 2 (claims by topic) +
  Pass 3 (cross-source verification) + Pass 4 (perspectives + summary)
  into one coordinated structured-output Gemini call. Pass 1 (topics) is
  provided by the caller; Pass 5 (entities) remains on news-worker.

  Uses gemini_news_claim_model (default gemini-3.5-flash) with a low
  thinking level — ~2-5x faster than gemini-2.5-pro at equal/better claim
  quality (benchmarked 2026-05-27). Called directly via the google-genai
  SDK rather than the shared langchain build_chain because thinking_level is
  only exposed by the consolidated SDK, not langchain-google-genai 3.x. This
  keeps the change isolated to the news endpoint — the other extraction
  services and the premium podcast pipeline are untouched.
  """
  if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not configured for news claim extraction")

  # Render the prompt with the same f-string template substitution langchain
  # used previously (verified equivalent: {{ }} -> { }, list values str()'d),
  # then call google-genai directly so we can pass thinking_level.
  prompt = NEWS_CLAIM_EXTRACT_PROMPT.format(
    headline=headline,
    sources=[s.model_dump() for s in sources],
    topics=topics,
  )

  try:
    client = genai.Client(
      api_key=settings.gemini_api_key,
      http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
    )
    config_kwargs: dict = {"temperature": settings.gemini_news_claim_temperature}
    # thinking_level is a Gemini-3+ control. Only attach it when set, so a
    # revert to a 2.5-era model (e.g. GEMINI_NEWS_CLAIM_MODEL=gemini-2.5-pro
    # with GEMINI_NEWS_CLAIM_THINKING_LEVEL unset) runs cleanly with no
    # thinking config and no error.
    thinking_level = (settings.gemini_news_claim_thinking_level or "").strip()
    if thinking_level:
      config_kwargs["thinking_config"] = types.ThinkingConfig(
        thinking_level=thinking_level,
      )
    config = types.GenerateContentConfig(**config_kwargs)
  except Exception as e:
    logger.error(f"Failed to build news extraction client/config: {e}")
    raise Exception("Error building extraction client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = client.models.generate_content(
        model=settings.gemini_news_claim_model,
        contents=prompt,
        config=config,
      )
      parsed = _parse_llm_response(response.text)
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
