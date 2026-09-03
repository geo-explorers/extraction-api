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

# Minimal role primer for the Claude fallback. All extraction logic +
# grounding rules live in NEWS_CLAIM_EXTRACT_PROMPT (the user message) — this
# only sets the system role and reinforces raw-JSON output. NOT a second copy
# of the extraction prompt.
_CLAUDE_SYSTEM_PROMPT = (
  "You are an expert news fact-extraction system. Follow the user's instructions "
  "exactly and with zero hallucination tolerance. Output ONLY a single valid JSON "
  "object matching the requested schema — no prose, no markdown code fences."
)


def _build_prompt(
  headline: str,
  sources: List[NewsArticleSource],
  topics: List[str],
) -> str:
  """Render NEWS_CLAIM_EXTRACT_PROMPT — the single source of truth shared by
  both the Gemini and Claude extraction paths (same f-string substitution
  langchain used previously: {{ }} -> { }, list values str()'d)."""
  return NEWS_CLAIM_EXTRACT_PROMPT.format(
    headline=headline,
    sources=[s.model_dump() for s in sources],
    topics=topics,
  )


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

  # Render the shared prompt, then call google-genai directly so we can pass
  # thinking_level (only exposed by the consolidated SDK, not langchain 3.x).
  prompt = _build_prompt(headline, sources, topics)

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


def extract_news_claims_claude(
  headline: str,
  sources: List[NewsArticleSource],
  topics: List[str],
) -> NewsClaimExtractResponse:
  """Fallback news-claim extraction running the SAME strong prompt on Claude.

  Identical contract to extract_news_claims() — same NEWS_CLAIM_EXTRACT_PROMPT,
  same NewsClaimExtractResponse schema — but invokes Anthropic Claude instead
  of Gemini. news-worker calls this only when the Gemini path errors out, so a
  Gemini failure no longer drops the pipeline onto a weaker, locally-defined
  prompt (the root cause of the 2026-06-10 inverted-claim incident). The
  prompt is NOT duplicated: it is built once by _build_prompt(), shared with
  the Gemini path.

  anthropic is imported lazily so that a missing/unsynced SDK can never break
  module import for the primary Gemini endpoint.
  """
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for Claude news claim fallback")

  prompt = _build_prompt(headline, sources, topics)

  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_MS / 1000,
    )
  except Exception as e:
    logger.error(f"Failed to build Claude news extraction client: {e}")
    raise Exception("Error building Claude extraction client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=settings.news_claim_claude_max_tokens,
        temperature=settings.gemini_news_claim_temperature,
        system=_CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
      )
      parsed = _parse_llm_response(_claude_text(message))
      return NewsClaimExtractResponse.model_validate(parsed)
    except Exception as e:
      last_error = e
      logger.warning(
        f"Claude news claim extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Claude news claim extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  # Unreachable — loop above either returns or raises
  raise Exception("Claude news claim extraction: unreachable code path")


def _claude_text(message) -> str:
  """Concatenate the text blocks of an Anthropic Messages response."""
  parts = [
    block.text
    for block in message.content
    if getattr(block, "type", None) == "text"
  ]
  return "".join(parts)


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
