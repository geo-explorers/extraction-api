"""Independent reject-only semantic review for grounded debate candidates."""

import json
import re

from google import genai
from google.genai import types

from src.api.schemas.news_claim_extract_schema import ExtractedClaim, NewsArticleSource
from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.schemas.news_debate_semantic_review_schema import (
  DebateSemanticReviewResponse,
  DebateSemanticVerdict,
)
from src.config.prompts.news_debate_semantic_review_prompt import (
  NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
_REQUEST_TIMEOUT_MS = 180_000
_CLAUDE_SYSTEM_PROMPT = (
  "You are a strict reject-only semantic reviewer for news debate cards. "
  "Use only the supplied material. Output only the requested JSON object."
)


def _build_review_prompt(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> str:
  factual_claims = [
    {
      "claim_index": index,
      "text": claim.text,
      "source_indices": claim.source_indices,
    }
    for index, claim in enumerate(claims)
  ]
  grounded_candidates = [
    {"candidate_index": index, **candidate.model_dump()}
    for index, candidate in enumerate(candidates)
  ]
  prior_axes = [
    {
      "proposition": candidate.text,
      "neutral_question": candidate.neutral_question,
    }
    for candidate in (prior_candidates or [])
  ]
  return NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT.format(
    headline=headline,
    claims=json.dumps(factual_claims, ensure_ascii=False),
    candidates=json.dumps(grounded_candidates, ensure_ascii=False),
    prior_axes=json.dumps(prior_axes, ensure_ascii=False),
    sources=json.dumps(
      [source.model_dump() for source in sources],
      ensure_ascii=False,
    ),
  )


def failed_semantic_gates(verdict: DebateSemanticVerdict) -> list[str]:
  """Compute failure reasons locally; never trust a model-authored pass flag."""
  failed: list[str] = []
  if not verdict.real_societal_debate:
    failed.append("NOT_SOCIETAL_DEBATE")
  if not verdict.raised_by_story:
    failed.append("NOT_FROM_STORY")
  if verdict.invented_facts or not verdict.no_invented_facts:
    failed.append("INVENTED_FACTS")
  if not verdict.distinct_axis or verdict.duplicate_of is not None:
    failed.append("DUPLICATE_AXIS")
  return failed


def apply_semantic_review(
  candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
  *,
  enforce: bool,
) -> list[GroundedDebateCandidate]:
  """Apply one complete verdict per candidate, failing closed when enforced."""
  verdicts_by_index: dict[int, list[DebateSemanticVerdict]] = {}
  for verdict in verdicts:
    verdicts_by_index.setdefault(verdict.candidate_index, []).append(verdict)

  accepted: list[GroundedDebateCandidate] = []
  for index, candidate in enumerate(candidates):
    matching = verdicts_by_index.get(index, [])
    if len(matching) != 1:
      failed = ["MISSING_VERDICT" if not matching else "DUPLICATE_VERDICT"]
    else:
      failed = failed_semantic_gates(matching[0])

    if failed:
      action = "Rejected" if enforce else "Shadow-rejected"
      logger.info(
        f"{action} debate candidate {index} at semantic review: {', '.join(failed)}"
      )
      if enforce:
        continue
    accepted.append(candidate)

  return accepted


def _gemini_review(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> DebateSemanticReviewResponse:
  prompt = _build_review_prompt(
    headline, sources, claims, candidates, prior_candidates
  )
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  config_kwargs: dict = {
    "temperature": settings.gemini_news_debate_review_temperature,
    "response_mime_type": "application/json",
    "response_schema": DebateSemanticReviewResponse,
  }
  thinking_level = (settings.gemini_news_debate_review_thinking_level or "").strip()
  if thinking_level:
    config_kwargs["thinking_config"] = types.ThinkingConfig(
      thinking_level=thinking_level,
    )
  config = types.GenerateContentConfig(**config_kwargs)

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = client.models.generate_content(
        model=settings.gemini_news_debate_review_model,
        contents=prompt,
        config=config,
      )
      return DebateSemanticReviewResponse.model_validate_json(response.text)
    except Exception as e:
      last_error = e
      logger.warning(
        f"News debate semantic review attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"News debate semantic review failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("News debate semantic review: unreachable code path")


def _parse_json(raw: str) -> dict:
  try:
    return json.loads(raw)
  except (json.JSONDecodeError, TypeError):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw or "", re.DOTALL | re.I)
    if not match:
      raise
    return json.loads(match.group(1))


def _claude_review(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> DebateSemanticReviewResponse:
  prompt = _build_review_prompt(
    headline, sources, claims, candidates, prior_candidates
  )
  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_MS / 1000,
    )
  except Exception as e:
    raise Exception("Error building Claude debate review client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=10000,
        temperature=settings.gemini_news_debate_review_temperature,
        system=_CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
      )
      raw = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
      )
      return DebateSemanticReviewResponse.model_validate(_parse_json(raw))
    except Exception as e:
      last_error = e
      logger.warning(
        f"Claude debate semantic review attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Claude debate semantic review failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("Claude debate semantic review: unreachable code path")


def review_news_debate_candidates(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  *,
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Review Gemini candidates and return accepted candidates plus the audit."""
  if not candidates:
    return [], []
  if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not configured for news debate semantic review")

  result = _gemini_review(
    headline, sources, claims, candidates, prior_candidates
  )
  accepted = apply_semantic_review(
    candidates,
    result.verdicts,
    enforce=settings.news_debate_semantic_review_enforced,
  )
  return accepted, result.verdicts


def review_news_debate_candidates_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  *,
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Review Claude candidates with the same reject-only semantic contract."""
  if not candidates:
    return [], []
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for debate semantic review")

  result = _claude_review(
    headline, sources, claims, candidates, prior_candidates
  )
  accepted = apply_semantic_review(
    candidates,
    result.verdicts,
    enforce=settings.news_debate_semantic_review_enforced,
  )
  return accepted, result.verdicts
