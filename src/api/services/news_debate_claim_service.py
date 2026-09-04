"""Debate generation for news stories, built on the product definition.

The model composes headline-style contested claims; deterministic checks here
enforce only the card contract (style, exactly two sides, duplicate collapse,
valid source indices) and the reject-only semantic review owns the judgment
calls. Valid candidates project down to the public ``ExtractedDebateClaim``.
"""

import json
import re
from typing import Iterable

from google import genai
from google.genai import types

from src.api.schemas.news_claim_extract_schema import (
  ExtractedClaim,
  ExtractedDebateClaim,
  NewsArticleSource,
  normalize_debate_claims,
)
from src.api.schemas.news_debate_claim_schema import (
  GroundedDebateCandidate,
  GroundedDebateResponse,
)
from src.api.schemas.news_debate_semantic_review_schema import DebateSemanticVerdict
from src.config.prompts.news_debate_claim_prompt import NEWS_DEBATE_CLAIM_PROMPT
from src.config.prompts.news_debate_completion_prompt import (
  NEWS_DEBATE_UNDERFILLED_RESCUE_PROMPT,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger
from src.api.services.news_debate_semantic_review_service import (
  failed_semantic_gates,
  review_news_debate_candidates,
  review_news_debate_candidates_claude,
)

logger = get_logger(__name__)

MAX_RETRIES = 3
_REQUEST_TIMEOUT_MS = 180_000
_HEDGE = re.compile(r"\b(?:may|might|could|some\s+argue|whether)\b", re.IGNORECASE)
_INFERENTIAL_FRAME = re.compile(
  r"\b(?:signals?|proves?|demonstrates?|indicates?|is\s+evidence\s+of)\b",
  re.IGNORECASE,
)

_CLAUDE_SYSTEM_PROMPT = (
  "You write news debate cards. Follow the user's definition and rules "
  "exactly. Output ONLY one valid JSON object matching the requested shape."
)


# Glyph variants that carry no difference in wording (curly vs straight quotes,
# en/em dashes, invisible marks). Folding them keeps the duplicate-collapse
# keys stable when the model and the rescue pass render the same question with
# different typography.
_TYPOGRAPHIC = str.maketrans({
  "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
  "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
  "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
  "―": "-", "−": "-",
  "…": "...",
  # Invisible formatting marks: never wording, and they defeat substring tests.
  # Written as escapes because the literal glyphs are unreadable in source.
  "\u00ad": "", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
})


def _normalized(value: str) -> str:
  """Normalize whitespace/case/glyphs for duplicate-collapse keys."""
  return " ".join((value or "").translate(_TYPOGRAPHIC).split()).casefold()


def _claims_context(claims: list[ExtractedClaim]) -> str:
  """Compact story-fact digest given to the prompts as reference material."""
  return json.dumps(
    [
      {
        "claim_index": index,
        "text": claim.text,
        "source_indices": claim.source_indices,
      }
      for index, claim in enumerate(claims)
    ],
    ensure_ascii=False,
  )


def _build_prompt(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> str:
  return NEWS_DEBATE_CLAIM_PROMPT.format(
    headline=headline,
    central_claims=_claims_context(claims),
    sources=json.dumps(
      [source.model_dump() for source in sources],
      ensure_ascii=False,
    ),
  )


def _build_underfilled_rescue_prompt(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  accepted_candidates: list[GroundedDebateCandidate],
  attempted_candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
) -> str:
  """Build a targeted completion prompt with the first pass's audit trail."""
  verdicts_by_index = {
    verdict.candidate_index: verdict
    for verdict in verdicts
  }
  attempted_axes = []
  for index, candidate in enumerate(attempted_candidates):
    verdict = verdicts_by_index.get(index)
    attempted_axes.append({
      "candidate_index": index,
      "proposition": candidate.text,
      "neutral_question": candidate.neutral_question,
      "semantic_failure_codes": (
        failed_semantic_gates(verdict) if verdict is not None else ["MISSING_VERDICT"]
      ),
    })

  survivor_count = len(accepted_candidates)
  return NEWS_DEBATE_UNDERFILLED_RESCUE_PROMPT.format(
    headline=headline,
    central_claims=_claims_context(claims),
    survivor_count=survivor_count,
    minimum_needed=max(0, 3 - survivor_count),
    maximum_new=max(0, 5 - survivor_count),
    surviving_candidates=json.dumps(
      [candidate.model_dump() for candidate in accepted_candidates],
      ensure_ascii=False,
    ),
    attempted_axes=json.dumps(attempted_axes, ensure_ascii=False),
    sources=json.dumps(
      [source.model_dump() for source in sources],
      ensure_ascii=False,
    ),
  )


def filter_grounded_debate_candidates(
  candidates: Iterable[GroundedDebateCandidate],
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[GroundedDebateCandidate]:
  """Enforce the deterministic card contract; judgment belongs to review.

  Checks only what the product definition makes mechanical: headline-style
  text (length, no questions or hedging), a neutral question for duplicate
  collapse, exactly two named sides, and known source indices (invalid ones
  are dropped, not fatal — attribution is best-effort). ``claims`` stays in
  the signature for caller stability; nothing here needs it.
  """
  del claims
  valid_source_indices = {source.index for source in sources}
  seen_questions: set[str] = set()
  seen_texts: set[str] = set()
  accepted: list[GroundedDebateCandidate] = []

  for position, candidate in enumerate(candidates):
    reason: str | None = None
    text = " ".join(candidate.text.split())
    question_key = _normalized(candidate.neutral_question)
    text_key = _normalized(text).rstrip(".")

    if not text:
      reason = "empty proposition"
    elif len(text.split()) > 20:
      reason = "proposition exceeds 20 words"
    elif "?" in text or _HEDGE.search(text):
      reason = "proposition is a question or contains hedging"
    elif _INFERENTIAL_FRAME.search(text):
      reason = "proposition uses evidentiary-summary framing"
    elif not question_key:
      reason = "missing neutral question"
    elif question_key in seen_questions or text_key in seen_texts:
      reason = "duplicate neutral question or proposition"
    elif len(candidate.opposing_positions) != 2 or any(
      not side.strip() for side in candidate.opposing_positions
    ):
      reason = "candidate does not provide exactly two opposing positions"

    if reason is not None:
      logger.info(f"Rejected debate candidate {position}: {reason}")
      continue

    candidate.text = text
    candidate.source_indices = [
      index
      for index in dict.fromkeys(candidate.source_indices)
      if index in valid_source_indices
    ]
    accepted.append(candidate)
    seen_questions.add(question_key)
    seen_texts.add(text_key)

  # Candidate generation is deliberately higher-recall than publication; the
  # semantic review chooses survivors and normalize_debate_claims caps at 5.
  return accepted[:8]


def project_debate_candidates(
  candidates: Iterable[GroundedDebateCandidate],
) -> list[ExtractedDebateClaim]:
  """Project reviewed internal candidates to the stable public API contract."""
  public = [
    ExtractedDebateClaim(
      text=candidate.text,
      source_indices=list(candidate.source_indices),
    )
    for candidate in candidates
  ]
  return normalize_debate_claims(public)


def validate_grounded_debate_candidates(
  candidates: Iterable[GroundedDebateCandidate],
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[ExtractedDebateClaim]:
  """Backward-compatible mechanical validation + public projection helper."""
  grounded = filter_grounded_debate_candidates(candidates, sources, claims)
  return project_debate_candidates(grounded)


def generate_news_debate_candidates(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[GroundedDebateCandidate]:
  """Generate candidates and apply deterministic grounding with Gemini."""
  if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not configured for news debate extraction")
  if not sources or not claims:
    return []

  prompt = _build_prompt(headline, sources, claims)
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  config_kwargs: dict = {
    "temperature": settings.gemini_news_debate_temperature,
    "response_mime_type": "application/json",
    "response_schema": GroundedDebateResponse,
  }
  thinking_level = (settings.gemini_news_debate_thinking_level or "").strip()
  if thinking_level:
    config_kwargs["thinking_config"] = types.ThinkingConfig(
      thinking_level=thinking_level,
    )
  config = types.GenerateContentConfig(**config_kwargs)

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = client.models.generate_content(
        model=settings.gemini_news_debate_model,
        contents=prompt,
        config=config,
      )
      parsed = GroundedDebateResponse.model_validate_json(response.text)
      return filter_grounded_debate_candidates(parsed.debate_claims, sources, claims)
    except Exception as e:
      last_error = e
      logger.warning(
        f"News debate extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"News debate extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("News debate extraction: unreachable code path")


def generate_news_debate_candidates_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[GroundedDebateCandidate]:
  """Generate candidates and apply the same deterministic gates with Claude."""
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for news debate extraction")
  if not sources or not claims:
    return []

  prompt = _build_prompt(headline, sources, claims)
  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_MS / 1000,
    )
  except Exception as e:
    raise Exception("Error building Claude news debate client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=8000,
        temperature=settings.gemini_news_debate_temperature,
        system=_CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
      )
      raw = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
      )
      parsed = GroundedDebateResponse.model_validate_json(raw)
      return filter_grounded_debate_candidates(parsed.debate_claims, sources, claims)
    except Exception as e:
      last_error = e
      logger.warning(
        f"Claude news debate extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Claude news debate extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("Claude news debate extraction: unreachable code path")


def _exclude_attempted_exact_axes(
  candidates: list[GroundedDebateCandidate],
  attempted_candidates: list[GroundedDebateCandidate],
  limit: int,
) -> list[GroundedDebateCandidate]:
  """Prevent completion from replaying even a rejected first-pass axis."""
  attempted_questions = {
    _normalized(candidate.neutral_question) for candidate in attempted_candidates
  }
  attempted_texts = {
    _normalized(candidate.text).rstrip(".") for candidate in attempted_candidates
  }
  return [
    candidate
    for candidate in candidates
    if _normalized(candidate.neutral_question) not in attempted_questions
    and _normalized(candidate.text).rstrip(".") not in attempted_texts
  ][:limit]


def generate_news_debate_underfilled_rescue(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  accepted_candidates: list[GroundedDebateCandidate],
  attempted_candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
) -> list[GroundedDebateCandidate]:
  """Ask Gemini for enough new axes to complete a 3-5 claim collection."""
  if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not configured for news debate completion")
  if not sources or not claims:
    return []

  max_new = max(0, 5 - len(accepted_candidates))
  if max_new == 0:
    return []
  prompt = _build_underfilled_rescue_prompt(
    headline,
    sources,
    claims,
    accepted_candidates,
    attempted_candidates,
    verdicts,
  )
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  config_kwargs: dict = {
    "temperature": settings.gemini_news_debate_temperature,
    "response_mime_type": "application/json",
    "response_schema": GroundedDebateResponse,
  }
  thinking_level = (settings.gemini_news_debate_thinking_level or "").strip()
  if thinking_level:
    config_kwargs["thinking_config"] = types.ThinkingConfig(
      thinking_level=thinking_level,
    )
  config = types.GenerateContentConfig(**config_kwargs)

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = client.models.generate_content(
        model=settings.gemini_news_debate_model,
        contents=prompt,
        config=config,
      )
      parsed = GroundedDebateResponse.model_validate_json(response.text)
      grounded = filter_grounded_debate_candidates(
        parsed.debate_claims, sources, claims
      )
      return _exclude_attempted_exact_axes(
        grounded, attempted_candidates, max_new
      )
    except Exception as e:
      last_error = e
      logger.warning(
        f"News debate completion attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"News debate completion failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("News debate completion: unreachable code path")


def generate_news_debate_underfilled_rescue_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  accepted_candidates: list[GroundedDebateCandidate],
  attempted_candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
) -> list[GroundedDebateCandidate]:
  """Ask Claude for the same conditional collection completion."""
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for news debate completion")
  if not sources or not claims:
    return []

  max_new = max(0, 5 - len(accepted_candidates))
  if max_new == 0:
    return []
  prompt = _build_underfilled_rescue_prompt(
    headline,
    sources,
    claims,
    accepted_candidates,
    attempted_candidates,
    verdicts,
  )
  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_MS / 1000,
    )
  except Exception as e:
    raise Exception("Error building Claude news debate completion client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=4000,
        temperature=settings.gemini_news_debate_temperature,
        system=_CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
      )
      raw = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
      )
      parsed = GroundedDebateResponse.model_validate_json(raw)
      grounded = filter_grounded_debate_candidates(
        parsed.debate_claims, sources, claims
      )
      return _exclude_attempted_exact_axes(
        grounded, attempted_candidates, max_new
      )
    except Exception as e:
      last_error = e
      logger.warning(
        f"Claude debate completion attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Claude debate completion failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception("Claude debate completion: unreachable code path")


def complete_underfilled_news_debate_candidates(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  attempted_candidates: list[GroundedDebateCandidate],
  accepted_candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Best-effort Gemini completion for an underfilled first-pass result.

  Zero survivors WITH attempted candidates gets one fresh generation + review
  draw: thin-supply stories sit at the 3-floor with no margin, so a single
  review flip otherwise zeroes a story whose axes pass cleanly on a redraw.
  The redraw is reviewed WITHOUT prior-axes context — an independent second
  opinion, unlike rescue, which must never replay rejected axes. One or two
  survivors get the focused rescue attempt with the first pass's audit trail.
  Zero survivors from zero candidates stays terminal. Either path spends at
  most one generation and one review call.
  """
  if len(accepted_candidates) >= 3:
    return accepted_candidates, []

  if len(accepted_candidates) == 0:
    if not settings.news_debate_zero_retry_enabled or not attempted_candidates:
      logger.info("News debate completion skipped: zero first-pass survivors")
      return accepted_candidates, []
    try:
      redraw = generate_news_debate_candidates(headline, sources, claims)
      retried, retry_verdicts = review_news_debate_candidates(
        headline, sources, claims, redraw
      )
      logger.info(
        f"News debate zero-retry: {len(redraw)} redraw candidates → "
        f"{len(retried)} accepted"
      )
      return retried[:5], retry_verdicts
    except Exception as e:
      logger.warning(f"News debate zero-retry skipped after failure: {e}")
      return accepted_candidates, []

  if not settings.news_debate_underfilled_rescue_enabled:
    return accepted_candidates, []

  try:
    rescued = generate_news_debate_underfilled_rescue(
      headline,
      sources,
      claims,
      accepted_candidates,
      attempted_candidates,
      verdicts,
    )
    if not rescued:
      logger.info("News debate completion found no additional grounded axes")
      return accepted_candidates, []

    reviewed, completion_verdicts = review_news_debate_candidates(
      headline,
      sources,
      claims,
      rescued,
      prior_candidates=attempted_candidates,
    )
    completed = [*accepted_candidates, *reviewed][:5]
    logger.info(
      f"News debate completion added {len(reviewed)} reviewed axes; "
      f"{len(completed)} total"
    )
    return completed, completion_verdicts
  except Exception as e:
    logger.warning(f"News debate completion skipped after failure: {e}")
    return accepted_candidates, []


def complete_underfilled_news_debate_candidates_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  attempted_candidates: list[GroundedDebateCandidate],
  accepted_candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Best-effort Claude completion with the same retry + rescue policy."""
  if len(accepted_candidates) >= 3:
    return accepted_candidates, []

  if len(accepted_candidates) == 0:
    if not settings.news_debate_zero_retry_enabled or not attempted_candidates:
      logger.info("Claude debate completion skipped: zero first-pass survivors")
      return accepted_candidates, []
    try:
      redraw = generate_news_debate_candidates_claude(headline, sources, claims)
      retried, retry_verdicts = review_news_debate_candidates_claude(
        headline, sources, claims, redraw
      )
      logger.info(
        f"Claude debate zero-retry: {len(redraw)} redraw candidates → "
        f"{len(retried)} accepted"
      )
      return retried[:5], retry_verdicts
    except Exception as e:
      logger.warning(f"Claude debate zero-retry skipped after failure: {e}")
      return accepted_candidates, []

  if not settings.news_debate_underfilled_rescue_enabled:
    return accepted_candidates, []

  try:
    rescued = generate_news_debate_underfilled_rescue_claude(
      headline,
      sources,
      claims,
      accepted_candidates,
      attempted_candidates,
      verdicts,
    )
    if not rescued:
      logger.info("Claude debate completion found no additional grounded axes")
      return accepted_candidates, []

    reviewed, completion_verdicts = review_news_debate_candidates_claude(
      headline,
      sources,
      claims,
      rescued,
      prior_candidates=attempted_candidates,
    )
    completed = [*accepted_candidates, *reviewed][:5]
    logger.info(
      f"Claude debate completion added {len(reviewed)} reviewed axes; "
      f"{len(completed)} total"
    )
    return completed, completion_verdicts
  except Exception as e:
    logger.warning(f"Claude debate completion skipped after failure: {e}")
    return accepted_candidates, []


def extract_news_debate_claims(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[ExtractedDebateClaim]:
  """Complete Gemini candidate -> grounding -> review -> completion pipeline."""
  candidates = generate_news_debate_candidates(headline, sources, claims)
  accepted, verdicts = review_news_debate_candidates(
    headline, sources, claims, candidates
  )
  accepted, _ = complete_underfilled_news_debate_candidates(
    headline, sources, claims, candidates, accepted, verdicts
  )
  return project_debate_candidates(accepted)


def extract_news_debate_claims_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
) -> list[ExtractedDebateClaim]:
  """Complete Claude candidate -> grounding -> review -> completion pipeline."""
  candidates = generate_news_debate_candidates_claude(headline, sources, claims)
  accepted, verdicts = review_news_debate_candidates_claude(
    headline, sources, claims, candidates
  )
  accepted, _ = complete_underfilled_news_debate_candidates_claude(
    headline, sources, claims, candidates, accepted, verdicts
  )
  return project_debate_candidates(accepted)
