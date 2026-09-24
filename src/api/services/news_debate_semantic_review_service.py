"""Independent reject-only semantic review for grounded debate candidates,
with a second opinion on the judgment gates.

The review is one call over the full sources. Two of its four gates are
judgment calls, and a rejected real debate is the product's costliest
mistake: given the same set the reviewer is near deterministic, but it judges
the set as a whole, so a draw with several off-story cards can take the
on-story ones down with them. When the first review leaves the set
underfilled, or with no card on the headline's disagreement while it
rejected one, the cards it rejected on the judgment gates alone get one more
reading on a smaller input (headline, story facts, those cards, the accepted
cards as context), so the second reading is independent of the set the first
one saw. Either reading passing a judgment gate passes it; the fact and
duplicate gates are objective and stand as first judged.
"""

import json
import re
from typing import Callable

from google import genai
from google.genai import types

from src.api.schemas.news_claim_extract_schema import (
  DEBATE_MIN,
  ExtractedClaim,
  NewsArticleSource,
)
from src.api.schemas.news_debate_claim_schema import (
  GroundedDebateCandidate,
  graded_strength,
  strongest_first,
)
from src.api.schemas.news_debate_semantic_review_schema import (
  DebateJudgmentOpinion,
  DebateJudgmentOpinionResponse,
  DebateSemanticReviewResponse,
  DebateSemanticVerdict,
)
from src.config.overrides import llm, prompts
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
_REQUEST_TIMEOUT_MS = 180_000
# The two gates a second opinion may overturn. The other two are objective.
JUDGMENT_GATES = frozenset({"NOT_SOCIETAL_DEBATE", "NOT_FROM_STORY"})
# What the reviewer sees of a candidate: the writer's draft, never the grade
# fields the review itself fills in.
_GRADE_FIELDS = {"on_headline", "strength"}


def _claims_json(claims: list[ExtractedClaim]) -> str:
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


def _candidates_json(candidates: list[GroundedDebateCandidate]) -> str:
  return json.dumps(
    [
      {"candidate_index": index, **candidate.model_dump(exclude=_GRADE_FIELDS)}
      for index, candidate in enumerate(candidates)
    ],
    ensure_ascii=False,
  )


def _axes_json(candidates: list[GroundedDebateCandidate]) -> str:
  return json.dumps(
    [
      {
        "proposition": candidate.text,
        "neutral_question": candidate.neutral_question,
      }
      for candidate in candidates
    ],
    ensure_ascii=False,
  )


def _build_review_prompt(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> str:
  return prompts.get("news_debate_semantic_review").format(
    headline=headline,
    claims=_claims_json(claims),
    candidates=_candidates_json(candidates),
    prior_axes=_axes_json(prior_candidates or []),
    sources=json.dumps(
      [source.model_dump() for source in sources],
      ensure_ascii=False,
    ),
  )


def _build_opinion_prompt(
  headline: str,
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  accepted: list[GroundedDebateCandidate],
) -> str:
  """The second opinion's input: no sources, the rejected cards alone, the
  accepted cards as context. A different input is what makes it a second
  reading rather than a replay."""
  return prompts.get("news_debate_semantic_review.judgment_opinion").format(
    headline=headline,
    claims=_claims_json(claims),
    candidates=_candidates_json(candidates),
    accepted=_axes_json(accepted),
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


def _verdicts_by_index(
  verdicts: list[DebateSemanticVerdict],
) -> dict[int, list[DebateSemanticVerdict]]:
  by_index: dict[int, list[DebateSemanticVerdict]] = {}
  for verdict in verdicts:
    by_index.setdefault(verdict.candidate_index, []).append(verdict)
  return by_index


def apply_semantic_review(
  candidates: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
  *,
  enforce: bool,
) -> list[GroundedDebateCandidate]:
  """Apply one complete verdict per candidate, failing closed when enforced."""
  verdicts_by_index = _verdicts_by_index(verdicts)

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
    # The review's grade travels with the card from here: it orders the
    # published set and becomes the public claim's confidence.
    verdict = matching[0] if len(matching) == 1 else None
    candidate.on_headline = verdict.on_headline if verdict else False
    candidate.strength = (
      graded_strength(verdict.strength, verdict.on_headline) if verdict else 0.0
    )
    accepted.append(candidate)

  # Strongest first, the writer's own order breaking ties, so every later
  # cap (completion, projection) keeps the best-graded cards.
  return strongest_first(accepted, len(accepted))


def judgment_rejected(
  verdicts: list[DebateSemanticVerdict],
) -> list[DebateSemanticVerdict]:
  """The verdicts a second opinion may overturn: one verdict for the index,
  failed, and failed on the judgment gates alone."""
  return [
    matching[0]
    for matching in _verdicts_by_index(verdicts).values()
    if len(matching) == 1
    and (failed := failed_semantic_gates(matching[0]))
    and JUDGMENT_GATES.issuperset(failed)
  ]


def needs_second_opinion(
  accepted: list[GroundedDebateCandidate],
  rejected: list[DebateSemanticVerdict],
) -> bool:
  """Underfilled, or nothing accepted on the headline's disagreement while a
  card on it was rejected. A full set on the headline never pays for it."""
  if not rejected:
    return False
  if len(accepted) < DEBATE_MIN:
    return True
  return not any(candidate.on_headline for candidate in accepted) and any(
    verdict.on_headline for verdict in rejected
  )


def merge_second_opinion(
  verdict: DebateSemanticVerdict, opinion: DebateJudgmentOpinion
) -> DebateSemanticVerdict:
  """Either reading passing a judgment gate passes it. When the merged
  verdict now passes, the second opinion's grade and reasons replace the
  first's, which graded and explained a card it had rejected."""
  merged = verdict.model_copy(
    update={
      "on_headline": verdict.on_headline or opinion.on_headline,
      "real_societal_debate": (
        verdict.real_societal_debate or opinion.real_societal_debate
      ),
      "raised_by_story": verdict.raised_by_story or opinion.raised_by_story,
    }
  )
  if not failed_semantic_gates(merged):
    merged = merged.model_copy(
      update={
        "headline_analysis": opinion.headline_analysis,
        "debate_analysis": opinion.debate_analysis,
        "story_analysis": opinion.story_analysis,
        "strength": opinion.strength,
      }
    )
  return merged.model_copy(update={"failure_codes": failed_semantic_gates(merged)})


def second_opinion(
  opinion_call: Callable[..., DebateJudgmentOpinionResponse],
  headline: str,
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  accepted: list[GroundedDebateCandidate],
  verdicts: list[DebateSemanticVerdict],
  rejected: list[DebateSemanticVerdict],
) -> list[DebateSemanticVerdict]:
  """Re-judge the judgment-rejected cards once and fold the opinions into
  the verdicts. Best-effort: a failed call leaves the first verdicts."""
  indices = [
    verdict.candidate_index
    for verdict in rejected
    if 0 <= verdict.candidate_index < len(candidates)
  ]
  if not indices:
    return verdicts
  try:
    opinions = opinion_call(
      headline, claims, [candidates[index] for index in indices], accepted
    ).verdicts
  except Exception as e:
    logger.warning(f"News debate second opinion skipped after failure: {e}")
    return verdicts

  # The opinion indexes the recheck list; map it back to the candidate index.
  by_candidate = {
    indices[opinion.candidate_index]: opinion
    for opinion in opinions
    if 0 <= opinion.candidate_index < len(indices)
  }
  merged = [
    merge_second_opinion(verdict, by_candidate[verdict.candidate_index])
    if verdict.candidate_index in by_candidate
    else verdict
    for verdict in verdicts
  ]
  overturned = sum(
    1
    for verdict in merged
    if verdict.candidate_index in by_candidate and not failed_semantic_gates(verdict)
  )
  logger.info(
    f"News debate second opinion: {len(indices)} rechecked, {overturned} passed"
  )
  return merged


def _gemini_json(prompt: str, schema, label: str):
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  config_kwargs: dict = {
    "temperature": llm.get("gemini_news_debate_review_temperature"),
    "response_mime_type": "application/json",
    "response_schema": schema,
  }
  thinking_level = (llm.get("gemini_news_debate_review_thinking_level") or "").strip()
  if thinking_level:
    config_kwargs["thinking_config"] = types.ThinkingConfig(
      thinking_level=thinking_level,
    )
  config = types.GenerateContentConfig(**config_kwargs)

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      response = client.models.generate_content(
        model=llm.get("gemini_news_debate_review_model"),
        contents=prompt,
        config=config,
      )
      return schema.model_validate_json(response.text)
    except Exception as e:
      last_error = e
      logger.warning(f"{label} attempt {attempt}/{MAX_RETRIES} failed: {e}")
      if attempt == MAX_RETRIES:
        raise Exception(f"{label} failed after {MAX_RETRIES} attempts") from last_error

  raise Exception(f"{label}: unreachable code path")


def _gemini_review(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> DebateSemanticReviewResponse:
  return _gemini_json(
    _build_review_prompt(headline, sources, claims, candidates, prior_candidates),
    DebateSemanticReviewResponse,
    "News debate semantic review",
  )


def _gemini_opinion(
  headline: str,
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  accepted: list[GroundedDebateCandidate],
) -> DebateJudgmentOpinionResponse:
  return _gemini_json(
    _build_opinion_prompt(headline, claims, candidates, accepted),
    DebateJudgmentOpinionResponse,
    "News debate second opinion",
  )


def _parse_json(raw: str) -> dict:
  try:
    return json.loads(raw)
  except (json.JSONDecodeError, TypeError):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw or "", re.DOTALL | re.I)
    if not match:
      raise
    return json.loads(match.group(1))


def _claude_json(prompt: str, schema, label: str):
  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_MS / 1000,
    )
  except Exception as e:
    raise Exception(f"Error building Claude client for {label}") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=llm.get("news_claim_claude_model"),
        max_tokens=10000,
        temperature=llm.get("gemini_news_debate_review_temperature"),
        system=prompts.get("news_debate_semantic_review.claude_system"),
        messages=[{"role": "user", "content": prompt}],
      )
      raw = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
      )
      return schema.model_validate(_parse_json(raw))
    except Exception as e:
      last_error = e
      logger.warning(f"Claude {label} attempt {attempt}/{MAX_RETRIES} failed: {e}")
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Claude {label} failed after {MAX_RETRIES} attempts"
        ) from last_error

  raise Exception(f"Claude {label}: unreachable code path")


def _claude_review(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None = None,
) -> DebateSemanticReviewResponse:
  return _claude_json(
    _build_review_prompt(headline, sources, claims, candidates, prior_candidates),
    DebateSemanticReviewResponse,
    "debate semantic review",
  )


def _claude_opinion(
  headline: str,
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  accepted: list[GroundedDebateCandidate],
) -> DebateJudgmentOpinionResponse:
  return _claude_json(
    _build_opinion_prompt(headline, claims, candidates, accepted),
    DebateJudgmentOpinionResponse,
    "debate second opinion",
  )


def _review(
  review_call: Callable[..., DebateSemanticReviewResponse],
  opinion_call: Callable[..., DebateJudgmentOpinionResponse],
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  prior_candidates: list[GroundedDebateCandidate] | None,
  second_opinion_allowed: bool,
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  verdicts = review_call(headline, sources, claims, candidates, prior_candidates).verdicts
  accepted = apply_semantic_review(
    candidates, verdicts, enforce=settings.news_debate_semantic_review_enforced
  )
  rejected = judgment_rejected(verdicts)
  if (
    second_opinion_allowed
    and settings.news_debate_second_opinion_enabled
    and needs_second_opinion(accepted, rejected)
  ):
    verdicts = second_opinion(
      opinion_call, headline, claims, candidates, accepted, verdicts, rejected
    )
    accepted = apply_semantic_review(
      candidates, verdicts, enforce=settings.news_debate_semantic_review_enforced
    )
  return accepted, verdicts


def review_news_debate_candidates(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  *,
  prior_candidates: list[GroundedDebateCandidate] | None = None,
  second_opinion: bool = True,
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Review Gemini candidates and return accepted candidates plus the audit.

  ``second_opinion`` is the first review's to spend; the completion pass
  turns it off so it stays within its own two-call budget."""
  if not candidates:
    return [], []
  if not settings.gemini_api_key:
    raise Exception("GEMINI_API_KEY not configured for news debate semantic review")

  return _review(
    _gemini_review,
    _gemini_opinion,
    headline,
    sources,
    claims,
    candidates,
    prior_candidates,
    second_opinion,
  )


def review_news_debate_candidates_claude(
  headline: str,
  sources: list[NewsArticleSource],
  claims: list[ExtractedClaim],
  candidates: list[GroundedDebateCandidate],
  *,
  prior_candidates: list[GroundedDebateCandidate] | None = None,
  second_opinion: bool = True,
) -> tuple[list[GroundedDebateCandidate], list[DebateSemanticVerdict]]:
  """Review Claude candidates with the same reject-only semantic contract."""
  if not candidates:
    return [], []
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for debate semantic review")

  return _review(
    _claude_review,
    _claude_opinion,
    headline,
    sources,
    claims,
    candidates,
    prior_candidates,
    second_opinion,
  )
