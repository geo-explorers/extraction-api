"""Overview-topic extraction (news Pass 1).

Ported from news-worker's injection branch (commands/enrich.ts on
feat/inject-urls-async-pipeline), where this ran client-side on Claude before a
separate claim-extraction POST. It now lives here so the topic pass and the
claim pass can run under one task (news.extract_topics_and_claims): the consumer
sends {headline, sources}, this produces the ordered topic labels, and the claim
pass groups claims under them.

`extract_overview_topics` is the only function that calls an LLM. The rest
(content pooling, label validation) are pure and unit-tested with no LLM. This
module is engine-agnostic — it imports nothing from the task layer or Hatchet.
"""

import re
from typing import List

from src.api.schemas.news_claim_extract_schema import NewsArticleSource
from src.api.services.news_claim_extract_service import (
  _claude_text,
  _parse_llm_response,
)
from src.config.prompts.news_topics_extract_prompt import (
  NEWS_TOPICS_EXTRACT_PROMPT,
  NEWS_TOPICS_REGENERATE_PROMPT,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# ── Tuning constants (mirror enrich.ts exactly) ──────────────────────────
LABEL_MIN_WORDS = 2
LABEL_MAX_WORDS = 5
MIN_OVERVIEW_TOPICS = 3
MAX_OVERVIEW_TOPICS = 6

# Longest source >= this many chars (or a single source) -> use it alone;
# otherwise pool every source so a thin multi-source story still shows all
# angles. Mirrors enrich.ts LONG_PRIMARY_THRESHOLD.
_LONG_PRIMARY_THRESHOLD = 1500
# The prompt sees at most this many characters of pooled content (char-based
# slice, multibyte-safe — matches content.slice(0, 12_000) in enrich.ts).
_CONTENT_CHAR_LIMIT = 12_000
# Below this much pooled content there is nothing to extract from; degrade to
# the single fallback topic rather than failing the whole task.
_MIN_CONTENT_CHARS = 100
# A source with empty body still contributes its title if the title is long
# enough to carry signal (mirrors the title/description fallback in enrich.ts).
_TITLE_FALLBACK_MIN_CHARS = 50
_FALLBACK_TOPIC = "Overview"

MAX_RETRIES = 3
_REQUEST_TIMEOUT_S = 180.0
_TOPIC_MAX_TOKENS = 1024
# Role primer only; all extraction rules live in the user-message prompt.
_TOPIC_SYSTEM_PROMPT = "You are an expert news analyst. Return only valid JSON."


# ── Pure helpers (no LLM; unit-tested) ───────────────────────────────────


def check_label_issues(label: str, max_words: int = LABEL_MAX_WORDS) -> List[str]:
  """Validate one topic label against the structural rules (mirrors
  checkLabelIssues). Structural only — there is intentionally NO word
  blocklist; style is taught by the prompt's few-shot examples. Returns a list
  of human-readable issues; an empty list means the label is valid."""
  issues: List[str] = []
  trimmed = label.strip()
  if not trimmed:
    issues.append("empty")
    return issues
  # JS `trimmed.split(/\s+/)`; Python str.split() with no args splits on runs
  # of whitespace, so the word count matches.
  words = len(trimmed.split())
  if words < LABEL_MIN_WORDS:
    issues.append(f"{words} words (min {LABEL_MIN_WORDS})")
  if words > max_words:
    issues.append(f"{words} words (max {max_words})")
  # Whitespace on BOTH sides, so "Android"/"England"/"sandal" do not match.
  if re.search(r"\s+and\s+", trimmed, re.IGNORECASE):
    issues.append('uses "and" conjunction')
  return issues


def find_label_issues(
  labels: List[str], max_words: int = LABEL_MAX_WORDS
) -> List[dict]:
  """Return {label, issues} for every label that has at least one issue
  (mirrors findLabelIssues)."""
  result: List[dict] = []
  for label in labels:
    issues = check_label_issues(label, max_words)
    if issues:
      result.append({"label": label, "issues": issues})
  return result


def validate_overview_topics_response(value) -> List[str]:
  """Coerce a parsed LLM response into a list of non-empty topic strings
  (mirrors validateOverviewTopicsResponse). Accepts a bare JSON array OR a
  {"topics": [...]} wrapper (some models wrap). Raises on anything else so the
  caller's retry loop can try again."""
  if isinstance(value, list):
    return [t for t in value if isinstance(t, str) and t.strip() != ""]
  if isinstance(value, dict) and isinstance(value.get("topics"), list):
    return [t for t in value["topics"] if isinstance(t, str) and t.strip() != ""]
  raise ValueError("Invalid LLM response: expected array of topic strings")


def pool_source_content(headline: str, sources: List[NewsArticleSource]) -> str:
  """Adaptive content selection mirroring enrich.ts.

  - Each source contributes its body content (or its title, when the body is
    empty but the title carries some signal).
  - If the longest source is rich (>= _LONG_PRIMARY_THRESHOLD chars) or there is
    only one source, use that source alone — pooling syndicated copies just
    inflates redundant claims.
  - Otherwise pool ALL sources as "Source {{i}} ({{url}}):\\n{{text}}" joined by
    a separator, so a story whose sources are individually thin still exposes
    every distinct angle.
  - Last resort (no usable source text): the headline.

  Returns the pooled text BEFORE the prompt char-limit slice (the caller slices).
  """
  contents: List[tuple[str, str]] = []
  for s in sources:
    text = (s.content or "").strip()
    if not text:
      fallback = (s.title or "").strip()
      if len(fallback) > _TITLE_FALLBACK_MIN_CHARS:
        text = fallback
    if text:
      contents.append((s.url, text))

  if not contents:
    return f"{headline}\n\n"

  contents.sort(key=lambda kv: len(kv[1]), reverse=True)
  _, longest_text = contents[0]
  if len(longest_text) >= _LONG_PRIMARY_THRESHOLD or len(contents) == 1:
    return longest_text
  return "\n\n---\n\n".join(
    f"Source {i + 1} ({url}):\n{text}" for i, (url, text) in enumerate(contents)
  )


# ── LLM-calling logic ─────────────────────────────────────────────────────


def extract_overview_topics(
  headline: str, sources: List[NewsArticleSource]
) -> List[str]:
  """Pass 1: extract 3-6 ordered topic labels for a story.

  Pools source content, runs the injection topic prompt on Claude, then if any
  label fails structural validation, regenerates ONCE with targeted feedback
  (falling back to the original labels if that rewrite fails). Degrades to
  ["Overview"] when there is too little content to extract from, rather than
  failing the whole task.

  Synchronous and blocking (Anthropic SDK); the task layer offloads it with
  asyncio.to_thread so the worker event loop stays free.
  """
  pooled = pool_source_content(headline, sources)
  if len(pooled) < _MIN_CONTENT_CHARS:
    logger.warning(
      f"Insufficient content for topic extraction ({len(pooled)} chars); "
      f"returning ['{_FALLBACK_TOPIC}']"
    )
    return [_FALLBACK_TOPIC]

  content = pooled[:_CONTENT_CHAR_LIMIT]
  prompt = NEWS_TOPICS_EXTRACT_PROMPT.format(headline=headline, content=content)
  topics = _call_claude_topics(prompt)
  topics = topics[:MAX_OVERVIEW_TOPICS] if topics else [_FALLBACK_TOPIC]

  issues = find_label_issues(topics)
  if issues:
    logger.info(
      f"Topic labels failed validation ({len(issues)} issue(s)); regenerating once"
    )
    try:
      topics = regenerate_overview_topics_with_feedback(
        headline, content, topics, issues
      )
    except Exception as e:
      logger.warning(f"Topic regeneration failed, keeping original labels: {e}")
  return topics


def regenerate_overview_topics_with_feedback(
  headline: str,
  content: str,
  previous_topics: List[str],
  issues: List[dict],
) -> List[str]:
  """One-shot rewrite of malformed labels, feeding the specific violations back
  (mirrors regenerateOverviewTopicsWithFeedback). `content` is already sliced to
  the prompt char limit by the caller. Falls back to the original labels if the
  rewrite returns nothing."""
  feedback = "\n".join(
    f"- \"{i['label']}\" — {'; '.join(i['issues'])}" for i in issues
  )
  previous = "\n".join(f"{idx + 1}. {t}" for idx, t in enumerate(previous_topics))
  prompt = NEWS_TOPICS_REGENERATE_PROMPT.format(
    headline=headline, content=content, feedback=feedback, previous=previous
  )
  topics = _call_claude_topics(prompt)
  if not topics:
    return previous_topics
  return topics[:MAX_OVERVIEW_TOPICS]


def _call_claude_topics(prompt: str) -> List[str]:
  """Run one topic prompt on Claude and return validated topic strings.

  Retries transient errors up to MAX_RETRIES and raises on persistent failure;
  the task layer owns durable retry/backoff above this. anthropic is imported
  lazily so a missing SDK can never break module import. Uses the same Claude
  model as the news-claim fallback (news_claim_claude_model)."""
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for topic extraction")

  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_S,
    )
  except Exception as e:
    logger.error(f"Failed to build Claude topic extraction client: {e}")
    raise Exception("Error building topic extraction client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=_TOPIC_MAX_TOKENS,
        temperature=settings.gemini_news_claim_temperature,
        system=_TOPIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
      )
      parsed = _parse_llm_response(_claude_text(message))
      return validate_overview_topics_response(parsed)
    except Exception as e:
      last_error = e
      logger.warning(
        f"Topic extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Topic extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  # Unreachable — loop above either returns or raises
  raise Exception("Topic extraction: unreachable code path")
