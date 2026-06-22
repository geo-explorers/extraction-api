"""Story-level topic + entity extraction (news "Pass 5").

Ported from news-worker's extractTopicsAndEntities (injection branch). One Claude
call extracts whole-story topics + entities from headline + summary; topics are
then classified curated-vs-free by EXACT-name membership against the caller-
provided curated list and gated at CURATED_MIN_RELEVANCE (free/llm topics are
always kept). extraction-api stays stateless: the curated vocabulary is passed
IN, and geo_id mapping stays caller-side.

The curated list is sent as a cache_control system block (it repeats across a
batch run), mirroring the prompt caching the TS path used. The pure helpers
(validation, classification) are LLM-free and unit-tested; only
extract_story_topics_and_entities calls Claude. Engine-agnostic (no task layer).
"""

import math
from typing import List, Optional

from src.api.schemas.news_topics_entities_schema import (
  NewsTopicsEntitiesResponse,
  ExtractedStoryTopic,
  ExtractedStoryEntity,
)
from src.api.services.news_claim_extract_service import (
  _claude_text,
  _parse_llm_response,
)
from src.config.prompts.news_topics_entities_prompt import (
  SYSTEM_BASE,
  build_curated_system_block,
  build_user_prompt,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
_REQUEST_TIMEOUT_S = 180.0
_MAX_TOKENS = 2048
# Curated topics are kept only at/above this relevance; free (llm) topics are
# always kept. Mirrors CURATED_MIN_RELEVANCE in enrich.ts.
CURATED_MIN_RELEVANCE = 0.6


# ── Validation helpers (ported verbatim from enrich.ts) ──────────────────


def _expect_object(value, field: str) -> dict:
  # JS: throws for null, non-object, or array. A Python dict is the only pass.
  if not isinstance(value, dict):
    raise ValueError(f"Invalid LLM response: {field} must be an object")
  return value


def _expect_string(value, field: str) -> str:
  if not isinstance(value, str) or value.strip() == "":
    raise ValueError(f"Invalid LLM response: {field} must be a non-empty string")
  return value


def _optional_string(value) -> Optional[str]:
  return value if isinstance(value, str) else None


def _optional_number(value, field: str) -> Optional[float]:
  if value is None:
    return None
  # bool is excluded: JS typeof true === "boolean", not "number".
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or (isinstance(value, float) and math.isnan(value))
  ):
    raise ValueError(f"Invalid LLM response: {field} must be a number")
  return value


def validate_topics_entities_response(value) -> dict:
  """Coerce the parsed LLM JSON into {topics:[{name,relevance}],
  entities:[{name,type,role}]}. Accepts a `topics` array OR the legacy
  `curated_topics` + `free_topics` arrays. Mirrors validateTopicsEntitiesResponse."""
  root = _expect_object(value, "topicsEntitiesResponse")
  entities_raw = root.get("entities")
  entities_raw = entities_raw if isinstance(entities_raw, list) else []

  def parse_topic_array(arr, prefix):
    out = []
    for index, topic in enumerate(arr):
      item = _expect_object(topic, f"{prefix}[{index}]")
      rel = _optional_number(item.get("relevance"), f"{prefix}[{index}].relevance")
      out.append(
        {
          "name": _expect_string(item.get("name"), f"{prefix}[{index}].name"),
          "relevance": 0 if rel is None else rel,
        }
      )
    return out

  topics_raw = root.get("topics")
  topics_raw = topics_raw if isinstance(topics_raw, list) else []
  curated_raw = root.get("curated_topics")
  curated_raw = curated_raw if isinstance(curated_raw, list) else []
  free_raw = root.get("free_topics")
  free_raw = free_raw if isinstance(free_raw, list) else []
  all_topics = topics_raw if len(topics_raw) > 0 else [*curated_raw, *free_raw]

  entities = []
  for index, entity in enumerate(entities_raw):
    item = _expect_object(entity, f"entities[{index}]")
    role = _optional_string(item.get("role"))
    entities.append(
      {
        "name": _expect_string(item.get("name"), f"entities[{index}].name"),
        "type": _expect_string(item.get("type"), f"entities[{index}].type"),
        "role": "" if role is None else role,
      }
    )

  return {"topics": parse_topic_array(all_topics, "topics"), "entities": entities}


def classify_topics(
  topics: List[dict], curated_topic_names: List[str]
) -> List[dict]:
  """Classify each topic curated-vs-llm by exact-name membership, drop
  relevance<=0, and keep curated topics only at/above CURATED_MIN_RELEVANCE
  (free topics always kept). Mirrors enrich.ts:1013-1026."""
  curated_set = set(curated_topic_names)
  classified = []
  for t in topics:
    if not t["name"] or not (t["relevance"] > 0):
      continue
    source = "curated" if t["name"] in curated_set else "llm"
    if source == "llm" or t["relevance"] >= CURATED_MIN_RELEVANCE:
      classified.append(
        {"name": t["name"], "relevance": t["relevance"], "source": source}
      )
  return classified


# ── LLM-calling logic ─────────────────────────────────────────────────────


def extract_story_topics_and_entities(
  headline: str,
  summary: str,
  curated_topic_names: List[str],
) -> NewsTopicsEntitiesResponse:
  """Pass 5: extract story-level topics + entities on Claude, then classify/gate
  topics against the curated vocabulary. Synchronous/blocking (Anthropic SDK);
  the task layer offloads via asyncio.to_thread. anthropic is imported lazily."""
  if not settings.anthropic_api_key:
    raise Exception("ANTHROPIC_API_KEY not configured for topic/entity extraction")

  has_curated = len(curated_topic_names) > 0
  system_blocks: list = [{"type": "text", "text": SYSTEM_BASE}]
  if has_curated:
    # Cache the (large, batch-stable) curated list across requests.
    system_blocks.append(
      {
        "type": "text",
        "text": build_curated_system_block(curated_topic_names),
        "cache_control": {"type": "ephemeral"},
      }
    )
  user_prompt = build_user_prompt(headline, summary, has_curated)

  try:
    import anthropic

    client = anthropic.Anthropic(
      api_key=settings.anthropic_api_key,
      timeout=_REQUEST_TIMEOUT_S,
    )
  except Exception as e:
    logger.error(f"Failed to build Claude topic/entity client: {e}")
    raise Exception("Error building topic/entity extraction client") from e

  last_error: Exception | None = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      message = client.messages.create(
        model=settings.news_claim_claude_model,
        max_tokens=_MAX_TOKENS,
        temperature=settings.gemini_news_claim_temperature,
        system=system_blocks,
        messages=[{"role": "user", "content": user_prompt}],
      )
      parsed = _parse_llm_response(_claude_text(message))
      validated = validate_topics_entities_response(parsed)
      topics = classify_topics(validated["topics"], curated_topic_names)
      entities = [e for e in validated["entities"] if e["name"] and e["type"]]
      return NewsTopicsEntitiesResponse(
        topics=[ExtractedStoryTopic(**t) for t in topics],
        entities=[ExtractedStoryEntity(**e) for e in entities],
      )
    except Exception as e:
      last_error = e
      logger.warning(
        f"Topic/entity extraction attempt {attempt}/{MAX_RETRIES} failed: {e}"
      )
      if attempt == MAX_RETRIES:
        raise Exception(
          f"Topic/entity extraction failed after {MAX_RETRIES} attempts"
        ) from last_error

  # Unreachable — loop above either returns or raises
  raise Exception("Topic/entity extraction: unreachable code path")
