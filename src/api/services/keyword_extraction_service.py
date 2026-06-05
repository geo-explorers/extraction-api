import json
import re
from typing import Any, Dict, List
from src.config.prompts.keyword_extraction_prompt import KEYWORD_EXTRACTION_PROMPT
from src.api.utils import llm_model
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def extract_topics(
    episode: Dict[str, Any],
    topics_list: List[str],
    min_topics: int,
    max_topics: int
) -> List[str]:
  try:
    chain = llm_model.build_chain(
      prompt=KEYWORD_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  invoke_params = {
    "episode": episode,
    "topics_list": topics_list,
    "min_topics": min_topics,
    "max_topics": max_topics,
  }

  last_error = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      raw_response = chain.invoke(invoke_params)
      response = _parse_llm_response(raw_response)
      break
    except Exception as e:
      last_error = e
      logger.warning(f"Topic extraction attempt {attempt}/{MAX_RETRIES} failed: {e}")
      if attempt == MAX_RETRIES:
        raise Exception(f"Topic extraction failed after {MAX_RETRIES} attempts") from last_error

  try:
    topics = response["topics"]
  except KeyError:
    raise Exception("Failed extracting topics")

  # Hard validation against the curated topics list. Even with the prompt
  # explicitly forbidding invention, Gemini occasionally hallucinates a topic
  # that matches strong episode cues (e.g. emits "Cholesterol" when the title
  # mentions cholesterol). Drop any returned topic that isn't in the input
  # list so no LLM invention can ever leak past this boundary.
  allowed = set(topics_list)
  filtered = [t for t in topics if t in allowed]
  invented = [t for t in topics if t not in allowed]
  if invented:
    logger.warning(
      f"Dropped {len(invented)} LLM-invented topic(s) not in curated list: {invented}"
    )

  return filtered


def _parse_llm_response(raw_response: str) -> dict:
  """Parse a JSON response from the LLM, handling markdown fences."""
  try:
    return json.loads(raw_response)
  except (json.JSONDecodeError, TypeError):
    pass

  response_text = raw_response.strip()
  fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)

  if fenced_match:
    response_text = fenced_match.group(1)

  return json.loads(response_text)
