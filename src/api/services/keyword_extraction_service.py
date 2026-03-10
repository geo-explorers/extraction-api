import json
import re
from typing import Any, Dict, List, Tuple
from src.config.prompts.keyword_extraction_prompt import KEYWORD_EXTRACTION_PROMPT
from src.api.utils import llm_model
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 2


def extract_keyword_and_topics(
    episode: Dict[str, Any],
    topics_list: List[str],
    min_keywords: int,
    max_keywords: int,
    min_topics: int,
    max_topics: int
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
  try:
    chain = llm_model.build_chain(
      prompt=KEYWORD_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  invoke_params = {
    "episode": episode,
    "topics_list": topics_list,
    "min_keywords": min_keywords,
    "max_keywords": max_keywords,
    "min_topics": min_topics,
    "max_topics": max_topics,
  }

  last_error = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      raw_response = chain.invoke(invoke_params)
    except Exception as e:
      raise Exception("Failed invoking chain")

    try:
      response = _parse_llm_response(raw_response)
      break
    except Exception as e:
      last_error = e
      logger.warning(f"Parse attempt {attempt}/{MAX_RETRIES} failed. Raw response: {raw_response}")
      if attempt == MAX_RETRIES:
        raise Exception("Failed parsing response") from last_error
  
  
  try:
    keywords =  response["keywords"]
  except KeyError:
    raise Exception("Failed extracting keywords")

  try:
    topics =  response["topics"]
  except KeyError:
    raise Exception("Failed extracting topics")

  topic_keywords = response.get("topic_keywords", {})

  return keywords, topics, topic_keywords


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
