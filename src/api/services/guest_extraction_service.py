import json
from typing import Any, Dict, List
from src.config.prompts.guest_extraction_prompt import GUEST_EXTRACTION_PROMPT
from src.api.utils import llm_model
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def extract_podcast_guests(
    title: str,
    description: str,
    truncated_transcript: str = "",
) -> List[Dict[str, Any]]:
  try:
    chain = llm_model.build_chain(
      prompt=GUEST_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  last_error = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      raw_response = chain.invoke({
        "title": title,
        "description": description,
        "truncated_transcript": truncated_transcript,
      })
      response = json.loads(raw_response)
      break
    except Exception as e:
      last_error = e
      logger.warning(f"Guest extraction attempt {attempt}/{MAX_RETRIES} failed: {e}")
      if attempt == MAX_RETRIES:
        raise Exception(f"Guest extraction failed after {MAX_RETRIES} attempts") from last_error

  try:
    guests = response["guests"]
  except KeyError:
    raise Exception("Failed extracting guests")

  for guest in guests:
    if "name" not in guest or "urls" not in guest:
      raise Exception("Invalid guest format")
    if not isinstance(guest["urls"], list):
      raise Exception("Invalid guest format")
  return guests
