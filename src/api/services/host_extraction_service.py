import json
from typing import Any, Dict, List
from src.config.prompts.host_extraction_prompt import HOST_EXTRACTION_PROMPT
from src.api.utils import llm_model
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


MAX_RETRIES = 3


def extract_podcast_hosts(
    title: str,
    description: str,
    truncated_transcript: str,
    possible_hosts: List[str] | None = None,
) -> List[Dict[str, Any]]:
  logger.info(f"Host extraction request: title='{title[:50]}...' possible_hosts={possible_hosts}")

  try:
    chain = llm_model.build_chain(
      prompt=HOST_EXTRACTION_PROMPT
    )
  except Exception as e:
    logger.error(f"Error building chain: {e}")
    raise Exception("Error building chain")

  # Format possible hosts for prompt
  if possible_hosts:
    possible_hosts_text = ", ".join(possible_hosts)
  else:
    possible_hosts_text = "None provided"

  last_error = None
  for attempt in range(1, MAX_RETRIES + 1):
    try:
      raw_response = chain.invoke({
        "title": title,
        "description": description,
        "truncated_transcript": truncated_transcript,
        "possible_hosts": possible_hosts_text,
      })
      logger.debug(f"Raw LLM response: {raw_response[:500]}")
      response = json.loads(raw_response)
      break
    except Exception as e:
      last_error = e
      logger.warning(f"Host extraction attempt {attempt}/{MAX_RETRIES} failed: {e}")
      if attempt == MAX_RETRIES:
        raise Exception(f"Host extraction failed after {MAX_RETRIES} attempts") from last_error

  try:
    hosts = response["hosts"]
  except KeyError:
    logger.error(f"Failed extracting hosts from response: {response}")
    raise Exception("Failed extracting hosts")

  for host in hosts:
    if "name" not in host or "urls" not in host:
      logger.error(f"Invalid host format: {host}")
      raise Exception("Invalid host format")
    if not isinstance(host["urls"], list):
      logger.error(f"Invalid host urls format: {host}")
      raise Exception("Invalid host format")

  host_names = [h["name"] for h in hosts]
  logger.info(f"Host extraction complete: extracted {len(hosts)} hosts: {host_names}")

  return hosts
