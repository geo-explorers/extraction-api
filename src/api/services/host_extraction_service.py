import json
from typing import Any, Dict, List
from src.config.prompts.host_extraction_prompt import HOST_EXTRACTION_PROMPT
from src.api.utils import llm_model


def extract_podcast_hosts(
    title: str,
    description: str,
    truncated_transcript: str,
    possible_hosts: List[str] | None = None,
) -> List[Dict[str, Any]]:
  try:
    chain = llm_model.build_chain(
      prompt=HOST_EXTRACTION_PROMPT
    )
  except Exception as e:
    raise Exception("Error building chain")

  # Format possible hosts for prompt
  if possible_hosts:
    possible_hosts_text = ", ".join(possible_hosts)
  else:
    possible_hosts_text = "None provided"

  try:
    raw_response = chain.invoke({
      "title": title,
      "description": description,
      "truncated_transcript": truncated_transcript,
      "possible_hosts": possible_hosts_text,
    })
  except Exception as e:
    raise Exception("Failed invoking chain")

  try:
    response = json.loads(raw_response)
  except Exception as e:
    raise Exception("Failed parsing response")

  try:
    hosts = response["hosts"]
  except KeyError:
    raise Exception("Failed extracting hosts")

  for host in hosts:
    if "name" not in host or "urls" not in host:
      raise Exception("Invalid host format")
    if not isinstance(host["urls"], list):
      raise Exception("Invalid host format")
  return hosts
