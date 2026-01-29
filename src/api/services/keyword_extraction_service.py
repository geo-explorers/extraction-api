import json
import re
from typing import Any, Dict, List, Tuple
from src.config.prompts.keyword_extraction_prompt import KEYWORD_EXTRACTION_PROMPT
from src.api.utils import llm_model


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
  
  try:
    raw_response = chain.invoke({
      "episode": episode,
      "topics_list": topics_list,
      "min_keywords": min_keywords,
      "max_keywords": max_keywords,
      "min_topics": min_topics,
      "max_topics": max_topics,
    })
  except Exception as e:
    raise Exception("Failed invoking chain")
  
  try:
    response = json.loads(raw_response)
  except Exception as e:
    try:
      response_text = raw_response.strip()
      fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)

      # Strip markdown code fences if the LLM wrapped the JSON in a ```json block
      if fenced_match:
        response_text = fenced_match.group(1)

      response = json.loads(response_text)
    except Exception as e:
      raise Exception("Failed parsing response")
  
  
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
