import json
import re
from typing import Dict, List, Tuple
from src.config.prompts.claim_keyword_extraction_prompt import CLAIM_KEYWORD_EXTRACTION_PROMPT
from src.api.utils import llm_model
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3


def _parse_json_response(raw_response: str) -> dict:
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


def extract_claim_keywords_and_topics(
    claims: List[Dict[str, str]],
    title: str,
    description: str,
    topics_list: List[str],
    min_keywords: int,
    max_keywords: int,
    min_topics: int,
    max_topics: int
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    try:
        chain = llm_model.build_chain(
            prompt=CLAIM_KEYWORD_EXTRACTION_PROMPT
        )
    except Exception as e:
        raise Exception("Error building chain")

    # Format claims for the prompt
    claims_text = "\n".join([f'[{i}] id="{c["id"]}": "{c["text"]}"' for i, c in enumerate(claims, 1)])

    invoke_params = {
        "title": title,
        "description": description,
        "claims": claims_text,
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
            response = _parse_json_response(raw_response)
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Claim keyword extraction attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise Exception(f"Claim keyword extraction failed after {MAX_RETRIES} attempts") from last_error

    try:
        claim_keywords = response["claim_keywords"]
    except KeyError:
        raise Exception("Failed extracting claim_keywords")

    try:
        claim_topics = response["claim_topics"]
    except KeyError:
        raise Exception("Failed extracting claim_topics")

    topic_keywords = response.get("topic_keywords", {})

    return claim_keywords, claim_topics, topic_keywords
