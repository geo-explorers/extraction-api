import json
from typing import Any, Dict, List

from src.config.prompts.standalone_claim_keywords_prompt import STANDALONE_CLAIM_KEYWORDS_PROMPT
from src.config.settings import settings
from src.api.utils import llm_model


class TooManyClaimsError(ValueError):
    """Raised when request exceeds maximum allowed claims."""
    pass


def extract_standalone_claim_keywords(
    claims: List[Dict[str, Any]],
    min_keywords: int = 1,
    max_keywords: int = 5,
) -> Dict[str, List[str]]:
    """
    Extract keywords for each claim independently (no episode context).

    Args:
        claims: List of claim dicts with 'id' and 'text' keys
        min_keywords: Minimum keywords per claim (default 1)
        max_keywords: Maximum keywords per claim (default 5)

    Returns:
        Dict mapping claim IDs to lists of keywords

    Raises:
        TooManyClaimsError: If claims exceed max_claims setting
        Exception: For chain building, invocation, or parsing errors
    """
    max_claims = settings.standalone_claim_keywords_max_claims
    if len(claims) > max_claims:
        raise TooManyClaimsError(f"Request exceeds maximum of {max_claims} claims")

    try:
        chain = llm_model.build_chain(prompt=STANDALONE_CLAIM_KEYWORDS_PROMPT)
    except Exception:
        raise Exception("Error building chain")

    claims_json = json.dumps(claims, indent=2)

    try:
        raw_response = chain.invoke({
            "claims_json": claims_json,
            "min_keywords": min_keywords,
            "max_keywords": max_keywords,
        })
    except Exception:
        raise Exception("Failed invoking chain")

    try:
        response = json.loads(raw_response)
    except Exception:
        raise Exception("Failed parsing response")

    try:
        claim_keywords = response["claim_keywords"]
    except KeyError:
        raise Exception("Failed extracting claim_keywords from response")

    if not isinstance(claim_keywords, dict):
        raise Exception("Invalid response format: claim_keywords must be a dict")

    for claim_id, keywords in claim_keywords.items():
        if not isinstance(keywords, list):
            raise Exception(f"Invalid response format: keywords for {claim_id} must be a list")

    return claim_keywords
