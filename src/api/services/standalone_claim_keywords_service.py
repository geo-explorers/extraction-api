"""Standalone claim keywords extraction service using Gemini structured outputs."""

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config.settings import settings
from src.config.prompts.standalone_claim_keywords_prompt import STANDALONE_CLAIM_KEYWORDS_PROMPT
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class TooManyClaimsError(ValueError):
    """Raised when request exceeds maximum allowed claims."""
    pass


class ClaimKeywordsItem(BaseModel):
    """Keywords for a single claim."""
    id: str = Field(description="The claim ID")
    keywords: List[str] = Field(description="List of keywords for this claim")


class ClaimKeywordsResponse(BaseModel):
    """Complete response containing keywords for all claims."""
    results: List[ClaimKeywordsItem] = Field(description="List of keyword results for all claims")


def extract_standalone_claim_keywords(
    claims: List[Dict[str, Any]],
    min_keywords: int = 1,
    max_keywords: int = 5,
) -> Dict[str, List[str]]:
    """
    Extract keywords for each claim independently (no episode context).

    Uses Gemini's structured output feature for reliable JSON parsing.

    Args:
        claims: List of claim dicts with 'id' and 'text' keys
        min_keywords: Minimum keywords per claim (default 1)
        max_keywords: Maximum keywords per claim (default 5)

    Returns:
        Dict mapping claim IDs to lists of keywords

    Raises:
        TooManyClaimsError: If claims exceed max_claims setting
        Exception: For API or parsing errors
    """
    max_claims = settings.standalone_claim_keywords_max_claims
    if len(claims) > max_claims:
        raise TooManyClaimsError(f"Request exceeds maximum of {max_claims} claims")

    if not claims:
        return {}

    # Initialize Gemini client
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=settings.gemini_api_key)
    model_name = settings.gemini_extraction_model

    # Build prompt
    claims_json = json.dumps(claims, indent=2)
    prompt = STANDALONE_CLAIM_KEYWORDS_PROMPT.format(
        min_keywords=min_keywords,
        max_keywords=max_keywords,
        claims_json=claims_json,
    )

    # Configure safety settings
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE"
        ),
    ]

    # Call Gemini with structured output
    logger.info(
        f"Calling {model_name} for standalone claim keywords extraction "
        f"({len(claims)} claims, {len(prompt)} chars)"
    )

    response = None
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.gemini_extraction_temperature,
                max_output_tokens=8192,
                safety_settings=safety_settings,
                response_mime_type="application/json",
                response_schema=ClaimKeywordsResponse,
            )
        )

        if not response or not response.text:
            raise ValueError("Empty response from Gemini")

        logger.debug(f"Received response: {len(response.text)} chars")

        # Parse structured response
        validated_response = ClaimKeywordsResponse.model_validate_json(response.text)

        # Convert to dict format
        result: Dict[str, List[str]] = {}
        for item in validated_response.results:
            result[item.id] = item.keywords

        logger.info(f"Successfully extracted keywords for {len(result)} claims")
        return result

    except Exception as e:
        logger.error(f"Error in standalone claim keywords extraction: {e}", exc_info=True)
        if response:
            logger.error(f"Response text: {getattr(response, 'text', 'N/A')[:500]}")
        raise
