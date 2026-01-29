"""Premium claim extraction service using Gemini 3 Pro with structured outputs."""

from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config.settings import settings
from src.config.prompts.premium_claim_extraction_prompt import (
    PREMIUM_CLAIM_EXTRACTION_PROMPT
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimExtractionResult(BaseModel):
    """Structured output schema for claim extraction."""
    claims: List[str] = Field(
        description="List of factual, verifiable claims extracted from the transcript. "
        "Each claim should be self-contained (no pronouns), specific (include names, numbers, dates), "
        "and concise (5-40 words)."
    )


class PremiumClaimExtractor:
    """Extract claims using Gemini 3 Pro with structured outputs and full transcript context."""

    def __init__(self):
        """Initialize Gemini client with structured output support."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for premium extraction")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_premium_model
        logger.info(f"Initialized PremiumClaimExtractor with model {self.model_name} (structured outputs)")

    async def extract_claims_from_transcript(
        self,
        full_transcript: str
    ) -> List[str]:
        """
        Extract claims from full transcript using Gemini 3 Pro with structured outputs.

        Args:
            full_transcript: Complete podcast transcript text

        Returns:
            List of extracted claim strings

        Raises:
            Exception: If Gemini API call fails
        """
        response = None
        try:
            # Format prompt with transcript
            prompt = PREMIUM_CLAIM_EXTRACTION_PROMPT.format(
                transcript=full_transcript
            )

            # Call Gemini API with structured output configuration
            logger.info(
                f"Calling Gemini 3 Pro for claim extraction with structured outputs "
                f"({len(full_transcript)} chars)"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=settings.gemini_premium_temperature,
                    response_mime_type="application/json",
                    response_schema=ClaimExtractionResult,
                )
            )

            # Parse structured response using Pydantic
            result: ClaimExtractionResult = ClaimExtractionResult.model_validate_json(response.text)
            claims = result.claims

            logger.info(f"Extracted {len(claims)} claims from full transcript via structured outputs")
            return claims

        except Exception as e:
            logger.error(f"Error in premium claim extraction: {e}", exc_info=True)
            if response:
                logger.error(f"Response text: {getattr(response, 'text', 'N/A')[:500]}")
            return []
