"""
Gemini service for claim validation.

Uses Google's Gemini API to validate claims in batches and identify bad claims
that should be flagged for review or deletion.

Usage:
    from src.infrastructure.gemini_service import GeminiService

    service = GeminiService()
    results = await service.validate_claims_batch(claims, prompt)
"""

import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClaimValidationInput:
    """Input claim for validation."""
    claim_id: int
    claim_text: str
    confidence: float
    episode_id: int


@dataclass
class ClaimValidationResult:
    """Result of claim validation."""
    claim_id: int
    is_valid: bool


# Pydantic models for structured output
class ClaimValidationItem(BaseModel):
    """Single claim validation result in structured output."""
    id: int = Field(description="The claim ID")
    is_valid: bool = Field(description="Whether the claim is valid (true) or should be flagged (false)")


class ClaimValidationResponse(BaseModel):
    """Complete validation response containing all claim results."""
    results: List[ClaimValidationItem] = Field(description="List of validation results for all claims")


class SingleClaimValidationResult(BaseModel):
    """Result of single claim validation with explanation."""
    explanation: str = Field(description="Brief explanation of why the claim is or is not context-independent")
    is_valid: bool = Field(description="Whether the claim is valid (context-independent)")


class GeminiService:
    """
    Service for validating claims using Google Gemini API.

    Features:
    - Batch validation (multiple claims per API call)
    - Configurable models (gemini-1.5-flash or gemini-1.5-pro)
    - Custom prompt support
    - Automatic retries with exponential backoff
    - Rate limiting handling

    Example:
        ```python
        service = GeminiService()

        claims = [
            ClaimValidationInput(id=1, text="Bitcoin is a cryptocurrency", confidence=0.9, episode_id=123),
            ClaimValidationInput(id=2, text="The sky is blue", confidence=0.8, episode_id=123)
        ]

        prompt = "Identify claims that are not crypto-related..."
        results = await service.validate_claims_batch(claims, prompt)

        for result in results:
            if not result.is_valid:
                print(f"Claim {result.claim_id} is flagged")
        ```
    """

    def __init__(self):
        """Initialize Gemini service with API key and model configuration."""
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Please set the GEMINI_API_KEY environment variable."
            )

        # Initialize Gemini client with new SDK
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model
        self.batch_size = settings.gemini_validation_batch_size
        self.timeout = settings.gemini_timeout

        # Configure safety settings for the new SDK
        self.safety_settings = [
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

        logger.info(
            f"Initialized GeminiService with model={self.model_name}, "
            f"batch_size={self.batch_size}"
        )

    async def validate_claims_batch(
        self,
        claims: List[ClaimValidationInput],
        validation_prompt: str,
        max_retries: int = 3
    ) -> List[ClaimValidationResult]:
        """
        Validate a batch of claims using Gemini.

        Args:
            claims: List of claims to validate
            validation_prompt: User-provided prompt for validation logic
            max_retries: Maximum number of retry attempts on failure

        Returns:
            List of validation results for each claim

        Example:
            ```python
            claims = [ClaimValidationInput(id=1, text="...", confidence=0.9, episode_id=123)]
            prompt = "Flag claims that mention specific prices without dates"
            results = await service.validate_claims_batch(claims, prompt)
            ```
        """
        if not claims:
            logger.warning("No claims to validate")
            return []

        logger.info(f"Validating batch of {len(claims)} claims")

        # Build the full prompt
        full_prompt = self._build_validation_prompt(claims, validation_prompt)

        # Call Gemini with retries
        for attempt in range(max_retries):
            try:
                response = await self._call_gemini(full_prompt)
                results = self._parse_response(response, claims)

                logger.info(
                    f"Validated {len(claims)} claims: "
                    f"{sum(1 for r in results if not r.is_valid)} flagged as bad"
                )

                return results

            except Exception as e:
                logger.error(
                    f"Error validating claims (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to validate claims after {max_retries} attempts")
                    # Return all claims as valid (fail-safe: don't flag if we can't validate)
                    return [
                        ClaimValidationResult(
                            claim_id=claim.claim_id,
                            is_valid=True
                        )
                        for claim in claims
                    ]

        # Should never reach here, but just in case
        return []

    def _build_validation_prompt(
        self,
        claims: List[ClaimValidationInput],
        validation_prompt: str
    ) -> str:
        """
        Build the full prompt for Gemini including claims data.

        With structured outputs, we don't need to specify the response format
        as it's enforced by the schema.

        Format:
            {user_validation_prompt}

            Claims to validate (JSON array):
            [
                {"id": 1, "text": "...", "confidence": 0.9, "episode_id": 123},
                ...
            ]
        """
        claims_json = json.dumps(
            [
                {
                    "id": claim.claim_id,
                    "text": claim.claim_text,
                    "confidence": claim.confidence,
                    "episode_id": claim.episode_id
                }
                for claim in claims
            ],
            indent=2
        )

        prompt = f"""{validation_prompt}

Claims to validate (JSON array):
{claims_json}

For each claim, evaluate whether it is valid or should be flagged based on the criteria above."""

        return prompt

    async def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API with the given prompt using structured outputs.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            Raw JSON response text from Gemini
        """
        logger.debug(f"Calling Gemini with prompt length: {len(prompt)} chars")

        # Run in thread pool since the new SDK is synchronous
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Deterministic for validation
                    max_output_tokens=8192,
                    safety_settings=self.safety_settings,
                    response_mime_type="application/json",
                    response_json_schema=ClaimValidationResponse.model_json_schema(),
                )
            )
        )

        if not response or not response.text:
            raise ValueError("Empty response from Gemini")

        logger.debug(f"Received structured JSON response: {len(response.text)} chars")
        return response.text

    def _parse_response(
        self,
        response: str,
        claims: List[ClaimValidationInput]
    ) -> List[ClaimValidationResult]:
        """
        Parse Gemini's structured JSON response using Pydantic validation.

        Args:
            response: Raw JSON response text from Gemini
            claims: Original claims (for fallback)

        Returns:
            List of validation results
        """
        try:
            # Parse and validate response using Pydantic
            validated_response = ClaimValidationResponse.model_validate_json(response)

            # Convert to ClaimValidationResult objects
            results = []
            claim_ids_set = {claim.claim_id for claim in claims}

            for item in validated_response.results:
                if item.id not in claim_ids_set:
                    logger.warning(f"Unknown claim ID in response: {item.id}")
                    continue

                results.append(ClaimValidationResult(
                    claim_id=item.id,
                    is_valid=item.is_valid
                ))

            # Ensure all claims have results (add missing ones as valid)
            result_ids = {r.claim_id for r in results}
            for claim in claims:
                if claim.claim_id not in result_ids:
                    logger.warning(
                        f"Claim {claim.claim_id} missing from response, assuming valid"
                    )
                    results.append(ClaimValidationResult(
                        claim_id=claim.claim_id,
                        is_valid=True
                    ))

            return results

        except Exception as e:
            logger.error(f"Error parsing structured response: {e}", exc_info=True)
            logger.error(f"Response text: {response[:500]}...")

            # Fail-safe: return all as valid
            return [
                ClaimValidationResult(
                    claim_id=claim.claim_id,
                    is_valid=True
                )
                for claim in claims
            ]

    def get_validation_prompt(self) -> str:
        """
        Get the hardcoded validation prompt.

        Returns:
            Prompt text from src/config/validation_prompt.py

        Example:
            ```python
            service = GeminiService()
            prompt = service.get_validation_prompt()
            ```
        """
        from src.config.validation_prompt import VALIDATION_PROMPT

        if not VALIDATION_PROMPT or VALIDATION_PROMPT.strip().startswith("TODO"):
            raise ValueError(
                "Validation prompt not configured!\n"
                "Please update VALIDATION_PROMPT in src/config/validation_prompt.py"
            )

        logger.info(f"Loaded validation prompt ({len(VALIDATION_PROMPT)} chars)")
        return VALIDATION_PROMPT

    async def validate_single_claim(
        self,
        claim_text: str,
        validation_prompt: str,
        max_retries: int = 3
    ) -> SingleClaimValidationResult:
        """
        Validate a single claim for context independence.

        Args:
            claim_text: The claim text to validate
            validation_prompt: Prompt template with {claim_text} placeholder
            max_retries: Maximum number of retry attempts on failure

        Returns:
            SingleClaimValidationResult with explanation and is_valid flag

        Example:
            ```python
            result = await service.validate_single_claim(
                "Bitcoin reached $69,000 in November 2021",
                CLAIM_VALIDATION_PROMPT
            )
            print(f"Valid: {result.is_valid}, Reason: {result.explanation}")
            ```
        """
        # Format the prompt with the claim text
        full_prompt = validation_prompt.format(claim_text=claim_text)

        for attempt in range(max_retries):
            try:
                # Call Gemini with structured output for single claim validation
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=1024,
                            safety_settings=self.safety_settings,
                            response_mime_type="application/json",
                            response_json_schema=SingleClaimValidationResult.model_json_schema(),
                        )
                    )
                )

                if not response or not response.text:
                    raise ValueError("Empty response from Gemini")

                # Parse and validate response
                result = SingleClaimValidationResult.model_validate_json(response.text)
                return result

            except Exception as e:
                logger.error(
                    f"Error validating single claim (attempt {attempt + 1}/{max_retries}): {e}",
                    exc_info=True
                )

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    # Fail-safe: assume invalid (conservative approach for context independence)
                    logger.warning(
                        f"Failed to validate claim after {max_retries} attempts, marking as invalid"
                    )
                    return SingleClaimValidationResult(
                        explanation="Validation failed due to API error",
                        is_valid=False
                    )

        # Should never reach here
        return SingleClaimValidationResult(
            explanation="Validation failed unexpectedly",
            is_valid=False
        )
