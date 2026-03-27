"""Premium claim extraction service using Gemini 3 Pro with structured outputs."""

from typing import Dict, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config.prompts.key_takeaways_prompt import KEY_TAKEAWAYS_PROMPT
from src.config.settings import settings
from src.config.prompts.topics_of_discussion_extraction_prompt import (
    TOPICS_OF_DISCUSSION_PROMPT
)
from src.config.prompts.claim_extraction_prompt import (
    CLAIM_EXTRACTION_PROMPT
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Timeout for Gemini API calls (3 minutes for large transcripts)
GEMINI_TIMEOUT_SECONDS = 60 * 3

# Retry configuration for Gemini API calls
GEMINI_RETRY_ATTEMPTS = 7
GEMINI_RETRY_INITIAL_DELAY = 2.0
GEMINI_RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]


class ClaimExtractionResult(BaseModel):
    """Structured output schema for claim extraction."""
    claims: List[str] = Field(
        description="List of factual, verifiable claims extracted from the transcript. "
        "Each claim should be self-contained (no pronouns), specific (include names, numbers, dates), "
        "and concise (5-40 words)."
    )

class TopicDiscussionResult(BaseModel):
    """Structured output schema for topic extraction."""
    topics: List[str] = Field(
        description="List of concise, descriptive topic labels (3-10 words) "
        "representing distinct discussion segments in chronological order."
    )
class ClaimWithTopicBaseResult(BaseModel):
    claim: List[str] = Field(
        description="List of factual, verifiable claims extracted from the transcript that comes under the above topic."
    )
    topic: str = Field(
        description="The topic label associated with the claim"
    )

class ClaimWithTopicResult(BaseModel):
    """Structured output schema for claim with topic."""
    claim_topic: List[ClaimWithTopicBaseResult] = Field(
        description="List of claims, each associated with a specific topic of discussion. "
        "The claims are factual, verifiable, self-contained, specific, and concise. "
        "The topics are from the provided `topics_of_discussion` list."
    )

class KeyTakeawayResult(BaseModel):
    """Structured output schema for key takeaway extraction."""
    key_takeaways: List[str] = Field(
        description="List of the most important claims (key takeaways) selected from the provided claims. "
        "These claims are central to the episode's main thesis, express impact, risk, opportunity, "
        "define foundational concepts, or make explicit causal claims. "
        "Claims are ordered by importance and output verbatim as provided in the input."
    )


class PremiumClaimExtractor:
    """Extract claims using Gemini 3 Pro with structured outputs and full transcript context."""

    def __init__(self):
        """Initialize Gemini client with structured output support."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for premium extraction")

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    initial_delay=GEMINI_RETRY_INITIAL_DELAY,
                    attempts=GEMINI_RETRY_ATTEMPTS,
                    http_status_codes=GEMINI_RETRYABLE_STATUS_CODES,
                ),
                timeout=GEMINI_TIMEOUT_SECONDS * 1000,
            )
        )
        self.model_name = settings.gemini_premium_model
        logger.info(
            f"Initialized PremiumClaimExtractor with model {self.model_name} "
            f"(structured outputs, {GEMINI_RETRY_ATTEMPTS} retry attempts)"
        )
    
    async def extract_topics_of_discussion_from_episode(
        self,
        title: str,
        description: str,
        full_transcript: str
    ) -> List[str]:
        """
        Extract topics of discussion from title, description and transcript.

        Args:
            title: Podcast episode title
            description: Episode description
            full_transcript: Complete podcast transcript text

        Returns:
            List of extracted topic strings

        Raises:
            ValueError: If response is empty or contains no topics
            Exception: If Gemini API call fails (after SDK retries exhausted)
        """
        prompt = TOPICS_OF_DISCUSSION_PROMPT.format(
            title=title,
            description=description,
            transcript=full_transcript
        )
        logger.info(f"Calling {self.model_name} for topics of discussion extraction")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.gemini_premium_temperature,
                response_mime_type="application/json",
                response_schema=TopicDiscussionResult,
            )
        )

        if not response or not response.text or not response.text.strip():
            raise ValueError("Empty response from Gemini API during topic extraction")

        result: TopicDiscussionResult = TopicDiscussionResult.model_validate_json(response.text)

        if not result.topics:
            raise ValueError("Gemini returned empty topics list")

        logger.info(f"Extracted {len(result.topics)} topics from full transcript via structured outputs")
        return result.topics

    async def extract_claims_with_topics_from_transcript(
        self,
        full_transcript: str,
        topics_of_discussion: List[str]
    ) -> Dict[str, List[str]]:
        """
        Extract claims from full transcript and associate them with topics of discussion
        using LLM with structured outputs.

        Args:
            full_transcript: Complete podcast transcript text
            topics_of_discussion: An ordered list of topic labels extracted from the same episode

        Returns:
            A dictionary where keys are topic labels and values are lists of claims associated with that topic.

        Raises:
            ValueError: If response is empty or contains no claims
            Exception: If Gemini API call fails (after SDK retries exhausted)
        """
        prompt = CLAIM_EXTRACTION_PROMPT.format(
            transcript=full_transcript,
            topics_of_discussion=topics_of_discussion
        )

        logger.info(
            f"Calling {self.model_name} for claim extraction with topics of discussion "
            f"({len(full_transcript)} chars)"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.gemini_premium_temperature,
                response_mime_type="application/json",
                response_schema=ClaimWithTopicResult,
            )
        )

        if not response or not response.text or not response.text.strip():
            raise ValueError("Empty response from Gemini API during claim extraction")

        result: ClaimWithTopicResult = ClaimWithTopicResult.model_validate_json(response.text)

        if not result.claim_topic:
            raise ValueError("Gemini returned empty claims list")

        parsed_result: Dict[str, List[str]] = {}
        for claim_topic in result.claim_topic:
            if claim_topic.topic not in parsed_result:
                parsed_result[claim_topic.topic] = []
            parsed_result[claim_topic.topic].extend(claim_topic.claim)

        total_claims = sum(len(c) for c in parsed_result.values())
        logger.info(f"Extracted {total_claims} claims across {len(parsed_result)} topics via structured outputs")
        return parsed_result

    async def extract_key_takeaways_from_claims(
        self,
        topics_with_claims: str
    ) -> List[str]:
        """
        Extract key takeaways from a list of claims using LLM with structured outputs.

        Args:
            topics_with_claims: A formatted string of topics with their claims.

        Returns:
            List of key takeaway strings.

        Raises:
            ValueError: If response is empty or contains no key takeaways
            Exception: If Gemini API call fails (after SDK retries exhausted)
        """
        prompt = KEY_TAKEAWAYS_PROMPT.format(
            topics_with_claims=topics_with_claims
        )

        logger.info(
            f"Calling {self.model_name} for key takeaway extraction with structured outputs"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.gemini_premium_temperature,
                response_mime_type="application/json",
                response_schema=KeyTakeawayResult,
            )
        )

        if not response or not response.text or not response.text.strip():
            raise ValueError("Empty response from Gemini API during key takeaway extraction")

        result: KeyTakeawayResult = KeyTakeawayResult.model_validate_json(response.text)

        if not result.key_takeaways:
            raise ValueError("Gemini returned empty key takeaways list")

        logger.info(f"Extracted {len(result.key_takeaways)} key takeaways via structured outputs")
        return result.key_takeaways
