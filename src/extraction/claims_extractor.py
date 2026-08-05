"""Gemini client for the generalized claims.extract pipeline.

Mirrors PremiumClaimExtractor: structured outputs via response_schema (the
decode-constrained shape is also the hard guardrail against caller
custom_instructions altering the output contract), app-level retries on
transient HTTP errors, blocking SDK calls offloaded to a thread. Adds the
news service's thinking_level support (Gemini 3+ only; empty setting
disables it so 2.5-era model reverts run clean).

Prompts are built by claims_prompt_builder and passed in ready-rendered, so
this module stays provider-mechanics only.
"""

import asyncio
import time
from typing import List, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

GEMINI_TIMEOUT_SECONDS = 60 * 3

APP_MAX_RETRIES = 3
APP_RETRY_INITIAL_DELAY = 5.0
APP_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


# ── LLM-facing schemas (internal; the public contract lives in
# src/api/schemas/claims_extract_schema.py and is assembled in finalize) ─────


class LLMClaim(BaseModel):
    # `text` first and `claims` first in LLMExtraction: decode order preserves
    # the claims-first index discipline the prompt demands.
    text: str = Field(description="Self-contained, atomic, verifiable claim.")
    topic: str = Field(
        default="",
        description="Topic label this claim belongs to (empty in flat mode).",
    )
    document_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices of the provided documents supporting this claim.",
    )
    confidence: float = Field(
        default=0.8,
        description="0.9+ explicitly stated, 0.7-0.9 strongly implied, 0.5-0.7 inferred.",
    )
    is_factual: Optional[bool] = Field(
        default=None,
        description=(
            "True if the claim is a verifiable statement of fact, False if it is "
            "an unverifiable opinion or value judgement. Leave null unless "
            "factuality classification was requested."
        ),
    )


class LLMGroup(BaseModel):
    name: str = Field(description="Topic label naming this group of claims.")
    summary: str = Field(default="", description="Optional one-sentence group summary.")
    claim_indices: List[int] = Field(
        default_factory=list,
        description="0-based indices into the claims array; at least 2 per group.",
    )


class LLMQuote(BaseModel):
    text: str = Field(description="Verbatim quote from the document text.")
    speaker: Optional[str] = Field(
        default=None, description="Named speaker exactly as the document gives it."
    )
    claim_index: int = Field(
        description="0-based index of the claim this quote most directly supports."
    )
    document_index: Optional[int] = Field(
        default=None, description="0-based index of the document the quote came from."
    )


class LLMExtraction(BaseModel):
    """Superset structured output for the fused extraction call. Unrequested
    sections are prompted to stay empty and deterministically stripped in
    finalize either way."""
    claims: List[LLMClaim] = Field(default_factory=list)
    groups: List[LLMGroup] = Field(default_factory=list)
    quotes: List[LLMQuote] = Field(default_factory=list)
    summary: str = Field(default="")


class LLMTopics(BaseModel):
    topics: List[str] = Field(
        default_factory=list,
        description="Concise topic labels (3-10 words) in the order they appear.",
    )


class LLMTakeaways(BaseModel):
    takeaways: List[str] = Field(
        default_factory=list,
        description="Selected claims, verbatim, ordered to tell a coherent story.",
    )


class ClaimsExtractor:
    """Structured-output Gemini calls for the claims.extract DAG."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for claim extraction")

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000),
        )
        self.model_name = settings.claims_extract_model
        logger.info(
            f"Initialized ClaimsExtractor with model {self.model_name} "
            f"(structured outputs, {APP_MAX_RETRIES} app-level retries)"
        )

    def _config(self, response_schema: type[BaseModel]) -> types.GenerateContentConfig:
        config_kwargs: dict = {
            "temperature": settings.claims_extract_temperature,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        # thinking_level is a Gemini-3+ control; only attach when configured so
        # a revert to a 2.5-era model runs with no thinking config and no error.
        thinking_level = (settings.claims_extract_thinking_level or "").strip()
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
            )
        return types.GenerateContentConfig(**config_kwargs)

    async def _call_gemini(
        self, prompt: str, config: types.GenerateContentConfig, step_name: str
    ) -> str:
        """Call Gemini with app-level retries on transient errors (same policy
        as PremiumClaimExtractor). Returns the raw response text."""
        for attempt in range(1, APP_MAX_RETRIES + 1):
            try:
                start = time.time()
                # Blocking SDK call; offload so the worker event loop never stalls.
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                elapsed = time.time() - start

                if not response or not response.text or not response.text.strip():
                    raise ValueError(f"Empty response from Gemini API during {step_name}")

                if attempt > 1:
                    logger.info(
                        f"Gemini {step_name} succeeded on attempt "
                        f"{attempt}/{APP_MAX_RETRIES} ({elapsed:.1f}s)"
                    )
                return response.text

            except APIError as e:
                status_code = getattr(e, "code", None)
                if status_code in APP_RETRYABLE_STATUS_CODES and attempt < APP_MAX_RETRIES:
                    wait_time = APP_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini {step_name} failed (attempt {attempt}/{APP_MAX_RETRIES}): "
                        f"HTTP {status_code} - {e.message}. Retrying in {wait_time:.0f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(
                    f"Gemini {step_name} failed (attempt {attempt}/{APP_MAX_RETRIES}): "
                    f"HTTP {status_code} - {e.message}."
                )
                raise

            except ValueError:
                # Empty response is not retryable — fail immediately.
                raise

            except Exception as e:
                if attempt < APP_MAX_RETRIES:
                    wait_time = APP_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini {step_name} failed (attempt {attempt}/{APP_MAX_RETRIES}): "
                        f"{type(e).__name__}: {e}. Retrying in {wait_time:.0f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(
                    f"Gemini {step_name} failed (attempt {attempt}/{APP_MAX_RETRIES}): "
                    f"{type(e).__name__}: {e}. Max retries exhausted."
                )
                raise

    async def extract_topics(self, prompt: str) -> List[str]:
        """Topic labels for grouped extraction. Raises on an empty list — with
        grouping requested, zero topics is a model failure worth a task retry."""
        response_text = await self._call_gemini(
            prompt=prompt,
            config=self._config(LLMTopics),
            step_name="topic extraction",
        )
        result = LLMTopics.model_validate_json(response_text)
        if not result.topics:
            raise ValueError("Gemini returned empty topics list")
        logger.info(f"Extracted {len(result.topics)} topics")
        return result.topics

    async def extract_claims(self, prompt: str) -> LLMExtraction:
        """The fused extraction call (claims + groups + quotes + summary).

        An empty claims list is returned as-is (with a warning) rather than
        raised: a generalized endpoint can legitimately receive material with
        nothing extractable, and the caller should get an empty result, not a
        failed workflow run."""
        response_text = await self._call_gemini(
            prompt=prompt,
            config=self._config(LLMExtraction),
            step_name="claim extraction",
        )
        result = LLMExtraction.model_validate_json(response_text)
        if not result.claims:
            logger.warning("Claim extraction returned zero claims")
        else:
            logger.info(
                f"Extracted {len(result.claims)} claims, {len(result.groups)} groups, "
                f"{len(result.quotes)} quotes"
            )
        return result

    async def extract_takeaways(self, prompt: str) -> List[str]:
        """Key-takeaway selection over extracted claims. Raises on empty — a
        selection pass over a non-empty claim set must select something."""
        response_text = await self._call_gemini(
            prompt=prompt,
            config=self._config(LLMTakeaways),
            step_name="key takeaway extraction",
        )
        result = LLMTakeaways.model_validate_json(response_text)
        if not result.takeaways:
            raise ValueError("Gemini returned empty takeaways list")
        logger.info(f"Selected {len(result.takeaways)} key takeaways")
        return result.takeaways
