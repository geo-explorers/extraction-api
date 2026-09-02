"""Gemini judge for claims.judge_equivalence: is a candidate claim LOGICALLY EQUIVALENT to the
claim — same truth conditions, so accepting one commits you to the other and vice versa?

Provider mechanics mirror ClaimsExtractor (structured output via response_schema, blocking SDK
call offloaded to a thread, app-level retries on transient HTTP errors). The rubric is the
contract; "similar" and "very close" are explicitly NOT equivalent.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Literal, Optional, Sequence

from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

GEMINI_TIMEOUT_SECONDS = 90
APP_MAX_RETRIES = 3
APP_RETRY_INITIAL_DELAY = 5.0
APP_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMVerdict(BaseModel):
    candidate_index: int = Field(description="0-based index into the CANDIDATES list.")
    verdict: Literal["equivalent", "not_equivalent", "unsure"]
    rationale: str = Field(description="One sentence: the decisive difference, or why they are equivalent.")


class LLMJudgement(BaseModel):
    verdicts: List[LLMVerdict]


RUBRIC = """You compare a CLAIM against CANDIDATE claims and decide, for each candidate, whether it is
LOGICALLY EQUIVALENT to the claim: the two sentences have the same truth conditions, so any
situation that makes one true makes the other true, and any situation that makes one false makes
the other false. Accepting one commits a reader to the other, in both directions.

Equivalent:
- same referents (who / what / where), same quantities and units, same time reference, same
  polarity (affirmed vs denied), same modality and hedging ("may", "can", "is", "always")
- paraphrase, synonyms, word order, sentence structure, punctuation and capitalization do NOT
  matter; neither does extra wording that adds no information

Not equivalent (even when the texts are similar or very close):
- one entails the other but not the reverse ("X raised prices by 10%" vs "X raised prices")
- one is more specific or more general ("A and B did X" vs "A did X"; "in 2026" vs "recently")
- different hedging or modality ("X may cause Y" vs "X causes Y")
- different quantities, dates, places, or a different actor; opposite polarity
- one adds a cause, condition, or consequence the other lacks

"unsure" only when a candidate is too ambiguous for a careful reader to decide.

Judge EVERY candidate; return exactly one verdict per candidate index."""


def build_prompt(claim_text: str, candidates: Sequence[str]) -> str:
    listed = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidates))
    return f"{RUBRIC}\n\nCLAIM:\n{claim_text}\n\nCANDIDATES:\n{listed}\n"


class ClaimEquivalenceJudge:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is required for ClaimEquivalenceJudge")
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000)
        )
        self.model_name = model or settings.claims_equivalence_model

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=settings.claims_equivalence_temperature,
            response_mime_type="application/json",
            response_schema=LLMJudgement,
        )

    async def judge(self, claim_text: str, candidates: Sequence[str]) -> List[LLMVerdict]:
        """One call for all candidates. An index the model skips comes back as 'unsure'."""
        if not candidates:
            return []
        raw = await self._call_gemini(build_prompt(claim_text, candidates))
        parsed = LLMJudgement.model_validate(json.loads(raw))
        by_index = {v.candidate_index: v for v in parsed.verdicts if 0 <= v.candidate_index < len(candidates)}
        return [
            by_index.get(i, LLMVerdict(candidate_index=i, verdict="unsure", rationale="no verdict returned"))
            for i in range(len(candidates))
        ]

    async def _call_gemini(self, prompt: str) -> str:
        for attempt in range(1, APP_MAX_RETRIES + 1):
            try:
                start = time.time()
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=self._config(),
                )
                if not response or not response.text or not response.text.strip():
                    raise ValueError("Empty response from Gemini API during equivalence judgement")
                if attempt > 1:
                    logger.info(
                        f"Gemini equivalence judgement succeeded on attempt {attempt} ({time.time() - start:.1f}s)"
                    )
                return response.text
            except APIError as e:
                status_code = getattr(e, "code", None)
                if status_code in APP_RETRYABLE_STATUS_CODES and attempt < APP_MAX_RETRIES:
                    wait = APP_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Gemini equivalence judgement failed (attempt {attempt}/{APP_MAX_RETRIES}): "
                        f"HTTP {status_code}. Retrying in {wait:.0f}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
        raise RuntimeError("unreachable")
