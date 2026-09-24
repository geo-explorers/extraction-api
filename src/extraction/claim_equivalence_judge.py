"""Gemini judge for claims.judge_equivalence: is a candidate claim LOGICALLY EQUIVALENT to the
claim — same truth conditions, so accepting one commits you to the other and vice versa?

Provider mechanics mirror ClaimsExtractor (structured output via response_schema, blocking SDK
call offloaded to a thread, app-level retries on transient HTTP errors). The rubric is the
contract; "similar" and "very close" are explicitly NOT equivalent.

The model does not return a verdict. It returns, per candidate, what each sentence asserts and
whether each entails the other; `verdict_from` derives the verdict from those answers. Asked for a
verdict directly, the model judged by thematic correspondence ("the candidate's problems directly
correspond to trustworthiness and eligibility") and called related claims on the same side
equivalent; made to answer the two directional questions, it does not.
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

from src.config.overrides import llm, prompts
from src.config.prompts.claims_judge_equivalence_prompt import CLAIMS_JUDGE_EQUIVALENCE_RUBRIC
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

GEMINI_TIMEOUT_SECONDS = 90
APP_MAX_RETRIES = 3
APP_RETRY_INITIAL_DELAY = 5.0
APP_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


Relation = Literal[
    "same",  # identical or a pure rewording
    "claim_implies_candidate_only",  # the claim is more specific: it adds a cause, a group, a number…
    "candidate_implies_claim_only",  # the candidate is the more specific one
    "related",  # same topic or side, different assertion (intent vs effect, signal vs purpose…)
    "contradicts",
    "unrelated",
]


class LLMAssessment(BaseModel):
    """What the model reports per candidate. The verdict is NOT one of its fields: it is derived in
    code from the two directional answers, so the model has to do the entailment test rather than
    reach for a gut feeling of sameness."""

    candidate_index: int = Field(description="0-based index into the CANDIDATES list.")
    claim_asserts: str = Field(
        description="What the CLAIM asserts, in a few words, naming its kind: an effect, an intent or purpose, "
        "a necessity, a signal or evidence, a mechanism, a frequency, a scope, a value judgement…"
    )
    candidate_asserts: str = Field(description="The same for this CANDIDATE.")
    claim_implies_candidate: bool = Field(
        description="If the CLAIM is true, must the CANDIDATE be true? (Not 'would it be plausible' — must.)"
    )
    candidate_implies_claim: bool = Field(
        description="If the CANDIDATE is true, must the CLAIM be true? (Not 'would it be plausible' — must.)"
    )
    relation: Relation
    decisive_difference: str = Field(
        description="One sentence: the difference that breaks equivalence, or 'none' when the two are the same claim."
    )
    unsure: bool = Field(default=False, description="True only when a careful reader could not decide.")


class LLMJudgement(BaseModel):
    assessments: List[LLMAssessment]


class LLMVerdict(BaseModel):
    """The public verdict per candidate (what the task returns)."""

    candidate_index: int
    verdict: Literal["equivalent", "not_equivalent", "unsure"]
    rationale: str


def verdict_from(assessment: LLMAssessment) -> LLMVerdict:
    """Equivalent only when the model affirmed BOTH directions and called the relation `same`. Any
    single disagreement among the three is a difference the model itself found, so it wins."""
    if assessment.unsure:
        verdict: Literal["equivalent", "not_equivalent", "unsure"] = "unsure"
    elif (
        assessment.claim_implies_candidate
        and assessment.candidate_implies_claim
        and assessment.relation == "same"
    ):
        verdict = "equivalent"
    else:
        verdict = "not_equivalent"
    difference = assessment.decisive_difference.strip()
    if verdict == "equivalent":
        rationale = difference if difference and difference.lower() != "none" else "Same assertion, both directions hold."
    else:
        detail = difference if difference and difference.lower() != "none" else assessment.relation.replace("_", " ")
        rationale = f"{assessment.claim_asserts} vs {assessment.candidate_asserts}: {detail}"
    return LLMVerdict(candidate_index=assessment.candidate_index, verdict=verdict, rationale=rationale)


# RUBRIC lives in src/config/prompts/claims_judge_equivalence_prompt.py (prompt
# key claims_judge_equivalence.rubric); re-exported here for existing imports.
RUBRIC = CLAIMS_JUDGE_EQUIVALENCE_RUBRIC


def build_prompt(claim_text: str, candidates: Sequence[str]) -> str:
    listed = "\n".join(f"[{i}] {text}" for i, text in enumerate(candidates))
    rubric = prompts.get("claims_judge_equivalence.rubric")
    return f"{rubric}\n\nCLAIM:\n{claim_text}\n\nCANDIDATES:\n{listed}\n"


class ClaimEquivalenceJudge:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is required for ClaimEquivalenceJudge")
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_SECONDS * 1000)
        )
        self._explicit_model = model

    @property
    def model_name(self) -> str:
        # An explicit model (constructor arg or assignment; CLI/eval use) wins;
        # otherwise resolve per call so a run's llm_overrides apply.
        return self._explicit_model or llm.get("claims_equivalence_model")

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        self._explicit_model = value


    def _config(self) -> types.GenerateContentConfig:
        config_kwargs: dict = {
            "temperature": llm.get("claims_equivalence_temperature"),
            "response_mime_type": "application/json",
            "response_schema": LLMJudgement,
        }
        thinking_level = (llm.get("claims_equivalence_thinking_level") or "").strip()
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        return types.GenerateContentConfig(**config_kwargs)

    async def judge(self, claim_text: str, candidates: Sequence[str]) -> List[LLMVerdict]:
        """One call for all candidates. An index the model skips comes back as 'unsure'."""
        if not candidates:
            return []
        raw = await self._call_gemini(build_prompt(claim_text, candidates))
        parsed = LLMJudgement.model_validate(json.loads(raw))
        by_index = {
            a.candidate_index: verdict_from(a) for a in parsed.assessments if 0 <= a.candidate_index < len(candidates)
        }
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
