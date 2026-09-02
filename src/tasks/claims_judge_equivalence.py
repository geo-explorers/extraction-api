"""claims.judge_equivalence — which of these candidate claims mean EXACTLY what this claim means?

A pure judge: one Gemini call over claim + candidates, verdict per candidate, the equivalent
ones surfaced. No retrieval here — candidates come from wherever the caller found them
(typically geo-lens ``/caches/{handle}/query``), which keeps the two services loosely coupled.
DB-free: results return to the caller.
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from typing import List

from hatchet_sdk import Context

from src.api.schemas.claims_judge_equivalence_schema import (
    ClaimsJudgeEquivalenceInput,
    ClaimsJudgeEquivalenceResult,
    JudgedCandidate,
)
from src.extraction.claim_equivalence_judge import ClaimEquivalenceJudge, LLMVerdict
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

CLAIMS_JUDGE_MAX_PAYLOAD_BYTES = 512 * 1024


@lru_cache(maxsize=1)
def _judge() -> ClaimEquivalenceJudge:
    return ClaimEquivalenceJudge()


def assemble(input: ClaimsJudgeEquivalenceInput, verdicts: List[LLMVerdict], model: str) -> ClaimsJudgeEquivalenceResult:
    judged = [
        JudgedCandidate(index=i, id=c.id, text=c.text, verdict=v.verdict, rationale=v.rationale)
        for i, (c, v) in enumerate(zip(input.candidates, verdicts))
    ]
    return ClaimsJudgeEquivalenceResult(
        claim=input.claim,
        equivalent=[j for j in judged if j.verdict == "equivalent"],
        unsure=[j for j in judged if j.verdict == "unsure"],
        judged=judged,
        model_used=model,
    )


async def _handle(input: ClaimsJudgeEquivalenceInput, ctx: Context) -> ClaimsJudgeEquivalenceResult:
    judge = _judge()
    spend_guard.check_and_record("gemini")
    verdicts = await judge.judge(input.claim.text, [c.text for c in input.candidates])
    result = assemble(input, verdicts, judge.model_name)
    logger.info(
        f"claims.judge_equivalence: {len(input.candidates)} candidates, "
        f"{len(result.equivalent)} equivalent, {len(result.unsure)} unsure"
    )
    return result


CLAIMS_JUDGE_EQUIVALENCE_SPEC = TaskSpec(
    name="claims.judge_equivalence",
    input_model=ClaimsJudgeEquivalenceInput,
    output_model=ClaimsJudgeEquivalenceResult,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=2,
    execution_timeout=timedelta(minutes=5),
    max_payload_bytes=CLAIMS_JUDGE_MAX_PAYLOAD_BYTES,
)
