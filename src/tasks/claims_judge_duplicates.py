"""claims.judge_duplicates — find already-published claims that mean exactly the same thing.

Per input claim: gather candidates from geo-lens (vector + text strategies over its claims
cache, merged by id, best score kept, the claim's own id excluded), then ONE Gemini call
judges every candidate against the rubric in claim_dedup_judge. DB-free: results return to
the caller, who persists.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from functools import lru_cache
from typing import Dict, List, Sequence

from hatchet_sdk import Context

from src.api.schemas.claims_judge_duplicates_schema import (
    Candidate,
    ClaimIn,
    ClaimResult,
    ClaimsJudgeDuplicatesInput,
    ClaimsJudgeDuplicatesResult,
    Match,
)
from src.api.services.geo_lens_client import GeoLensClient, LensHit
from src.config.settings import settings
from src.extraction.claim_dedup_judge import ClaimDedupJudge
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

CLAIMS_JUDGE_MAX_PAYLOAD_BYTES = 1 * 1024 * 1024


@lru_cache(maxsize=1)
def _judge() -> ClaimDedupJudge:
    return ClaimDedupJudge()


@lru_cache(maxsize=1)
def _lens() -> GeoLensClient:
    return GeoLensClient()


def merge_candidates(hits_by_strategy: Dict[str, Sequence[LensHit]], exclude_id: str | None) -> List[Candidate]:
    """Union of every strategy's hits keyed by id; best score kept; the claim itself excluded;
    ordered best score first."""
    merged: Dict[str, Candidate] = {}
    for strategy, hits in hits_by_strategy.items():
        for h in hits:
            if exclude_id and h.id == exclude_id:
                continue
            existing = merged.get(h.id)
            if existing is None:
                merged[h.id] = Candidate(id=h.id, text=h.name, score=h.score, sources=[strategy])
            else:
                merged[h.id] = Candidate(
                    id=h.id,
                    text=existing.text or h.name,
                    score=max(existing.score, h.score),
                    sources=sorted(set(existing.sources) | {strategy}),
                )
    return sorted(merged.values(), key=lambda c: -c.score)


async def gather_candidates(lens: GeoLensClient, claim: ClaimIn, input: ClaimsJudgeDuplicatesInput, cache: str) -> List[Candidate]:
    filters = {"spaceIds": input.space_ids} if input.space_ids else {}

    async def one(strategy: str) -> Sequence[LensHit]:
        return await lens.query(
            cache,
            strategy,
            {"text": claim.text},
            k=input.k,
            min_score=input.min_score if strategy == "vector" else None,
            filters=filters,
        )

    results = await asyncio.gather(*(one(s) for s in input.strategies))
    return merge_candidates(dict(zip(input.strategies, results)), claim.id)


async def judge_claim(judge: ClaimDedupJudge, claim: ClaimIn, candidates: List[Candidate]) -> ClaimResult:
    if not candidates:
        return ClaimResult(claim=claim, candidates_considered=0, matches=[], same=[])
    spend_guard.check_and_record("gemini")
    verdicts = await judge.judge(claim.text, [c.text for c in candidates])
    matches = [
        Match(id=c.id, text=c.text, score=c.score, sources=c.sources, verdict=v.verdict, rationale=v.rationale)
        for c, v in zip(candidates, verdicts)
    ]
    return ClaimResult(
        claim=claim,
        candidates_considered=len(candidates),
        matches=matches,
        same=[m.id for m in matches if m.verdict == "same"],
        unsure=[m.id for m in matches if m.verdict == "unsure"],
    )


async def _handle(input: ClaimsJudgeDuplicatesInput, ctx: Context) -> ClaimsJudgeDuplicatesResult:
    cache = input.cache or settings.geo_lens_claims_cache
    lens, judge = _lens(), _judge()
    results: List[ClaimResult] = []
    llm_calls = 0
    for claim in input.claims:
        candidates = await gather_candidates(lens, claim, input, cache)
        result = await judge_claim(judge, claim, candidates)
        llm_calls += 1 if candidates else 0
        results.append(result)
    logger.info(
        f"claims.judge_duplicates: {len(input.claims)} claims, {llm_calls} LLM calls, "
        f"{sum(len(r.same) for r in results)} judged same"
    )
    return ClaimsJudgeDuplicatesResult(results=results, cache=cache, model_used=judge.model_name, llm_calls=llm_calls)


CLAIMS_JUDGE_DUPLICATES_SPEC = TaskSpec(
    name="claims.judge_duplicates",
    input_model=ClaimsJudgeDuplicatesInput,
    output_model=ClaimsJudgeDuplicatesResult,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=2,
    execution_timeout=timedelta(minutes=10),
    max_payload_bytes=CLAIMS_JUDGE_MAX_PAYLOAD_BYTES,
)
