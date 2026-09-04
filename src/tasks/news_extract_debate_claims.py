"""Standalone grounded-debate extraction as a checkpointed Hatchet DAG.

Three tasks — extract_debate_candidates -> review_debates ->
complete_underfilled_debates -> finalize — the exact debate chain that
news.extract_topics_and_claims used to run inline. Split into its own task
type so debate never sits on the fused task's critical path: the fused task
answers at pre-debate speed, and each consumer decides when to pay for
debates. The cron pipeline awaits this task right after the fused one
(sequential, latency-insensitive); the injector fires it in the background
and attaches the result while the editor reviews the story.

The consumer sends {headline, sources, claims} — the claims are the fused
task's own output passed back verbatim, because candidate generation grounds
in them and the reject-only semantic review judges against them. Grounding,
review gates, zero-retry, and the underfilled completion pass are all
unchanged: same services, same settings, same 0-or-3-5 contract.

Rate limits: candidate generation and semantic review each consume one
gemini_global unit; the conditional completion pass reserves two more for its
candidate and review calls; zero-survivor and full results are terminal and
finalize consumes none.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet
from src.api.schemas.news_debate_claims_task_schema import (
    NewsDebateClaimsRequest,
    NewsDebateClaimsResponse,
)
from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.schemas.news_debate_semantic_review_schema import DebateSemanticVerdict
from src.api.services.news_debate_claim_service import (
    complete_underfilled_news_debate_candidates,
    generate_news_debate_candidates,
    project_debate_candidates,
)
from src.api.services.news_debate_semantic_review_service import (
    review_news_debate_candidates,
)
from src.tasks.base import DEFAULT_MAX_PAYLOAD_BYTES
from src.config.settings import settings
from src.infrastructure.spend_guard import spend_guard
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Same shape as the fused news task's payloads plus the claims echo — the 5MB
# default is ample.
NEWS_DEBATE_CLAIMS_MAX_PAYLOAD_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
_GEMINI_RESCUE = [RateLimit(static_key="gemini_global", units=2)]
_STEP_TIMEOUT = timedelta(minutes=8)
_FINALIZE_TIMEOUT = timedelta(minutes=2)


news_debate_claims_workflow = hatchet.workflow(
    name="news.extract_debate_claims",
    input_validator=NewsDebateClaimsRequest,
)


@news_debate_claims_workflow.task(
    rate_limits=_GEMINI,
    execution_timeout=_STEP_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def extract_debate_candidates(
    input: NewsDebateClaimsRequest, ctx: Context
) -> dict:
    spend_guard.check_and_record("gemini")
    candidates = await asyncio.to_thread(
        generate_news_debate_candidates, input.headline, input.sources, input.claims
    )
    return {"candidates": [candidate.model_dump() for candidate in candidates]}


@news_debate_claims_workflow.task(
    parents=[extract_debate_candidates],
    rate_limits=_GEMINI,
    execution_timeout=_STEP_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def review_debates(
    input: NewsDebateClaimsRequest, ctx: Context
) -> dict:
    candidates = [
        GroundedDebateCandidate.model_validate(candidate)
        for candidate in ctx.task_output(extract_debate_candidates)["candidates"]
    ]
    if candidates:
        spend_guard.check_and_record("gemini")
        accepted, verdicts = await asyncio.to_thread(
            review_news_debate_candidates,
            input.headline,
            input.sources,
            input.claims,
            candidates,
        )
    else:
        accepted, verdicts = [], []
    debate_claims = project_debate_candidates(accepted)
    return {
        "accepted_candidates": [candidate.model_dump() for candidate in accepted],
        "debate_claims": [claim.model_dump() for claim in debate_claims],
        "semantic_verdicts": [verdict.model_dump() for verdict in verdicts],
    }


@news_debate_claims_workflow.task(
    parents=[extract_debate_candidates, review_debates],
    rate_limits=_GEMINI_RESCUE,
    execution_timeout=_STEP_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def complete_underfilled_debates(
    input: NewsDebateClaimsRequest, ctx: Context
) -> dict:
    review = ctx.task_output(review_debates)
    accepted = [
        GroundedDebateCandidate.model_validate(candidate)
        for candidate in review["accepted_candidates"]
    ]
    attempted = [
        GroundedDebateCandidate.model_validate(candidate)
        for candidate in ctx.task_output(extract_debate_candidates)["candidates"]
    ]
    # The retry/rescue POLICY lives in complete_underfilled_news_debate_candidates
    # (full collections and candidate-less zeros are terminal there). This early
    # return only mirrors it to avoid reserving Gemini spend units for a call
    # the service would refuse — keep the conditions in sync. Either branch
    # spends at most one generation and one review call (the reserved 2 units).
    # input.repair (consumer opt-out, default True) gates both passes: a
    # latency-bound caller takes the first review's verdict as final.
    zero_retry = (
        input.repair
        and len(accepted) == 0
        and len(attempted) > 0
        and settings.news_debate_zero_retry_enabled
    )
    rescue = (
        input.repair
        and 0 < len(accepted) < 3
        and settings.news_debate_underfilled_rescue_enabled
    )
    if not zero_retry and not rescue:
        return {
            "debate_claims": review["debate_claims"],
            "completion_semantic_verdicts": [],
        }

    verdicts = [
        DebateSemanticVerdict.model_validate(verdict)
        for verdict in review["semantic_verdicts"]
    ]
    # Reserve the maximum conditional cost: one completion generation call and,
    # only when that yields a mechanically valid candidate, one review call.
    for _ in range(2):
        spend_guard.check_and_record("gemini")
    accepted, completion_verdicts = await asyncio.to_thread(
        complete_underfilled_news_debate_candidates,
        input.headline,
        input.sources,
        input.claims,
        attempted,
        accepted,
        verdicts,
    )
    debate_claims = project_debate_candidates(accepted)
    return {
        "debate_claims": [claim.model_dump() for claim in debate_claims],
        "completion_semantic_verdicts": [
            verdict.model_dump() for verdict in completion_verdicts
        ],
    }


@news_debate_claims_workflow.task(
    parents=[complete_underfilled_debates],
    execution_timeout=_FINALIZE_TIMEOUT,
)
async def finalize(
    input: NewsDebateClaimsRequest, ctx: Context
) -> NewsDebateClaimsResponse:
    debate_claims = ctx.task_output(complete_underfilled_debates)["debate_claims"]
    logger.info(
        f"finalize news debates: {len(debate_claims)} debate claims "
        f"for '{input.headline[:60]}'"
    )
    return NewsDebateClaimsResponse(debate_claims=debate_claims)
