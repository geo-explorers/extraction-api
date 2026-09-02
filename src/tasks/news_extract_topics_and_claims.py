"""News topic + claim extraction as a checkpointed Hatchet DAG.

Six tasks — extract_topics -> extract_claims_fused ->
extract_debate_candidates -> review_debates -> complete_underfilled_debates ->
finalize — fuse what used
to be two separate operations (news-worker's client-side Pass 1 topic extraction
on Claude, then a POST to /extract/news/claims on Gemini) into one task type.
The consumer sends {headline, sources}; step 1 extracts the ordered topic labels,
step 2 feeds them into the EXISTING, UNCHANGED Gemini claim extraction, and
finalize merges the two outputs into one response.

Why a DAG and not one handler: the Gemini claim call is the long, dense step
(8-min timeout). With separate checkpointed steps, a Gemini failure or a worker
redeploy re-runs ONLY extract_claims_fused and never re-bills the already-
succeeded Claude topic call. The label validation + single feedback retry live
INSIDE extract_topics as plain Python (not a 4th step) — bounded to one extra
Claude call, only on the rare label-violation path.

Rate limits: the topic step consumes one claude_global unit; the factual,
debate-candidate, and semantic-review steps each consume one gemini_global
unit. The conditional underfilled pass (one or two survivors) reserves two
more Gemini units for its candidate and review calls; zero-survivor and full
results are terminal and finalize consumes none.
Engine-agnostic: Hatchet wiring is confined to
the @workflow.task decorators here plus base.py/worker.py, exactly like the
podcast DAG — the handlers themselves import only Context.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet
from src.api.schemas.news_topics_and_claims_schema import (
    NewsTopicsAndClaimsRequest,
    NewsTopicsAndClaimsResponse,
)
from src.api.schemas.news_claim_extract_schema import ExtractedClaim
from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.schemas.news_debate_semantic_review_schema import DebateSemanticVerdict
from src.api.services.news_topics_extract_service import extract_overview_topics
from src.api.services.news_claim_extract_service import extract_news_claims_factual
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

# News payloads are small (~2-5KB/source x up to ~10 sources); the 5MB default is
# ample. Do NOT inherit podcast's 8MB cap (that is transcript-specific).
NEWS_TOPICS_AND_CLAIMS_MAX_PAYLOAD_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
_CLAUDE = [RateLimit(static_key="claude_global", units=1)]
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
_GEMINI_RESCUE = [RateLimit(static_key="gemini_global", units=2)]
_TOPIC_TIMEOUT = timedelta(minutes=3)
_CLAIM_TIMEOUT = timedelta(minutes=8)
_FINALIZE_TIMEOUT = timedelta(minutes=2)


def _derive_topics(step1_topics: list[str], claims: list[dict]) -> list[str]:
    """The authoritative topic list is what the claim pass ACTUALLY used: the
    distinct claim.topic values in first-seen order. This captures topics the
    claim prompt relabeled one level broader or added to home orphan facts
    (mirrors news-worker's `[...new Set(fresh.claims.map(c => c.topic))]`). Falls
    back to the Pass-1 labels when the story produced no claims."""
    seen: list[str] = []
    for claim in claims:
        topic = claim.get("topic")
        if topic and topic not in seen:
            seen.append(topic)
    return seen or step1_topics


news_topics_and_claims_workflow = hatchet.workflow(
    name="news.extract_topics_and_claims",
    input_validator=NewsTopicsAndClaimsRequest,
)


@news_topics_and_claims_workflow.task(
    rate_limits=_CLAUDE, execution_timeout=_TOPIC_TIMEOUT, retries=3, backoff_factor=2.0
)
async def extract_topics(input: NewsTopicsAndClaimsRequest, ctx: Context) -> dict:
    spend_guard.check_and_record("claude")
    # Blocking Anthropic SDK call (incl. its own bounded feedback retry); offload
    # so the worker event loop stays free for other concurrent runs.
    topics = await asyncio.to_thread(
        extract_overview_topics, input.headline, input.sources
    )
    return {"topics": topics}


@news_topics_and_claims_workflow.task(
    parents=[extract_topics],
    rate_limits=_GEMINI,
    execution_timeout=_CLAIM_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def extract_claims_fused(
    input: NewsTopicsAndClaimsRequest, ctx: Context
) -> dict:
    topic_list = ctx.task_output(extract_topics)["topics"]
    spend_guard.check_and_record("gemini")
    # First pass: factual claims/collections/summary. The exported prompt forces
    # debate_claims empty; the next checkpointed step exclusively owns them.
    resp = await asyncio.to_thread(
        extract_news_claims_factual, input.headline, input.sources, topic_list
    )
    return {"claims_result": resp.model_dump()}


@news_topics_and_claims_workflow.task(
    parents=[extract_claims_fused],
    rate_limits=_GEMINI,
    execution_timeout=_CLAIM_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def extract_debate_candidates(
    input: NewsTopicsAndClaimsRequest, ctx: Context
) -> dict:
    cr = ctx.task_output(extract_claims_fused)["claims_result"]
    claims = [ExtractedClaim.model_validate(claim) for claim in cr["claims"]]
    spend_guard.check_and_record("gemini")
    candidates = await asyncio.to_thread(
        generate_news_debate_candidates, input.headline, input.sources, claims
    )
    return {"candidates": [candidate.model_dump() for candidate in candidates]}


@news_topics_and_claims_workflow.task(
    parents=[extract_claims_fused, extract_debate_candidates],
    rate_limits=_GEMINI,
    execution_timeout=_CLAIM_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def review_debates(
    input: NewsTopicsAndClaimsRequest, ctx: Context
) -> dict:
    cr = ctx.task_output(extract_claims_fused)["claims_result"]
    claims = [ExtractedClaim.model_validate(claim) for claim in cr["claims"]]
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
            claims,
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


@news_topics_and_claims_workflow.task(
    parents=[extract_claims_fused, extract_debate_candidates, review_debates],
    rate_limits=_GEMINI_RESCUE,
    execution_timeout=_CLAIM_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def complete_underfilled_debates(
    input: NewsTopicsAndClaimsRequest, ctx: Context
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
    zero_retry = (
        len(accepted) == 0
        and len(attempted) > 0
        and settings.news_debate_zero_retry_enabled
    )
    rescue = (
        0 < len(accepted) < 3
        and settings.news_debate_underfilled_rescue_enabled
    )
    if not zero_retry and not rescue:
        return {
            "debate_claims": review["debate_claims"],
            "completion_semantic_verdicts": [],
        }

    cr = ctx.task_output(extract_claims_fused)["claims_result"]
    claims = [ExtractedClaim.model_validate(claim) for claim in cr["claims"]]
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
        claims,
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


@news_topics_and_claims_workflow.task(
    parents=[extract_topics, extract_claims_fused, complete_underfilled_debates],
    execution_timeout=_FINALIZE_TIMEOUT,
)
async def finalize(
    input: NewsTopicsAndClaimsRequest, ctx: Context
) -> NewsTopicsAndClaimsResponse:
    step1_topics = ctx.task_output(extract_topics)["topics"]
    cr = ctx.task_output(extract_claims_fused)["claims_result"]
    debate_claims = ctx.task_output(complete_underfilled_debates)["debate_claims"]

    topics = _derive_topics(step1_topics, cr["claims"])

    logger.info(
        f"finalize news story: {len(cr['claims'])} claims, {len(topics)} topics "
        f"({len(step1_topics)} from step 1), {len(debate_claims)} debate claims"
    )
    return NewsTopicsAndClaimsResponse(
        topics=topics,
        claims=cr["claims"],
        quotes=cr["quotes"],
        collections=cr["collections"],
        collection_order=cr["collection_order"],
        debate_claims=debate_claims,
        summary=cr["summary"],
    )
