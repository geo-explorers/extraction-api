"""News topic + claim extraction as a checkpointed Hatchet DAG.

Three tasks — extract_topics -> extract_claims_fused -> finalize — fuse what used
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

Rate limits: the topic step consumes one claude_global unit, the claim step one
gemini_global unit, finalize none. Engine-agnostic: Hatchet wiring is confined to
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
from src.api.services.news_topics_extract_service import extract_overview_topics
from src.api.services.news_claim_extract_service import extract_news_claims
from src.tasks.base import DEFAULT_MAX_PAYLOAD_BYTES
from src.infrastructure.spend_guard import spend_guard
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# News payloads are small (~2-5KB/source x up to ~10 sources); the 5MB default is
# ample. Do NOT inherit podcast's 8MB cap (that is transcript-specific).
NEWS_TOPICS_AND_CLAIMS_MAX_PAYLOAD_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
_CLAUDE = [RateLimit(static_key="claude_global", units=1)]
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
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
    # The EXISTING Gemini claim service, unchanged — it already takes a topic
    # list and runs NEWS_CLAIM_EXTRACT_PROMPT. model_dump() so the step output is
    # a JSON-serializable dict for ctx.task_output / checkpointing.
    resp = await asyncio.to_thread(
        extract_news_claims, input.headline, input.sources, topic_list
    )
    return {"claims_result": resp.model_dump()}


@news_topics_and_claims_workflow.task(
    parents=[extract_topics, extract_claims_fused],
    execution_timeout=_FINALIZE_TIMEOUT,
)
async def finalize(
    input: NewsTopicsAndClaimsRequest, ctx: Context
) -> NewsTopicsAndClaimsResponse:
    step1_topics = ctx.task_output(extract_topics)["topics"]
    cr = ctx.task_output(extract_claims_fused)["claims_result"]

    topics = _derive_topics(step1_topics, cr["claims"])

    logger.info(
        f"finalize news story: {len(cr['claims'])} claims, {len(topics)} topics "
        f"({len(step1_topics)} from step 1)"
    )
    return NewsTopicsAndClaimsResponse(
        topics=topics,
        claims=cr["claims"],
        quotes=cr["quotes"],
        collections=cr["collections"],
        collection_order=cr["collection_order"],
        summary=cr["summary"],
    )
