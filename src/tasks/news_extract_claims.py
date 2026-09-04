"""News claim + grounded-debate extraction tasks (Gemini + Claude).

Two task types mirroring the existing HTTP endpoints exactly — fallback stays
caller-orchestrated (news-worker decides when to fail over), so we do NOT chain
Gemini -> Claude inside the service. Each provider performs one factual call,
one candidate call, and one semantic-review call, plus at most two calls for a
conditional two-survivor completion pass. Each task wraps the composite service in
`asyncio.to_thread` so the blocking SDK calls never stall the worker event loop,
and conservatively reserves five units from the relevant global provider limit.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.news_claim_extract_schema import (
    NewsClaimExtractRequest,
    NewsClaimExtractResponse,
)
from src.api.services.news_claim_extract_service import (
    extract_news_claims,
    extract_news_claims_claude,
)
from src.config.settings import settings
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

_TIMEOUT = timedelta(minutes=8)


async def _handle_gemini(
    input: NewsClaimExtractRequest, ctx: Context
) -> NewsClaimExtractResponse:
    # Composite service: factual + candidate + review + up to two rescue calls.
    reserved_calls = 5 if settings.news_debate_underfilled_rescue_enabled else 3
    for _ in range(reserved_calls):
        spend_guard.check_and_record("gemini")
    # extract_news_claims is a blocking sync function; offload it so the worker
    # event loop stays free for other concurrent runs.
    return await asyncio.to_thread(
        extract_news_claims, input.headline, input.sources, input.topics
    )


async def _handle_claude(
    input: NewsClaimExtractRequest, ctx: Context
) -> NewsClaimExtractResponse:
    reserved_calls = 5 if settings.news_debate_underfilled_rescue_enabled else 3
    for _ in range(reserved_calls):
        spend_guard.check_and_record("claude")
    return await asyncio.to_thread(
        extract_news_claims_claude, input.headline, input.sources, input.topics
    )


NEWS_EXTRACT_CLAIMS_SPEC = TaskSpec(
    name="news.extract_claims",
    input_model=NewsClaimExtractRequest,
    output_model=NewsClaimExtractResponse,
    handler=_handle_gemini,
    rate_limit_key="gemini_global",
    rate_limit_units=5,
    retries=3,
    execution_timeout=_TIMEOUT,
)

NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC = TaskSpec(
    name="news.extract_claims_claude",
    input_model=NewsClaimExtractRequest,
    output_model=NewsClaimExtractResponse,
    handler=_handle_claude,
    rate_limit_key="claude_global",
    rate_limit_units=5,
    retries=3,
    execution_timeout=_TIMEOUT,
)
