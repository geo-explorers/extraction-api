"""News claim extraction tasks (Gemini + Claude).

Two task types mirroring the existing HTTP endpoints exactly — fallback stays
caller-orchestrated (news-worker decides when to fail over), so we do NOT chain
Gemini -> Claude inside the service. Each wraps the existing synchronous service
function in `asyncio.to_thread` so the blocking google-genai / anthropic SDK
calls never stall the worker's event loop, and consumes the relevant global
provider rate-limit key.
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
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

_TIMEOUT = timedelta(minutes=8)


async def _handle_gemini(
    input: NewsClaimExtractRequest, ctx: Context
) -> NewsClaimExtractResponse:
    spend_guard.check_and_record("gemini")
    # extract_news_claims is a blocking sync function; offload it so the worker
    # event loop stays free for other concurrent runs.
    return await asyncio.to_thread(
        extract_news_claims, input.headline, input.sources, input.topics
    )


async def _handle_claude(
    input: NewsClaimExtractRequest, ctx: Context
) -> NewsClaimExtractResponse:
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
    rate_limit_units=1,
    retries=3,
    execution_timeout=_TIMEOUT,
)

NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC = TaskSpec(
    name="news.extract_claims_claude",
    input_model=NewsClaimExtractRequest,
    output_model=NewsClaimExtractResponse,
    handler=_handle_claude,
    rate_limit_key="claude_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=_TIMEOUT,
)
