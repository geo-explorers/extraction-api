"""Collection review as a standalone task (news.review_collections).

Runs on the fused task's output and rebuilds the story's blocks from its
claims — group, rescue, merge, check (see the service). Its own task type,
not a step of news.extract_topics_and_claims, for the same reason debates
are: the fused task keeps answering at extraction speed, and each consumer
decides when to wait — the cron pipeline right away, the injector before
prepare-ops while entity resolution, covers and the debate lane run.

Reserves four gemini_global units for the common path (group, rescue, merge,
check); the source checks and the one regroup a rejected check earns are
cheap and rare enough to ride on those.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.news_collection_review_schema import (
    NewsCollectionReviewRequest,
    NewsCollectionReviewResponse,
)
from src.api.services.news_collection_review_service import review_collections
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

# Four calls on the common path at ~10-30s each, one source check per composed
# sentence, and a regroup + second check when the first check rejects.
_TIMEOUT = timedelta(minutes=6)
_GEMINI_CALLS = 4


async def _handle(
    input: NewsCollectionReviewRequest, ctx: Context
) -> NewsCollectionReviewResponse:
    for _ in range(_GEMINI_CALLS):
        spend_guard.check_and_record("gemini")
    return await asyncio.to_thread(
        review_collections,
        input.headline,
        input.sources,
        input.claims,
        input.quotes,
        input.collections,
        input.collection_order,
        input.thinking_level,
    )


NEWS_REVIEW_COLLECTIONS_SPEC = TaskSpec(
    name="news.review_collections",
    input_model=NewsCollectionReviewRequest,
    output_model=NewsCollectionReviewResponse,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=_GEMINI_CALLS,
    retries=2,
    execution_timeout=_TIMEOUT,
)
