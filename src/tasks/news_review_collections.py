"""Collection review as a standalone task (news.review_collections).

Runs on the fused task's output and removes claims that say the same thing
inside one collection (see the service). Its own task type, not a step of
news.extract_topics_and_claims, for the same reason debates are: the fused
task keeps answering at extraction speed, and each consumer decides when to
wait — the cron pipeline right away, the injector before prepare-ops while
entity resolution, covers and the debate lane run.

Reserves two gemini_global units: the review call and, when it composed a
merged sentence, one source check.
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

_TIMEOUT = timedelta(minutes=4)


async def _handle(
    input: NewsCollectionReviewRequest, ctx: Context
) -> NewsCollectionReviewResponse:
    for _ in range(2):
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
    rate_limit_units=2,
    retries=2,
    execution_timeout=_TIMEOUT,
)
