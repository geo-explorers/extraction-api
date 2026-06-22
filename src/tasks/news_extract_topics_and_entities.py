"""news.extract_topics_and_entities task — story-level curated/free topics + entities.

A single Claude call (news "Pass 5"), wrapping the synchronous service via
asyncio.to_thread and consuming the global Claude rate limit. The caller passes
its curated topic vocabulary in the payload; extraction-api stays stateless
(geo_id mapping stays caller-side). Distinct from news.extract_topics_and_claims,
which produces the per-claim grouping topics.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.news_topics_entities_schema import (
    NewsTopicsEntitiesRequest,
    NewsTopicsEntitiesResponse,
)
from src.api.services.news_topics_entities_service import (
    extract_story_topics_and_entities,
)
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

_TIMEOUT = timedelta(minutes=3)


async def _handle(
    input: NewsTopicsEntitiesRequest, ctx: Context
) -> NewsTopicsEntitiesResponse:
    spend_guard.check_and_record("claude")
    return await asyncio.to_thread(
        extract_story_topics_and_entities,
        input.headline,
        input.summary,
        input.curated_topic_names,
    )


NEWS_EXTRACT_TOPICS_AND_ENTITIES_SPEC = TaskSpec(
    name="news.extract_topics_and_entities",
    input_model=NewsTopicsEntitiesRequest,
    output_model=NewsTopicsEntitiesResponse,
    handler=_handle,
    rate_limit_key="claude_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=_TIMEOUT,
)
