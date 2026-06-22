"""keyword.extract task — keyword + topic extraction for an episode.

Wraps the existing synchronous keyword service (offloaded via asyncio.to_thread
so the blocking langchain call never stalls the worker loop) and consumes the
global Gemini rate limit. The HTTP /extract/keywords endpoint stays live during
the transition; this task type is the path pg-migrations enqueues.
"""

import asyncio
from datetime import timedelta
from typing import Dict, List

from pydantic import BaseModel, Field
from hatchet_sdk import Context

from src.api.schemas.keyword_extraction_schema import KeywordExtractionRequest
from src.api.services.keyword_extraction_service import extract_keyword_and_topics
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec


class KeywordExtractResult(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    topic_keywords: Dict[str, List[str]] = Field(default_factory=dict)


async def _handle(input: KeywordExtractionRequest, ctx: Context) -> KeywordExtractResult:
    spend_guard.check_and_record("gemini")
    keywords, topics, topic_keywords = await asyncio.to_thread(
        extract_keyword_and_topics,
        input.episode,
        input.topics_list,
        input.min_keywords,
        input.max_keywords,
        input.min_topics,
        input.max_topics,
    )
    return KeywordExtractResult(
        keywords=keywords, topics=topics, topic_keywords=topic_keywords
    )


KEYWORD_EXTRACT_SPEC = TaskSpec(
    name="keyword.extract",
    input_model=KeywordExtractionRequest,
    output_model=KeywordExtractResult,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=timedelta(minutes=3),
)
