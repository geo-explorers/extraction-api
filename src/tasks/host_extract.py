"""host.extract task — podcast host extraction for an episode.

Wraps the synchronous host service (offloaded via asyncio.to_thread) and
consumes the global Gemini rate limit. Mirrors the /extract/hosts endpoint's
result shape ({hosts: [{name, urls}]}); supports the optional possible_hosts hint.
"""

import asyncio
from datetime import timedelta
from typing import List

from pydantic import BaseModel, Field
from hatchet_sdk import Context

from src.api.schemas.host_extraction_schema import HostExtractionRequest
from src.api.services.host_extraction_service import extract_podcast_hosts
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec


class HostItem(BaseModel):
    name: str
    urls: List[str] = Field(default_factory=list)


class HostExtractResult(BaseModel):
    hosts: List[HostItem] = Field(default_factory=list)


async def _handle(input: HostExtractionRequest, ctx: Context) -> HostExtractResult:
    spend_guard.check_and_record("gemini")
    hosts = await asyncio.to_thread(
        extract_podcast_hosts,
        input.title,
        input.description,
        input.truncated_transcript,
        input.possible_hosts,
    )
    return HostExtractResult(hosts=hosts)


HOST_EXTRACT_SPEC = TaskSpec(
    name="host.extract",
    input_model=HostExtractionRequest,
    output_model=HostExtractResult,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=timedelta(minutes=3),
)
