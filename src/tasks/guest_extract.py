"""guest.extract task — podcast guest extraction for an episode.

Wraps the synchronous guest service (offloaded via asyncio.to_thread) and
consumes the global Gemini rate limit. Mirrors the /extract/guests endpoint's
result shape ({guests: [{name, urls}]}).
"""

import asyncio
from datetime import timedelta
from typing import List

from pydantic import BaseModel, Field
from hatchet_sdk import Context

from src.api.schemas.guest_extraction_schema import GuestExtractionRequest
from src.api.services.guest_extraction_service import extract_podcast_guests
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec


class GuestItem(BaseModel):
    name: str
    urls: List[str] = Field(default_factory=list)


class GuestExtractResult(BaseModel):
    guests: List[GuestItem] = Field(default_factory=list)


async def _handle(input: GuestExtractionRequest, ctx: Context) -> GuestExtractResult:
    spend_guard.check_and_record("gemini")
    guests = await asyncio.to_thread(
        extract_podcast_guests,
        input.title,
        input.description,
        input.truncated_transcript,
    )
    return GuestExtractResult(guests=guests)


GUEST_EXTRACT_SPEC = TaskSpec(
    name="guest.extract",
    input_model=GuestExtractionRequest,
    output_model=GuestExtractResult,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=timedelta(minutes=3),
)
