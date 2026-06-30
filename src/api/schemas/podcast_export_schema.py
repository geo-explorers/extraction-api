"""Schemas for the podcast.export task.

The request mirrors the postgres_to_geo `POST /api/export` body field-for-field
so the task handler is a pure pass-through forwarder; the result mirrors that
endpoint's success `data` block. Keeping these identical means the task adds
durability + single-consumer queueing around the existing publish without
reshaping its contract.
"""

from pydantic import BaseModel, Field
from typing import List


class PodcastExportRequest(BaseModel):
    podcast_name: List[str] = Field(
        ..., description="Podcast names to export (the export selects their recent episodes)"
    )
    limit: int = Field(..., ge=1, description="Episodes-per-podcast cap")
    num_episodes: int = Field(..., ge=1, description="Overall episode cap")
    date_filter: str = Field(
        ..., description="YYYY-MM-DD; only episodes with air_date after this are exported"
    )


class PodcastExportResult(BaseModel):
    success: bool = False
    episodes_processed: int = 0
    ops_created: int = 0
    duration_ms: int = 0
    message: str = ""
