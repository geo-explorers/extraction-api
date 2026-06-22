"""Schemas for the story-level topic + entity task (news.extract_topics_and_entities).

These are whole-story curated/free topics and related entities (news "Pass 5"),
distinct from the per-claim grouping topics in news.extract_topics_and_claims.
The caller passes its curated vocabulary in `curated_topic_names`; the response
carries each topic's `source` (curated|llm) but NO geo_id — the caller attaches
geo_id from its own name->geoId map (extraction-api stays stateless).
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class NewsTopicsEntitiesRequest(BaseModel):
  headline: str
  summary: str = ""
  curated_topic_names: List[str] = Field(
    default_factory=list,
    description="Caller's curated topic vocabulary (exact, case-sensitive). Empty -> free-form topics only.",
  )


class ExtractedStoryTopic(BaseModel):
  name: str
  relevance: float
  source: Literal["curated", "llm"]


class ExtractedStoryEntity(BaseModel):
  name: str
  type: str
  role: str = ""


class NewsTopicsEntitiesResponse(BaseModel):
  topics: List[ExtractedStoryTopic] = Field(default_factory=list)
  entities: List[ExtractedStoryEntity] = Field(default_factory=list)
