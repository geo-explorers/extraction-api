from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class NewsArticleSource(BaseModel):
  index: int
  url: str
  title: str
  publisher: Optional[str] = None
  published_at: Optional[str] = None
  content: str


class NewsClaimExtractRequest(BaseModel):
  headline: str
  sources: List[NewsArticleSource]
  topics: List[str] = Field(
    default_factory=list,
    description="Topic labels already extracted for this story (Pass 1 output). Required for topic-grouped claim emission."
  )


# ── Response types ─────────────────────────────────────────────────────


class ExtractedClaim(BaseModel):
  text: str
  topic: str
  source_indices: List[int] = Field(default_factory=list)
  confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class ExtractedQuote(BaseModel):
  text: str
  speaker: Optional[str] = None
  claim_index: int = Field(
    ge=0,
    description="0-based index into the 'claims' array"
  )


class ExtractedCollection(BaseModel):
  name: str
  type: Literal["topic", "perspective"]
  summary: str = ""
  claim_indices: List[int] = Field(default_factory=list)


class NewsClaimExtractResponse(BaseModel):
  claims: List[ExtractedClaim] = Field(default_factory=list)
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection] = Field(default_factory=list)
  collection_order: List[str] = Field(default_factory=list)
  summary: str = ""
