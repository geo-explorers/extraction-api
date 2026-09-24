"""Schemas for the fused news topic + claim task (news.extract_topics_and_claims).

The consumer sends only {headline, sources}: unlike NewsClaimExtractRequest,
there is NO topics field, because the task extracts the topics itself (step 1)
before grouping claims under them (step 2). The response adds a top-level
`topics` list to the existing claim-response shape so downstream consumers keep
the topic labels without a second call. Claim/quote/collection types are reused
verbatim, so the claim portion stays byte-identical to /extract/news/claims.
"""

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin
from typing import List, Optional

from src.api.schemas.news_claim_extract_schema import (
  AnchorReport,
  NewsArticleSource,
  ExtractedClaim,
  ExtractedQuote,
  ExtractedCollection,
  ExtractedDebateClaim,
  normalize_debate_claims,
)


class NewsTopicsAndClaimsRequest(OverridesMixin):
  headline: str
  sources: List[NewsArticleSource]


class NewsTopicsAndClaimsResponse(BaseModel):
  # Ordered topic labels. The finalize step derives these from the topics the
  # claim pass actually used (so relabeled/added topics are reflected), falling
  # back to the Pass-1 labels when there are no claims.
  topics: List[str] = Field(default_factory=list)
  claims: List[ExtractedClaim] = Field(default_factory=list)
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection] = Field(default_factory=list)
  collection_order: List[str] = Field(default_factory=list)
  debate_claims: List[ExtractedDebateClaim] = Field(default_factory=list)
  summary: str = ""
  # What the source-anchor guard repaired or dropped in the claim pass, so
  # the consumer can log it; absent from a server that predates the guard.
  anchor_report: Optional[AnchorReport] = None

  # Same repair as NewsClaimExtractResponse: this model is also constructed
  # directly (task finalize, HTTP layer), so the contract must hold here too.
  @model_validator(mode="after")
  def _enforce_debate_claim_contract(self) -> "NewsTopicsAndClaimsResponse":
    self.debate_claims = normalize_debate_claims(self.debate_claims)
    return self
