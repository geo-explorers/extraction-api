"""Schemas for the standalone debate task (news.extract_debate_claims).

Split out of news.extract_topics_and_claims so debate never sits on the fused
task's critical path: the injector needs claims fast (an editor is waiting)
and collects debates asynchronously, while the cron pipeline simply awaits the
two tasks in sequence. The consumer passes back the claims the fused task
returned — generation grounds candidates in them and the semantic review
judges against them — so this task adds no extraction of its own.
"""

from pydantic import BaseModel, Field, model_validator
from typing import List

from src.api.schemas.news_claim_extract_schema import (
  NewsArticleSource,
  ExtractedClaim,
  ExtractedDebateClaim,
  normalize_debate_claims,
)


class NewsDebateClaimsRequest(BaseModel):
  headline: str
  sources: List[NewsArticleSource]
  # The fused task's claims, passed back verbatim: candidate generation grounds
  # in them and the reviewer needs the same factual context the consumer saw.
  claims: List[ExtractedClaim]
  # Whether the server may run its repair passes (zero-retry redraw +
  # underfilled rescue) when the first review leaves fewer than three
  # survivors. Each pass adds a full generation + review round (~45-60s), so a
  # latency-bound consumer (the news injector, whose editor is waiting on the
  # response) sends False to answer right after the first review — the 3-floor
  # then empties thin collections instead of repairing them. The response
  # contract is identical either way (0 or 3-5); default True keeps every
  # existing caller, including the cron pipeline, unchanged.
  repair: bool = True


class NewsDebateClaimsResponse(BaseModel):
  debate_claims: List[ExtractedDebateClaim] = Field(default_factory=list)

  # Same repair as NewsTopicsAndClaimsResponse: the 0-or-3-5 contract must
  # hold on every door this shape leaves through.
  @model_validator(mode="after")
  def _enforce_debate_claim_contract(self) -> "NewsDebateClaimsResponse":
    self.debate_claims = normalize_debate_claims(self.debate_claims)
    return self
