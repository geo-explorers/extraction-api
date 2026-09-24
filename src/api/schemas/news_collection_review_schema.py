"""Schemas for the collection review task (news.review_collections).

Runs AFTER news.extract_topics_and_claims on that task's own output and
removes the one defect readers report most: two claims in one collection that
say the same thing (a restatement, or one fact cut into pieces). It merges
such claims, moves a claim whose collection is left alone into the collection
that covers it, and returns the same shape it was given, re-numbered.

Its own task type, like news.extract_debate_claims, so each consumer decides
when to pay for it: the cron pipeline awaits it right after the fused task; the
injector enqueues it as soon as the claims arrive and awaits it before
prepare-ops, overlapped with entity resolution, covers and the debate wait.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field
from src.api.schemas.overrides_schema import OverridesMixin

from src.api.schemas.news_claim_extract_schema import (
  ExtractedClaim,
  ExtractedCollection,
  ExtractedQuote,
  NewsArticleSource,
)


class NewsCollectionReviewRequest(OverridesMixin):
  headline: str
  # The story's sources, for the source check on every merged sentence the
  # review composes — a merge that joins two facts can imply a relation the
  # sources never state, and only the sources can show that.
  sources: List[NewsArticleSource]
  # The fused task's output, passed back verbatim.
  claims: List[ExtractedClaim]
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection]
  collection_order: List[str] = Field(default_factory=list)
  # How hard the reviewer thinks. Measured on 30 prod stories against an
  # independent judge: "high" leaves ~3% of two-claim collections with a
  # same-fact pair at ~33s, "medium" ~5% at ~23s, "low" ~7% at ~8s (from
  # 9-12% unreviewed). The pipeline sends "high" (nobody waits); the injector
  # sends "medium" and overlaps the wait.
  thinking_level: Literal["low", "medium", "high"] = "high"


class CollectionReviewReport(BaseModel):
  # False when the review was discarded (a step failed, the check rejected
  # the grouping twice, or a core claim would have been dropped) and the
  # input was returned untouched — the extraction's own grouping.
  applied: bool
  # Blocks in the result.
  blocks: int = 0
  # New claims written from the sources to give a lone claim company.
  rescued_claims: int = 0
  merges: int = 0
  # Claims folded into another claim (a merge of three counts two).
  merged_claims: int = 0
  # Lone claims folded into a block whose heading the check found true of them.
  moves: int = 0
  # Lone claims nothing could home (never a core claim — that discards).
  dropped_claims: int = 0
  # "ok" | "repaired" (one regroup) | "rejected" (twice → discarded) | "".
  check: str = ""
  # The reading order the check returned for the blocks: "same" (as grouped),
  # "changed", "kept" (no usable order came back), "refused" (the block it
  # put first shares no word with the headline — the grouped order stands).
  order: str = ""
  # Human-readable refusals: a rescue the sources did not back, a merge that
  # lost a name, a block the check faulted.
  rejected: List[str] = Field(default_factory=list)
  seconds: float = 0.0
  # Seconds per step ("group", "merge", "rescue", "check"), for the latency
  # decisions the injector makes from prod logs.
  steps: Dict[str, float] = Field(default_factory=dict)


class NewsCollectionReviewResponse(BaseModel):
  claims: List[ExtractedClaim]
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection]
  collection_order: List[str] = Field(default_factory=list)
  review: CollectionReviewReport
