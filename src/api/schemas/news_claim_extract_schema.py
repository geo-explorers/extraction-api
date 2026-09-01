from pydantic import BaseModel, Field, model_validator
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
  importance: Optional[float] = Field(
    ge=0.0, le=1.0, default=None,
    description="How central this claim is to the story (1.0 = the core event itself). Graded relative to THIS story's claim set; consumers rank/cap claims by it."
  )


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


class ExtractedDebateClaim(BaseModel):
  """A composed debatable proposition (Step 8 of the prompt): headline-like,
  taking a definite side of a dispute that significant groups genuinely
  argue — factual, causal, predictive, evaluative, or prescriptive.
  Deliberately outside the claims array so factual scoring, dedup and caps
  never touch it — the consumer publishes these under their own collection."""
  text: str
  source_indices: List[int] = Field(
    default_factory=list,
    description="Sources showing the contested question this proposition answers"
  )
  # Not requested from the model — a self-graded score on a composed
  # proposition means nothing. The default exists so consumers whose claim
  # shape requires a confidence can map these uniformly.
  confidence: float = Field(ge=0.0, le=1.0, default=0.8)


def normalize_debate_claims(
  items: List[ExtractedDebateClaim],
) -> List[ExtractedDebateClaim]:
  """Deterministic enforcement of the Step-8 contract: 0 or 2-4 claims.

  The prompt asks for this, but a count is exactly the kind of instruction a
  model occasionally ignores — and rejecting the response would re-run the
  whole billed extraction over a count violation (validation happens inside
  the service's retry loop). So the contract is repaired, never refused:
  duplicates collapse first (so a pair of identical texts becomes zero, not
  a surviving "pair"), then the strongest four survive (the model lists in
  order of strength), and a lone remainder becomes none at all.
  """
  seen: set[str] = set()
  unique: List[ExtractedDebateClaim] = []
  for c in items:
    key = " ".join((c.text or "").split()).casefold()
    if key and key not in seen:
      seen.add(key)
      unique.append(c)
  unique = unique[:4]
  return [] if len(unique) == 1 else unique


class NewsClaimExtractResponse(BaseModel):
  claims: List[ExtractedClaim] = Field(default_factory=list)
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection] = Field(default_factory=list)
  collection_order: List[str] = Field(default_factory=list)
  debate_claims: List[ExtractedDebateClaim] = Field(default_factory=list)
  summary: str = ""

  @model_validator(mode="after")
  def _enforce_debate_claim_contract(self) -> "NewsClaimExtractResponse":
    self.debate_claims = normalize_debate_claims(self.debate_claims)
    return self
