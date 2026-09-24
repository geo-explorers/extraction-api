from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin
from typing import List, Literal, Optional


class NewsArticleSource(BaseModel):
  index: int
  url: str
  title: str
  publisher: Optional[str] = None
  published_at: Optional[str] = None
  content: str


class NewsClaimExtractRequest(OverridesMixin):
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


# The Debate collection contract: 0 cards, or DEBATE_MIN..DEBATE_MAX of them.
# Two is a publishable collection (Armando 2026-09-07); four is the editors'
# number (the master debate prompt returns four) and the point past which
# the checker's weaker passes were showing up (2026-09-24).
DEBATE_MIN = 2
DEBATE_MAX = 4


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
  # The semantic review's strength grade (0-1: how strongly the card meets
  # the definition for this story), not the writer's self-assessment. It
  # orders the published set and is what consumers store as the claim's
  # confidence. The default covers a claim that reached this shape ungraded.
  confidence: float = Field(ge=0.0, le=1.0, default=0.8)


def normalize_debate_claims(
  items: List[ExtractedDebateClaim],
) -> List[ExtractedDebateClaim]:
  """Deterministic enforcement of the Debate collection contract: 0 or 2-4.

  Duplicates collapse first; the survivors are ordered by confidence (the
  review's grade), highest first, with the producer's own order breaking
  ties, and the strongest DEBATE_MAX are kept. An underfilled result becomes
  empty so the consumer omits the Debate collection rather than padding it
  with weak or mirrored claims.
  """
  seen: set[str] = set()
  unique: List[ExtractedDebateClaim] = []
  for c in items:
    key = " ".join((c.text or "").split()).casefold()
    if key and key not in seen:
      seen.add(key)
      unique.append(c)
  unique = sorted(unique, key=lambda c: -c.confidence)[:DEBATE_MAX]
  return unique if len(unique) >= DEBATE_MIN else []


class AnchorReport(BaseModel):
  """What the source-anchor guard did to a claim set: every date, year and
  number in every claim must be one a source states or one code resolved
  from a source's weekday against its publication date. A claim the model
  could not repair in one round is dropped rather than published with a
  guessed date; its text is kept here so the drop can be audited."""
  checked: int = 0
  flagged: int = 0
  repaired: int = 0
  dropped: int = 0
  dropped_claims: List[str] = Field(default_factory=list)
  problems: List[str] = Field(default_factory=list)


class NewsClaimExtractResponse(BaseModel):
  claims: List[ExtractedClaim] = Field(default_factory=list)
  quotes: List[ExtractedQuote] = Field(default_factory=list)
  collections: List[ExtractedCollection] = Field(default_factory=list)
  collection_order: List[str] = Field(default_factory=list)
  debate_claims: List[ExtractedDebateClaim] = Field(default_factory=list)
  summary: str = ""
  anchor_report: Optional[AnchorReport] = None

  @model_validator(mode="after")
  def _enforce_debate_claim_contract(self) -> "NewsClaimExtractResponse":
    self.debate_claims = normalize_debate_claims(self.debate_claims)
    return self
