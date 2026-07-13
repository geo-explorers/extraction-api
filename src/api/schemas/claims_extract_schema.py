"""Contracts for the generalized claim extraction DAG (claims.extract).

One media-neutral input for any text-based source — debate transcripts,
research papers, news articles, podcast transcripts, talks, generic documents.
Documents keep stable 0-based indices so a future chunking layer can split
content internally and still map provenance back to `document_indices`
without a contract change.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Literal, Optional

MediaType = Literal["debate", "news", "podcast", "research_paper", "talk", "generic"]

# "plain" covers any text, including transcripts with inline "Name: ..." speaker
# labels. The vendor formats are parsed with the existing TranscriptParser and
# rendered as "speaker: text" lines so quote attribution survives.
DocumentFormat = Literal["plain", "podscribe", "bankless", "assembly"]

# ~1M tokens of input; chunking is a future layer, so oversized corpora are
# rejected at enqueue rather than silently truncated.
MAX_TOTAL_CONTENT_CHARS = 4_000_000
CUSTOM_INSTRUCTIONS_MAX_CHARS = 2_000
FOCUS_TOPIC_MAX_CHARS = 100


class InputDocument(BaseModel):
  id: Optional[str] = None  # caller-opaque; echoed back in document_ids
  title: Optional[str] = None
  content: str = Field(min_length=1)
  format: DocumentFormat = "plain"
  publisher: Optional[str] = None
  published_at: Optional[str] = None  # ISO date; used to resolve relative dates
  url: Optional[str] = None
  metadata: Dict[str, str] = Field(default_factory=dict)


class ClaimsExtractInput(BaseModel):
  media_type: MediaType  # unknown values are rejected at enqueue (422)
  documents: List[InputDocument] = Field(min_length=1, max_length=50)
  title: Optional[str] = None  # debate motion / episode title / headline
  context: str = Field(default="", max_length=5_000)
  grouping: bool = True  # false skips the topics pass entirely (flat claims)
  include_quotes: bool = False
  include_summary: bool = False
  include_takeaways: bool = False
  focus_topics: List[str] = Field(default_factory=list, max_length=20)
  max_claims: Optional[int] = Field(default=None, ge=1, le=500)
  min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
  language: str = Field(default="en", max_length=32)
  custom_instructions: str = Field(
    default="", max_length=CUSTOM_INSTRUCTIONS_MAX_CHARS
  )

  @model_validator(mode="after")
  def _validate_caps(self) -> "ClaimsExtractInput":
    total = sum(len(d.content) for d in self.documents)
    if total > MAX_TOTAL_CONTENT_CHARS:
      raise ValueError(
        f"Total document content is {total} chars; the maximum is "
        f"{MAX_TOTAL_CONTENT_CHARS} (chunking is not supported yet)"
      )
    for topic in self.focus_topics:
      if len(topic) > FOCUS_TOPIC_MAX_CHARS:
        raise ValueError(
          f"focus_topics entries must be <= {FOCUS_TOPIC_MAX_CHARS} chars"
        )
    return self


# ── Response types ─────────────────────────────────────────────────────


class ExtractedClaimOut(BaseModel):
  text: str
  topic: Optional[str] = None  # None when grouping=false
  document_indices: List[int] = Field(default_factory=list)
  confidence: float = Field(ge=0.0, le=1.0, default=0.8)


class ExtractedQuoteOut(BaseModel):
  text: str
  speaker: Optional[str] = None
  claim_index: int = Field(
    ge=0,
    description="0-based index into the final 'claims' array"
  )
  document_index: Optional[int] = None


class ClaimGroup(BaseModel):
  name: str
  summary: str = ""
  claim_indices: List[int] = Field(default_factory=list)


class TakeawayOut(BaseModel):
  text: str
  claim_index: Optional[int] = None  # exact-text link; None if unmatched


class ClaimsExtractResult(BaseModel):
  media_type: str
  grouping: bool
  topics: List[str] = Field(default_factory=list)  # [] when grouping=false
  claims: List[ExtractedClaimOut] = Field(default_factory=list)  # always flat
  groups: List[ClaimGroup] = Field(default_factory=list)  # only when grouping
  quotes: List[ExtractedQuoteOut] = Field(default_factory=list)
  takeaways: List[TakeawayOut] = Field(default_factory=list)
  summary: str = ""
  document_ids: List[Optional[str]] = Field(default_factory=list)
  claims_extracted: int = 0
  model_used: str = ""
