"""Public contract for claims.score_highlights.

Extracted claims and the discussion they were drawn from in; a highlight score
per claim out. The score is the probability that the list of claims would
misrepresent the discussion without this claim — that it carries a central
point, decisive evidence or a key step that no other claim does — so a consumer
can show the few claims that matter instead of all of them.

Scoring is deliberately separate from claims.extract: a claim's role can only
be judged with the whole discussion in view, and claims can be re-scored
without being re-extracted. The task never drops or reorders claims; what to
do with a score (a threshold, the top N) is the consumer's decision.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin

# The rubric is written per media type (prompt keys
# claims_score_highlights.<media_type>.*); unknown values are rejected at
# enqueue (422).
MediaType = Literal["debate"]

# Every claim is scored against the whole content, so a run costs
# claims x content. These caps keep one request far under the provider's
# input limit and one run's total bounded.
MAX_DOCUMENTS = 200
MAX_CLAIMS = 100
MAX_TOTAL_CONTENT_CHARS = 400_000


class HighlightDocument(BaseModel):
    id: Optional[str] = None  # caller-opaque
    # Who said it. Shown to the model as the passage's label, so role and
    # attribution can be judged; a passage without one reads "Unknown speaker".
    speaker: Optional[str] = Field(default=None, max_length=200)
    content: str = Field(min_length=1)


class HighlightClaim(BaseModel):
    id: Optional[str] = Field(default=None, description="Caller-opaque id; echoed back.")
    text: str = Field(min_length=1, max_length=2000)
    # The documents the claim was drawn from (0-based, as claims.extract
    # returns them). They attribute the claim to its speaker.
    document_indices: List[int] = Field(default_factory=list)


class ClaimsScoreHighlightsInput(OverridesMixin):
    media_type: MediaType
    title: Optional[str] = Field(default=None, max_length=1000)  # the debate motion
    context: str = Field(default="", max_length=5_000)  # e.g. who argues which side
    documents: List[HighlightDocument] = Field(min_length=1, max_length=MAX_DOCUMENTS)  # in spoken order
    claims: List[HighlightClaim] = Field(min_length=1, max_length=MAX_CLAIMS)

    @model_validator(mode="after")
    def _validate_caps_and_indices(self) -> "ClaimsScoreHighlightsInput":
        total = sum(len(d.content) for d in self.documents)
        if total > MAX_TOTAL_CONTENT_CHARS:
            raise ValueError(
                f"Total document content is {total} chars; the maximum is {MAX_TOTAL_CONTENT_CHARS}"
            )
        for position, claim in enumerate(self.claims):
            bad = [i for i in claim.document_indices if not 0 <= i < len(self.documents)]
            if bad:
                raise ValueError(
                    f"claims[{position}].document_indices {bad} are out of range "
                    f"for {len(self.documents)} documents"
                )
        return self


class ScoredClaim(BaseModel):
    index: int = Field(description="Position in the input `claims` list.")
    id: Optional[str]
    text: str
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Highlight score: the probability that the list would misrepresent the "
        "discussion without this claim; null when it could not be scored (see `error`).",
    )
    decisions: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability of yes per decision asked. One today, `essential`, which is the score.",
    )
    error: Optional[str] = None


class ClaimsScoreHighlightsResult(BaseModel):
    claims: List[ScoredClaim] = Field(description="Every input claim with its score, in input order.")
    claims_scored: int
    provider: str
    model_used: str
