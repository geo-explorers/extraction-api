"""Internal structured output for reject-only debate semantic review.

Four gates, mirroring the product definition of a debate claim: a real
societal debate, raised by this story, on its own axis, asserting no facts
the story does not contain. Application code computes acceptance from the
booleans; the analysis strings are short audit explanations retained in the
Hatchet checkpoint for observability and for the rescue pass's audit trail.
"""

from pydantic import BaseModel, Field


class DebateSemanticVerdict(BaseModel):
  # Analysis precedes each decision because Gemini structured output follows
  # schema order.
  candidate_index: int = Field(description="0-based index into the candidate list")

  debate_analysis: str = Field(default="")
  real_societal_debate: bool = Field(
    default=False,
    description="Clear, large or significant groups genuinely debate this question, in society or online",
  )

  story_analysis: str = Field(default="")
  raised_by_story: bool = Field(
    default=False,
    description="A reader of this story would recognize the debate as raised by it",
  )

  invented_facts_analysis: str = Field(default="")
  invented_facts: list[str] = Field(
    default_factory=list,
    description="Factual assertions in the proposition that the supplied material does not contain",
  )
  no_invented_facts: bool = Field(default=False)

  distinctness_analysis: str = Field(default="")
  distinct_axis: bool = Field(default=False)
  duplicate_of: int | None = Field(
    default=None,
    description="Earlier candidate index sharing the same neutral debate axis",
  )

  # Graded after the four gates (schema order): how strongly the card meets
  # the definition for this story. Orders the published set; the public
  # claim's confidence. Ignored for a rejected card.
  strength: float = Field(
    default=0.0,
    ge=0.0,
    le=1.0,
    description="0.0-1.0: on the story's central disagreement, a real near-even split, wording both sides accept, clear in five seconds",
  )

  failure_codes: list[str] = Field(default_factory=list)


class DebateSemanticReviewResponse(BaseModel):
  verdicts: list[DebateSemanticVerdict] = Field(default_factory=list)
