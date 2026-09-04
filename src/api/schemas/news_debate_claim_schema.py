"""Internal schema for the news debate pass.

The candidate carries only what the product definition needs: the card itself,
the neutral question behind it (drives mirrored/duplicate collapse), the two
real sides (proves a debate exists), and which sources raise it (feeds the
public ``ExtractedDebateClaim.source_indices`` attribution). The class keeps
its historical name so the service, task, and test layers stay stable.
"""

from pydantic import BaseModel, Field


class GroundedDebateCandidate(BaseModel):
  # Keep the reasoning fields before ``text``. Gemini structured output follows
  # schema order, so the model settles the question and sides before writing
  # the polished card.
  neutral_question: str = Field(
    default="",
    description="Neutral question used to detect mirrored or duplicate motions",
  )
  opposing_positions: list[str] = Field(
    default_factory=list,
    description="Exactly two short, genuinely held opposing positions",
  )
  source_indices: list[int] = Field(
    default_factory=list,
    description="Indices of the sources this debate arises from",
  )
  text: str = Field(
    default="",
    description="Final concise headline-style proposition, no more than 20 words",
  )


class GroundedDebateResponse(BaseModel):
  debate_claims: list[GroundedDebateCandidate] = Field(default_factory=list)
