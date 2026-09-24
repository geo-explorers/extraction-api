"""Internal schema for the news debate pass.

The writer's draft carries only what the product definition needs: the card
itself, the neutral question behind it (drives mirrored/duplicate collapse),
the two real sides (proves a debate exists), and which sources raise it
(feeds the public ``ExtractedDebateClaim.source_indices`` attribution). The
candidate the service and task layers pass around is that draft plus the
review's strength grade, which the writer must never be asked for.
"""

from typing import Iterable

from pydantic import BaseModel, Field


class DebateCandidateDraft(BaseModel):
  """What the writer returns per card. This is the writer's response schema,
  so it carries nothing the writer should fill in about its own quality."""

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


class GroundedDebateCandidate(DebateCandidateDraft):
  """A draft past the deterministic filter, carrying the review's grade.

  Both fields are set by the semantic review. ``on_headline`` says the card's
  disagreement is the one the headline reports; ``strength`` is the review's
  0-1 grade folded with it by ``graded_strength``. The grade orders the
  published set and becomes the public claim's confidence. The class keeps
  its historical name so the service, task, and test layers stay stable.
  """

  on_headline: bool = Field(default=False)
  strength: float = Field(default=0.0, ge=0.0, le=1.0)


class GroundedDebateResponse(BaseModel):
  debate_claims: list[DebateCandidateDraft] = Field(default_factory=list)


def graded_strength(strength: float, on_headline: bool) -> float:
  """The review's grade as one number: the upper half of the scale for a card
  on the headline's disagreement, the lower half for a card off it.

  Half the grade is where the card sits, half is how strong the review found
  it, so every on-headline card outranks every off-headline card and the
  review's grade orders cards within each half. Armando's approval story is
  the case: the war card graded 0.95 and the approval card 0.4, so the set
  led with the war. Folding keeps that order inside the public confidence,
  which is what every later cap (completion, projection, the consumer's
  clamp) sorts by."""
  return round(0.5 * strength + (0.5 if on_headline else 0.0), 3)


def strongest_first(
  candidates: Iterable[GroundedDebateCandidate], limit: int
) -> list[GroundedDebateCandidate]:
  """The ``limit`` highest-graded candidates, the producer's order breaking
  ties (a stable sort), so ungraded lists keep their strongest-first order."""
  return sorted(candidates, key=lambda c: -c.strength)[:limit]
