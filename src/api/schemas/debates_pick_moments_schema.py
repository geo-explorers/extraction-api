"""Public contract for debates.pick_moments (GEO-3638).

A debate's punctuated transcript and its claims in, the four clips of its moment
reel out. A reel plays four clips back to back, alternating speakers, telling
the debate's story in about a minute to a viewer who never saw it.

The reel used to be stitched together from clips cut around extracted claims,
which were extracted to fill the knowledge graph. Several claims shared one
stretch of speech, one debater often had a single usable clip, and the picker
kept the order said over alternation. Here the reel is chosen as a whole, from
the transcript, for the story. Each clip is a run of whole sentences from one
turn, and is linked to the claim of its speaker it best supports, so a viewer
still votes on a claim.

Every rule below is checked here before the answer is returned. A caller should
check them again against its own recordings.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin

MediaType = Literal["debate"]
Role = Literal["opening", "answer", "rebuttal", "closing"]

MOMENTS = 4
MAX_SENTENCES = 1_500
MAX_CLAIMS = 80
MAX_TOTAL_CHARS = 200_000


class MomentSpeaker(BaseModel):
    slot: int
    name: str = Field(min_length=1, max_length=200)
    # The side of the motion the speaker argues, as the caller knows it ("Agree", "Disagree").
    side: Optional[str] = Field(default=None, max_length=100)


class MomentSentence(BaseModel):
    speaker: int = Field(description="The speaker's slot.")
    turn: int = Field(ge=0, description="The turn it was said in. A clip never spans two turns.")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=4000)


class MomentClaim(BaseModel):
    index: int = Field(description="Caller's claim index; echoed back as `claim_index`.")
    speaker: int
    text: str = Field(min_length=1, max_length=2000)


class AvoidRange(BaseModel):
    """A stretch an earlier reel used that the caller refused, and why. No clip may overlap it."""

    start_sentence: int = Field(ge=0)
    end_sentence: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=300)


class DebatesPickMomentsInput(OverridesMixin):
    media_type: MediaType
    title: str = Field(min_length=1, max_length=1000)  # the motion
    speakers: List[MomentSpeaker] = Field(min_length=2, max_length=2)
    sentences: List[MomentSentence] = Field(min_length=4, max_length=MAX_SENTENCES)  # in spoken order
    claims: List[MomentClaim] = Field(min_length=2, max_length=MAX_CLAIMS)
    # A clip's length bounds, first sentence's start to last sentence's end.
    clip_min_ms: int = Field(default=6_000, ge=1_000)
    clip_max_ms: int = Field(default=30_000, ge=2_000)
    # What to aim for. The reel aims at about four times the middle of this range.
    target_min_ms: int = Field(default=10_000, ge=1_000)
    target_max_ms: int = Field(default=18_000, ge=2_000)
    # Clips the caller refused after checking an earlier answer itself (for example, judged by
    # claims.score_clips to argue the other side). The new reel avoids them.
    avoid: List[AvoidRange] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _validate(self) -> "DebatesPickMomentsInput":
        slots = {s.slot for s in self.speakers}
        if len(slots) != 2:
            raise ValueError("speakers need two different slots")
        if any(s.speaker not in slots for s in self.sentences) or any(c.speaker not in slots for c in self.claims):
            raise ValueError("every sentence and claim must belong to one of the two speakers")
        if len({c.index for c in self.claims}) != len(self.claims):
            raise ValueError("claim indices must be unique")
        if sum(len(s.text) for s in self.sentences) > MAX_TOTAL_CHARS:
            raise ValueError(f"the transcript is over {MAX_TOTAL_CHARS} chars")
        if not (self.clip_min_ms <= self.target_min_ms <= self.target_max_ms <= self.clip_max_ms):
            raise ValueError("need clip_min_ms <= target_min_ms <= target_max_ms <= clip_max_ms")
        if any(not 0 <= a.start_sentence <= a.end_sentence < len(self.sentences) for a in self.avoid):
            raise ValueError("an avoid range is outside the sentences")
        if any(s.end_ms < s.start_ms for s in self.sentences):
            raise ValueError("a sentence ends before it starts")
        return self


class PickedMoment(BaseModel):
    speaker: int
    start_sentence: int = Field(description="Position in the input `sentences` of the clip's first sentence.")
    end_sentence: int = Field(description="Position of its last sentence, inclusive.")
    start_ms: int
    end_ms: int
    claim_index: int = Field(description="The linked claim's caller index.")
    claim_fit: float = Field(ge=0.0, le=1.0, description="How well the clip supports the linked claim.")
    role: Role
    answers: Optional[int] = Field(
        default=None, description="Position in `moments` of the earlier clip this one answers, or null."
    )
    why: str


class DebatesPickMomentsResult(BaseModel):
    story: str = Field(description="One sentence: the arc the four clips tell, in play order.")
    moments: List[PickedMoment] = Field(description="Exactly four, in play order, alternating speakers.")
    provider: str
    model_used: str
    attempts: int = Field(description="How many answers were asked for before one passed every rule.")
