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
still votes on a claim. The moments are chosen first and the claim linked after, so
a punchline or a question that no claim records can still be picked.

By default (`link_claims: "after"`, GEO-3692) the picker never sees the claims: the
reel is picked for its story, then a separate call links each clip to a claim of its
own speaker, a speaker's two clips to different claims. A clip that call cannot link
gets its speaker's unused claim with the most words in common and `claim_fit` 0. The
answer has the same shape either way. `link_claims: "in_pick"` restores the picker
that is shown the claims and links them itself, per request.

Claude offers up to three candidate reels; when more than one passes every rule, a
second call judges them on the clips' words alone, as a viewer meets them.

Every rule below is checked here before the answer is returned. A caller should
check them again against its own recordings.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin

MediaType = Literal["debate"]
Role = Literal["opening", "answer", "rebuttal", "closing"]
# How the clips are linked to claims (GEO-3692). "after": the picker never sees the claims, and a
# separate call links each picked clip. "in_pick": the picker is shown the claims and links them in
# its own answer, as before GEO-3692.
LinkClaims = Literal["after", "in_pick"]

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
    clip_max_ms: int = Field(default=22_000, ge=2_000)
    # What to aim for. The reel aims at about four times the middle of this range. 14-20 s since
    # GEO-3741: at 10-18 s Sonnet 5.5 picked near the bottom (13 Nov: clips 12.3 s, reels ~49 s,
    # against 17.8 s and ~71 s on Sonnet 4.6), under the agreed minute or so per debate.
    target_min_ms: int = Field(default=14_000, ge=1_000)
    target_max_ms: int = Field(default=20_000, ge=2_000)
    # The shortest a reel may play, its four clips as heard back to back (GEO-3716). A reel under it
    # is sent back to be picked longer (GEO-3741); one still under it when the answers run out is
    # refused, or returned with `short_reel` set, as the moments_pick_short_reel setting says.
    min_reel_ms: int = Field(default=50_000, ge=0)
    # Clips the caller refused after checking an earlier answer itself (for example, judged by
    # claims.score_clips to argue the other side). The new reel avoids them.
    avoid: List[AvoidRange] = Field(default_factory=list, max_length=20)
    link_claims: LinkClaims = Field(
        default="after",
        description='How clips are linked to claims. "after" (default): the reel is picked without '
        "the claims, then each clip is linked to a claim of its own speaker by a separate call. "
        '"in_pick": the picker is shown the claims and links them itself, as before GEO-3692. Either '
        "way every moment has a claim_index of its own speaker, distinct between one speaker's clips.",
    )

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
        if self.min_reel_ms > MOMENTS * self.target_max_ms:
            raise ValueError(f"need min_reel_ms <= {MOMENTS} * target_max_ms, the longest a reel should run")
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
    claim_index: int = Field(description="The linked claim's caller index: always one of this speaker's claims, different from their other clip's.")
    claim_fit: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the clip supports the linked claim. 0 when, under link_claims \"after\", "
        "the link call gave no usable link and the claim was chosen by words in common.",
    )
    role: Role
    answers: Optional[int] = Field(
        default=None, description="Position in `moments` of the earlier clip this one answers, or null."
    )
    why: str
    trimmed_from_sentence: Optional[int] = Field(
        default=None,
        description="The model picked this clip from an earlier sentence, over the hard limit; it "
        "was shortened from its start to fit, because no earlier end fitted cleanly. role and "
        "why (and claim and claim_fit under link_claims \"in_pick\") were given for the longer "
        "clip. Only on a reel kept as a last "
        "resort: an over-long clip is sent back to be picked again.",
    )
    trimmed_to_sentence: Optional[int] = Field(
        default=None,
        description="The model picked this clip to a later sentence, over the hard limit; it was "
        "shortened from its end to fit, keeping its setup. role and why (and claim and claim_fit "
        "under link_claims \"in_pick\") were given for the longer clip. Only on a reel kept as a "
        "last resort.",
    )
    ended_from_sentence: Optional[int] = Field(
        default=None,
        description="The sentence the picker ended this clip on, when the clip ran on into the rest of a "
        "sentence the transcript broke in two (GEO-3716, link_claims \"after\" only): the sentences "
        "after it in its turn that open mid-sentence or are marked a split half were added, within "
        "the hard limit. role and why were given for the clip as picked.",
    )


class DebatesPickMomentsResult(BaseModel):
    story: str = Field(description="One sentence: the arc the four clips tell, in play order.")
    moments: List[PickedMoment] = Field(description="Exactly four, in play order, alternating speakers.")
    provider: str
    model_used: str
    attempts: int = Field(description="Which answer the reel came from (1 for the first): the one that passed every rule, else the passing one whose best reel had the fewest soft problems, or for a last resort the one (with the fewest problems in its best reel) that broke only the first clip's opening rules (first_clip_problems) or ran a clip over the limit (cut to fit: trimmed_to_sentence, trimmed_from_sentence).")
    candidates: int = Field(default=1, description="How many reels the pick was made from: those with no soft problem when any passed, else all that passed. More than one means the judge picked.")
    judge_why: Optional[str] = Field(default=None, description="Why the judge picked this reel, when it judged.")
    reel_ms: int = Field(
        default=0,
        description="How long the reel plays: its four clips back to back, each from the first word heard "
        "once the filler geo-chat cuts is gone, as the sentences' times place it (GEO-3716).",
    )
    short_reel: bool = Field(
        default=False,
        description="True when the reel plays (reel_ms) under the input's min_reel_ms, after every send-back, "
        "and was returned rather than refused (moments_pick_short_reel \"flag\"; the default refuses, GEO-3741).",
    )
    first_clip_problems: List[str] = Field(
        default_factory=list,
        description="Empty unless no answer had a reel whose first clip keeps its rules (it starts a "
        "sentence and a thought, not on a connective or filler): then the reel was kept as a last "
        "resort, and these are the rules its first clip breaks.",
    )
