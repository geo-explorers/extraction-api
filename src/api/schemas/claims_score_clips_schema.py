"""Public contract for claims.score_clips.

A debate's candidate clips in, a judgement of each clip as a viewer hears it out.
claims.score_highlights scores the *claim*, a sentence the extractor wrote; this
scores the *clip*, the words the speaker actually says between two cut points.
The two differ exactly where moment reels go wrong (GEO-3614): a clip that reads
out the motion its speaker argues against, stops one sentence before the
punchline, opens on a "that" whose referent is outside it, or carries a misheard
word that reverses its meaning, can belong to a claim that scores well.

For each clip: how strongly it would hold a stranger as a reel's opener, judged
against the debate's other clips; whether it argues its speaker's own side as
heard; whether it makes sense alone; whether it ends after its point lands; and
which other clip, if any, it answers. What to do with these (weights, a picker)
is the caller's decision; nothing is dropped or reordered here.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.claims_score_highlights_schema import HighlightDocument
from src.api.schemas.overrides_schema import OverridesMixin

MediaType = Literal["debate"]

# Each clip is judged with the whole transcript in view, so a run costs
# clips x transcript. A reel's candidate pool is about four clips per debater.
MAX_DOCUMENTS = 200
MAX_CLIPS = 24
MAX_TOTAL_CONTENT_CHARS = 400_000


class ScoreClip(BaseModel):
    id: Optional[str] = Field(default=None, description="Caller-opaque id; echoed back.")
    # Who says it. Clips by the same speaker are never offered as each other's prompt.
    speaker: str = Field(min_length=1, max_length=200)
    # The speaker's side of the motion as the caller knows it ("Agree", "Disagree"),
    # so faithfulness is judged against the side they argue, not guessed.
    side: Optional[str] = Field(default=None, max_length=100)
    # The clip's words exactly as transcribed between its cut points.
    text: str = Field(min_length=1, max_length=4000)
    # The transcript passage the clip sits in (0-based), so the model can read
    # what comes just before and after it. Optional; without it the model
    # finds the clip in the transcript by its words.
    document_index: Optional[int] = None
    # Clip length; shown so a long, slow clip can lose to a short, sharp one.
    duration_ms: Optional[int] = Field(default=None, ge=0)


class ClaimsScoreClipsInput(OverridesMixin):
    media_type: MediaType
    title: Optional[str] = Field(default=None, max_length=1000)  # the debate motion
    context: str = Field(default="", max_length=5_000)  # e.g. who argues which side
    documents: List[HighlightDocument] = Field(min_length=1, max_length=MAX_DOCUMENTS)  # in spoken order
    clips: List[ScoreClip] = Field(min_length=1, max_length=MAX_CLIPS)

    @model_validator(mode="after")
    def _validate_caps_and_indices(self) -> "ClaimsScoreClipsInput":
        total = sum(len(d.content) for d in self.documents)
        if total > MAX_TOTAL_CONTENT_CHARS:
            raise ValueError(
                f"Total document content is {total} chars; the maximum is {MAX_TOTAL_CONTENT_CHARS}"
            )
        for position, clip in enumerate(self.clips):
            if clip.document_index is not None and not 0 <= clip.document_index < len(self.documents):
                raise ValueError(
                    f"clips[{position}].document_index {clip.document_index} is out of range "
                    f"for {len(self.documents)} documents"
                )
        return self


class ScoredClip(BaseModel):
    index: int = Field(description="Position in the input `clips` list.")
    id: Optional[str]
    hook: float = Field(
        ge=0.0,
        le=1.0,
        description="How strongly the clip would hold a stranger as the reel's opener, comparable "
        "within this debate only: the clip's percentile, among this run's clips, on its mean "
        "probability over the shuffled which-clip-opens questions.",
    )
    decisions: Dict[str, float] = Field(
        description="Every decision asked, 0-1. `own_side`, `standalone` and `complete` are "
        "probabilities of yes. `opener` is the clip's mean probability over the shuffled "
        "which-clip-opens questions (it sums to one over the run's clips; read it through "
        "`hook`). `answers_elsewhere` is the probability that the clip replies to a point that "
        "none of the other clips contains.",
    )
    responds_to: Optional[int] = Field(
        description="The input index of the other speaker's clip this one replies to, when that "
        "is the most probable answer; null when it stands on its own or replies to something no "
        "other clip contains (see `answers_elsewhere`).",
    )


class ClaimsScoreClipsResult(BaseModel):
    clips: List[ScoredClip] = Field(description="Every input clip with its judgement, in input order.")
    # Always the number of clips; kept so a consumer can check the result is whole.
    clips_scored: int
    provider: str
    model_used: str
