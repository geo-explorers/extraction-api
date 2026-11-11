"""Public contract for transcript.punctuate (GEO-3638).

An automatic transcript in, where each sentence ends out. Whisper loses its
punctuation partway through some debates: once one window comes back as a
run-on, the next is prompted with it and follows suit. Moment cuts, subtitles
and claim extraction all need sentence ends.

The answer is word positions, never text: the words cannot change, a long
transcript cannot be cut off at the output limit, and checking the answer is a
bounds check. Each turn's last word always ends a sentence.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from src.api.schemas.overrides_schema import OverridesMixin

MediaType = Literal["debate"]
Mark = Literal[".", "?", "!"]

MAX_TURNS = 60
MAX_WORDS_PER_TURN = 4_000
MAX_TOTAL_WORDS = 15_000


class PunctuateTurn(BaseModel):
    speaker: str = Field(min_length=1, max_length=200)
    # The turn's words as transcribed, in order. Punctuation already on a word is
    # shown to the model as a hint, not trusted.
    words: List[str] = Field(min_length=1, max_length=MAX_WORDS_PER_TURN)


class TranscriptPunctuateInput(OverridesMixin):
    media_type: MediaType
    title: Optional[str] = Field(default=None, max_length=1000)  # the debate motion
    turns: List[PunctuateTurn] = Field(min_length=1, max_length=MAX_TURNS)

    @model_validator(mode="after")
    def _cap_total(self) -> "TranscriptPunctuateInput":
        total = sum(len(t.words) for t in self.turns)
        if total > MAX_TOTAL_WORDS:
            raise ValueError(f"{total} words; the maximum is {MAX_TOTAL_WORDS}")
        return self


class SentenceEnd(BaseModel):
    word: int = Field(ge=0, description="Position in the turn's `words` of the sentence's last word.")
    mark: Mark


class PunctuatedTurn(BaseModel):
    ends: List[SentenceEnd] = Field(
        description="Every sentence end in the turn, in order. The last is always the turn's last word."
    )


class TranscriptPunctuateResult(BaseModel):
    turns: List[PunctuatedTurn] = Field(description="One per input turn, in input order.")
    sentences: int
    provider: str
    model_used: str
