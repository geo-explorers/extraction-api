"""Model-agnostic decision interface.

A decision model answers named questions about a piece of content with
probabilities instead of text: a yes/no question gets the probability of yes,
a choice question gets a probability per option, a scale question gets a
position on an ordered rubric. Tasks build questions from the types here and
read typed answers back; which provider computes them is a deployment
setting, so the model behind a task can be swapped without touching the task.

To add a provider, implement `DecisionModel` and register it in
src/decisions/__init__.py. A provider that has no native probabilities (a chat
model asked for structured output, say) still fits: it only has to return a
number per question in the shapes below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Union


@dataclass(frozen=True)
class YesNo:
    """A yes/no question. `yes` and `no` optionally say what counts as each;
    they are given together or not at all."""

    instructions: str
    yes: Optional[str] = None
    no: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.yes is None) != (self.no is None):
            raise ValueError("YesNo criteria come as a pair: give both `yes` and `no`, or neither")


@dataclass(frozen=True)
class Choice:
    """Pick one of `options`: name -> when it applies (None lets the name
    speak for itself)."""

    instructions: str
    options: Mapping[str, Optional[str]]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("Choice needs at least one option")


@dataclass(frozen=True)
class Scale:
    """Rate on an ordered rubric: `levels` describe the rungs from the lowest
    to the highest, two or more. Only the descriptions reach the model — a
    rung has a position, not a name — and the answer is that position,
    probability-weighted, as a number from 0 (surely the lowest) to 1."""

    instructions: str
    levels: Sequence[str]

    def __post_init__(self) -> None:
        # A bare string is a Sequence too, and would be sent as one level per character.
        if isinstance(self.levels, str) or len(self.levels) < 2:
            raise ValueError("Scale needs at least two levels, as a list of descriptions")


Question = Union[YesNo, Choice, Scale]


@dataclass(frozen=True)
class YesNoAnswer:
    probability: float  # of yes, 0..1


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str  # the most probable option
    probabilities: Dict[str, float]  # one per option, summing to about 1


@dataclass(frozen=True)
class ScaleAnswer:
    score: float  # 0..1: the probability-weighted position on the scale
    probabilities: List[float]  # one per level, lowest first, summing to about 1


Answer = Union[YesNoAnswer, ChoiceAnswer, ScaleAnswer]


class DecisionError(Exception):
    """The provider could not answer: a failed request (after its own retries)
    or a response that does not answer every question in the expected shape.

    `retryable` says whether asking again later could succeed — a rate limit
    or an outage — as opposed to a request the provider will keep refusing
    (bad credentials, an unknown model, a malformed answer). A task uses it to
    stop a run at the first unretryable failure instead of repeating it for
    every claim."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class DecisionModel(Protocol):
    # Names the provider for the spend guard and logs, e.g. "perplexity".
    provider: str

    async def decide(
        self,
        content: Mapping[str, str],
        questions: Mapping[str, Question],
        *,
        model: str,
    ) -> Dict[str, Answer]:
        """Answer every question about `content` (named sections, in reading
        order). Returns one answer per question name, typed by its question
        (YesNo -> YesNoAnswer, Choice -> ChoiceAnswer with a probability for
        every option, Scale -> ScaleAnswer with a probability for every
        level). Raises DecisionError rather than return a partial or
        malformed set."""
        ...
