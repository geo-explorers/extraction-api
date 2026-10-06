"""Model-agnostic decision interface.

A decision model answers named questions about a piece of content with
probabilities instead of text: a yes/no question gets the probability of yes,
a choice question gets a probability per option. Tasks build questions from
the types here and read typed answers back; which provider computes them is a
deployment setting, so the model behind a task can be swapped without touching
the task.

To add a provider, implement `DecisionModel` and register it in
src/decisions/__init__.py. A provider that has no native probabilities (a chat
model asked for structured output, say) still fits: it only has to return a
number per question in the shapes below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Union


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


Question = Union[YesNo, Choice]


@dataclass(frozen=True)
class YesNoAnswer:
    probability: float  # of yes, 0..1


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str  # the most probable option
    probabilities: Dict[str, float]  # one per option, summing to about 1


Answer = Union[YesNoAnswer, ChoiceAnswer]


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
        (YesNo -> YesNoAnswer, Choice -> ChoiceAnswer, with a probability for
        every option). Raises DecisionError rather than return a partial or
        malformed set."""
        ...
