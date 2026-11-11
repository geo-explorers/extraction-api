"""One Claude conversation that must end in a JSON answer the caller accepts.

Shared by transcript.punctuate and debates.pick_moments (GEO-3638), which ask
Claude to write an answer rather than choose among options. The caller passes a
`check` that parses the answer and returns its problems, hard and soft:

- an answer with hard problems is refused;
- an answer with only soft problems is acceptable but worth improving.

A refused or improvable answer is shown its problems and asked again in the same
conversation, up to `attempts` answers in all. The best answer seen wins: the
first with no problems at all, otherwise the latest with only soft ones. Only
when no answer passed the hard rules does the run fail.

Failures are final (`FinalDecisionError`, which Hatchet does not retry) when
asking again cannot help: no acceptable answer, an answer cut off at the token
limit, no API key, the spend guard. A failed request (network, rate limit,
overload) stays retryable, so the engine runs the task again.

The model and its knobs are the `moments_claude_*` settings, overridable per run.
The system prompt and the first message are marked for prompt caching, so a send
back or an engine retry within minutes reads the transcript from cache.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, TypeVar

from hatchet_sdk.exceptions import NonRetryableException

from src.config.overrides import llm
from src.config.settings import settings
from src.decisions.base import DecisionError
from src.infrastructure.claude import claude_text, parse_json
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import SpendLimitExceeded, spend_guard

logger = get_logger(__name__)

# What results name as their provider.
PROVIDER = "anthropic"
# The spend guard's key for Claude: the news Claude paths count under it too, so one hourly
# budget covers every Claude call this service makes.
SPEND_KEY = "claude"
# One request's ceiling. The client does not retry on its own (max_retries=0): the engine retries
# the run, so a task's execution timeout can be sized as attempts x this.
REQUEST_TIMEOUT_S = 180

T = TypeVar("T")

# Parses Claude's answer: (value, hard problems, soft problems).
Check = Callable[[Any], Tuple[Optional[T], List[str], List[str]]]


class FinalDecisionError(DecisionError, NonRetryableException):
    """No answer can come of running the task again: Hatchet fails the run without a retry."""

    def __init__(self, message: str):
        super().__init__(message, retryable=False)


def execution_timeout_s(attempts: int) -> int:
    """A task's execution timeout for `attempts` answers: every request at its ceiling, plus a
    minute for the rest of the run."""
    return attempts * REQUEST_TIMEOUT_S + 60


def _client():
    import anthropic

    if not settings.anthropic_api_key:
        raise FinalDecisionError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0
    )


def _retry_message(problems: List[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your answer breaks these rules:\n"
        f"{listed}\n\n"
        "Answer again with the whole corrected JSON, and nothing else."
    )


def _cached(text: str) -> List[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


async def ask_json(
    system: str,
    user: str,
    check: Check[T],
    *,
    task: str,
    attempts: int,
    client: Any = None,
) -> Tuple[T, str, int]:
    """Ask until an answer has no problems, or `attempts` run out. Returns (value, model, answers
    asked for); the value is the best answer seen (see the module doc)."""
    model = llm.get("moments_claude_model")
    client = client or _client()
    messages: List[dict] = [{"role": "user", "content": _cached(user)}]
    best: Optional[Tuple[T, int]] = None
    problems: List[str] = []
    for attempt in range(1, attempts + 1):
        try:
            spend_guard.check_and_record(SPEND_KEY)
        except SpendLimitExceeded as e:
            raise FinalDecisionError(f"{task}: spend guard tripped: {e}") from e
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=llm.get("moments_claude_max_tokens"),
                temperature=llm.get("moments_claude_temperature"),
                system=_cached(system),
                messages=messages,
            )
        except Exception as e:  # network, rate limit, overload: the engine retries the run
            raise DecisionError(f"{task}: Claude request failed: {e}", retryable=True) from e
        raw = claude_text(message)
        if getattr(message, "stop_reason", None) == "max_tokens":
            # Every part of the answer is required, so asking again gives one as long.
            if best is not None:
                break
            raise FinalDecisionError(f"{task}: the answer was cut off at the token limit")
        try:
            value, hard, soft = check(parse_json(raw))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            # Not JSON (JSONDecodeError is a ValueError), or JSON a check could not read: either
            # way the answer is refused and sent back, never a crash Hatchet would rerun.
            value, hard, soft = None, [f"The answer is not valid JSON in the asked shape ({e})."], []
        if not hard:
            best = (value, attempt)
            if not soft:
                return value, model, attempt
        problems = hard + soft
        if attempt < attempts:
            logger.warning(f"{task}: answer {attempt}/{attempts} sent back: {problems}")
            messages += [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": _retry_message(problems)},
            ]
    if best is not None:
        return best[0], model, best[1]
    raise FinalDecisionError(f"{task}: no acceptable answer after {attempts} attempts: {problems}")


__all__ = [
    "PROVIDER",
    "SPEND_KEY",
    "FinalDecisionError",
    "ask_json",
    "execution_timeout_s",
    "parse_json",
]
