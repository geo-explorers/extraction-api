"""One Claude conversation that must end in a JSON answer the caller accepts.

Shared by transcript.punctuate and debates.pick_moments (GEO-3638), which ask
Claude to write an answer rather than choose among options. The caller passes a
`check` that parses the answer and returns its problems, hard and soft:

- an answer with hard problems is refused;
- an answer with only soft problems is acceptable but worth improving.

A refused or improvable answer is shown its problems and asked again in the same
conversation, up to `attempts` answers in all. The best answer seen wins: the
first with no problems at all, otherwise the latest with only soft ones. A check
may also return a value with hard problems: that answer is refused and sent back
like any other, but kept as a last resort, returned only when no answer passed
the hard rules. Only when none passed and there is no last resort does the run
fail. A caller that passes `rank` keeps, among answers with only soft problems
(and among last resorts), the one it ranks best, the latest among equals: a
send-back that comes back worse then does not replace the better answer.

A caller that passes a JSON `schema` gets Anthropic structured outputs
(`output_config.format`): the reply is constrained to JSON in that shape, so the
model cannot spend its tokens on prose around the answer (debates.pick_moments
hit the token limit on 8 of 20 runs on 11 Nov writing it).

An answer cut off at the token limit is logged with its tail. With
`cut_off_retries`, it is sent back that many times, told to answer shorter with
the JSON alone. An acceptable answer seen before the cut-off is kept at once;
a last resort only once no retry is left; with neither the run fails.

Failures are final (`FinalDecisionError`, which Hatchet does not retry) when
asking again cannot help: no acceptable answer, an answer cut off at the token
limit, no API key, the spend guard, a request Claude rejects (a 4xx such as a
schema it cannot compile). A failed request (network, rate limit, overload)
stays retryable, so the engine runs the task again. Once an answer is held
(acceptable, or a last resort), a later spend-guard trip or failed request ends
the conversation and returns that answer instead: a rerun would throw it away.

A caller that passes a `deadline` (deadline.Deadline, GEO-3697) gets an answer by
then or a final failure: no answer is asked for once too little time is left
(deadline.MIN_REQUEST_S), so the run keeps what it holds, and each request's
timeout is capped at the time left. With nothing held at that point, or when a
request fails with too little time left, the run fails as final: a rerun would
have no time to answer in. Each such stop is logged as "shortened by the deadline".

The model and its knobs are settings, overridable per run, named by the caller's
`config`: PUNCTUATE (`moments_claude_*`) for transcript.punctuate, PICK
(`moments_pick_claude_*`, GEO-3702) for every call of debates.pick_moments. A
temperature goes only to a model that takes one (`takes_temperature`): Claude
Sonnet 5.x and the other newer models reject it. An effort, where the config has
one set, goes in `output_config.effort`, beside a schema's `format`. A model that
thinks may answer with thinking blocks before its text; the answer is read from its
text blocks alone, and only that text is sent back with a refused answer.

An answer Claude declines (stop reason "refusal") is logged with its category and
ends the conversation: the answer held is kept, else the run fails as final, as
asking the same again would be declined again.
The system prompt and the first message are marked for prompt caching, so a send
back or an engine retry within minutes reads the transcript from cache.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from hatchet_sdk.exceptions import NonRetryableException

from src.config.overrides import llm
from src.config.settings import settings
from src.decisions.base import DecisionError
from src.infrastructure.claude import claude_text, parse_json
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import SpendLimitExceeded, spend_guard
from src.pipeline.deadline import Deadline

logger = get_logger(__name__)

# What results name as their provider.
PROVIDER = "anthropic"
# The spend guard's key for Claude: the news Claude paths count under it too, so one hourly
# budget covers every Claude call this service makes.
SPEND_KEY = "claude"
# One request's ceiling. The client does not retry on its own (max_retries=0): the engine retries
# the run, so a task's execution timeout can be sized as attempts x this. 180 s cut off slow but
# healthy answers during a story-reel backfill (GEO-3638), failing the whole run.
REQUEST_TIMEOUT_S = 300
# The 4xx statuses worth running the task again for (timeout, conflict, rate limit). Any other 4xx
# (a schema structured outputs rejects, a model without it, a prompt too long) fails the same way
# on every run, so it is final. 5xx, 529 overloaded and connection errors stay retryable.
RETRYABLE_4XX = {408, 409, 429}

T = TypeVar("T")


@dataclass(frozen=True)
class ClaudeConfig:
    """The names of the settings one kind of call runs on, read through overrides.llm so a run
    may override them. `effort` is None for a config that never sends one."""

    model: str
    temperature: str
    max_tokens: str
    effort: Optional[str] = None


# transcript.punctuate: claude-sonnet-4-6 at temperature 0.2, as since GEO-3638.
PUNCTUATE = ClaudeConfig("moments_claude_model", "moments_claude_temperature", "moments_claude_max_tokens")
# debates.pick_moments, its judge and its claim-link call (GEO-3702).
PICK = ClaudeConfig(
    "moments_pick_claude_model",
    "moments_pick_claude_temperature",
    "moments_pick_claude_max_tokens",
    effort="moments_pick_claude_effort",
)

# The model families that take a sampling temperature. Claude Sonnet 5 and 5.5, Opus 4.7 and
# later, Haiku 5.5, Fable and Mythos reject a non-default one with a 400, and a model this list
# does not know is taken to be as new.
_TAKES_TEMPERATURE = ("claude-3", "claude-sonnet-4", "claude-haiku-4", "claude-opus-4")
_TAKES_NO_TEMPERATURE = ("claude-opus-4-7", "claude-opus-4-8")


def takes_temperature(model: str) -> bool:
    """Whether `model` accepts a sampling temperature."""
    return model.startswith(_TAKES_TEMPERATURE) and not model.startswith(_TAKES_NO_TEMPERATURE)


def _request(config: ClaudeConfig, schema: Optional[dict]) -> dict:
    """The model and its knobs for one request, from `config`'s settings: a temperature only for
    a model that takes one, an effort only when one is set (beside a schema's format)."""
    model = llm.get(config.model)
    request: dict = {"model": model, "max_tokens": llm.get(config.max_tokens)}
    if takes_temperature(model):
        request["temperature"] = llm.get(config.temperature)
    output_config: dict = {}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    effort = llm.get(config.effort) if config.effort else None
    if effort:
        output_config["effort"] = effort
    if output_config:
        request["output_config"] = output_config
    return request

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


# How much of a cut-off answer's end is logged: enough to see whether it was prose or JSON, never
# the whole answer (it can quote the transcript).
CUT_OFF_TAIL_CHARS = 500

CUT_OFF_MESSAGE = (
    "Your answer was cut off at the token limit before it ended. Answer again, shorter: "
    "the whole JSON object and nothing else, with no text before or after it."
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
    config: ClaudeConfig,
    client: Any = None,
    schema: Optional[dict] = None,
    cut_off_retries: int = 0,
    rank: Optional[Callable[[T], Any]] = None,
    deadline: Optional[Deadline] = None,
) -> Tuple[T, str, int]:
    """Ask until an answer has no problems, or `attempts` run out. Returns (value, model, answers
    asked for); the value is the best answer seen (see the module doc). `config` names the
    settings the requests run on (PUNCTUATE or PICK). `schema`, a JSON schema,
    constrains the reply to it (structured outputs); `cut_off_retries` is how many answers cut off
    at the token limit are sent back to be answered shorter, within `attempts`. `rank` (lower is
    better) chooses among answers with only soft problems, and among last resorts; without it the
    latest wins. `deadline` stops asking once too little time is left and caps each request's
    timeout at the time left (see the module doc)."""

    def no_worse(value: T, held: Optional[Tuple[Any, ...]]) -> bool:
        return held is None or rank is None or rank(value) <= rank(held[0])

    # Read once, so every answer in the conversation is asked of the same model the same way.
    shape = _request(config, schema)
    model = shape["model"]
    client = client or _client()
    messages: List[dict] = [{"role": "user", "content": _cached(user)}]
    best: Optional[Tuple[T, int]] = None
    # The last resort kept, with the problems it was refused for (logged if it is returned).
    last_resort: Optional[Tuple[T, int, List[str]]] = None
    problems: List[str] = []
    for attempt in range(1, attempts + 1):
        if deadline is not None and not deadline.allows_request():
            left = deadline.left()
            if best is not None or last_resort is not None:
                logger.warning(
                    f"{task}: shortened by the deadline: {left:.0f} s left, so answer {attempt} was "
                    "not asked for; keeping what is held"
                )
                break
            raise FinalDecisionError(
                f"{task}: shortened by the deadline: {left:.0f} s left before answer {attempt} and no "
                f"acceptable answer held: {problems}"
            )
        try:
            spend_guard.check_and_record(SPEND_KEY)
        except SpendLimitExceeded as e:
            # An answer already held is still the best seen: keep it rather than fail the run.
            if best is not None or last_resort is not None:
                logger.warning(f"{task}: spend guard tripped before answer {attempt}; keeping what is held: {e}")
                break
            raise FinalDecisionError(f"{task}: spend guard tripped: {e}") from e
        # Under a deadline the request may take only the time left: the client's own timeout, and
        # a wall-clock cap around the whole call (the client's timeout bounds each read, not the
        # call).
        timeout = deadline.request_timeout(REQUEST_TIMEOUT_S) if deadline is not None else None
        try:
            request = client.messages.create(
                system=_cached(system),
                messages=messages,
                **shape,
                **({"timeout": timeout} if timeout is not None else {}),
            )
            message = await (asyncio.wait_for(request, timeout) if timeout is not None else request)
        except Exception as e:  # network, rate limit, overload: the engine retries the run
            out_of_time = deadline is not None and not deadline.allows_request()
            # A send-back that fails leaves the answer already held, which a rerun would throw away.
            if best is not None or last_resort is not None:
                note = "shortened by the deadline: " if out_of_time else ""
                logger.warning(f"{task}: {note}answer {attempt} request failed; keeping what is held: {e}")
                break
            # A rerun would start with no time left.
            if out_of_time:
                raise FinalDecisionError(
                    f"{task}: shortened by the deadline: answer {attempt} request failed with "
                    f"{deadline.left():.0f} s left and no acceptable answer held: {e!r}"
                ) from e
            if _rejected(e):
                raise FinalDecisionError(f"{task}: Claude rejected the request: {e}") from e
            raise DecisionError(f"{task}: Claude request failed: {e}", retryable=True) from e
        _log_usage(task, attempt, model, message)
        raw = claude_text(message)
        if getattr(message, "stop_reason", None) == "refusal":
            category = getattr(getattr(message, "stop_details", None), "category", None)
            logger.warning(f"{task}: answer {attempt}/{attempts} declined by Claude (category {category})")
            if best is not None or last_resort is not None:
                break
            raise FinalDecisionError(f"{task}: Claude declined to answer (category {category})")
        if getattr(message, "stop_reason", None) == "max_tokens":
            logger.warning(
                f"{task}: answer {attempt}/{attempts} cut off at the token limit after "
                f"{len(raw)} chars; it ended: {raw[-CUT_OFF_TAIL_CHARS:]!r}"
            )
            if best is not None:
                break
            # With only a last resort, a retry left can still bring an answer that keeps every rule.
            if cut_off_retries > 0 and attempt < attempts:
                cut_off_retries -= 1
                messages += [
                    {"role": "assistant", "content": raw or "(empty)"},
                    {"role": "user", "content": CUT_OFF_MESSAGE},
                ]
                continue
            if last_resort is not None:
                break
            raise FinalDecisionError(f"{task}: the answer was cut off at the token limit")
        try:
            value, hard, soft = check(parse_json(raw))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
            # Not JSON (JSONDecodeError is a ValueError), or JSON a check could not read: either
            # way the answer is refused and sent back, never a crash Hatchet would rerun.
            value, hard, soft = None, [f"The answer is not valid JSON in the asked shape ({e})."], []
        if not hard:
            if not soft:
                return value, model, attempt
            if no_worse(value, best):
                best = (value, attempt)
        elif value is not None and no_worse(value, last_resort):
            last_resort = (value, attempt, hard)
        # Hard problems alone while there are any: soft nudges sent with them pull the answer
        # in more directions at once (an earlier opener makes a clip longer).
        problems = hard or soft
        if attempt < attempts:
            logger.warning(f"{task}: answer {attempt}/{attempts} sent back: {problems}")
            messages += [
                {"role": "assistant", "content": raw or "(empty)"},
                {"role": "user", "content": _retry_message(problems)},
            ]
    if best is not None:
        return best[0], model, best[1]
    if last_resort is not None:
        logger.warning(
            f"{task}: no answer kept every rule; keeping answer {last_resort[1]} as a last "
            f"resort: {last_resort[2]}"
        )
        return last_resort[0], model, last_resort[1]
    raise FinalDecisionError(f"{task}: no acceptable answer after {attempts} attempts: {problems}")


def _rejected(e: Exception) -> bool:
    """Whether Claude refused the request itself (a 4xx not in RETRYABLE_4XX)."""
    import anthropic

    return isinstance(e, anthropic.APIStatusError) and 400 <= e.status_code < 500 and e.status_code not in RETRYABLE_4XX


def _log_usage(task: str, attempt: int, model: str, message: Any) -> None:
    """One line per answer with its model and token counts, so what a run costs can be read from
    its logs (output tokens count a thinking model's thinking too). The model is named because a
    debate's punctuation and its pick run on different ones, priced differently."""
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    counts = ", ".join(
        f"{name} {getattr(usage, name)}"
        for name in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")
        if isinstance(getattr(usage, name, None), int)
    )
    logger.info(f"{task}: answer {attempt} ({model}) used {counts}")


__all__ = [
    "PROVIDER",
    "SPEND_KEY",
    "PICK",
    "PUNCTUATE",
    "ClaudeConfig",
    "FinalDecisionError",
    "ask_json",
    "execution_timeout_s",
    "parse_json",
    "takes_temperature",
]
