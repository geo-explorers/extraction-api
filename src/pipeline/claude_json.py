"""One Claude conversation that must end in a JSON answer the caller accepts.

Shared by transcript.punctuate and debates.pick_moments (GEO-3638), which ask
Claude to write an answer rather than choose among options. The caller passes a
`check` that parses the answer and returns the problems with it. When there are
problems, Claude is shown them and asked again in the same conversation, up to
`attempts` answers in all. A provider failure or an answer still wrong after the
last attempt raises DecisionError, so the task engine can retry the run and a
consumer can fall back.

The model and its knobs are the `moments_claude_*` settings, overridable per run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from src.config.overrides import llm
from src.config.settings import settings
from src.decisions.base import DecisionError
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import SpendLimitExceeded, spend_guard

logger = get_logger(__name__)

PROVIDER = "anthropic"
REQUEST_TIMEOUT_S = 180

T = TypeVar("T")

# Parses Claude's answer. Returns (value, problems): an accepted answer has no problems.
Check = Callable[[Any], Tuple[Optional[T], List[str]]]


def parse_json(raw: str) -> Any:
    """The JSON in `raw`: the whole text, or the first fenced block."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw or "", re.DOTALL | re.I)
        if not match:
            start, end = (raw or "").find("{"), (raw or "").rfind("}")
            if start == -1 or end <= start:
                raise
            return json.loads(raw[start : end + 1])
        return json.loads(match.group(1))


def _client():
    import anthropic

    if not settings.anthropic_api_key:
        raise DecisionError("ANTHROPIC_API_KEY is not set")
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key, timeout=REQUEST_TIMEOUT_S)


def _retry_message(problems: List[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "Your answer breaks these rules:\n"
        f"{listed}\n\n"
        "Answer again with the whole corrected JSON, and nothing else."
    )


async def ask_json(
    system: str,
    user: str,
    check: Check[T],
    *,
    task: str,
    attempts: int = 2,
    client: Any = None,
) -> Tuple[T, str, int]:
    """Ask until `check` accepts the answer. Returns (value, model, answers asked for)."""
    model = llm.get("moments_claude_model")
    client = client or _client()
    messages: List[dict] = [{"role": "user", "content": user}]
    problems: List[str] = []
    for attempt in range(1, attempts + 1):
        try:
            spend_guard.check_and_record(PROVIDER)
        except SpendLimitExceeded as e:
            raise DecisionError(f"{task}: spend guard tripped: {e}") from e
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=llm.get("moments_claude_max_tokens"),
                temperature=llm.get("moments_claude_temperature"),
                system=system,
                messages=messages,
            )
        except Exception as e:  # network, rate limit, overload: the engine retries the run
            raise DecisionError(f"{task}: Claude request failed: {e}", retryable=True) from e
        raw = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        if getattr(message, "stop_reason", None) == "max_tokens":
            problems = ["The answer was cut off at the token limit. Answer more briefly."]
        else:
            try:
                value, problems = check(parse_json(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                value, problems = None, [f"The answer is not valid JSON in the asked shape ({e})."]
            if not problems:
                return value, model, attempt
        logger.warning(f"{task}: answer {attempt}/{attempts} refused: {problems}")
        messages += [{"role": "assistant", "content": raw or "(empty)"}, {"role": "user", "content": _retry_message(problems)}]
    raise DecisionError(f"{task}: no acceptable answer after {attempts} attempts: {problems}", retryable=True)


__all__ = ["PROVIDER", "ask_json", "parse_json"]
