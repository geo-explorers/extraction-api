"""Perplexity Decisions API behind the DecisionModel interface.

POST https://api.perplexity.ai/v1/decisions takes the content as `state` and
named questions, and returns a probability per question instead of text. The
wire names differ from ours: a yes/no question is type `noul` with criteria
keyed "true"/"false", and its answer is the `noul` field; a scale question is
type `score` with the level descriptions as an ordered list, and its answer
carries a probability per level index ("0", "1", …) and a probability-weighted
index as `score`.

Contract read from https://docs.perplexity.ai/docs/decisions/quickstart
(2026-10-02): 10 requests/second per organization, 429 carries Retry-After in
seconds, 5xx (504 after about a minute) is retryable, and a 404/405/504 body
is not JSON — so the status is checked before the body is parsed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Optional

import requests

from src.config.settings import settings
from src.decisions.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionError,
    Question,
    Scale,
    ScaleAnswer,
    YesNo,
    YesNoAnswer,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

PERPLEXITY_DECISIONS_URL = "https://api.perplexity.ai/v1/decisions"
# The docs' own figure: 30 s covers any request under the input limit.
TIMEOUT_SECONDS = 30
# Attempts for an outage or a dropped connection (5xx, 408, network errors).
MAX_ATTEMPTS = 3
# Attempts when the organisation's 10 requests/second are exhausted. A 429 is
# not a failure of this request, only of its timing, and the task runs several
# requests at once, so it gets a longer budget and waits what the server asks.
MAX_RATE_LIMIT_ATTEMPTS = 8
RETRY_INITIAL_DELAY = 2.0
# A Retry-After longer than this is treated as the backoff instead: the task's
# whole run has a budget of minutes, and one request must not sleep it away.
MAX_RETRY_AFTER_SECONDS = 30.0
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# A `score` question takes 2 to 10 levels.
MAX_SCALE_LEVELS = 10


def to_wire(question: Question) -> Dict[str, Any]:
    if isinstance(question, YesNo):
        wire: Dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        if question.yes is not None:
            wire["criteria"] = {"true": question.yes, "false": question.no}
        return wire
    if isinstance(question, Scale):
        if len(question.levels) > MAX_SCALE_LEVELS:
            raise DecisionError(f"Perplexity scores at most {MAX_SCALE_LEVELS} levels; got {len(question.levels)}")
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": list(question.levels),
        }
    return {
        "type": "choice",
        "instructions": question.instructions,
        "criteria": dict(question.options),
    }


def _is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def from_wire(name: str, question: Question, answers: Mapping[str, Any]) -> Answer:
    raw = answers.get(name)
    if not isinstance(raw, dict):
        raise DecisionError(f"Perplexity returned no answer for question {name!r}")
    if isinstance(question, YesNo):
        probability = raw.get("noul")
        if not _is_probability(probability):
            raise DecisionError(f"Perplexity answer for {name!r} has no usable `noul`: {raw!r}")
        return YesNoAnswer(probability=float(probability))
    probabilities = raw.get("probabilities")
    if isinstance(question, Scale):
        # Levels come back keyed by their index as a string. The position is
        # recomputed from them rather than read from `score`, so the answer is
        # consistent with its own probabilities whatever the provider rounds.
        keys = [str(i) for i in range(len(question.levels))]
        if not isinstance(probabilities, dict) or not all(_is_probability(probabilities.get(k)) for k in keys):
            raise DecisionError(f"Perplexity answer for {name!r} lacks a probability per level: {raw!r}")
        per_level = [float(probabilities[k]) for k in keys]
        top = len(per_level) - 1
        score = sum(p * i for i, p in enumerate(per_level)) / top
        return ScaleAnswer(score=min(1.0, max(0.0, score)), probabilities=per_level)
    if not isinstance(probabilities, dict) or not all(
        _is_probability(probabilities.get(option)) for option in question.options
    ):
        raise DecisionError(f"Perplexity answer for {name!r} lacks a probability per option: {raw!r}")
    choice = raw.get("choice")
    if choice not in question.options:
        raise DecisionError(f"Perplexity answer for {name!r} chose an unknown option: {choice!r}")
    return ChoiceAnswer(
        choice=choice,
        probabilities={option: float(probabilities[option]) for option in question.options},
    )


def _retry_after_seconds(response: requests.Response) -> Optional[float]:
    """The server's Retry-After in seconds, capped; None when absent or not a
    number (an HTTP-date form falls back to the backoff)."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(max(0.0, seconds), MAX_RETRY_AFTER_SECONDS)


def _error_message(response: requests.Response) -> str:
    try:
        error = response.json().get("error")
        return str(error.get("message")) if isinstance(error, dict) else response.text[:200]
    except (ValueError, AttributeError):
        return response.text[:200]


class PerplexityDecisionModel:
    provider = "perplexity"

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or settings.perplexity_api_key
        if not key:
            raise ValueError("PERPLEXITY_API_KEY is required for the perplexity decision model")
        self._headers = {"Authorization": f"Bearer {key}"}

    async def decide(
        self,
        content: Mapping[str, str],
        questions: Mapping[str, Question],
        *,
        model: str,
    ) -> Dict[str, Answer]:
        body = {
            "model": model,
            "state": dict(content),
            "questions": {name: to_wire(question) for name, question in questions.items()},
        }
        payload = await self._post(body)
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise DecisionError("Perplexity decisions response has no `answers` object")
        usage = payload.get("usage") or {}
        logger.debug(f"perplexity decisions: {len(questions)} questions, {usage.get('input_tokens')} input tokens")
        return {name: from_wire(name, question, answers) for name, question in questions.items()}

    async def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        # Two budgets, counted separately: a 429 spends from the rate-limit one, every
        # other transient failure from the outage one, so a busy minute cannot use up
        # the retries an outage would need or the other way round.
        failures = 0
        rate_limited = 0
        while True:
            attempt = failures + rate_limited + 1
            backoff = RETRY_INITIAL_DELAY * (2**failures)
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    PERPLEXITY_DECISIONS_URL,
                    json=body,
                    headers=self._headers,
                    timeout=TIMEOUT_SECONDS,
                )
            except requests.RequestException as e:
                failures += 1
                if failures < MAX_ATTEMPTS:
                    logger.warning(
                        f"Perplexity decisions request failed (attempt {attempt}): {e}. Retrying in {backoff:.0f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise DecisionError(f"Perplexity decisions request failed: {e}", retryable=True) from e

            status = response.status_code
            if status == 200:
                try:
                    payload = response.json()
                except ValueError as e:
                    raise DecisionError("Perplexity decisions returned a 200 that is not JSON") from e
                if not isinstance(payload, dict):
                    raise DecisionError("Perplexity decisions returned a 200 that is not a JSON object")
                return payload

            if status == 429:
                rate_limited += 1
                if rate_limited < MAX_RATE_LIMIT_ATTEMPTS:
                    retry_after = _retry_after_seconds(response)
                    wait = retry_after if retry_after is not None else backoff
                    logger.warning(
                        f"Perplexity decisions rate limited (attempt {attempt}): waiting {wait:.0f}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
            elif status in RETRYABLE_STATUS_CODES:
                failures += 1
                if failures < MAX_ATTEMPTS:
                    logger.warning(
                        f"Perplexity decisions failed (attempt {attempt}): HTTP {status}. Retrying in {backoff:.0f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue

            raise DecisionError(
                f"Perplexity decisions returned HTTP {status}: {_error_message(response)} "
                f"(request id {response.headers.get('x-request-id')})",
                retryable=status in RETRYABLE_STATUS_CODES,
            )
