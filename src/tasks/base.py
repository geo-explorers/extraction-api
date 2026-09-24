"""Generic task-layer contract.

A `TaskSpec` describes a task type uniformly: input/output Pydantic schemas,
retry policy, rate-limit key, execution timeout, and a plain-async `handler`.
`build_task(spec)` is the SINGLE place that touches the Hatchet decorator API —
handlers never import Hatchet internals beyond the `Context` type — so swapping
the engine later means rewriting only this file, not the task implementations.
"""

import functools
from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

from pydantic import BaseModel
from hatchet_sdk import Context, RateLimit

from src.config.overrides import activate, overrides_of
from src.hatchet_client import hatchet
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

TIn = TypeVar("TIn", bound=BaseModel)
TOut = TypeVar("TOut", bound=BaseModel)

# Default payload ceiling. Podcast transcripts can be large, so this is generous;
# the facade rejects anything bigger before it ever reaches the queue.
DEFAULT_MAX_PAYLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@dataclass
class TaskSpec(Generic[TIn, TOut]):
    """Engine-neutral description of one task type."""

    name: str
    input_model: Type[TIn]
    output_model: Type[TOut]
    handler: Callable[[TIn, Context], Awaitable[TOut]]
    rate_limit_key: Optional[str] = None
    rate_limit_units: int = 1
    retries: int = 3
    backoff_factor: float = 2.0
    backoff_max_seconds: int = 60
    execution_timeout: timedelta = timedelta(minutes=10)
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES
    # Engine-enforced ceiling on concurrent runs of this task type (global,
    # across all workers). An int N is passed straight to Hatchet as a constant
    # limit with the GROUP_ROUND_ROBIN strategy: at most N run at once and the
    # rest QUEUE (they are not cancelled and not rejected). `concurrency=1` makes
    # the type a single-consumer queue. None leaves it unlimited.
    concurrency: Optional[int] = None


def with_overrides(fn=None, *, label: Optional[str] = None):
    """Run a task step under the override scope its input carries.

    The input's `prompt_overrides` / `llm_overrides` (OverridesMixin) are
    activated for the duration of the step, so every prompts.get / llm.get
    down the call stack sees them. On exit the scope summary — which overrides
    applied, which went unused, which prompts and settings were read — goes to
    the worker log and, when overrides were given, to the run's own log in the
    Hatchet dashboard.

    build_task applies this to every standalone handler; DAG steps declare it
    themselves, under the `@workflow.task(...)` decorator, because each step is
    its own Hatchet execution with its own copy of the workflow input.
    """

    def decorate(handler):
        name = label or getattr(handler, "__name__", "task")

        @functools.wraps(handler)
        async def wrapper(input, ctx: Context):
            prompt_ov, llm_ov = overrides_of(input)
            with activate(prompt_ov, llm_ov) as scope:
                try:
                    return await handler(input, ctx)
                finally:
                    line = scope.summary(name)
                    logger.info(line)
                    if scope.given:
                        _ctx_log(ctx, line)

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _ctx_log(ctx: Context, line: str) -> None:
    """Best-effort line into the run's dashboard log (a test Context stub may
    not have .log, and a failed log must never fail the step)."""
    log = getattr(ctx, "log", None)
    if not callable(log):
        return
    try:
        log(line)
    except Exception:  # noqa: BLE001
        pass


def build_task(spec: TaskSpec):
    """Register a TaskSpec as a Hatchet standalone task and return the task object.

    All Hatchet-specific wiring (decorator, rate limits, retry/backoff/timeout)
    lives here. The wrapped runner awaits the spec's handler under the input's
    override scope; Hatchet validates the input against `input_model` and
    serializes the returned model.
    """
    kwargs: dict = {
        "name": spec.name,
        "input_validator": spec.input_model,
        "retries": spec.retries,
        "backoff_factor": spec.backoff_factor,
        "backoff_max_seconds": spec.backoff_max_seconds,
        "execution_timeout": spec.execution_timeout,
    }
    if spec.concurrency is not None:
        kwargs["concurrency"] = spec.concurrency
    if spec.rate_limit_key:
        kwargs["rate_limits"] = [
            RateLimit(static_key=spec.rate_limit_key, units=spec.rate_limit_units)
        ]

    handler = with_overrides(spec.handler, label=spec.name)

    @hatchet.task(**kwargs)
    async def _runner(input: spec.input_model, ctx: Context) -> spec.output_model:
        return await handler(input, ctx)

    return _runner
