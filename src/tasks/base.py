"""Generic task-layer contract.

A `TaskSpec` describes a task type uniformly: input/output Pydantic schemas,
retry policy, rate-limit key, execution timeout, and a plain-async `handler`.
`build_task(spec)` is the SINGLE place that touches the Hatchet decorator API —
handlers never import Hatchet internals beyond the `Context` type — so swapping
the engine later means rewriting only this file, not the task implementations.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Awaitable, Callable, Generic, Optional, Type, TypeVar

from pydantic import BaseModel
from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet

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


def build_task(spec: TaskSpec):
    """Register a TaskSpec as a Hatchet standalone task and return the task object.

    All Hatchet-specific wiring (decorator, rate limits, retry/backoff/timeout)
    lives here. The wrapped runner simply awaits the spec's handler; Hatchet
    validates the input against `input_model` and serializes the returned model.
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

    @hatchet.task(**kwargs)
    async def _runner(input: spec.input_model, ctx: Context) -> spec.output_model:
        return await spec.handler(input, ctx)

    return _runner
