"""Central task registry.

Builds every task type (standalone tasks and multi-step workflows) and exposes
lookups for the worker (register all) and the API facade (enqueue by name).

A RegisteredTask carries only what the facade needs — the input schema, payload
cap, and a `runnable` with `.run(input, wait_for_result=False)` — so the facade
is agnostic to whether a type is a standalone task or a DAG workflow.

Importing this module constructs the Hatchet client and so requires
HATCHET_CLIENT_TOKEN. The API facade imports it LAZILY inside request handlers,
keeping API boot independent of Hatchet configuration.
"""

from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel

from src.tasks.base import TaskSpec, build_task, DEFAULT_MAX_PAYLOAD_BYTES
from src.tasks.ping import PING_SPEC
from src.tasks.news_extract_claims import (
    NEWS_EXTRACT_CLAIMS_SPEC,
    NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC,
)
from src.tasks.keyword_extract import KEYWORD_EXTRACT_SPEC
from src.tasks.guest_extract import GUEST_EXTRACT_SPEC
from src.tasks.host_extract import HOST_EXTRACT_SPEC
from src.tasks.podcast_extract_claims import (
    podcast_workflow,
    PodcastExtractInput,
    PodcastExtractResult,
    PODCAST_MAX_PAYLOAD_BYTES,
)

# Standalone single tasks, declared as TaskSpecs and built via build_task.
_STANDALONE_SPECS: list[TaskSpec] = [
    PING_SPEC,
    NEWS_EXTRACT_CLAIMS_SPEC,
    NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC,
    KEYWORD_EXTRACT_SPEC,
    GUEST_EXTRACT_SPEC,
    HOST_EXTRACT_SPEC,
]


@dataclass
class RegisteredTask:
    name: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    max_payload_bytes: int
    runnable: object  # exposes .run(input, wait_for_result=False)


def _build_registry() -> dict[str, RegisteredTask]:
    registry: dict[str, RegisteredTask] = {}

    for spec in _STANDALONE_SPECS:
        registry[spec.name] = RegisteredTask(
            name=spec.name,
            input_model=spec.input_model,
            output_model=spec.output_model,
            max_payload_bytes=spec.max_payload_bytes,
            runnable=build_task(spec),
        )

    # Multi-step DAG workflow(s).
    registry["podcast.extract_claims"] = RegisteredTask(
        name="podcast.extract_claims",
        input_model=PodcastExtractInput,
        output_model=PodcastExtractResult,
        max_payload_bytes=PODCAST_MAX_PAYLOAD_BYTES,
        runnable=podcast_workflow,
    )
    return registry


_REGISTRY: dict[str, RegisteredTask] = _build_registry()


def get_task(name: str) -> Optional[RegisteredTask]:
    """Look up a registered task by type name (None if unknown)."""
    return _REGISTRY.get(name)


def all_tasks() -> list:
    """All runnable task/workflow objects, for worker registration."""
    return [entry.runnable for entry in _REGISTRY.values()]


def task_names() -> list[str]:
    return list(_REGISTRY.keys())
