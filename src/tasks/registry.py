"""Central task registry.

Builds every TaskSpec into a Hatchet task and exposes lookups for the worker
(register all) and the API facade (enqueue by name).

Importing this module constructs the Hatchet client (via build_task) and so
requires HATCHET_CLIENT_TOKEN. The API facade therefore imports it LAZILY inside
request handlers, keeping API boot independent of Hatchet configuration.
"""

from dataclasses import dataclass
from typing import Optional

from src.tasks.base import TaskSpec, build_task
from src.tasks.ping import PING_SPEC
from src.tasks.news_extract_claims import (
    NEWS_EXTRACT_CLAIMS_SPEC,
    NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC,
)

# All task specs in the system. Add new specs here.
SPECS: list[TaskSpec] = [
    PING_SPEC,
    NEWS_EXTRACT_CLAIMS_SPEC,
    NEWS_EXTRACT_CLAIMS_CLAUDE_SPEC,
]


@dataclass
class RegisteredTask:
    spec: TaskSpec
    task: object  # Hatchet Standalone task object


_REGISTRY: dict[str, RegisteredTask] = {
    spec.name: RegisteredTask(spec=spec, task=build_task(spec)) for spec in SPECS
}


def get_task(name: str) -> Optional[RegisteredTask]:
    """Look up a registered task by type name (None if unknown)."""
    return _REGISTRY.get(name)


def all_tasks() -> list:
    """All built Hatchet task objects, for worker registration."""
    return [entry.task for entry in _REGISTRY.values()]
