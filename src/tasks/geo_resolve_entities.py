"""geo.resolve_entities task — policy-driven name→Geo-entity resolution.

Read-only lookups against the knowledge graph: items reference caller-defined
policies (required type id, optional required tag, threshold), and each name
resolves to `resolved` / `not_found` / `ambiguous` (with candidates) / `error`.
The service never guesses between qualifying candidates and never writes —
entity creation is a publisher-side concern, never a task-side one.

Domain-agnostic BY DESIGN: policy keys are opaque caller labels; which types,
tags, and names mean what is entirely the caller's business. No LLM involved —
no rate-limit key, no spend guard.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.geo_resolve_schema import (
    GeoResolveRequest,
    GeoResolveResponse,
)
from src.api.services.geo_resolve_service import resolve_entities
from src.tasks.base import TaskSpec

_TIMEOUT = timedelta(minutes=5)


async def _handle(input: GeoResolveRequest, ctx: Context) -> GeoResolveResponse:
    return await asyncio.to_thread(resolve_entities, input)


GEO_RESOLVE_ENTITIES_SPEC = TaskSpec(
    name="geo.resolve_entities",
    input_model=GeoResolveRequest,
    output_model=GeoResolveResponse,
    handler=_handle,
    retries=3,
    execution_timeout=_TIMEOUT,
)
