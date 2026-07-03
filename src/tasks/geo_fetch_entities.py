"""geo.fetch_entities — standalone, reusable generic entity read.

Wraps `hypergraph_client.fetch_entities` as a Hatchet task so any pipeline can
fetch entities of any type by name + query, independently of the DAG. The
`entity_type` field is the single replaceable axis; the same task serves every
type with no per-type code.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.geo_spaces_schema import (
    GeoFetchEntitiesRequest,
    GeoFetchEntitiesResponse,
)
from src.api.services import hypergraph_client
from src.tasks.base import TaskSpec


async def _handle(
    input: GeoFetchEntitiesRequest, ctx: Context
) -> GeoFetchEntitiesResponse:
    entities = await asyncio.to_thread(
        hypergraph_client.fetch_entities, input.type_id, input.query
    )
    return GeoFetchEntitiesResponse(
        type_id=input.type_id,
        entities=entities,
        count=len(entities),
    )


GEO_FETCH_ENTITIES_SPEC = TaskSpec(
    name="geo.fetch_entities",
    input_model=GeoFetchEntitiesRequest,
    output_model=GeoFetchEntitiesResponse,
    handler=_handle,
    retries=3,  # idempotent read
    execution_timeout=timedelta(minutes=3),
)
