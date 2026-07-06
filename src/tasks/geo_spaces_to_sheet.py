"""Geo entity -> space assignment -> Google Sheet, as a checkpointed Hatchet DAG.

Four tasks in a fan-in ("diamond") shape:

    fetch_spaces  ─┐
                   ├─▶ assign_spaces ─▶ export_sheet   (terminal → workflow output)
    fetch_entities ┘

`fetch_spaces` and `fetch_entities` are independent roots (they run in parallel);
`assign_spaces` waits for both, and `export_sheet` renders the assignment output.
Each step is a checkpoint, so a crash or redeploy resumes at the failed step
instead of re-running completed graph reads or the (billed) Gemini assignment.

All real logic lives in engine-agnostic services (hypergraph_client,
space_assignment_service, sheets_service); these steps only wire
`ctx.task_output` and offload the blocking service calls via `asyncio.to_thread`
— the same pattern as the podcast and news DAGs. The Gemini step consumes one
`gemini_global` rate-limit unit; the two reads and the Sheet write consume none.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet
from src.api.schemas.geo_spaces_schema import (
    Entity,
    GeoSpaceAssignInput,
    GeoSpaceAssignResult,
    Space,
)
from src.api.services import hypergraph_client, sheets_service
from src.api.services.space_assignment_service import assign_spaces as assign_spaces_service
from src.tasks.base import DEFAULT_MAX_PAYLOAD_BYTES
from src.infrastructure.spend_guard import spend_guard
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Space/entity payloads are small (ids + names + a few properties); the 5MB
# default is ample. Do NOT inherit podcast's 8MB cap (that is transcript-specific).
GEO_MAX_PAYLOAD_BYTES = DEFAULT_MAX_PAYLOAD_BYTES
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
_FETCH_TIMEOUT = timedelta(minutes=5)
_ASSIGN_TIMEOUT = timedelta(minutes=15)  # pooled parallel Gemini batches for large entity sets
_EXPORT_TIMEOUT = timedelta(minutes=2)

_SHEET_COLUMNS = ["Entity", "Type", "Assigned Spaces"]


geo_workflow = hatchet.workflow(
    name="geo.assign_spaces_to_sheet",
    input_validator=GeoSpaceAssignInput,
)


@geo_workflow.task(execution_timeout=timedelta(minutes=2), retries=3, backoff_factor=2.0)
async def fetch_spaces(input: GeoSpaceAssignInput, ctx: Context) -> dict:
    # Blocking HTTP read; offload so the worker event loop stays free. model_dump
    # so the step output is a JSON-serializable dict for ctx.task_output /
    # checkpointing.
    spaces = await asyncio.to_thread(hypergraph_client.fetch_spaces, input.space_ids)
    return {"spaces": [s.model_dump() for s in spaces]}


@geo_workflow.task(execution_timeout=_FETCH_TIMEOUT, retries=3, backoff_factor=2.0)
async def fetch_entities(input: GeoSpaceAssignInput, ctx: Context) -> dict:
    entities = await asyncio.to_thread(
        hypergraph_client.fetch_entities, input.type_id, input.entity_query
    )
    return {"entities": [e.model_dump() for e in entities]}


@geo_workflow.task(
    parents=[fetch_spaces, fetch_entities],
    rate_limits=_GEMINI,
    execution_timeout=_ASSIGN_TIMEOUT,
    retries=3,
    backoff_factor=2.0,
)
async def assign_spaces(input: GeoSpaceAssignInput, ctx: Context) -> dict:
    spaces = [Space(**s) for s in ctx.task_output(fetch_spaces)["spaces"]]
    entities = [Entity(**e) for e in ctx.task_output(fetch_entities)["entities"]]
    spend_guard.check_and_record("gemini")
    rows = await asyncio.to_thread(assign_spaces_service, spaces, entities)
    return {"rows": [r.model_dump() for r in rows]}


@geo_workflow.task(parents=[assign_spaces], execution_timeout=_EXPORT_TIMEOUT, retries=0)
async def export_sheet(input: GeoSpaceAssignInput, ctx: Context) -> GeoSpaceAssignResult:
    rows_data = ctx.task_output(assign_spaces)["rows"]
    # Title from the fetched type's human name when available (type_id is a UUID).
    type_label = rows_data[0]["entity_type"] if rows_data else input.type_id
    title = input.sheet_title or f"Geo · {type_label} → Spaces"
    sheet_rows = [
        [r["entity_name"], r["entity_type"], ", ".join(r["assigned_space_names"])]
        for r in rows_data
    ]
    # Creating a new spreadsheet is non-idempotent (retries=0). Blocking gspread
    # calls offloaded to a thread.
    spreadsheet_id, url = await asyncio.to_thread(
        sheets_service.create_table_sheet,
        title,
        _SHEET_COLUMNS,
        sheet_rows,
        share_email=input.share_email,
        folder_id=input.folder_id,
    )
    assigned = sum(1 for r in rows_data if r["assigned_space_ids"])
    logger.info(
        f"geo.assign_spaces_to_sheet done: {len(rows_data)} entities, "
        f"{assigned} assigned -> {url}"
    )
    return GeoSpaceAssignResult(
        spreadsheet_id=spreadsheet_id,
        spreadsheet_url=url,
        entity_count=len(rows_data),
        assigned_count=assigned,
    )
