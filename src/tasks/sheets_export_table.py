"""sheets.export_table — standalone, reusable 'tabular data -> new Google Sheet'.

Wraps the generic `sheets_service.create_table_sheet` as a Hatchet task so any
pipeline can push a header + rows to a fresh spreadsheet independently of the DAG.
Knows nothing about entities/spaces — pure columns + rows.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.geo_spaces_schema import (
    SheetTableExportRequest,
    SheetTableExportResult,
)
from src.api.services import sheets_service
from src.tasks.base import TaskSpec


async def _handle(
    input: SheetTableExportRequest, ctx: Context
) -> SheetTableExportResult:
    spreadsheet_id, url = await asyncio.to_thread(
        sheets_service.create_table_sheet,
        input.title,
        input.columns,
        input.rows,
        share_email=input.share_email,
        folder_id=input.folder_id,
    )
    return SheetTableExportResult(
        spreadsheet_id=spreadsheet_id,
        spreadsheet_url=url,
        row_count=len(input.rows),
    )


SHEETS_EXPORT_TABLE_SPEC = TaskSpec(
    name="sheets.export_table",
    input_model=SheetTableExportRequest,
    output_model=SheetTableExportResult,
    handler=_handle,
    retries=0,  # creating a spreadsheet is non-idempotent (a retry duplicates)
    execution_timeout=timedelta(minutes=2),
)
