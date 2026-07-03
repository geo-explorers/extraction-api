"""Generic 'tabular data -> new Google Sheet' service (gspread + service account).

Knows nothing about entities or spaces: pure ``columns`` + ``rows``. Any pipeline
can reuse it. Engine-agnostic (no Hatchet import); the gspread calls are blocking,
so callers on an event loop offload via ``asyncio.to_thread``.

A headless worker authenticates with a service-account key (inline JSON preferred,
file path fallback). The created spreadsheet lives in the service account's own
Drive, so it is shared with ``share_email`` and/or created inside ``folder_id`` to
be visible to a human.
"""

import json

import gspread
from google.oauth2.service_account import Credentials

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Sheets scope for writing values; Drive scope for create/share/folder placement.
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _load_service_account_info() -> dict:
    """Load the service-account key: inline JSON (preferred) or a file path."""
    if settings.google_service_account_json:
        return json.loads(settings.google_service_account_json)
    if settings.google_service_account_file:
        with open(settings.google_service_account_file, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(
        "No Google service-account credentials: set GOOGLE_SERVICE_ACCOUNT_JSON "
        "(inline) or GOOGLE_SERVICE_ACCOUNT_FILE (path)."
    )


def _client() -> gspread.Client:
    creds = Credentials.from_service_account_info(
        _load_service_account_info(), scopes=_SCOPES
    )
    return gspread.authorize(creds)


def create_table_sheet(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    *,
    share_email: str | None = None,
    folder_id: str | None = None,
) -> tuple[str, str]:
    """Create a NEW spreadsheet, write ``columns`` as a header + ``rows``,
    optionally share it, and return ``(spreadsheet_id, spreadsheet_url)``.

    Synchronous (blocking gspread calls); offload via ``asyncio.to_thread`` on an
    event loop. Creating a new spreadsheet is NOT idempotent — a retry makes a
    duplicate — so callers use ``retries=0``.
    """
    gc = _client()
    share_email = (
        share_email if share_email is not None else settings.google_sheets_share_email
    )
    folder_id = folder_id if folder_id is not None else settings.google_drive_folder_id

    sh = gc.create(title, folder_id=folder_id) if folder_id else gc.create(title)
    if share_email:
        sh.share(share_email, perm_type="user", role="writer", notify=False)

    values = [list(columns)] + [list(r) for r in rows]
    sh.sheet1.update(range_name="A1", values=values, value_input_option="RAW")

    logger.info(
        f"create_table_sheet: created '{title}' ({len(rows)} rows) -> {sh.url}"
    )
    return sh.id, sh.url
