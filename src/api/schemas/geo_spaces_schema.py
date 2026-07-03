"""Pydantic contracts for the geo spaces -> entities -> sheet pipeline.

`Space` and `Entity` mirror the Geo/Hypergraph knowledge-graph read API
(https://testnet-api.geobrowser.io/graphql): a space's human name/description
live on its `page`; an entity carries `types { id name }` and `spaceIds`, and is
selected by a Geo *type entity id* (32-hex), not a type name. These models are
deliberately generic and shared across the standalone tasks (`geo.fetch_entities`,
`sheets.export_table`) and the `geo.assign_spaces_to_sheet` DAG — nothing here
imports Hatchet.
"""

from typing import Any

from pydantic import BaseModel, Field


# --- Core graph shapes -------------------------------------------------------


class Space(BaseModel):
    """A canonical Geo space, with enough detail for the assignment LLM to judge
    fit. `name`/`description` come from the space's `page`."""

    id: str
    name: str
    description: str = ""
    # Optional signal for the assignment LLM; not populated by the default read.
    entity_types: list[str] = Field(default_factory=list)


class EntityQuery(BaseModel):
    """Generic, parameterized query for `fetch_entities`. The `type_id` (Geo type
    entity id) is passed separately as the single replaceable axis; this carries
    the rest so one query serves every type with no per-type code."""

    space_id: str | None = Field(
        default=None, description="Restrict to entities in this Geo space id"
    )
    page_size: int = Field(
        default=500, ge=1, le=1000, description="Entities per page (GraphQL `first`)"
    )
    max_entities: int | None = Field(
        default=None,
        ge=1,
        description="Cap on total entities fetched across pages; None = fetch all (bounded by a safety ceiling)",
    )


class Entity(BaseModel):
    id: str
    name: str
    description: str = ""
    type: str = ""  # human-readable type name(s), comma-joined (from `types.name`)
    type_ids: list[str] = Field(default_factory=list)
    space_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class AssignedRow(BaseModel):
    """One table-ready row: an entity plus the spaces the LLM assigned to it
    (0 or more)."""

    entity_id: str
    entity_name: str
    entity_type: str
    assigned_space_ids: list[str] = Field(default_factory=list)
    assigned_space_names: list[str] = Field(default_factory=list)


# --- geo.fetch_entities (standalone reusable task) ---------------------------


class GeoFetchEntitiesRequest(BaseModel):
    type_id: str = Field(
        ...,
        description="Geo type entity id to fetch (e.g. Project = 484a18c5030a499cb0f2ef588ff16d50)",
    )
    query: EntityQuery = Field(default_factory=EntityQuery)


class GeoFetchEntitiesResponse(BaseModel):
    type_id: str
    entities: list[Entity] = Field(default_factory=list)
    count: int = 0


# --- sheets.export_table (standalone reusable task; fully generic) -----------


class SheetTableExportRequest(BaseModel):
    """Generic tabular payload -> new Google Sheet. Knows nothing about
    entities/spaces; any pipeline can reuse it."""

    title: str = Field(..., description="Title of the spreadsheet to create")
    columns: list[str] = Field(..., description="Header row")
    rows: list[list[str]] = Field(
        default_factory=list, description="Data rows (each a list of cell strings)"
    )
    share_email: str | None = Field(default=None, description="Override settings.google_sheets_share_email")
    folder_id: str | None = Field(default=None, description="Override settings.google_drive_folder_id")


class SheetTableExportResult(BaseModel):
    spreadsheet_id: str
    spreadsheet_url: str
    row_count: int = 0


# --- geo.assign_spaces_to_sheet (the 4-task DAG) -----------------------------


class GeoSpaceAssignInput(BaseModel):
    type_id: str = Field(
        ..., description="Geo type entity id to fetch and assign spaces to"
    )
    # Bound the assign path by default so the LLM step stays feasible; the
    # standalone geo.fetch_entities task fetches all (max_entities=None).
    entity_query: EntityQuery = Field(
        default_factory=lambda: EntityQuery(max_entities=2000)
    )
    space_ids: list[str] | None = Field(
        default=None,
        description="Canonical space ids to assign against; None = subspaces of the Geo root space (dynamic)",
    )
    sheet_title: str | None = Field(
        default=None,
        description="Title for the created Google Sheet; defaults to a generated title",
    )
    share_email: str | None = Field(default=None, description="Override settings share email")
    folder_id: str | None = Field(default=None, description="Override settings Drive folder id")


class GeoSpaceAssignResult(BaseModel):
    spreadsheet_id: str
    spreadsheet_url: str
    entity_count: int = 0
    assigned_count: int = 0  # entities with >= 1 space assigned
