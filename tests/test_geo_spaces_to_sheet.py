"""Unit tests for the geo spaces -> entities -> sheet pipeline.

No network and no Hatchet engine: registry wiring, the generic entity query
builder + canonical-space-id resolution, the aliased spaces-query builder, the
space-assignment join (LLM mocked), the service-account credential loader, and
the prompt builder.
"""

import json

import pytest


# --- Registry wiring ---------------------------------------------------------


def test_geo_dag_and_standalone_tasks_registered():
    from src.tasks.registry import get_task, task_names
    from src.tasks.base import DEFAULT_MAX_PAYLOAD_BYTES
    from src.api.schemas.geo_spaces_schema import (
        GeoSpaceAssignInput,
        GeoSpaceAssignResult,
        GeoFetchEntitiesRequest,
        GeoFetchEntitiesResponse,
        SheetTableExportRequest,
        SheetTableExportResult,
    )

    dag = get_task("geo.assign_spaces_to_sheet")
    assert dag is not None
    assert dag.input_model is GeoSpaceAssignInput
    assert dag.output_model is GeoSpaceAssignResult
    assert dag.runnable is not None
    assert dag.max_payload_bytes == DEFAULT_MAX_PAYLOAD_BYTES

    fetch = get_task("geo.fetch_entities")
    assert fetch is not None
    assert fetch.input_model is GeoFetchEntitiesRequest
    assert fetch.output_model is GeoFetchEntitiesResponse

    export = get_task("sheets.export_table")
    assert export is not None
    assert export.input_model is SheetTableExportRequest
    assert export.output_model is SheetTableExportResult

    names = {"geo.assign_spaces_to_sheet", "geo.fetch_entities", "sheets.export_table"}
    assert names.issubset(set(task_names()))


def test_standalone_specs_retry_policy():
    from src.tasks.geo_fetch_entities import GEO_FETCH_ENTITIES_SPEC
    from src.tasks.sheets_export_table import SHEETS_EXPORT_TABLE_SPEC

    # Reads are idempotent -> retry; sheet creation is not -> no retry.
    assert GEO_FETCH_ENTITIES_SPEC.retries == 3
    assert SHEETS_EXPORT_TABLE_SPEC.retries == 0


# --- Generic entity query builder --------------------------------------------


def test_build_entities_variables_shape():
    from src.api.services.hypergraph_client import build_entities_variables

    v = build_entities_variables("484a18c5030a499cb0f2ef588ff16d50")
    assert v.keys() == {"typeId", "spaceId", "first"}
    assert v["typeId"] == "484a18c5030a499cb0f2ef588ff16d50"
    assert v["spaceId"] is None  # no space filter by default
    assert v["first"] == 50  # EntityQuery default limit


def test_build_entities_variables_with_space_and_limit():
    from src.api.services.hypergraph_client import build_entities_variables
    from src.api.schemas.geo_spaces_schema import EntityQuery

    v = build_entities_variables(
        "t1", EntityQuery(space_id="c9f267dcb0d270718c2a3c45a64afd32", limit=10)
    )
    assert v["typeId"] == "t1"
    assert v["spaceId"] == "c9f267dcb0d270718c2a3c45a64afd32"
    assert v["first"] == 10


def test_canonical_space_ids_default_and_override(monkeypatch):
    import src.api.services.hypergraph_client as hg

    # Default: the built-in set (includes the Geo space id).
    monkeypatch.setattr(hg.settings, "geo_canonical_space_ids", None)
    default_ids = hg.canonical_space_ids()
    assert "a19c345ab9866679b001d7d2138d88a1" in default_ids  # Geo
    assert len(default_ids) >= 10

    # Override: comma-separated, whitespace-trimmed, empties dropped.
    monkeypatch.setattr(hg.settings, "geo_canonical_space_ids", "a, b ,, c")
    assert hg.canonical_space_ids() == ["a", "b", "c"]


def test_build_spaces_query_aliases_each_id():
    from src.api.services.hypergraph_client import build_spaces_query

    q = build_spaces_query(["a1", "b2"])
    assert 's0: space(id: "a1")' in q
    assert 's1: space(id: "b2")' in q
    assert "page { name description }" in q


# --- Space assignment join (LLM mocked) --------------------------------------


def _spaces():
    from src.api.schemas.geo_spaces_schema import Space

    return [Space(id="s1", name="Crypto"), Space(id="s2", name="Technology")]


def _entities():
    from src.api.schemas.geo_spaces_schema import Entity

    return [
        Entity(id="e1", name="Bitcoin", type="Project"),
        Entity(id="e2", name="Widget", type="Project"),
    ]


def test_assign_spaces_joins_ids_to_names(monkeypatch):
    import src.api.services.space_assignment_service as svc
    from src.api.services.llm_classify_service import ItemAssignment

    def fake_classify(*, prompt, model=None, temperature=None):
        return [
            ItemAssignment(item_id="e1", category_ids=["s1", "s2", "s1", "bogus"]),
            ItemAssignment(item_id="e2", category_ids=[]),
            ItemAssignment(item_id="ghost", category_ids=["s1"]),  # unknown -> ignored
        ]

    monkeypatch.setattr(svc, "classify_items", fake_classify)

    rows = svc.assign_spaces(_spaces(), _entities())
    # Input order preserved; the unknown "ghost" id is dropped.
    assert [r.entity_id for r in rows] == ["e1", "e2"]
    r1 = rows[0]
    # dedup ("s1" twice) + drop invalid ("bogus"); ids joined to names in order.
    assert r1.assigned_space_ids == ["s1", "s2"]
    assert r1.assigned_space_names == ["Crypto", "Technology"]
    # 0 spaces is a valid outcome.
    assert rows[1].assigned_space_ids == []


def test_assign_spaces_no_spaces_skips_llm(monkeypatch):
    import src.api.services.space_assignment_service as svc

    def boom(**kwargs):
        raise AssertionError("classify_items must not run with an empty vocabulary")

    monkeypatch.setattr(svc, "classify_items", boom)
    rows = svc.assign_spaces([], _entities())
    assert [r.entity_id for r in rows] == ["e1", "e2"]
    assert all(r.assigned_space_ids == [] for r in rows)


def test_assign_spaces_no_entities_returns_empty():
    from src.api.services.space_assignment_service import assign_spaces

    assert assign_spaces(_spaces(), []) == []


# --- Service-account credential loader ---------------------------------------


def test_credential_loader_prefers_inline_json(monkeypatch):
    import src.api.services.sheets_service as sheets

    monkeypatch.setattr(
        sheets.settings,
        "google_service_account_json",
        '{"type": "service_account", "x": 1}',
    )
    monkeypatch.setattr(sheets.settings, "google_service_account_file", None)
    assert sheets._load_service_account_info() == {"type": "service_account", "x": 1}


def test_credential_loader_falls_back_to_file(monkeypatch, tmp_path):
    import src.api.services.sheets_service as sheets

    p = tmp_path / "sa.json"
    p.write_text(json.dumps({"type": "service_account", "y": 2}))
    monkeypatch.setattr(sheets.settings, "google_service_account_json", None)
    monkeypatch.setattr(sheets.settings, "google_service_account_file", str(p))
    assert sheets._load_service_account_info() == {"type": "service_account", "y": 2}


def test_credential_loader_raises_when_unset(monkeypatch):
    import src.api.services.sheets_service as sheets

    monkeypatch.setattr(sheets.settings, "google_service_account_json", None)
    monkeypatch.setattr(sheets.settings, "google_service_account_file", None)
    with pytest.raises(ValueError):
        sheets._load_service_account_info()


# --- Prompt builder ----------------------------------------------------------


def test_prompt_includes_space_ids_entity_ids_and_contract():
    from src.config.prompts.space_assignment_prompt import build_space_assignment_prompt

    prompt = build_space_assignment_prompt(_spaces(), _entities())
    assert 'id="s1"' in prompt and 'id="s2"' in prompt
    assert '"id": "e1"' in prompt and '"id": "e2"' in prompt
    # The model must return item_id + category_ids.
    assert "item_id" in prompt and "category_ids" in prompt
