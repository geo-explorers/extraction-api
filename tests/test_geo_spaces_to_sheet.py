"""Unit tests for the geo spaces -> entities -> sheet pipeline.

No network and no Hatchet engine: registry wiring, the generic entity query
builder, dynamic canonical-space resolution (subspaces of root), the cursor
pagination loop + page-size ladder + truncation guard, the batched
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


def test_dag_default_bounds_entities_but_standalone_fetches_all():
    from src.api.schemas.geo_spaces_schema import (
        GeoSpaceAssignInput,
        GeoFetchEntitiesRequest,
    )

    dag_in = GeoSpaceAssignInput(type_id="t")
    assert dag_in.entity_query.max_entities == 2000  # assign path is bounded
    fetch_in = GeoFetchEntitiesRequest(type_id="t")
    assert fetch_in.query.max_entities is None  # standalone fetches all


# --- Generic entity query builder --------------------------------------------


def test_build_entities_variables_shape():
    from src.api.services.hypergraph_client import build_entities_variables

    v = build_entities_variables("484a18c5030a499cb0f2ef588ff16d50")
    assert v.keys() == {"typeId", "spaceId", "first", "after", "filter"}
    assert v["typeId"] == "484a18c5030a499cb0f2ef588ff16d50"
    assert v["spaceId"] is None and v["after"] is None and v["filter"] is None
    assert v["first"] == 500  # EntityQuery default page_size


def test_build_entities_variables_with_space_and_page_size():
    from src.api.services.hypergraph_client import build_entities_variables
    from src.api.schemas.geo_spaces_schema import EntityQuery

    v = build_entities_variables(
        "t1", EntityQuery(space_id="c9f267dcb0d270718c2a3c45a64afd32", page_size=10), after="cur"
    )
    assert v["spaceId"] == "c9f267dcb0d270718c2a3c45a64afd32"
    assert v["first"] == 10
    assert v["after"] == "cur"


def test_page_size_ladder():
    from src.api.services.hypergraph_client import _page_size_ladder

    assert _page_size_ladder(500) == [500, 250, 100]
    assert _page_size_ladder(1000) == [1000, 500, 250, 100]
    assert _page_size_ladder(100) == [100]
    assert _page_size_ladder(50) == [50]


def test_build_entities_variables_passes_filter_verbatim():
    from src.api.services.hypergraph_client import build_entities_variables
    from src.api.schemas.geo_spaces_schema import EntityQuery

    filt = {"relations": {"some": {"toEntityId": {"is": "7f79"}}}}
    v = build_entities_variables("t1", EntityQuery(filter=filt))
    assert v["filter"] == filt  # arbitrary EntityFilter, passed through untouched
    assert build_entities_variables("t1")["filter"] is None  # no filter -> None


# --- Dynamic canonical spaces (subspaces of root) ----------------------------


def test_resolve_canonical_space_ids_override(monkeypatch):
    import src.api.services.hypergraph_client as hg

    monkeypatch.setattr(hg.settings, "geo_canonical_space_ids", "a, b ,, c")

    def no_net(*a, **k):
        raise AssertionError("override must not hit the network")

    monkeypatch.setattr(hg, "_post", no_net)
    assert hg.resolve_canonical_space_ids() == ["a", "b", "c"]


def test_resolve_canonical_space_ids_dynamic(monkeypatch):
    import src.api.services.hypergraph_client as hg

    monkeypatch.setattr(hg.settings, "geo_canonical_space_ids", None)
    monkeypatch.setattr(hg.settings, "geo_root_space_id", "ROOT")
    captured = {}

    def fake_post(query, variables):
        captured.update(variables)
        return {"subspaces": [{"childSpaceId": "s1"}, {"childSpaceId": "s2"}, {"childSpaceId": None}]}

    monkeypatch.setattr(hg, "_post", fake_post)
    ids = hg.resolve_canonical_space_ids()
    assert ids == ["s1", "s2"]  # None child dropped
    assert captured["parentSpaceId"] == "ROOT"


# --- Cursor pagination -------------------------------------------------------


def _node(i):
    return {
        "id": f"e{i}",
        "name": f"n{i}",
        "description": "",
        "spaceIds": [],
        "types": [{"id": "t", "name": "T"}],
    }


def test_fetch_entities_paginates_by_cursor(monkeypatch):
    import src.api.services.hypergraph_client as hg
    from src.api.schemas.geo_spaces_schema import EntityQuery

    pages = {
        None: {"entitiesConnection": {"totalCount": 3, "pageInfo": {"hasNextPage": True, "endCursor": "c1"}, "nodes": [_node(1), _node(2)]}},
        "c1": {"entitiesConnection": {"totalCount": 3, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [_node(3)]}},
    }
    seen_cursors = []

    def fake_post(query, variables):
        seen_cursors.append(variables.get("after"))
        return pages[variables.get("after")]

    monkeypatch.setattr(hg, "_post", fake_post)
    monkeypatch.setattr(hg.time, "sleep", lambda s: None)

    out = hg.fetch_entities("T", EntityQuery(page_size=2))
    assert [e.id for e in out] == ["e1", "e2", "e3"]
    assert seen_cursors == [None, "c1"]  # advanced by endCursor


def test_fetch_entities_respects_max_entities(monkeypatch):
    import src.api.services.hypergraph_client as hg
    from src.api.schemas.geo_spaces_schema import EntityQuery

    def fake_post(query, variables):
        return {"entitiesConnection": {"totalCount": 10, "pageInfo": {"hasNextPage": True, "endCursor": "c"}, "nodes": [_node(1), _node(2), _node(3)]}}

    monkeypatch.setattr(hg, "_post", fake_post)
    monkeypatch.setattr(hg.time, "sleep", lambda s: None)

    out = hg.fetch_entities("T", EntityQuery(page_size=3, max_entities=2))
    assert [e.id for e in out] == ["e1", "e2"]  # capped mid-page, no truncation error


def test_fetch_entities_truncation_guard(monkeypatch):
    import src.api.services.hypergraph_client as hg
    from src.api.schemas.geo_spaces_schema import EntityQuery

    # totalCount says 5 but the page reports hasNextPage=False with only 2 nodes:
    # a silent truncation. Every page size in the ladder raises, so fetch re-raises.
    def fake_post(query, variables):
        return {"entitiesConnection": {"totalCount": 5, "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [_node(1), _node(2)]}}

    monkeypatch.setattr(hg, "_post", fake_post)
    monkeypatch.setattr(hg.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="truncation"):
        hg.fetch_entities("T", EntityQuery(page_size=500))


# --- Space assignment join + batching (LLM mocked) ---------------------------


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
    assert [r.entity_id for r in rows] == ["e1", "e2"]
    r1 = rows[0]
    assert r1.assigned_space_ids == ["s1", "s2"]  # dedup + drop invalid
    assert r1.assigned_space_names == ["Crypto", "Technology"]
    assert rows[1].assigned_space_ids == []


def test_assign_spaces_batches_large_sets(monkeypatch):
    import src.api.services.space_assignment_service as svc
    from src.api.services.llm_classify_service import ItemAssignment

    monkeypatch.setattr(svc.settings, "space_assignment_batch_size", 1)
    calls = []

    def fake_classify(*, prompt, model=None, temperature=None):
        eid = "e1" if '"id": "e1"' in prompt else "e2"
        calls.append(eid)
        return [ItemAssignment(item_id=eid, category_ids=["s1"])]

    monkeypatch.setattr(svc, "classify_items", fake_classify)

    rows = svc.assign_spaces(_spaces(), _entities())
    assert calls == ["e1", "e2"]  # one Gemini call per entity (batch_size=1)
    assert all(r.assigned_space_ids == ["s1"] for r in rows)


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
    assert "item_id" in prompt and "category_ids" in prompt
