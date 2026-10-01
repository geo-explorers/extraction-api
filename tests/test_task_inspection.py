"""Run inspection routes (GET /tasks, /tasks/{id}/input|steps|logs) and the
read-only API key scope that gates them, against a fake Hatchet client."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from src.api.auth import ApiKey, configured_keys, describe, parse_api_keys, resolve_key
from hatchet_sdk.clients.rest.exceptions import ApiException
from src.config.settings import settings

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def _enum(v):
    return NS(value=v)


def fake_details(run_id="run-1"):
    tasks = [
        NS(metadata=NS(id="t-topics"), task_external_id="t-topics", display_name="topics-1790000000001", step_id="s1",
           workflow_name=None, status=_enum("COMPLETED"), started_at=T0, finished_at=T0 + timedelta(seconds=3),
           created_at=T0, attempt=1, retry_count=0, input={"media_type": "debate"}, output={"topics": []}, error_message=""),
        NS(metadata=NS(id="t-extract"), task_external_id="t-extract", display_name="extract-1790000000002", step_id="s2",
           workflow_name="claims.extract", status=_enum("COMPLETED"), started_at=T0 + timedelta(seconds=3), finished_at=T0 + timedelta(seconds=20),
           created_at=T0, attempt=1, retry_count=0, input={}, output={"extraction": {"claims": [{"text": "c1"}]}}, error_message=""),
        NS(metadata=NS(id="t-finalize"), task_external_id="t-finalize", display_name="finalize-1790000000003", step_id="s3",
           workflow_name="claims.extract", status=_enum("FAILED"), started_at=T0 + timedelta(seconds=20), finished_at=T0 + timedelta(seconds=21),
           created_at=T0, attempt=2, retry_count=1, input={}, output=None, error_message="boom"),
    ]
    shape = [  # WorkflowRunShapeItemForWorkflowRunDetails: step-id graph with the code-level step name
        NS(task_external_id="t-topics", step_id="s1", task_name="topics", children_step_ids=["s2"]),
        NS(task_external_id="t-extract", step_id="s2", task_name="extract", children_step_ids=["s3"]),
        NS(task_external_id="t-finalize", step_id="s3", task_name="finalize", children_step_ids=[]),
    ]
    run = NS(metadata=NS(id=run_id), status=_enum("FAILED"), started_at=T0, finished_at=T0 + timedelta(seconds=21),
             input={"media_type": "debate", "documents": [{"content": "A: hi"}]}, output={"claims": []}, error_message="boom",
             display_name="claims.extract-1790000000000")
    return NS(run=run, tasks=tasks, shape=shape)


def fake_rows():
    def row(run_id, name, typ, status, created, ext_id):
        return NS(metadata=NS(id=ext_id), workflow_run_external_id=run_id, workflow_name=name, display_name=f"{name}-x",
                  type=_enum(typ), status=_enum(status), created_at=created, started_at=created, finished_at=None)
    return NS(rows=[
        row("run-1", "claims.extract", "TASK", "COMPLETED", T0, "t-topics"),       # child step row
        row("run-1", "claims.extract", "DAG", "FAILED", T0, "run-1"),              # the run row wins
        row("run-2", "claims.extract", "DAG", "COMPLETED", T0 + timedelta(hours=1), "run-2"),
        row("run-3", "news.extract_claims", "TASK", "COMPLETED", T0, "run-3"),     # other type, filtered out
    ])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy-key")
    monkeypatch.setattr(settings, "api_keys", "agent-ops:agent-key,armando:ro:ro-key")
    import src.tasks.registry  # noqa: F401  build the registry with the real client before stubbing it
    import src.hatchet_client as hc

    calls = {"list": [], "logs": [], "workflows": []}
    state = {"outage": False}

    def runs_get(run_id):
        if state["outage"]:
            raise ApiException(status=401, reason="token expired")
        if run_id != "run-1":
            raise ApiException(status=404, reason="not found")
        return fake_details(run_id)

    async def runs_aio_get(run_id):
        return runs_get(run_id)

    def runs_list(**kwargs):
        calls["list"].append(kwargs)
        return fake_rows()

    def workflows_list(workflow_name=None, **_):
        calls["workflows"].append(workflow_name)
        rows = [NS(metadata=NS(id="wf-claims"), name="claims.extract"), NS(metadata=NS(id="wf-claims-old"), name="claims.extract"),
                NS(metadata=NS(id="wf-other"), name="claims.extract.v2")]
        return NS(rows=[r for r in rows if r.name.startswith(workflow_name or "")])

    async def logs_aio_list(task_id, limit=1000, **_):
        calls["logs"].append((task_id, limit))
        if task_id == "t-finalize":
            raise ApiException(status=404, reason="no logs")
        if state["outage"]:
            raise ApiException(status=503, reason="engine down")
        return NS(rows=[NS(created_at=T0, level=_enum("INFO"), message=f"overrides[claims.extract:{task_id[2:]}] applied=['claims_extract.core'] unused=[]")])

    monkeypatch.setattr(hc, "hatchet", NS(runs=NS(get=runs_get, aio_get=runs_aio_get, list=runs_list),
                                          logs=NS(aio_list=logs_aio_list), workflows=NS(list=workflows_list)))
    from src.api.main import app

    c = TestClient(app)
    c.calls = calls
    c.state = state
    return c


RO = {"X-API-Key": "ro-key"}
RW = {"X-API-Key": "agent-key"}
LEGACY = {"X-API-Key": "legacy-key"}


# ── key scope ────────────────────────────────────────────────────────────────


def test_parse_scope_variants_and_fail_closed_on_typos():
    problems = []
    keys = parse_api_keys("a:ro:k1, b:RW:k2, c:k3, k4, d:xx:k5, ro:k6", problems)
    assert keys == [
        ApiKey("a", "k1", "ro"), ApiKey("b", "k2", "rw"), ApiKey("c", "k3", "rw"), ApiKey("key-4", "k4", "rw"),
    ]
    # an unknown scope and a scope-as-label are rejected, never admitted as read-write keys
    assert len(problems) == 2 and "unknown scope 'xx'" in problems[0] and "'ro'" in problems[1]
    assert resolve_key("xx:k5", keys) is None and resolve_key("k5", keys) is None and resolve_key("k6", keys) is None
    assert resolve_key("k1", keys).read_only and not resolve_key("k2", keys).read_only
    assert "armando (ro)" in describe([ApiKey("armando", "x", "ro")])


def test_configured_keys_reports_problems(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy")
    monkeypatch.setattr(settings, "api_keys", "ok:ro:k1,bad:readonly:k2")
    problems = []
    assert [k.label for k in configured_keys(settings, problems)] == ["api_key", "ok"]
    assert problems and "readonly" in problems[0]


def test_read_only_key_can_get_but_not_post_or_extract(client):
    assert client.get("/prompts", headers=RO).status_code == 200
    assert client.get("/tasks/run-1", headers=RO).status_code == 200
    r = client.post("/tasks", headers=RO, json={"type": "ping", "payload": {}})
    assert r.status_code == 403 and "read-only" in r.text
    assert client.post("/extract/guests", headers=RO, json={"title": "t", "description": "d"}).status_code == 403


def test_read_write_and_legacy_keys_are_unchanged(client, monkeypatch):
    import src.tasks.registry as registry

    entry = registry.get_task("ping")
    monkeypatch.setattr(entry, "runnable", NS(run=lambda *a, **k: NS(workflow_run_id="run-9")))
    for h in (RW, LEGACY):
        assert client.get("/tasks/run-1", headers=h).status_code == 200
        assert client.post("/tasks", headers=h, json={"type": "ping", "payload": {"message": "hi"}}).status_code == 201


# ── existing status route keeps its exact shape ──────────────────────────────


def test_status_route_shape_unchanged(client):
    body = client.get("/tasks/run-1", headers=LEGACY).json()
    assert set(body) == {"id", "status", "result", "error", "started_at", "finished_at"}
    assert body["status"] == "FAILED" and body["result"] == {"claims": []} and body["error"] == "boom"


# ── inspection routes ────────────────────────────────────────────────────────


def test_list_runs_filters_server_side_dedupes_rows_and_orders_newest_first(client):
    r = client.get("/tasks", params={"type": "claims.extract", "limit": 10, "since_hours": 48}, headers=RO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["id"] for x in body["runs"]] == ["run-2", "run-1"]
    assert body["runs"][1]["status"] == "FAILED"  # the DAG row, not the child step row
    assert all("_dag" not in x for x in body["runs"])
    kw = client.calls["list"][-1]
    assert kw["workflow_ids"] == ["wf-claims", "wf-claims-old"]  # exact-name matches only, server-side filter
    assert kw["include_payloads"] is False and kw["limit"] == 50 and kw["statuses"] is None
    assert kw["since"] <= datetime.now(timezone.utc) - timedelta(hours=47)
    assert client.get("/tasks", headers=RO).status_code == 422  # type is required
    assert client.get("/tasks", params={"type": "claims.extract", "since_hours": 168}, headers=RO).status_code == 422  # under the SDK's 7-day warning
    assert client.get("/tasks", params={"type": "does.not.exist"}, headers=RO).status_code == 404
    assert client.get("/tasks", params={"type": "claims.extract", "status": "bogus"}, headers=RO).status_code == 422


def test_list_runs_without_dag_rows_keeps_the_most_telling_status(client, monkeypatch):
    import src.hatchet_client as hc

    def rows(**_):
        mk = lambda ext, st, created: NS(metadata=NS(id=ext), workflow_run_external_id="run-x", workflow_name="claims.extract",
                                         display_name="x", type=_enum("TASK"), status=_enum(st), created_at=created, started_at=created, finished_at=None)
        return NS(rows=[mk("t1", "COMPLETED", T0), mk("t2", "FAILED", T0), mk("t3", "COMPLETED", T0)])

    monkeypatch.setattr(hc.hatchet.runs, "list", rows)
    body = client.get("/tasks", params={"type": "claims.extract"}, headers=RO).json()
    assert [(r["id"], r["status"]) for r in body["runs"]] == [("run-x", "FAILED")]


def test_inspection_routes_distinguish_missing_run_from_outage(client):
    assert client.get("/tasks/nope/steps", headers=RO).status_code == 404
    client.state["outage"] = True
    for path in ("/tasks/run-1/input", "/tasks/run-1/steps", "/tasks/run-1/logs"):
        r = client.get(path, headers=RO)
        assert r.status_code == 503 and "unavailable" in r.text, path
    # the frozen legacy route keeps mapping every failure to 404 for its pollers
    assert client.get("/tasks/run-1", headers=LEGACY).status_code == 404


def test_steps_order_puts_never_started_steps_last(client, monkeypatch):
    import src.hatchet_client as hc

    d = fake_details()
    d.tasks[2].started_at = None  # finalize was cancelled before it started
    d.tasks[2].status = _enum("CANCELLED")
    monkeypatch.setattr(hc.hatchet.runs, "get", lambda run_id: d)
    body = client.get("/tasks/run-1/steps", headers=RO).json()
    assert [s["name"] for s in body["steps"]] == ["topics", "extract", "finalize"]


def test_input_route_returns_the_run_input(client):
    body = client.get("/tasks/run-1/input", headers=RO).json()
    assert body == {"id": "run-1", "type": "claims.extract", "status": "FAILED",
                    "input": {"media_type": "debate", "documents": [{"content": "A: hi"}]}}
    assert client.get("/tasks/nope/input", headers=RO).status_code == 404


def test_input_route_unwraps_the_task_envelope_of_a_standalone_run(monkeypatch, client):
    # runs.get leaves run.input empty for a standalone task; the task row carries
    # Hatchet's envelope with the payload under "input".
    import src.hatchet_client as hc

    task = NS(metadata=NS(id="t-ping"), task_external_id="t-ping", display_name="ping-1790000000009", step_id="s1",
              workflow_name=None, status=_enum("COMPLETED"), started_at=T0, finished_at=T0, created_at=T0, attempt=1,
              retry_count=0, output={"reply": "pong"}, error_message="",
              input={"input": {"message": "hi", "prompt_overrides": {}, "llm_overrides": {}}, "parents": {},
                     "triggered_by": "manual", "triggers": {"filter_payload": None}, "overrides": None, "user_data": None})
    run = NS(metadata=NS(id="run-p"), status=_enum("COMPLETED"), started_at=T0, finished_at=T0, input=None,
             output={"reply": "pong"}, error_message="", display_name="ping-1790000000009")
    monkeypatch.setattr(hc.hatchet.runs, "get", lambda run_id: NS(run=run, tasks=[task], shape=[]))
    body = client.get("/tasks/run-p/input", headers=RO).json()
    assert body["type"] == "ping"
    assert body["input"] == {"message": "hi", "prompt_overrides": {}, "llm_overrides": {}}
    # the engine may also put the envelope on run.input itself (seen on hatchet-lite)
    run.input = task.input
    assert client.get("/tasks/run-p/input", headers=RO).json()["input"] == {"message": "hi", "prompt_overrides": {}, "llm_overrides": {}}
    # a payload that merely has an "input" field is not an envelope
    run.input = {"input": "x", "parents": []}
    assert client.get("/tasks/run-p/input", headers=RO).json()["input"] == {"input": "x", "parents": []}


def test_steps_route_exposes_every_step_output_and_parents(client):
    body = client.get("/tasks/run-1/steps", headers=RO).json()
    assert body["status"] == "FAILED"
    steps = {s["name"]: s for s in body["steps"]}
    assert list(steps) == ["topics", "extract", "finalize"]  # chronological
    assert steps["topics"]["parents"] == [] and steps["extract"]["parents"] == ["topics"]
    assert steps["finalize"]["parents"] == ["extract"]
    assert steps["extract"]["output"] == {"extraction": {"claims": [{"text": "c1"}]}}
    assert steps["finalize"] | {"output": None, "error": "boom", "attempt": 2, "retry_count": 1} == steps["finalize"]
    assert steps["topics"]["started_at"] == T0.isoformat()


def test_logs_route_groups_lines_by_step_and_tolerates_a_missing_store(client):
    body = client.get("/tasks/run-1/logs", params={"limit_per_step": 50}, headers=RO).json()
    assert [s["name"] for s in body["steps"]] == ["topics", "extract", "finalize"]  # execution order
    by_name = {s["name"]: s for s in body["steps"]}
    assert by_name["topics"]["lines"][0]["message"].startswith("overrides[claims.extract:topics] applied=")
    assert by_name["topics"]["lines"][0]["level"] == "INFO"
    assert by_name["finalize"]["lines"] == [] and "no log lines" in by_name["finalize"]["error"]
    assert all(limit == 50 for _, limit in client.calls["logs"])
