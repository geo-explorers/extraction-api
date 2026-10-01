"""Task enqueue/status facade.

The HTTP contract TypeScript consumers use to run tasks on the Hatchet plane
without a Hatchet SDK:

    POST /tasks                 -> 201 {id, type, status:"queued"}
    GET  /tasks/{run_id}        -> {id, status, result?, error?}
    GET  /tasks?type=…          -> recent runs of a task type (ids, statuses, timings)
    GET  /tasks/{run_id}/input  -> the exact input the run received
    GET  /tasks/{run_id}/steps  -> every step (DAG task) with its output / error
    GET  /tasks/{run_id}/logs   -> the run's log lines, grouped by step

The three inspection routes exist so an agent with a read-only key can study
what production ran — the article bodies a news run was given, what each DAG
step produced, which prompt overrides applied — without a Hatchet token.

Hatchet imports are done LAZILY inside the handlers so a missing
HATCHET_CLIENT_TOKEN degrades only these endpoints, never the whole API (which
also serves the sync extraction endpoints).
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ValidationError

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


class EnqueueRequest(BaseModel):
    type: str
    payload: dict
    # Accepted for forward-compatibility; wiring to Hatchet's dedup key is a
    # follow-up to verify against a live engine.
    idempotency_key: str | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
def enqueue_task(req: EnqueueRequest, request: Request) -> dict:
    """Validate and enqueue a task run; returns the run id to poll."""
    from src.tasks.registry import get_task  # lazy: keep API boot Hatchet-free

    entry = get_task(req.type)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown task type: {req.type}",
        )

    raw = json.dumps(req.payload).encode()
    if len(raw) > entry.max_payload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Payload {len(raw)} bytes exceeds cap {entry.max_payload_bytes}",
        )

    try:
        input_obj = entry.input_model(**req.payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid payload for {req.type}: {e.errors()}",
        )

    ref = entry.runnable.run(input_obj, wait_for_result=False)
    caller = getattr(request.state, "api_caller", "unknown")
    logger.info(f"Enqueued {req.type} -> run {ref.workflow_run_id} (caller={caller})")
    return {"id": ref.workflow_run_id, "type": req.type, "status": "queued"}


# Windows of 7 days or more make the SDK warn about engine performance.
MAX_SINCE_HOURS = 24 * 7 - 1
_STEP_SUFFIX = re.compile(r"-\d+$")
# Hatchet's internal task-input envelope carries the payload under "input".
_ENVELOPE_KEYS = frozenset({"parents", "triggered_by", "triggers"})
# When a run is represented by several rows, the row with the most telling status wins.
_STATUS_RANK = {"FAILED": 0, "CANCELLED": 1, "RUNNING": 2, "QUEUED": 3, "COMPLETED": 4}


def _status_value(obj: Any) -> Optional[str]:
    st = getattr(obj, "status", None)
    return getattr(st, "value", str(st)) if st is not None else None


def _iso(dt: Any) -> Optional[str]:
    return dt.isoformat() if dt else None


def _hatchet_error(e: Exception, what: str) -> HTTPException:
    """Map an SDK failure for the inspection routes: an engine 404 means the
    run does not exist; anything else (expired token, engine down, 5xx) is an
    outage and must not read as "not found"."""
    from hatchet_sdk.clients.rest.exceptions import ApiException  # lazy

    if isinstance(e, ApiException) and getattr(e, "status", None) == 404:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found")
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"{what} unavailable: {e}")


def _run_details(run_id: str):
    from src.hatchet_client import hatchet  # lazy: keep API boot Hatchet-free

    try:
        return hatchet.runs.get(run_id)
    except Exception as e:
        raise _hatchet_error(e, f"Run {run_id}")


async def _run_details_async(run_id: str):
    from src.hatchet_client import hatchet  # lazy

    try:
        return await hatchet.runs.aio_get(run_id)
    except Exception as e:
        raise _hatchet_error(e, f"Run {run_id}")


def _step_name(task: Any) -> str:
    """Hatchet names a task "<step>-<timestamp>"; strip the timestamp."""
    name = getattr(task, "display_name", None) or getattr(task, "step_id", None) or ""
    return _STEP_SUFFIX.sub("", name)


def _step_namer(details: Any) -> Callable[[Any], str]:
    """One resolver for every route, so /steps and /logs agree on a step's
    name: the DAG shape's task_name (the code-level step name) by step id,
    else the display name minus its timestamp."""
    name_by_step = {
        getattr(i, "step_id", None): getattr(i, "task_name", None)
        for i in (getattr(details, "shape", None) or [])
    }
    return lambda t: name_by_step.get(getattr(t, "step_id", None)) or _step_name(t)


def _parents_by_step(details: Any) -> dict:
    """step id -> parent step ids, from the run shape (children_step_ids)."""
    parents: dict = {}
    for item in getattr(details, "shape", None) or []:
        for child in getattr(item, "children_step_ids", None) or []:
            parents.setdefault(child, []).append(getattr(item, "step_id", None))
    return parents


def _run_type(details: Any) -> Optional[str]:
    """The task type of a run. runs.get leaves workflow_name unset on the task
    rows, so fall back to the run's display name ("<type>-<timestamp>")."""
    for t in details.tasks or []:
        if getattr(t, "workflow_name", None):
            return t.workflow_name
    name = getattr(details.run, "display_name", None)
    return _STEP_SUFFIX.sub("", name) if name else None


def _run_input(details: Any) -> Any:
    """The payload the run was enqueued with. For a DAG run runs.get fills
    run.input with the payload; for a standalone task the engine hands back
    Hatchet's internal envelope ({input, parents, triggered_by, triggers, ...})
    — on run.input or, when that is empty, on the task row — which is unwrapped
    only when every envelope key is present."""
    raw = details.run.input
    if not raw and details.tasks:
        raw = getattr(details.tasks[0], "input", None)
    if isinstance(raw, dict) and "input" in raw and _ENVELOPE_KEYS <= set(raw):
        raw = raw["input"]
    return raw or None


def _ordered_tasks(details: Any) -> list:
    """The run's tasks in execution order: by start time, with steps that never
    started (cancelled, skipped) last — runs.get returns them unordered."""
    floor = datetime.min.replace(tzinfo=timezone.utc)
    namer = _step_namer(details)
    return sorted(
        details.tasks or [],
        key=lambda t: (
            getattr(t, "started_at", None) is None,
            getattr(t, "started_at", None) or getattr(t, "created_at", None) or floor,
            namer(t),
        ),
    )


@router.get("")
def list_task_runs(
    type: str = Query(..., description="Task type, e.g. news.extract_topics_and_claims"),
    limit: int = Query(20, ge=1, le=200),
    since_hours: int = Query(24, ge=1, le=MAX_SINCE_HOURS),
    status_filter: Optional[str] = Query(None, alias="status", description="QUEUED|RUNNING|COMPLETED|FAILED|CANCELLED"),
) -> dict:
    """Recent runs of one task type, newest first, without payloads. Use the
    ids with GET /tasks/{id}/input, /steps and /logs."""
    from src.hatchet_client import hatchet  # lazy
    from src.tasks.registry import get_task  # lazy
    from hatchet_sdk import V1TaskStatus

    if get_task(type) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown task type: {type}")
    statuses = None
    if status_filter:
        try:
            statuses = [V1TaskStatus(status_filter.upper())]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown status {status_filter!r}; use one of {[s.value for s in V1TaskStatus]}",
            )
    try:
        # Filter server-side by workflow id: a type-agnostic page would miss a
        # rarer type behind a busier one.
        workflow_ids = [
            w.metadata.id for w in hatchet.workflows.list(workflow_name=type).rows or [] if w.name == type
        ]
        if not workflow_ids:
            return {"type": type, "since_hours": since_hours, "runs": []}
        summary = hatchet.runs.list(
            since=datetime.now(timezone.utc) - timedelta(hours=since_hours),
            statuses=statuses,
            workflow_ids=workflow_ids,
            limit=min(limit * 5, 1000),  # a run may be listed with its step rows; collapsed below
            include_payloads=False,
        )
    except Exception as e:
        raise _hatchet_error(e, "Run listing")

    runs: dict = {}
    for row in summary.rows or []:
        if (getattr(row, "workflow_name", None) or type) != type:
            continue  # defensive: the engine filtered by workflow id already
        run_id = getattr(row, "workflow_run_external_id", None) or row.metadata.id
        row_type = getattr(getattr(row, "type", None), "value", None)
        current = runs.get(run_id)
        # One entry per run: a DAG row wins; among step rows the most telling
        # status wins (FAILED over COMPLETED), never just the first seen.
        if current is not None and current["_dag"]:
            continue
        if current is not None and row_type != "DAG":
            if _STATUS_RANK.get(_status_value(row) or "", 9) >= _STATUS_RANK.get(current["status"] or "", 9):
                continue
        runs[run_id] = {
            "id": run_id,
            "type": type,
            "status": _status_value(row),
            "created_at": _iso(getattr(row, "created_at", None)),
            "started_at": _iso(getattr(row, "started_at", None)),
            "finished_at": _iso(getattr(row, "finished_at", None)),
            "_dag": row_type == "DAG",
        }
    ordered = sorted(runs.values(), key=lambda r: r["created_at"] or "", reverse=True)[:limit]
    for r in ordered:
        r.pop("_dag", None)
    return {"type": type, "since_hours": since_hours, "runs": ordered}


@router.get("/stats")
def task_stats(type: str) -> dict:
    """Active-run counts for a task type: how many are QUEUED vs RUNNING.

    The export consumer polls this to alert when the single-consumer
    `podcast.export` queue backs up (publish slower than its trigger cadence).
    Counts the type's currently-active runs from the last 24h (the engine's
    default window; an export's 40-min timeout keeps live runs well inside it).

    NOTE: declared BEFORE `GET /{run_id}` so the literal `/stats` path is not
    captured as a run id.
    """
    from src.hatchet_client import hatchet  # lazy
    from hatchet_sdk import V1TaskStatus

    try:
        summary = hatchet.runs.list(
            statuses=[V1TaskStatus.QUEUED, V1TaskStatus.RUNNING]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not list runs: {e}",
        )

    queued = running = 0
    for row in summary.rows:
        name = row.workflow_name or row.display_name or ""
        if name != type and not name.startswith(type):
            continue
        st = getattr(row.status, "value", str(row.status))
        if st == "QUEUED":
            queued += 1
        elif st == "RUNNING":
            running += 1
    return {"type": type, "queued": queued, "running": running}


@router.get("/{run_id}")
def get_task_status(run_id: str) -> dict:
    """Return the current status (and result/error when terminal) of a run.

    FROZEN CONTRACT: news-worker and postgres_to_geo poll this. Response keys
    and the "any failure is 404" mapping stay as they are; the inspection
    routes below are where the 404/503 distinction lives.

    Mapping verified against hatchet-sdk's V1WorkflowRunDetails: `.run` carries
    status (a V1TaskStatus enum), output (dict), and error_message. For DAG
    workflows `run.output` holds the terminal task's output; per-step outputs
    are served by GET /tasks/{run_id}/steps.
    """
    from src.hatchet_client import hatchet  # lazy

    try:
        details = hatchet.runs.get(run_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found or unavailable: {e}",
        )

    run = details.run
    return {
        "id": run_id,
        "status": _status_value(run),
        "result": run.output or None,
        "error": run.error_message or None,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


@router.get("/{run_id}/input")
def get_task_input(run_id: str) -> dict:
    """The exact input the run received: for news runs that includes every
    source article's body; for claims.extract the documents and knobs."""
    details = _run_details(run_id)
    return {
        "id": run_id,
        "type": _run_type(details),
        "status": _status_value(details.run),
        "input": _run_input(details),
    }


@router.get("/{run_id}/steps")
def get_task_steps(run_id: str) -> dict:
    """Every step of the run — the DAG's intermediate data — with its output
    or error. A standalone task is one step. `parents` follows the DAG shape."""
    details = _run_details(run_id)
    namer = _step_namer(details)
    parents = _parents_by_step(details)
    name_of_step = {getattr(t, "step_id", None): namer(t) for t in details.tasks or []}

    steps = []
    for t in _ordered_tasks(details):
        tid = getattr(t, "task_external_id", None) or t.metadata.id
        steps.append({
            "name": namer(t),
            "task_id": tid,
            "status": _status_value(t),
            "parents": [name_of_step.get(p, p) for p in parents.get(getattr(t, "step_id", None), [])],
            "started_at": _iso(getattr(t, "started_at", None)),
            "finished_at": _iso(getattr(t, "finished_at", None)),
            "attempt": getattr(t, "attempt", None),
            "retry_count": getattr(t, "retry_count", None),
            "output": getattr(t, "output", None) or None,
            "error": getattr(t, "error_message", None) or None,
        })
    return {"id": run_id, "status": _status_value(details.run), "steps": steps}


@router.get("/{run_id}/logs")
async def get_task_logs(run_id: str, limit_per_step: int = Query(500, ge=1, le=5000)) -> dict:
    """The run's log lines grouped by step, in execution order — including the
    `overrides[...] applied=[...] unused=[...]` summary each step writes.
    Steps are fetched concurrently; an engine 404 for one step's log store is
    reported on that step, any other failure is a 503."""
    from src.hatchet_client import hatchet  # lazy
    from hatchet_sdk.clients.rest.exceptions import ApiException

    details = await _run_details_async(run_id)
    namer = _step_namer(details)
    tasks = _ordered_tasks(details)
    ids = [getattr(t, "task_external_id", None) or t.metadata.id for t in tasks]
    results = await asyncio.gather(
        *(hatchet.logs.aio_list(tid, limit=limit_per_step) for tid in ids), return_exceptions=True
    )

    steps = []
    for t, tid, res in zip(tasks, ids, results):
        if isinstance(res, Exception):
            if isinstance(res, ApiException) and getattr(res, "status", None) == 404:
                steps.append({"name": namer(t), "task_id": tid, "lines": [], "error": "no log lines stored for this step"})
                continue
            raise _hatchet_error(res, f"Logs of run {run_id}")
        steps.append({
            "name": namer(t),
            "task_id": tid,
            "lines": [
                {
                    "at": _iso(getattr(r, "created_at", None)),
                    "level": getattr(getattr(r, "level", None), "value", getattr(r, "level", None)),
                    "message": getattr(r, "message", ""),
                }
                for r in (getattr(res, "rows", None) or [])
            ],
        })
    return {"id": run_id, "status": _status_value(details.run), "steps": steps}
