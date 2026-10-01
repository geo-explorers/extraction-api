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

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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


def _status_value(obj: Any) -> Optional[str]:
    st = getattr(obj, "status", None)
    return getattr(st, "value", str(st)) if st is not None else None


def _iso(dt: Any) -> Optional[str]:
    return dt.isoformat() if dt else None


def _run_details(run_id: str):
    """hatchet.runs.get or a 404 the caller can act on (lazy import keeps API boot Hatchet-free)."""
    from src.hatchet_client import hatchet  # lazy

    try:
        return hatchet.runs.get(run_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found or unavailable: {e}",
        )


_STEP_SUFFIX = re.compile(r"-\d+$")


def _step_name(task: Any) -> str:
    """Hatchet names a DAG task "<step>-<timestamp>"; strip the timestamp so the
    step reads as its code name (topics, extract, finalize, ...)."""
    name = getattr(task, "display_name", None) or getattr(task, "step_id", None) or ""
    return _STEP_SUFFIX.sub("", name)


def _run_type(details: Any) -> Optional[str]:
    """The task type of a run. runs.get leaves workflow_name unset on the task
    rows, so fall back to the run's display name ("<type>-<timestamp>")."""
    for t in details.tasks or []:
        if getattr(t, "workflow_name", None):
            return t.workflow_name
    name = getattr(details.run, "display_name", None)
    return _STEP_SUFFIX.sub("", name) if name else None


_ENVELOPE_KEYS = frozenset({"parents", "triggered_by", "triggers"})


def _run_input(details: Any) -> Any:
    """The payload the run was enqueued with. runs.get fills run.input for DAG
    runs; for a standalone task it is empty and the task row carries Hatchet's
    internal envelope ({input, parents, triggered_by, ...}), so unwrap that."""
    raw = details.run.input
    if not raw and details.tasks:
        raw = getattr(details.tasks[0], "input", None)
    if isinstance(raw, dict) and "input" in raw and _ENVELOPE_KEYS & set(raw):
        raw = raw["input"]
    return raw or None


def _ordered_tasks(details: Any) -> list:
    """The run's tasks in execution order (runs.get returns them unordered)."""
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        details.tasks or [],
        key=lambda t: getattr(t, "started_at", None) or getattr(t, "created_at", None) or floor,
    )


@router.get("")
def list_task_runs(
    type: str = Query(..., description="Task type, e.g. news.extract_topics_and_claims"),
    limit: int = Query(20, ge=1, le=200),
    since_hours: int = Query(24, ge=1, le=24 * 14),
    status_filter: Optional[str] = Query(None, alias="status", description="QUEUED|RUNNING|COMPLETED|FAILED|CANCELLED"),
) -> dict:
    """Recent runs of one task type, newest first, without payloads. Use the
    ids with GET /tasks/{id}/input, /steps and /logs."""
    from src.hatchet_client import hatchet  # lazy
    from hatchet_sdk import V1TaskStatus

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
        summary = hatchet.runs.list(
            since=datetime.now(timezone.utc) - timedelta(hours=since_hours),
            statuses=statuses,
            limit=min(limit * 5, 1000),  # rows include DAG child tasks; filtered below
            include_payloads=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not list runs: {e}",
        )

    runs: dict[str, dict] = {}
    for row in summary.rows:
        if (getattr(row, "workflow_name", None) or "") != type:
            continue
        run_id = getattr(row, "workflow_run_external_id", None) or row.metadata.id
        row_type = getattr(getattr(row, "type", None), "value", None)
        # A DAG run lists one row per step plus (depending on engine version)
        # one for the run itself; keep one entry per run, preferring the run row.
        if run_id in runs and row_type != "DAG":
            continue
        runs[run_id] = {
            "id": run_id,
            "type": type,
            "status": _status_value(row),
            "created_at": _iso(getattr(row, "created_at", None)),
            "started_at": _iso(getattr(row, "started_at", None)),
            "finished_at": _iso(getattr(row, "finished_at", None)),
        }
    ordered = sorted(runs.values(), key=lambda r: r["created_at"] or "", reverse=True)[:limit]
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

    Mapping verified against hatchet-sdk's V1WorkflowRunDetails: `.run` carries
    status (a V1TaskStatus enum), output (dict), and error_message. For DAG
    workflows `run.output` holds the terminal task's output; per-step outputs
    are available under `details.tasks` if needed later.
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
    run_status = getattr(run.status, "value", str(run.status))
    return {
        "id": run_id,
        "status": run_status,
        "result": run.output or None,
        "error": run.error_message or None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
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
    tasks = list(details.tasks or [])

    # DAG shape (WorkflowRunShapeItemForWorkflowRunDetails): each item names a
    # step (task_name, step_id) and the step ids of its children.
    shape = list(getattr(details, "shape", None) or [])
    name_by_step = {getattr(i, "step_id", None): getattr(i, "task_name", None) for i in shape}
    parents_by_step: dict[str, list[str]] = {}
    for item in shape:
        for child in getattr(item, "children_step_ids", None) or []:
            parents_by_step.setdefault(child, []).append(getattr(item, "step_id", None))

    def step_name(t: Any) -> str:
        return name_by_step.get(getattr(t, "step_id", None)) or _step_name(t)

    steps = []
    for t in _ordered_tasks(details):
        tid = getattr(t, "task_external_id", None) or t.metadata.id
        steps.append({
            "name": step_name(t),
            "task_id": tid,
            "status": _status_value(t),
            "parents": [name_by_step.get(p, p) for p in parents_by_step.get(getattr(t, "step_id", None), [])],
            "started_at": _iso(getattr(t, "started_at", None)),
            "finished_at": _iso(getattr(t, "finished_at", None)),
            "attempt": getattr(t, "attempt", None),
            "retry_count": getattr(t, "retry_count", None),
            "output": getattr(t, "output", None) or None,
            "error": getattr(t, "error_message", None) or None,
        })
    return {"id": run_id, "status": _status_value(details.run), "steps": steps}


@router.get("/{run_id}/logs")
def get_task_logs(run_id: str, limit_per_step: int = Query(500, ge=1, le=5000)) -> dict:
    """The run's log lines grouped by step — including the
    `overrides[...] applied=[...] unused=[...]` summary each step writes."""
    from src.hatchet_client import hatchet  # lazy

    details = _run_details(run_id)
    steps = []
    for t in _ordered_tasks(details):
        tid = getattr(t, "task_external_id", None) or t.metadata.id
        try:
            rows = list(getattr(hatchet.logs.list(tid, limit=limit_per_step), "rows", None) or [])
        except Exception as e:  # a step with no log store entry must not fail the whole run
            rows, err = [], str(e)
        else:
            err = None
        steps.append({
            "name": _step_name(t),
            "task_id": tid,
            "lines": [
                {
                    "at": _iso(getattr(r, "created_at", None)),
                    "level": getattr(getattr(r, "level", None), "value", getattr(r, "level", None)),
                    "message": getattr(r, "message", ""),
                }
                for r in rows
            ],
            **({"error": err} if err else {}),
        })
    return {"id": run_id, "status": _status_value(details.run), "steps": steps}
