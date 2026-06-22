"""Task enqueue/status facade.

The HTTP contract TypeScript consumers use to run tasks on the Hatchet plane
without a Hatchet SDK:

    POST /tasks            -> 201 {id, type, status:"queued"}
    GET  /tasks/{run_id}   -> {id, status, result?, error?}

Hatchet imports are done LAZILY inside the handlers so a missing
HATCHET_CLIENT_TOKEN degrades only these endpoints, never the whole API (which
also serves the sync extraction endpoints).
"""

import json

from fastapi import APIRouter, HTTPException, status
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
def enqueue_task(req: EnqueueRequest) -> dict:
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
    logger.info(f"Enqueued {req.type} -> run {ref.workflow_run_id}")
    return {"id": ref.workflow_run_id, "type": req.type, "status": "queued"}


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
