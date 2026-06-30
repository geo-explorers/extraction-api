"""podcast.export task — durable, single-consumer trigger for Geo publishing.

A thin forwarder: it POSTs to the postgres_to_geo `/api/export` service (which
does the real ~25-minute on-chain publish) and returns its stats. It exists so
the ETL stops holding a 25-min synchronous HTTP connection through Railway's
public edge (which 502s on the idle socket); instead the ETL enqueues + polls,
and the worker makes the long call over Railway PRIVATE networking where there
is no edge proxy to time out.

Two deliberate settings:
  * concurrency=1 — engine-enforced, GLOBAL single-consumer queue. At most one
    export runs at a time; additional triggers QUEUE (not cancelled, not
    rejected). This is the whole "never two exports at once, queue the rest"
    requirement, native to Hatchet.
  * retries=0 — /api/export is not idempotent (a second run can re-publish), so
    a failure must NOT auto-retry. The caller turns a failure into an alert.

The handler is DB-free and wallet-free (it only makes an HTTP call), so it runs
on the existing stateless extraction-worker.
"""

import asyncio
from datetime import timedelta

import requests
from hatchet_sdk import Context

from src.config.settings import settings
from src.api.schemas.podcast_export_schema import (
    PodcastExportRequest,
    PodcastExportResult,
)
from src.tasks.base import TaskSpec
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# 10s to connect, 40min to read — just under the task execution_timeout so the
# HTTP client gives up cleanly before Hatchet kills the step.
_HTTP_TIMEOUT = (10, 2400)


async def _handle(input: PodcastExportRequest, ctx: Context) -> PodcastExportResult:
    url = f"{settings.postgrestogeo_url.rstrip('/')}/api/export"
    headers = {
        "X-API-Key": settings.postgrestogeo_api_key or "",
        "Content-Type": "application/json",
    }
    logger.info(
        f"podcast.export -> POST {url} | {len(input.podcast_name)} podcast(s), "
        f"date_filter={input.date_filter}, limit={input.limit}"
    )
    # Blocking call (~25 min over private networking); offload so the worker
    # event loop stays free for other concurrent task types.
    resp = await asyncio.to_thread(
        requests.post, url, json=input.model_dump(), headers=headers, timeout=_HTTP_TIMEOUT
    )
    resp.raise_for_status()  # non-2xx -> task FAILED (terminal; retries=0)

    body = resp.json()
    data = body.get("data") or {}
    result = PodcastExportResult(
        success=bool(body.get("success", False)),
        episodes_processed=int(data.get("episodes_processed", 0)),
        ops_created=int(data.get("ops_created", 0)),
        duration_ms=int(data.get("duration_ms", 0)),
        message=str(body.get("message", "")),
    )
    logger.info(
        f"podcast.export done: {result.episodes_processed} episodes, "
        f"{result.ops_created} ops, {result.duration_ms}ms"
    )
    return result


PODCAST_EXPORT_SPEC = TaskSpec(
    name="podcast.export",
    input_model=PodcastExportRequest,
    output_model=PodcastExportResult,
    handler=_handle,
    retries=0,  # /api/export is non-idempotent — never auto-retry
    execution_timeout=timedelta(minutes=40),
    concurrency=1,  # single-consumer queue (engine-enforced, global)
)
