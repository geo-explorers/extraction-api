"""debates.pick_moments — the four clips of a debate's moment reel (GEO-3638).

geo-chat sends the punctuated transcript as sentences, the debate's claims and
its motion; Claude picks four alternating clips that tell the debate's story,
each a run of whole sentences linked to one of its speaker's claims. Every
mechanical rule is checked before the answer returns (a broken answer is sent
back to be fixed), so a failure means no acceptable reel and the caller falls
back to its own picker. DB-free.
"""

from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.debates_pick_moments_schema import DebatesPickMomentsInput, DebatesPickMomentsResult
from src.infrastructure.logger import get_logger
from src.pipeline.claude_json import execution_timeout_s
from src.pipeline.debates_pick_moments_core import ATTEMPTS, TASK, pick_moments
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

DEBATES_PICK_MOMENTS_MAX_PAYLOAD_BYTES = 1024 * 1024


async def _handle(input: DebatesPickMomentsInput, ctx: Context) -> DebatesPickMomentsResult:
    result = await pick_moments(input)
    total = sum(m.end_ms - m.start_ms for m in result.moments) / 1000
    logger.info(f"{TASK}: 4 moments, {total:.1f} s, {result.attempts} answer(s) ({result.model_used})")
    return result


DEBATES_PICK_MOMENTS_SPEC = TaskSpec(
    name=TASK,
    input_model=DebatesPickMomentsInput,
    output_model=DebatesPickMomentsResult,
    handler=_handle,
    # Shares the Claude budget with the news fallback: one run at a time, so a
    # backfill cannot starve news.
    rate_limit_key="claude_global",
    # One unit per Claude call the run can make, as the news tasks reserve.
    rate_limit_units=ATTEMPTS,
    retries=2,
    execution_timeout=timedelta(seconds=execution_timeout_s(ATTEMPTS)),
    max_payload_bytes=DEBATES_PICK_MOMENTS_MAX_PAYLOAD_BYTES,
    concurrency=1,
)
