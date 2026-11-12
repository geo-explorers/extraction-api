"""debates.pick_moments — the four clips of a debate's moment reel (GEO-3638).

geo-chat sends the punctuated transcript as sentences, the debate's claims and
its motion; Claude offers candidate reels of four alternating clips that tell the
debate's story, each clip a run of whole sentences, and a judge picks among those
that pass. By default (link_claims "after", GEO-3692) the picker never sees the
claims: a third call then links each clip to one of its speaker's claims, falling
back to the claim it shares most words with, so a reel is never lost to its links.
link_claims "in_pick" restores the picker that links them itself. Every mechanical rule is checked
before the answer returns (a broken answer is sent back to be fixed), so a failure
means no acceptable reel and the caller falls back to its own picker. DB-free.
"""

from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.debates_pick_moments_schema import DebatesPickMomentsInput, DebatesPickMomentsResult
from src.infrastructure.logger import get_logger
from src.pipeline.claude_json import execution_timeout_s
from src.pipeline.debates_pick_moments_core import CALLS, TASK, pick_moments, reel_ms
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

DEBATES_PICK_MOMENTS_MAX_PAYLOAD_BYTES = 1024 * 1024


async def _handle(input: DebatesPickMomentsInput, ctx: Context) -> DebatesPickMomentsResult:
    result = await pick_moments(input)
    total = reel_ms(result.moments) / 1000
    logger.info(
        f"{TASK}: 4 moments, {total:.1f} s, {result.attempts} answer(s), "
        f"judged from {result.candidates} reel(s), claims linked {input.link_claims} ({result.model_used})"
    )
    return result


DEBATES_PICK_MOMENTS_SPEC = TaskSpec(
    name=TASK,
    input_model=DebatesPickMomentsInput,
    output_model=DebatesPickMomentsResult,
    handler=_handle,
    # Shares the Claude budget with the news fallback. Four runs at a time: one
    # was too slow for a backfill batch (each run is 1-3 min of Claude calls), and
    # four stays far under claude_global's 100 calls/min, so news is not starved.
    rate_limit_key="claude_global",
    # One unit per Claude call the run can make, as the news tasks reserve: 3 answers, 2 judge
    # answers and 2 link answers (GEO-3692).
    rate_limit_units=CALLS,
    retries=2,
    execution_timeout=timedelta(seconds=execution_timeout_s(CALLS)),
    max_payload_bytes=DEBATES_PICK_MOMENTS_MAX_PAYLOAD_BYTES,
    concurrency=4,
)
