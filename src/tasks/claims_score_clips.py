"""claims.score_clips — how does each candidate clip of a debate land as heard?

The clip-level companion to claims.score_highlights (GEO-3614). That task scores
the extracted *claim*; this one judges the *clip*, the words a viewer hears
between two cut points: whether it argues its speaker's side as heard, makes
sense alone, ends after its point, which other clip it answers, and how strongly
it would open a reel compared with the debate's other clips. Same decision
model, provider and model setting as claims.score_highlights, so the two move
together. DB-free: results return to the caller.

All or nothing: a run judges every clip or fails, so the engine retries it and a
consumer never picks a reel from a half-judged pool.
"""

from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.claims_score_clips_schema import ClaimsScoreClipsInput, ClaimsScoreClipsResult
from src.config.overrides import llm
from src.config.settings import settings
from src.decisions import get_decision_model
from src.infrastructure.logger import get_logger
from src.pipeline.claims_score_clips_core import TASK, assemble_result, decide_clips
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

CLAIMS_SCORE_CLIPS_MAX_PAYLOAD_BYTES = 1024 * 1024


async def _handle(input: ClaimsScoreClipsInput, ctx: Context) -> ClaimsScoreClipsResult:
    decider = get_decision_model(settings.claims_highlights_provider)
    model = llm.get("claims_highlights_model")
    decisions = await decide_clips(input, decider, model)
    result = assemble_result(input, decisions, decider.provider, model)
    logger.info(f"{TASK}: {result.clips_scored} clips scored ({decider.provider}/{model})")
    return result


CLAIMS_SCORE_CLIPS_SPEC = TaskSpec(
    name=TASK,
    input_model=ClaimsScoreClipsInput,
    output_model=ClaimsScoreClipsResult,
    handler=_handle,
    # Shares the decision provider's rate limit and request bound with
    # claims.score_highlights, which runs for the same debates.
    rate_limit_key="decisions_global",
    rate_limit_units=1,
    retries=2,
    execution_timeout=timedelta(minutes=10),
    max_payload_bytes=CLAIMS_SCORE_CLIPS_MAX_PAYLOAD_BYTES,
    concurrency=2,
)
