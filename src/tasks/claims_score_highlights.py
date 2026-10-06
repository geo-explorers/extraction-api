"""claims.score_highlights — which of these extracted claims carry the discussion?

A pure scorer: the discussion and its extracted claims in, a highlight score
per claim out. One decision-model request per claim, each with the whole
discussion in view. The model sits behind the DecisionModel interface
(src/decisions), chosen by CLAIMS_HIGHLIGHTS_PROVIDER, so it can be swapped
without touching this task. DB-free: results return to the caller.
"""

from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.claims_score_highlights_schema import (
    ClaimsScoreHighlightsInput,
    ClaimsScoreHighlightsResult,
)
from src.config.overrides import llm
from src.config.settings import settings
from src.decisions import DecisionError, get_decision_model
from src.infrastructure.logger import get_logger
from src.pipeline.claims_score_highlights_core import assemble_result, decide_claims
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

CLAIMS_SCORE_HIGHLIGHTS_MAX_PAYLOAD_BYTES = 1024 * 1024


async def _handle(input: ClaimsScoreHighlightsInput, ctx: Context) -> ClaimsScoreHighlightsResult:
    decider = get_decision_model(settings.claims_highlights_provider)
    model = llm.get("claims_highlights_model")
    outcomes = await decide_claims(input, decider, model)
    result = assemble_result(input, outcomes, decider.provider, model)
    if result.claims_scored == 0:
        # Nothing answered: a bad key, a bad model name, an outage. Fail the
        # run so the engine retries it instead of returning a page of nulls.
        raise next(o for o in outcomes if isinstance(o, DecisionError))
    logger.info(
        f"claims.score_highlights: {result.claims_scored}/{len(input.claims)} claims scored "
        f"({decider.provider}/{model})"
    )
    return result


CLAIMS_SCORE_HIGHLIGHTS_SPEC = TaskSpec(
    name="claims.score_highlights",
    input_model=ClaimsScoreHighlightsInput,
    output_model=ClaimsScoreHighlightsResult,
    handler=_handle,
    rate_limit_key="decisions_global",
    rate_limit_units=1,
    retries=2,
    execution_timeout=timedelta(minutes=10),
    max_payload_bytes=CLAIMS_SCORE_HIGHLIGHTS_MAX_PAYLOAD_BYTES,
)
