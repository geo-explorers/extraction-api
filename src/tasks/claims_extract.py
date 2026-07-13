"""Generalized claim extraction as a checkpointed Hatchet DAG (claims.extract).

One pipeline for any text-based media — debate transcripts, research papers,
news articles, podcast transcripts, talks, generic documents — parameterized
by media_type presets, structured knobs, and fenced caller instructions
(see ClaimsExtractInput).

Four tasks — topics -> extract -> takeaways -> finalize — so a redeploy or
crash resumes at the failed step without re-billing completed Gemini calls.
Branching uses the early-return pattern (not skip_if): grouping=false makes
`topics` return an empty list with NO LLM call, and include_takeaways=false
does the same in `takeaways`. Typical cost is therefore 2 Gemini calls, 1 for
flat extraction without takeaways, 3 maximum.

This module is DB-free: results are returned to the caller, who persists.
"""

from datetime import timedelta
from functools import lru_cache

from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet
from src.config.settings import settings
from src.api.schemas.claims_extract_schema import (
    ClaimsExtractInput,
    ClaimsExtractResult,
)
from src.extraction.claims_extractor import ClaimsExtractor
from src.extraction.claims_prompt_builder import (
    build_extract_prompt,
    build_takeaways_prompt,
    build_topics_prompt,
)
from src.pipeline.claims_extract_core import assemble_result
from src.infrastructure.spend_guard import spend_guard
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Transcript-scale inputs (multi-hour debates); matches the podcast cap.
CLAIMS_EXTRACT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
_STEP_TIMEOUT = timedelta(minutes=5)
# The fused call emits claims + groups + quotes + summary over dense material.
_EXTRACT_TIMEOUT = timedelta(minutes=10)


@lru_cache(maxsize=1)
def _extractor() -> ClaimsExtractor:
    return ClaimsExtractor()


# ── DAG ──────────────────────────────────────────────────────────────────────

claims_workflow = hatchet.workflow(
    name="claims.extract",
    input_validator=ClaimsExtractInput,
)


@claims_workflow.task(
    rate_limits=_GEMINI, execution_timeout=_STEP_TIMEOUT, retries=3, backoff_factor=2.0
)
async def topics(input: ClaimsExtractInput, ctx: Context) -> dict:
    if not input.grouping:
        return {"topics": []}
    spend_guard.check_and_record("gemini")
    result = await _extractor().extract_topics(build_topics_prompt(input))
    return {"topics": result}


@claims_workflow.task(
    parents=[topics], rate_limits=_GEMINI, execution_timeout=_EXTRACT_TIMEOUT,
    retries=3, backoff_factor=2.0,
)
async def extract(input: ClaimsExtractInput, ctx: Context) -> dict:
    topic_list = ctx.task_output(topics)["topics"]
    spend_guard.check_and_record("gemini")
    extraction = await _extractor().extract_claims(
        build_extract_prompt(input, topic_list)
    )
    return {"extraction": extraction.model_dump()}


@claims_workflow.task(
    parents=[extract], rate_limits=_GEMINI, execution_timeout=_STEP_TIMEOUT,
    retries=3, backoff_factor=2.0,
)
async def takeaways(input: ClaimsExtractInput, ctx: Context) -> dict:
    if not input.include_takeaways:
        return {"takeaways": []}
    claims = ctx.task_output(extract)["extraction"]["claims"]
    if not claims:
        return {"takeaways": []}
    spend_guard.check_and_record("gemini")
    result = await _extractor().extract_takeaways(
        build_takeaways_prompt(input, claims)
    )
    return {"takeaways": result}


@claims_workflow.task(
    parents=[extract, takeaways], execution_timeout=timedelta(minutes=2)
)
async def finalize(input: ClaimsExtractInput, ctx: Context) -> ClaimsExtractResult:
    extraction = ctx.task_output(extract)["extraction"]
    takeaway_texts = ctx.task_output(takeaways)["takeaways"]

    result = assemble_result(
        input,
        extraction,
        takeaway_texts,
        model_used=settings.claims_extract_model,
    )
    logger.info(
        f"finalize claims.extract ({input.media_type}, grouping={input.grouping}): "
        f"{result.claims_extracted} claims, {len(result.groups)} groups, "
        f"{len(result.quotes)} quotes, {len(result.takeaways)} takeaways"
    )
    return result
