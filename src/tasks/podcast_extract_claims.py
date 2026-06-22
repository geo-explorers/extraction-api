"""Podcast premium claim extraction as a checkpointed Hatchet DAG.

Four tasks — topics -> claims -> takeaways -> finalize — so a redeploy or crash
mid-pipeline resumes at the failed step instead of re-running completed Gemini
calls. The three LLM steps each consume one `gemini_global` rate-limit unit
(matching today's 3 calls/episode). `finalize` returns key takeaways already
resolved to their claim_order, so the caller never re-runs the fragile string
match.

This module is DB-free: it calls the extractor/parser/core helpers directly and
never imports the database-coupled pipeline module. Episode loading and result
persistence are the caller's responsibility (the payload carries the transcript;
pg-migrations persists the result).
"""

from datetime import timedelta
from functools import lru_cache

from pydantic import BaseModel, Field
from hatchet_sdk import Context, RateLimit

from src.hatchet_client import hatchet
from src.config.settings import settings
from src.preprocessing.transcript_parser import TranscriptParser
from src.extraction.premium_claim_extractor import PremiumClaimExtractor
from src.pipeline.premium_extraction_core import (
    build_claim_topics,
    format_topics_with_claims,
    link_takeaways_to_claims,
)
from src.infrastructure.spend_guard import spend_guard
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Transcripts up to ~1M chars (~1MB); 8MB leaves headroom for JSON overhead.
PODCAST_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_GEMINI = [RateLimit(static_key="gemini_global", units=1)]
_STEP_TIMEOUT = timedelta(minutes=5)


# ── Contracts ──────────────────────────────────────────────────────────────


class PodcastExtractInput(BaseModel):
    episode_id: int
    title: str
    description: str = ""
    transcript: str
    transcript_format: str  # "podscribe" | "bankless" | "assembly"


class ResultClaim(BaseModel):
    claim_text: str
    topic: str
    episode_id: int
    claim_order: int
    # Always null in prod (ENABLE_EMBEDDINGS=false); kept so the caller can
    # populate pgvector later without a contract change.
    embedding: list[float] | None = None


class ResultTakeaway(BaseModel):
    text: str
    claim_order: int | None = None


class PodcastExtractResult(BaseModel):
    episode_id: int
    claims: list[ResultClaim] = Field(default_factory=list)
    key_takeaways: list[ResultTakeaway] = Field(default_factory=list)
    topic_of_discussion: list[str] = Field(default_factory=list)
    claims_extracted: int = 0
    model_used: str = ""


# ── Lazy singletons (one per worker process; constructed on first run) ───────


@lru_cache(maxsize=1)
def _extractor() -> PremiumClaimExtractor:
    return PremiumClaimExtractor()


@lru_cache(maxsize=1)
def _parser() -> TranscriptParser:
    return TranscriptParser()


def _full_text(input: PodcastExtractInput) -> str:
    return _parser().parse(input.transcript, format=input.transcript_format).full_text


# ── DAG ──────────────────────────────────────────────────────────────────────

podcast_workflow = hatchet.workflow(
    name="podcast.extract_claims",
    input_validator=PodcastExtractInput,
)


@podcast_workflow.task(
    rate_limits=_GEMINI, execution_timeout=_STEP_TIMEOUT, retries=3, backoff_factor=2.0
)
async def topics(input: PodcastExtractInput, ctx: Context) -> dict:
    spend_guard.check_and_record("gemini")
    result = await _extractor().extract_topics_of_discussion_from_episode(
        title=input.title,
        description=input.description,
        full_transcript=_full_text(input),
    )
    return {"topics": result}


@podcast_workflow.task(
    parents=[topics], rate_limits=_GEMINI, execution_timeout=_STEP_TIMEOUT,
    retries=3, backoff_factor=2.0,
)
async def claims(input: PodcastExtractInput, ctx: Context) -> dict:
    topic_list = ctx.task_output(topics)["topics"]
    if not topic_list:
        return {"claims_with_topics": {}}
    spend_guard.check_and_record("gemini")
    cwt = await _extractor().extract_claims_with_topics_from_transcript(
        full_transcript=_full_text(input),
        topics_of_discussion=topic_list,
    )
    return {"claims_with_topics": cwt}


@podcast_workflow.task(
    parents=[topics, claims], rate_limits=_GEMINI, execution_timeout=_STEP_TIMEOUT,
    retries=3, backoff_factor=2.0,
)
async def takeaways(input: PodcastExtractInput, ctx: Context) -> dict:
    topic_list = ctx.task_output(topics)["topics"]
    cwt = ctx.task_output(claims)["claims_with_topics"]
    _, filtered, claim_topics, _ = build_claim_topics(topic_list, cwt, input.episode_id)
    if not claim_topics:
        return {"key_takeaways": []}
    spend_guard.check_and_record("gemini")
    kt = await _extractor().extract_key_takeaways_from_claims(
        topics_with_claims=format_topics_with_claims(filtered)
    )
    return {"key_takeaways": kt}


@podcast_workflow.task(parents=[topics, claims, takeaways], execution_timeout=timedelta(minutes=2))
async def finalize(input: PodcastExtractInput, ctx: Context) -> PodcastExtractResult:
    topic_list = ctx.task_output(topics)["topics"]
    cwt = ctx.task_output(claims)["claims_with_topics"]
    kt = ctx.task_output(takeaways)["key_takeaways"]

    ordered_topics, _, claim_topics, claims_extracted = build_claim_topics(
        topic_list, cwt, input.episode_id
    )
    takeaway_links = link_takeaways_to_claims(kt, claim_topics)

    logger.info(
        f"finalize episode {input.episode_id}: {len(claim_topics)} claims, "
        f"{len(ordered_topics)} topics, {len(takeaway_links)} takeaways"
    )
    return PodcastExtractResult(
        episode_id=input.episode_id,
        claims=[
            ResultClaim(
                claim_text=c.claim_text,
                topic=c.topic,
                episode_id=c.episode_id,
                claim_order=c.claim_order,
            )
            for c in claim_topics
        ],
        key_takeaways=[
            ResultTakeaway(text=link.text, claim_order=link.claim_order)
            for link in takeaway_links
        ],
        topic_of_discussion=ordered_topics,
        claims_extracted=claims_extracted,
        model_used=settings.gemini_premium_model,
    )
