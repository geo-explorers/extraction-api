"""transcript.punctuate — where does each sentence of an automatic transcript end? (GEO-3638)

geo-chat sends a debate's turns as Whisper heard them and gets back each
sentence's last word and mark. It never gets text back, so no word can change.
Claude writes the answer; DB-free, results return to the caller.
"""

from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.transcript_punctuate_schema import TranscriptPunctuateInput, TranscriptPunctuateResult
from src.infrastructure.logger import get_logger
from src.pipeline.transcript_punctuate_core import TASK, punctuate
from src.tasks.base import TaskSpec

logger = get_logger(__name__)

TRANSCRIPT_PUNCTUATE_MAX_PAYLOAD_BYTES = 1024 * 1024


async def _handle(input: TranscriptPunctuateInput, ctx: Context) -> TranscriptPunctuateResult:
    result = await punctuate(input)
    words = sum(len(t.words) for t in input.turns)
    logger.info(f"{TASK}: {result.sentences} sentences over {words} words ({result.model_used})")
    return result


TRANSCRIPT_PUNCTUATE_SPEC = TaskSpec(
    name=TASK,
    input_model=TranscriptPunctuateInput,
    output_model=TranscriptPunctuateResult,
    handler=_handle,
    # The news Claude fallback shares this budget, so a backfill burst must not
    # starve it: one run at a time.
    rate_limit_key="claude_global",
    rate_limit_units=1,
    retries=2,
    execution_timeout=timedelta(minutes=5),
    max_payload_bytes=TRANSCRIPT_PUNCTUATE_MAX_PAYLOAD_BYTES,
    concurrency=1,
)
