"""claims.link_entities task — vocabulary-grounded per-claim topic/person annotation.

The vocabulary-selection ANNOTATION FAMILY: a single Gemini call over a
batch of claims plus caller-supplied FACETS (named, numbered vocabularies);
returns per-claim, per-facet vocabulary indices. Facets are data — a new
vocabulary-style annotation field costs the caller a facet config and this
service nothing. Annotation families with a different logic shape (claim<->
claim stance, span alignment, per-claim classification) belong in sibling
tasks with their own homogeneous prompts, never in new facets here.

Domain-agnostic BY DESIGN: the task knows nothing about what the claims are
about or which app produced them — the domain arrives entirely through the
inputs. Stateless on Geo (no ids in or out) — callers enforce policies like
"curated only" / "existing only" simply by what they put in each facet's
vocabulary, and map indices back to Geo entities on their side.
"""

import asyncio
from datetime import timedelta

from hatchet_sdk import Context

from src.api.schemas.claims_link_entities_schema import (
    ClaimsLinkEntitiesRequest,
    ClaimsLinkEntitiesResponse,
)
from src.api.services.claims_link_entities_service import link_claim_entities
from src.infrastructure.spend_guard import spend_guard
from src.tasks.base import TaskSpec

_TIMEOUT = timedelta(minutes=3)


async def _handle(
    input: ClaimsLinkEntitiesRequest, ctx: Context
) -> ClaimsLinkEntitiesResponse:
    # Requests whose facets are all empty short-circuit inside the service
    # without an LLM call; only record spend when a call can actually happen.
    if any(f.vocabulary for f in input.facets):
        spend_guard.check_and_record("gemini")
    return await asyncio.to_thread(link_claim_entities, input)


CLAIMS_LINK_ENTITIES_SPEC = TaskSpec(
    name="claims.link_entities",
    input_model=ClaimsLinkEntitiesRequest,
    output_model=ClaimsLinkEntitiesResponse,
    handler=_handle,
    rate_limit_key="gemini_global",
    rate_limit_units=1,
    retries=3,
    execution_timeout=_TIMEOUT,
)
