"""claims.link_entities service — one Gemini call, deterministic post-validation.

The LLM's only job is index selection; everything defensible is enforced in
`validate_links_response`: unknown facet keys and indices dropped, duplicates
deduped, per-facet caps applied, and the output canonicalized to exactly one
entry per request claim (in request order) carrying every requested facet key.
Facets with empty vocabularies never reach the model; an all-empty request
short-circuits without an LLM call.

Follows the news_claim_extract_service google-genai pattern (direct SDK for
thinking_level support; app-level retries).
"""

import json
import re
from typing import Iterable, List

from google import genai
from google.genai import types

from src.api.schemas.claims_link_entities_schema import (
    ClaimLink,
    ClaimsLinkEntitiesRequest,
    ClaimsLinkEntitiesResponse,
)
from src.config.prompts.claims_link_entities_prompt import build_user_prompt
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3

# Short task: a few hundred short strings in, a small index map out.
_REQUEST_TIMEOUT_MS = 120_000


def _parse_llm_response(text: str) -> dict:
    """Strip optional markdown fences and parse JSON (news-service convention)."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    return json.loads(cleaned)


def _canonical_indices(raw: object, valid: set[int], cap: int) -> List[int]:
    """Coerce, filter to known vocabulary indices, dedupe preserving order, cap."""
    out: List[int] = []
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return out
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx in valid and idx not in out:
            out.append(idx)
        if len(out) >= cap:
            break
    return out


def _empty_links(request: ClaimsLinkEntitiesRequest) -> List[ClaimLink]:
    keys = [f.key for f in request.facets]
    return [
        ClaimLink(claim_index=c.index, selections={k: [] for k in keys})
        for c in request.claims
    ]


def validate_links_response(
    raw: dict, request: ClaimsLinkEntitiesRequest
) -> List[ClaimLink]:
    """Canonicalize the model output against the request.

    Guarantees: exactly one ClaimLink per request claim, in request order,
    keyed by the claim's own index; every entry carries every requested facet
    key; only that facet's vocabulary indices survive; per-facet caps enforced.
    A malformed or partial model response degrades to empty selections for the
    affected claims/facets rather than failing the task.
    """
    facet_valid = {
        f.key: ({v.index for v in f.vocabulary}, f.max_per_claim)
        for f in request.facets
    }

    by_claim: dict[int, ClaimLink] = {}
    entries = raw.get("links", []) if isinstance(raw, dict) else []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                claim_index = int(entry.get("claim_index"))
            except (TypeError, ValueError):
                continue
            if claim_index in by_claim:  # first entry wins on duplicates
                continue
            raw_selections = entry.get("selections")
            raw_selections = raw_selections if isinstance(raw_selections, dict) else {}
            selections = {
                key: _canonical_indices(raw_selections.get(key), valid, cap)
                for key, (valid, cap) in facet_valid.items()
            }  # unknown facet keys in the response are simply ignored
            by_claim[claim_index] = ClaimLink(
                claim_index=claim_index, selections=selections
            )

    empty = {k: [] for k in facet_valid}
    return [
        by_claim.get(
            c.index, ClaimLink(claim_index=c.index, selections=dict(empty))
        )
        for c in request.claims
    ]


def link_claim_entities(
    request: ClaimsLinkEntitiesRequest,
) -> ClaimsLinkEntitiesResponse:
    """Annotate each claim against every facet vocabulary via one Gemini call."""
    populated = [f for f in request.facets if f.vocabulary]
    # Nothing to select from → deterministic empty annotation, no LLM spend.
    if not populated:
        return ClaimsLinkEntitiesResponse(links=_empty_links(request), model_used="")

    if not settings.gemini_api_key:
        raise Exception("GEMINI_API_KEY not configured for claim entity linking")

    # Only populated facets reach the model; empty ones are restored as empty
    # selections by the validator (which works off the FULL facet list).
    prompt = build_user_prompt(
        claims=request.claims, facets=populated, context=request.context
    )

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
    )
    config_kwargs: dict = {"temperature": settings.claims_link_temperature}
    # Gemini 3+ control; attach only when set so 2.5-era model overrides run clean.
    thinking_level = (settings.claims_link_thinking_level or "").strip()
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level,
        )
    config = types.GenerateContentConfig(**config_kwargs)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=settings.claims_link_model,
                contents=prompt,
                config=config,
            )
            parsed = _parse_llm_response(response.text)
            return ClaimsLinkEntitiesResponse(
                links=validate_links_response(parsed, request),
                model_used=settings.claims_link_model,
            )
        except Exception as e:
            last_error = e
            logger.warning(
                f"Claim entity linking attempt {attempt}/{MAX_RETRIES} failed: {e}"
            )

    raise Exception(
        f"Claim entity linking failed after {MAX_RETRIES} attempts"
    ) from last_error
