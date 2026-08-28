"""DB-free deterministic post-processing for the claims.extract DAG.

Pure transforms from the raw LLM extraction to the public result contract —
no LLM, database, or Hatchet dependencies, so the fiddliest logic in the
pipeline (index remapping after filtering) is fully unit-testable.

Fixed order inside assemble_result (changing it corrupts indices):
sanitize rows -> strip unrequested sections -> min_confidence filter ->
max_claims truncation -> remap quote/group indices -> drop small groups ->
derive topics -> link takeaways.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.api.schemas.claims_extract_schema import (
    ClaimsExtractInput,
    ClaimsExtractResult,
    ClaimGroup,
    ExtractedClaimOut,
    ExtractedQuoteOut,
    TakeawayOut,
)
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Groups with fewer claims than this are dropped (the news prompt's HARD RULE:
# no collection with fewer than 2 claims).
MIN_CLAIMS_PER_GROUP = 2

# A debate's title is its motion — a claim that already exists and is linked to
# the debate. Re-extracting it would publish a duplicate Claim entity, so
# claims that merely restate the title are dropped. Token-set overlap at or
# above this ratio counts as a restatement ("... is an effective tool for
# mental health support." vs the same words without the period).
TITLE_RESTATEMENT_MIN_OVERLAP = 0.85

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def restates_title(claim_text: str, title: Optional[str]) -> bool:
    """True when the claim is the title (or a near-verbatim rewording of it):
    Jaccard overlap of word sets >= TITLE_RESTATEMENT_MIN_OVERLAP. Anything
    that adds a qualifier ("...does not apply to everyone") keeps enough new
    words to fall below the bar and survives."""
    if not title:
        return False
    a, b = _tokens(claim_text), _tokens(title)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= TITLE_RESTATEMENT_MIN_OVERLAP


def sanitize_claims(raw_claims: List[dict], num_documents: int) -> List[ExtractedClaimOut]:
    """Coerce raw LLM claim rows into the public model, dropping junk rows and
    out-of-range document indices rather than failing the whole run."""
    claims: List[ExtractedClaimOut] = []
    for row in raw_claims:
        try:
            claim = ExtractedClaimOut.model_validate(row)
        except Exception:
            logger.warning(f"Dropping malformed claim row: {row!r}")
            continue
        if not claim.text.strip():
            continue
        claim.document_indices = [
            i for i in claim.document_indices if 0 <= i < num_documents
        ]
        claims.append(claim)
    return claims


def filter_and_reindex_claims(
    claims: List[ExtractedClaimOut],
    *,
    min_confidence: float,
    max_claims: Optional[int],
    exclude_title: Optional[str] = None,
) -> Tuple[List[ExtractedClaimOut], Dict[int, int]]:
    """Apply the confidence floor, the title-restatement guard, then the claim
    budget, preserving narrative order. Returns the surviving claims and an
    old-index -> new-index map for remapping quotes and groups."""
    kept: List[ExtractedClaimOut] = []
    index_map: Dict[int, int] = {}
    for old_index, claim in enumerate(claims):
        if claim.confidence < min_confidence:
            continue
        if restates_title(claim.text, exclude_title):
            logger.info(f"Dropping claim that restates the title: {claim.text!r}")
            continue
        if max_claims is not None and len(kept) >= max_claims:
            break
        index_map[old_index] = len(kept)
        kept.append(claim)
    return kept, index_map


def remap_quotes(
    raw_quotes: List[dict],
    index_map: Dict[int, int],
    num_documents: int,
) -> List[ExtractedQuoteOut]:
    """Rewrite quote claim_index values to the final claims array; quotes
    referencing dropped or out-of-range claims are dropped with them."""
    quotes: List[ExtractedQuoteOut] = []
    for row in raw_quotes:
        try:
            quote = ExtractedQuoteOut.model_validate(row)
        except Exception:
            logger.warning(f"Dropping malformed quote row: {row!r}")
            continue
        if quote.claim_index not in index_map:
            continue
        quote.claim_index = index_map[quote.claim_index]
        if quote.document_index is not None and not (
            0 <= quote.document_index < num_documents
        ):
            quote.document_index = None
        quotes.append(quote)
    return quotes


def remap_groups(
    raw_groups: List[dict],
    index_map: Dict[int, int],
) -> List[ClaimGroup]:
    """Rewrite group claim_indices to the final claims array, dropping
    references to filtered claims (small groups are dropped separately)."""
    groups: List[ClaimGroup] = []
    for row in raw_groups:
        try:
            group = ClaimGroup.model_validate(row)
        except Exception:
            logger.warning(f"Dropping malformed group row: {row!r}")
            continue
        group.claim_indices = [
            index_map[i] for i in group.claim_indices if i in index_map
        ]
        groups.append(group)
    return groups


def drop_small_groups(groups: List[ClaimGroup]) -> List[ClaimGroup]:
    return [g for g in groups if len(g.claim_indices) >= MIN_CLAIMS_PER_GROUP]


def synthesize_groups_from_topics(claims: List[ExtractedClaimOut]) -> List[ClaimGroup]:
    """Fallback when grouping was requested but the model emitted no usable
    groups: rebuild them from the claims' own topic labels in first-seen order
    (the podcast pipeline's grouping model)."""
    indices_by_topic: Dict[str, List[int]] = {}
    for i, claim in enumerate(claims):
        topic = (claim.topic or "").strip()
        if topic:
            indices_by_topic.setdefault(topic, []).append(i)
    return [
        ClaimGroup(name=topic, claim_indices=indices)
        for topic, indices in indices_by_topic.items()
    ]


def derive_topics(groups: List[ClaimGroup], claims: List[ExtractedClaimOut]) -> List[str]:
    """The authoritative topic list: surviving group names in output order,
    falling back to distinct claim topics in first-seen order (the fused news
    DAG's _derive_topics pattern)."""
    if groups:
        return [g.name for g in groups]
    seen: List[str] = []
    for claim in claims:
        topic = (claim.topic or "").strip()
        if topic and topic not in seen:
            seen.append(topic)
    return seen


def link_takeaways_by_text(
    takeaways: List[str],
    claims: List[ExtractedClaimOut],
) -> List[TakeawayOut]:
    """Resolve each takeaway to the index of the claim it restates by exact
    text match (done here, in-process, so consumers never re-run the fragile
    string match). Unmatched -> claim_index=None."""
    index_by_text = {c.text: i for i, c in enumerate(claims)}
    return [
        TakeawayOut(text=t, claim_index=index_by_text.get(t)) for t in takeaways
    ]


def assemble_result(
    input: ClaimsExtractInput,
    extraction: dict,
    takeaways: List[str],
    model_used: str,
) -> ClaimsExtractResult:
    """Deterministic assembly of the public result from the raw LLM extraction
    dict ({claims, groups, quotes, summary}) and the takeaway texts."""
    num_documents = len(input.documents)

    claims = sanitize_claims(extraction.get("claims", []), num_documents)

    # Strip unrequested sections even if the model emitted them.
    raw_groups = extraction.get("groups", []) if input.grouping else []
    raw_quotes = extraction.get("quotes", []) if input.include_quotes else []
    summary = extraction.get("summary", "") if input.include_summary else ""
    if not input.grouping:
        for claim in claims:
            claim.topic = None
    if not input.classify_factuality:
        for claim in claims:
            claim.is_factual = None

    # Only debates carry a motion as the title; a news headline or episode
    # title is not a claim that already exists elsewhere.
    claims, index_map = filter_and_reindex_claims(
        claims,
        min_confidence=input.min_confidence,
        max_claims=input.max_claims,
        exclude_title=input.title if input.media_type == "debate" else None,
    )

    quotes = remap_quotes(raw_quotes, index_map, num_documents)
    groups = drop_small_groups(remap_groups(raw_groups, index_map))
    if input.grouping and not groups:
        groups = drop_small_groups(synthesize_groups_from_topics(claims))

    topics = derive_topics(groups, claims) if input.grouping else []
    takeaway_links = (
        link_takeaways_by_text(takeaways, claims) if input.include_takeaways else []
    )

    return ClaimsExtractResult(
        media_type=input.media_type,
        grouping=input.grouping,
        topics=topics,
        claims=claims,
        groups=groups,
        quotes=quotes,
        takeaways=takeaway_links,
        summary=summary or "",
        document_ids=[d.id for d in input.documents],
        claims_extracted=len(claims),
        model_used=model_used,
    )
