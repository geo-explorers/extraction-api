"""geo.resolve_entities service — read-only name→entity resolution under policies.

Per unique (name, policy) pair: one name-filtered entitiesConnection query
(optionally tag-restricted), then client-side type filtering, normalized
scoring, and conservative status assignment:

  resolved   exactly one candidate at/above the threshold (or a unique exact
             normalized match)
  ambiguous  several candidates qualify — top candidates returned, caller decides
  not_found  nothing qualifies
  error      the query for this item failed (other items are unaffected)

Query shapes are the PROVEN subset only (verified live 2026-07-22): the name
filter and the Tags-relation filter compose correctly; server-side type
filters (`typeIds`, the typeId shortcut arg combined with `filter`, and
type-via-relations) return wrong/empty results or time out, so the TYPE check
happens client-side against each node's `types`. Matching semantics port
news-worker's resolver: normalized exact match wins, else Levenshtein
similarity against a configurable threshold (default 0.85).

Read-only; never creates entities. Engine-agnostic; blocking (callers offload
via asyncio.to_thread). The candidate-fetch function is injectable for tests.
"""

import re
import time
from typing import Callable, List, Optional

from src.api.schemas.geo_resolve_schema import (
    GeoResolveRequest,
    GeoResolveResponse,
    Resolution,
    ResolveCandidate,
    ResolvePolicy,
)
from src.api.services.geo_selection_strategies import get_strategy
from src.api.services.hypergraph_client import _post
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# The Geo "Tags" relation type id — how marker tags (e.g. curated) attach to
# entities. A protocol-level constant, not a caller domain concept.
TAGS_RELATION_TYPE_ID = "257090341ba5406f94e4d4af90042fba"

_AMBIGUOUS_CANDIDATES = 3
_QUERY_THROTTLE_SECONDS = 0.1

_NAME_QUERY = """
query ResolveByName($filter: EntityFilter, $first: Int!) {
  entitiesConnection(filter: $filter, first: $first) {
    nodes { id name spaceIds types { id name } }
  }
}
"""


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (news-worker semantics)."""
    s = (name or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def name_match_score(query: str, candidate: str) -> float:
    """1.0 on normalized equality, else Levenshtein similarity in [0, 1]."""
    qn, cn = normalize_name(query), normalize_name(candidate)
    if not qn or not cn:
        return 0.0
    if qn == cn:
        return 1.0
    longest = max(len(qn), len(cn))
    return 1.0 - (_levenshtein(qn, cn) / longest)


def fetch_candidates(name: str, restrict_to_tag: Optional[str]) -> list[dict]:
    """Fetch candidate nodes by name: exact case-insensitive first, substring
    fallback only when exact yields nothing. Substring-first fails short names
    ("AI" matches thousands of containing names and the true match can fall
    outside the page — found live 2026-07-22). Optionally restricted to
    entities carrying a Tags relation to `restrict_to_tag`.
    """

    def _query(name_filter: dict) -> list[dict]:
        flt: dict = {"name": name_filter}
        if restrict_to_tag:
            flt["relations"] = {
                "some": {
                    "typeId": {"is": TAGS_RELATION_TYPE_ID},
                    "toEntityId": {"is": restrict_to_tag},
                }
            }
        data = _post(
            _NAME_QUERY, {"filter": flt, "first": settings.geo_resolve_max_candidates}
        )
        return (data.get("entitiesConnection") or {}).get("nodes") or []

    # Exact tier: indexed `in` over case variants (isInsensitive forces an
    # unindexed lower(name) scan — 45s+ timeouts, measured live 2026-07-22).
    variants = list(dict.fromkeys(
        [name, name.lower(), name.upper(), name.title(),
         name[:1].upper() + name[1:].lower() if name else name]
    ))
    exact = _query({"in": variants})
    if exact:
        return exact
    return _query({"includesInsensitive": name})


def _resolve_one(
    name: str,
    policy: ResolvePolicy,
    fetch: Callable[[str, Optional[str]], list[dict]],
    strategy_factory: Callable = get_strategy,
) -> Resolution:
    threshold = (
        policy.match_threshold
        if policy.match_threshold is not None
        else settings.geo_resolve_match_threshold
    )
    nodes = fetch(name, policy.restrict_to_tag)

    scored: List[ResolveCandidate] = []
    for node in nodes:
        type_ids = [t.get("id") for t in (node.get("types") or [])]
        if policy.type_id not in type_ids:
            continue  # client-side type gate (server-side filters unreliable)
        score = name_match_score(name, node.get("name") or "")
        scored.append(
            ResolveCandidate(
                geo_id=str(node["id"]),
                name=node.get("name") or "",
                score=round(score, 4),
                space_ids=node.get("spaceIds") or [],
            )
        )
    scored.sort(key=lambda c: -c.score)

    # The cascade breaks ties WITHIN the best name-match tier: exact matches
    # when any exist, else everything at/above the threshold. A featured fuzzy
    # match must never beat an exact match.
    exact = [c for c in scored if c.score >= 1.0]
    qualifying = [c for c in scored if c.score >= threshold]
    pool = exact if exact else qualifying

    if not pool:
        return Resolution(index=-1, status="not_found")
    if len(pool) == 1:
        best = pool[0]
        return Resolution(
            index=-1, status="resolved", geo_id=best.geo_id,
            matched_name=best.name, score=best.score,
        )

    status, winner_id = strategy_factory(policy.selection).select(
        [c.geo_id for c in pool]
    )
    if status == "resolved" and winner_id:
        best = next(c for c in pool if c.geo_id == winner_id)
        return Resolution(
            index=-1, status="resolved", geo_id=best.geo_id,
            matched_name=best.name, score=best.score,
        )
    return Resolution(
        index=-1, status="ambiguous", candidates=pool[:_AMBIGUOUS_CANDIDATES]
    )


def resolve_entities(
    request: GeoResolveRequest,
    fetch: Callable[[str, Optional[str]], list[dict]] = fetch_candidates,
    strategy_factory: Callable = get_strategy,
) -> GeoResolveResponse:
    """Resolve every item; one query per unique (normalized name, policy) pair."""
    cache: dict[tuple[str, str], Resolution] = {}
    resolutions: List[Resolution] = []
    for item in request.items:
        key = (normalize_name(item.name), item.policy)
        if key not in cache:
            policy = request.policies[item.policy]
            try:
                cache[key] = _resolve_one(item.name, policy, fetch, strategy_factory)
            except Exception as e:  # per-item isolation: one bad name ≠ failed batch
                logger.warning(f"geo_resolve: query failed for '{item.name}': {e}")
                cache[key] = Resolution(
                    index=-1, status="error", error=str(e)[:200]
                )
            if fetch is fetch_candidates:
                time.sleep(_QUERY_THROTTLE_SECONDS)
        resolutions.append(cache[key].model_copy(update={"index": item.index}))

    counts: dict[str, int] = {}
    for r in resolutions:
        counts[r.status] = counts.get(r.status, 0) + 1
    logger.info(f"geo_resolve: {len(resolutions)} items → {counts}")
    return GeoResolveResponse(resolutions=resolutions)
