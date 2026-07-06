"""Read-side client for the Geo/Hypergraph knowledge graph (GraphQL).

A headless Hatchet worker cannot use the HyperGraph MCP, so it queries the Geo
GraphQL API directly over HTTP. Endpoint + robustness patterns are grounded in
the geo-explorers publishing code (postgres_to_geo), verified live against
https://testnet-api.geobrowser.io/graphql (mainnet: https://api.geobrowser.io/graphql).
Reads are UNAUTHENTICATED.

Design notes:
  * Canonical spaces are the **subspaces of the Geo root space**, fetched
    dynamically via `subspaces(condition:{parentSpaceId})` -> `childSpaceId`, then
    resolved to name/description via `spaces(filter:{id:{in}}){ page {...} }`.
    (No hardcoded list of space ids.)
  * Entities are read with the `entitiesConnection` cursor-paginated query using
    the `typeId`/`spaceId` shortcut args, selecting `types { id name }` directly
    (avoids the silent 100-cap on generic `relations`, see postgres_to_geo PR #13).
  * `_post` mirrors postgres_to_geo's `fetchWithRetry` (PRs #10/#11): 5 retries,
    exponential backoff + jitter, retry on network errors / 429 / 502 / 503 / 504
    and on HTTP-200 responses carrying GraphQL errors with null `data` (a
    transient the indexer emits under load).
  * Bulk fetch mirrors `searchEntities` (PR #14): a descending page-size ladder
    retried on a truncation error, a `totalCount` guard against silent
    under-fetching, and a 200ms inter-page throttle.

Engine-agnostic (no Hatchet import). The blocking calls are offloaded by callers
via ``asyncio.to_thread``.
"""

import random
import time

import requests

from src.api.schemas.geo_spaces_schema import Entity, EntityQuery, Space
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# HTTP + retry policy (mirrors postgres_to_geo fetchWithRetry).
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_RETRIES = 5
_BASE_DELAY = 1.0
_RETRYABLE_STATUS = {429, 502, 503, 504}

# Bulk entity fetch controls (mirrors searchEntities).
_FALLBACK_PAGE_SIZES = [500, 250, 100]
_PAGE_THROTTLE_SECONDS = 0.2
_SAFETY_MAX_ENTITIES = 50_000


_SUBSPACES_QUERY = """
query RootSubspaces($parentSpaceId: UUID!, $first: Int!) {
  subspaces(condition: { parentSpaceId: $parentSpaceId }, first: $first) {
    childSpaceId
  }
}
"""

_SPACES_BY_ID_QUERY = """
query SpacesById($ids: [UUID!]) {
  spaces(filter: { id: { in: $ids } }) {
    id
    page { name description }
  }
}
"""

# entitiesConnection with typeId/spaceId shortcut args + cursor pagination. The
# `filter` is a caller-supplied Postgraphile EntityFilter, passed as a typed
# GraphQL variable (safe, arbitrary — null means no filter). `types { id name }`
# gives human-readable type names for the LLM + the sheet.
_ENTITIES_QUERY = """
query Entities($typeId: UUID, $spaceId: UUID, $first: Int!, $after: Cursor, $filter: EntityFilter) {
  entitiesConnection(typeId: $typeId, spaceId: $spaceId, first: $first, after: $after, filter: $filter) {
    totalCount
    pageInfo { hasNextPage endCursor }
    nodes { id name description spaceIds types { id name } }
  }
}
"""


def _post(query: str, variables: dict) -> dict:
    """POST a GraphQL query with retries; return the ``data`` object.

    Retries transient failures (network errors, 429/502/503/504, and HTTP-200
    responses carrying GraphQL errors with null ``data``) with exponential
    backoff + jitter. GraphQL errors that arrive WITH data are non-transient and
    raised immediately.
    """
    headers = {"Content-Type": "application/json"}
    if settings.hypergraph_api_key:
        headers["Authorization"] = f"Bearer {settings.hypergraph_api_key}"

    last_error: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.post(
                settings.hypergraph_graphql_url,
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            if resp.status_code == 200:
                body = resp.json()
                errors = body.get("errors")
                if errors and body.get("data") is None:
                    last_error = RuntimeError(f"Geo GraphQL transient (null data): {errors}")
                elif errors:
                    raise RuntimeError(f"Geo GraphQL errors: {errors}")
                else:
                    return body.get("data") or {}
            elif resp.status_code in _RETRYABLE_STATUS:
                last_error = RuntimeError(f"Geo GraphQL HTTP {resp.status_code}")
            else:
                resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e

        if attempt < _RETRIES - 1:
            delay = _BASE_DELAY * (2 ** attempt) * (0.5 + random.random() * 0.5)
            logger.warning(
                f"Geo GraphQL retry {attempt + 1}/{_RETRIES - 1} in {delay:.1f}s: {last_error}"
            )
            time.sleep(delay)

    raise last_error or RuntimeError("Geo GraphQL request failed")


# --- Spaces (canonical = subspaces of root) ----------------------------------


def resolve_canonical_space_ids() -> list[str]:
    """The default canonical space ids: an explicit GEO_CANONICAL_SPACE_IDS
    override if set, otherwise the **subspaces of the Geo root space**, fetched
    dynamically (no hardcoded list)."""
    override = settings.geo_canonical_space_ids
    if override:
        return [s.strip() for s in override.split(",") if s.strip()]
    data = _post(
        _SUBSPACES_QUERY,
        {"parentSpaceId": settings.geo_root_space_id, "first": 500},
    )
    ids = [
        r["childSpaceId"]
        for r in (data.get("subspaces") or [])
        if r.get("childSpaceId")
    ]
    logger.info(
        f"canonical spaces = {len(ids)} subspaces of root {settings.geo_root_space_id}"
    )
    return ids


def fetch_spaces(space_ids: list[str] | None = None) -> list[Space]:
    """Fetch spaces (name + description) for the assignment vocabulary. When
    ``space_ids`` is None, the canonical set = subspaces of the Geo root space
    (resolved dynamically)."""
    ids = space_ids if space_ids is not None else resolve_canonical_space_ids()
    if not ids:
        return []
    data = _post(_SPACES_BY_ID_QUERY, {"ids": ids})
    out: list[Space] = []
    for s in data.get("spaces") or []:
        page = s.get("page") or {}
        out.append(
            Space(
                id=str(s["id"]),
                name=page.get("name") or "",
                description=page.get("description") or "",
            )
        )
    logger.info(f"fetch_spaces: {len(out)}/{len(ids)} spaces resolved")
    return out


# --- Entities (cursor-paginated) ---------------------------------------------


def build_entities_variables(
    type_id: str, query: EntityQuery | None = None, after: str | None = None
) -> dict:
    """Build the GraphQL variables for one entities page. Pure (no I/O), so it is
    unit-testable: the same key shape for every ``type_id``."""
    q = query or EntityQuery()
    return {
        "typeId": type_id,
        "spaceId": q.space_id,
        "first": q.page_size,
        "after": after,
        "filter": q.filter,
    }


def _entity_from_node(node: dict, type_id: str) -> Entity:
    types = node.get("types") or []
    # Dedup type names while preserving order (an entity can carry the same type
    # via several relations).
    type_names: list[str] = []
    for t in types:
        n = t.get("name")
        if n and n not in type_names:
            type_names.append(n)
    return Entity(
        id=str(node["id"]),
        name=node.get("name") or "",
        description=node.get("description") or "",
        type=", ".join(type_names),
        type_ids=[t["id"] for t in types if t.get("id")],
        space_ids=node.get("spaceIds") or [],
    )


def _page_size_ladder(start: int) -> list[int]:
    """The starting page size, then the standard smaller fallbacks below it. The
    whole fetch is retried at the next smaller size on a truncation error (the
    indexer is more reliable at smaller pages under load)."""
    return [start] + [s for s in _FALLBACK_PAGE_SIZES if s < start]


def _fetch_entities_once(type_id: str, q: EntityQuery, page_size: int) -> list[Entity]:
    collected: list[Entity] = []
    after: str | None = None
    pages = 0
    while True:
        variables = {
            "typeId": type_id,
            "spaceId": q.space_id,
            "first": page_size,
            "after": after,
            "filter": q.filter,
        }
        conn = _post(_ENTITIES_QUERY, variables).get("entitiesConnection") or {}
        for node in conn.get("nodes") or []:
            collected.append(_entity_from_node(node, type_id))
            if q.max_entities is not None and len(collected) >= q.max_entities:
                logger.info(f"fetch_entities: reached max_entities={q.max_entities}")
                return collected[: q.max_entities]
            if len(collected) >= _SAFETY_MAX_ENTITIES:
                logger.warning(
                    f"fetch_entities: hit safety ceiling {_SAFETY_MAX_ENTITIES} "
                    f"(type_id={type_id}); returning a truncated set"
                )
                return collected
        pages += 1
        page_info = conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            total = conn.get("totalCount")
            # Truncation guard (postgres_to_geo PR #14): a full fetch that stops
            # short of totalCount means the indexer lied about hasNextPage — fail
            # so the caller retries at a smaller page size instead of silently
            # under-fetching.
            if q.max_entities is None and isinstance(total, int) and len(collected) < total:
                raise RuntimeError(
                    f"fetch_entities: silent pagination truncation at "
                    f"{len(collected)}/{total} (type_id={type_id}, page_size={page_size})"
                )
            break
        after = page_info.get("endCursor")
        if not after:
            break
        time.sleep(_PAGE_THROTTLE_SECONDS)
    logger.info(
        f"fetch_entities(type_id={type_id}): {len(collected)} entities across {pages} page(s)"
    )
    return collected


def fetch_entities(type_id: str, query: EntityQuery | None = None) -> list[Entity]:
    """Generic, parameterized entity read with robust cursor pagination.

    Pages through ``entitiesConnection`` (typeId/spaceId shortcut args) until
    complete, retrying transient failures per request and falling back to smaller
    page sizes on a truncation error. ``query.max_entities`` caps the total; None
    fetches all (bounded by a safety ceiling)."""
    q = query or EntityQuery()
    last_error: Exception | None = None
    for page_size in _page_size_ladder(q.page_size):
        try:
            return _fetch_entities_once(type_id, q, page_size)
        except RuntimeError as e:
            last_error = e
            logger.warning(f"fetch_entities: retrying at a smaller page size after: {e}")
    raise last_error or RuntimeError("fetch_entities failed")
