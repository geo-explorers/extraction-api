"""Read-side client for the Geo/Hypergraph knowledge graph (GraphQL).

A headless Hatchet worker cannot use the HyperGraph MCP, so it queries the Geo
GraphQL API directly over HTTP. Endpoint + query shapes match the geo-explorers
reference code (postgres_to_geo, spreadsheet-to-geo, geo_tech_demo):

  * Endpoint: https://testnet-api.geobrowser.io/graphql (mainnet:
    https://api.geobrowser.io/graphql). Reads are UNAUTHENTICATED.
  * A space's human name/description live on ``space(id){ page { name description } }``.
    There is no "list all spaces" query, so canonical spaces are resolved by id
    (aliased into a single request).
  * Entities are read with the ``entities(typeId, spaceId, first)`` list query,
    which returns ``id name description spaceIds types { id name }``. The type
    filter is a Geo *type entity id* (32-hex), not a type name.

Two generic reads:
  * fetch_spaces   — canonical spaces (+ detail) for the assignment vocabulary
  * fetch_entities — entities of a Geo type id, parameterized by type_id + query

Engine-agnostic (no Hatchet import). The ``requests`` calls are blocking, so
callers on an event loop offload via ``asyncio.to_thread``.
"""

import requests

from src.api.schemas.geo_spaces_schema import Entity, EntityQuery, Space
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_HTTP_TIMEOUT = (10, 120)  # (connect, read) seconds

# Default canonical spaces (Geo space ids, 32-hex) used when the caller passes
# none and GEO_CANONICAL_SPACE_IDS is unset. Sourced from the live Geo graph.
_DEFAULT_CANONICAL_SPACE_IDS = [
    "a19c345ab9866679b001d7d2138d88a1",  # Geo
    "41e851610e13a19441c4d980f2f2ce6b",  # AI
    "c9f267dcb0d270718c2a3c45a64afd32",  # Crypto
    "52c7ae149838b6d47ce0f3b2a5974546",  # Health
    "d69608290513c2a91102c939b3265bd7",  # Industries
    "870e3b3068661e6280fad2ab456829bc",  # Technology
    "9b611b848b12491b9b6b43f3cf019b8b",  # Software
    "720eb279c64d56735dccd17a2a416ba2",  # Geo Education
    "5d3e53b46f2dd38caa231ccc763212f5",  # Healthcare
    "b5a31f8182b042437ede0f84ee02f104",  # Podcast App
]

# The `entities` list query with the top-level typeId/spaceId shortcut args.
# `types { id name }` gives human-readable type names for the LLM + the sheet.
_ENTITIES_QUERY = """
query Entities($typeId: UUID, $spaceId: UUID, $first: Int!) {
  entities(typeId: $typeId, spaceId: $spaceId, first: $first) {
    id
    name
    description
    spaceIds
    types { id name }
  }
}
"""


def _post(query: str, variables: dict) -> dict:
    """POST a GraphQL query and return the ``data`` object, raising on transport
    or GraphQL-level errors. Reads are unauthenticated; a key is sent only if one
    is explicitly configured."""
    headers = {"Content-Type": "application/json"}
    if settings.hypergraph_api_key:
        headers["Authorization"] = f"Bearer {settings.hypergraph_api_key}"
    resp = requests.post(
        settings.hypergraph_graphql_url,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"Geo GraphQL errors: {body['errors']}")
    return body.get("data") or {}


def canonical_space_ids() -> list[str]:
    """The default set of canonical space ids: GEO_CANONICAL_SPACE_IDS
    (comma-separated) when set, else the built-in default set."""
    raw = settings.geo_canonical_space_ids
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return list(_DEFAULT_CANONICAL_SPACE_IDS)


def build_spaces_query(ids: list[str]) -> str:
    """Build a single request that aliases one ``space(id)`` selection per id
    (there is no list-all-spaces query on the Geo API). Pure — no I/O."""
    parts = [
        f'  s{i}: space(id: "{sid}") {{ id page {{ name description }} }}'
        for i, sid in enumerate(ids)
    ]
    return "query Spaces {\n" + "\n".join(parts) + "\n}"


def fetch_spaces(space_ids: list[str] | None = None) -> list[Space]:
    """Fetch canonical spaces (name + description from each space's ``page``).
    When ``space_ids`` is None, the configured canonical set is used."""
    ids = space_ids or canonical_space_ids()
    if not ids:
        return []
    data = _post(build_spaces_query(ids), {})
    out: list[Space] = []
    for i, sid in enumerate(ids):
        node = data.get(f"s{i}")
        if not node:
            continue  # unknown / missing space id
        page = node.get("page") or {}
        out.append(
            Space(
                id=str(node.get("id") or sid),
                name=page.get("name") or "",
                description=page.get("description") or "",
            )
        )
    logger.info(f"fetch_spaces: {len(out)}/{len(ids)} spaces resolved")
    return out


def build_entities_variables(type_id: str, query: EntityQuery | None = None) -> dict:
    """Build the GraphQL variables for an entity read. Pure (no I/O), so it is
    unit-testable: the same key shape for every ``type_id``."""
    q = query or EntityQuery()
    return {"typeId": type_id, "spaceId": q.space_id, "first": q.limit}


def fetch_entities(type_id: str, query: EntityQuery | None = None) -> list[Entity]:
    """Generic, parameterized entity read. ``type_id`` (a Geo type entity id)
    selects the type; ``query`` carries the space filter + limit. One query serves
    every type — no per-type code."""
    variables = build_entities_variables(type_id, query)
    data = _post(_ENTITIES_QUERY, variables)
    out: list[Entity] = []
    for e in data.get("entities") or []:
        types = e.get("types") or []
        # Dedup type names while preserving order (an entity can carry the same
        # type via several relations, e.g. "Project" twice).
        type_names: list[str] = []
        for t in types:
            n = t.get("name")
            if n and n not in type_names:
                type_names.append(n)
        out.append(
            Entity(
                id=str(e["id"]),
                name=e.get("name") or "",
                description=e.get("description") or "",
                type=", ".join(type_names),
                type_ids=[t["id"] for t in types if t.get("id")],
                space_ids=e.get("spaceIds") or [],
            )
        )
    logger.info(f"fetch_entities(type_id={type_id}): {len(out)} entities")
    return out
