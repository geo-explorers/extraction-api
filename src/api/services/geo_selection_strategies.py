"""Candidate-selection strategies for geo.resolve_entities (strategy pattern).

When several graph entities qualify for one name (the duplicate problem), a
SELECTION STRATEGY decides the outcome. Callers pick one per policy:

  team_priority (default)  Apply the team-wide canonical-selection rules that
                           all teams agreed on for deduplication (the same
                           cascade used by the merge tooling and the podcast
                           publisher — geo-merge-topics/src/select_canonical.ts,
                           postgres_to_geo buildPriorityComparator):
                           exclusions first (personal-/dataset-space residents,
                           unless Root-resident or a space's representative
                           topic), then: represents a canonical space > lives
                           in Root > properly placed (not catch-all-only) >
                           featured > scored (≥2 scored eligible → AMBIGUOUS,
                           the escalation rule) > curated > backlinks > data
                           count > older > id. Deterministic floor.

  strict                   Never choose between qualifying candidates: unique
                           match resolves, anything else is `ambiguous`.

The cascade needs per-candidate signals (tags, Score, backlinks, data counts,
representative-topic status); they are fetched lazily — only when a strategy
that needs them actually faces >1 candidate. Signal ids (tags/Score/dataset
spaces/catch-all) are team-wide GRAPH constants, exposed as settings defaults —
not caller domain concepts. Canonical spaces resolve dynamically as the Root
space's subspaces (no hardcoded list).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from src.api.services.hypergraph_client import _post, resolve_canonical_space_ids
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

TAGS_RELATION_TYPE_ID = "257090341ba5406f94e4d4af90042fba"

_META_QUERY = """
query CandidateMeta($id: UUID!, $score: UUID!, $tagsType: UUID!, $marks: [UUID!]) {
  e: entities(filter: { id: { is: $id } }) {
    id name spaceIds createdAt
    score: values(filter: { propertyId: { is: $score } }) { nodes { id } }
    marks: relations(filter: { typeId: { is: $tagsType } toEntityId: { in: $marks } }) {
      nodes { toEntityId }
    }
  }
  bl: relationsConnection(filter: { toEntityId: { is: $id } }) { totalCount }
  vc: valuesConnection(filter: { entityId: { is: $id } }) { totalCount }
  rc: relationsConnection(filter: { fromEntityId: { is: $id } }) { totalCount }
  rep: spaces(filter: { topicId: { is: $id } }) { id }
}
"""

_SPACE_TYPES_QUERY = """
query SpaceTypes($ids: [UUID!]) { spaces(filter: { id: { in: $ids } }) { id type } }
"""


@dataclass
class CandidateMeta:
    """Selection signals for one candidate (merge-tool parity)."""

    geo_id: str
    found: bool = False
    space_ids: List[str] = field(default_factory=list)
    represents_spaces: List[str] = field(default_factory=list)
    created_at: float = float("inf")
    backlink_count: int = 0
    data_count: int = 0
    has_featured: bool = False
    has_curated: bool = False
    scored: bool = False


def fetch_candidate_meta(geo_ids: List[str]) -> Dict[str, CandidateMeta]:
    """Fetch cascade signals per candidate (the merge tool's meta query, ported)."""
    out: Dict[str, CandidateMeta] = {}
    marks = [settings.geo_featured_tag_id, settings.geo_curated_tag_id]
    for gid in dict.fromkeys(geo_ids):
        data = _post(
            _META_QUERY,
            {
                "id": gid,
                "score": settings.geo_score_property_id,
                "tagsType": TAGS_RELATION_TYPE_ID,
                "marks": marks,
            },
        )
        rows = data.get("e") or []
        if not rows:
            out[gid] = CandidateMeta(geo_id=gid)
            continue
        e = rows[0]
        mark_ids = {n["toEntityId"] for n in (e.get("marks") or {}).get("nodes") or []}
        out[gid] = CandidateMeta(
            geo_id=gid,
            found=True,
            space_ids=e.get("spaceIds") or [],
            represents_spaces=[s["id"] for s in data.get("rep") or []],
            created_at=float(e.get("createdAt") or float("inf")),
            backlink_count=int((data.get("bl") or {}).get("totalCount") or 0),
            data_count=int((data.get("vc") or {}).get("totalCount") or 0)
            + int((data.get("rc") or {}).get("totalCount") or 0),
            has_featured=settings.geo_featured_tag_id in mark_ids,
            has_curated=settings.geo_curated_tag_id in mark_ids,
            scored=bool((e.get("score") or {}).get("nodes")),
        )
    return out


def fetch_personal_space_ids(space_ids: List[str]) -> set[str]:
    """Which of these spaces are PERSONAL (queried, never assumed)."""
    unique = [s for s in dict.fromkeys(space_ids) if s]
    if not unique:
        return set()
    data = _post(_SPACE_TYPES_QUERY, {"ids": unique})
    return {
        s["id"] for s in data.get("spaces") or [] if (s.get("type") or "") == "PERSONAL"
    }


# --- Strategies ---------------------------------------------------------------


class SelectionOutcome(Tuple):
    """(status, winner_geo_id | None) — plain tuple alias for readability."""


class SelectionStrategy(Protocol):
    name: str

    def select(self, candidate_ids: List[str]) -> Tuple[str, Optional[str]]:
        """Given >1 qualifying candidate ids (best name-match tier), return
        ("resolved", winner_id) or ("ambiguous", None)."""
        ...


class StrictStrategy:
    """Never guess between qualifying candidates."""

    name = "strict"

    def select(self, candidate_ids: List[str]) -> Tuple[str, Optional[str]]:
        return ("ambiguous", None)


class TeamPriorityStrategy:
    """The team-wide deduplication cascade (see module docstring)."""

    name = "team_priority"

    def __init__(self, meta_fetcher=fetch_candidate_meta, personal_fetcher=fetch_personal_space_ids):
        self._fetch_meta = meta_fetcher
        self._fetch_personal = personal_fetcher

    def select(self, candidate_ids: List[str]) -> Tuple[str, Optional[str]]:
        meta = self._fetch_meta(candidate_ids)
        candidates = [meta[c] for c in candidate_ids if meta[c].found]
        if not candidates:
            return ("ambiguous", None)

        all_spaces = [s for m in candidates for s in m.space_ids]
        personal = self._fetch_personal(all_spaces)
        dataset = set(settings.geo_dataset_space_ids_list)
        root = settings.geo_root_space_id
        try:
            canonical = set(resolve_canonical_space_ids())
        except Exception as e:
            logger.warning(f"canonical-space resolution failed, degrading: {e}")
            canonical = set()
        catchall = settings.geo_catchall_space_id

        def excluded(m: CandidateMeta) -> bool:
            # Personal-/dataset-space residents are ineligible — UNLESS Root-
            # resident or a space's representative topic (the v3 Root-priority
            # rule: unmanaged residencies don't disqualify a canonical).
            if root in m.space_ids or m.represents_spaces:
                return False
            return any(s in personal or s in dataset for s in m.space_ids)

        eligible = [m for m in candidates if not excluded(m)] or candidates

        # Escalation rule: ≥2 scored eligible candidates → human decision.
        if sum(1 for m in eligible if m.scored) >= 2:
            return ("ambiguous", None)

        def sort_key(m: CandidateMeta):
            return (
                0 if any(s in canonical for s in m.represents_spaces) else 1,
                0 if root in m.space_ids else 1,
                0 if [s for s in m.space_ids if s != catchall] else 1,
                0 if m.has_featured else 1,
                0 if m.scored else 1,
                0 if m.has_curated else 1,
                -m.backlink_count,
                -m.data_count,
                m.created_at,
                m.geo_id,
            )

        winner = sorted(eligible, key=sort_key)[0]
        return ("resolved", winner.geo_id)


def get_strategy(name: str) -> SelectionStrategy:
    if name == "strict":
        return StrictStrategy()
    return TeamPriorityStrategy()
