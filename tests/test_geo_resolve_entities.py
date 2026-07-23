"""Unit tests for geo.resolve_entities (network-free).

Exercises scoring/normalization, the status decision ladder (unique exact,
same-name twins → ambiguous, unique fuzzy, multi-fuzzy → ambiguous,
not_found), client-side type gating, per-item error isolation, request-level
dedupe, threshold overrides, and schema constraints — via an injected fake
candidate fetcher.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas.geo_resolve_schema import (
    GeoResolveRequest,
    ResolveItem,
    ResolvePolicy,
)
from src.api.services.geo_resolve_service import (
    name_match_score,
    normalize_name,
    resolve_entities,
)

TOPIC_TYPE = "5ef5a5860f274d8e8f6c59ae5b3e89e2"  # real Geo Topic type id
OTHER_TYPE = "00000000000000000000000000000001"  # any non-matching type
CURATED_TAG = "7f796eb5bfc5449c98649bf7d996a2ca"


def _node(gid: str, name: str, type_id: str = TOPIC_TYPE, spaces=("s1",)) -> dict:
    return {
        "id": gid,
        "name": name,
        "spaceIds": list(spaces),
        "types": [{"id": type_id, "name": "T"}],
    }


def _request(items, policies=None) -> GeoResolveRequest:
    return GeoResolveRequest(
        items=items,
        policies=policies
        or {"topic": ResolvePolicy(type_id=TOPIC_TYPE, restrict_to_tag=CURATED_TAG)},
    )


# ── scoring & normalization ───────────────────────────────────────────────


def test_normalize_strips_punctuation_and_case():
    assert normalize_name("  U.S. Politics! ") == "u s politics"


def test_score_exact_and_fuzzy():
    assert name_match_score("Bitcoin", "bitcoin") == 1.0
    assert name_match_score("Bitcoin", "Bitcoin ETF") < 1.0
    assert name_match_score("", "x") == 0.0


# ── status ladder ─────────────────────────────────────────────────────────


def test_unique_exact_match_resolves():
    fake = lambda name, tag: [_node("g1", "Bitcoin"), _node("g2", "Bitcoin ETF")]
    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")]), fetch=fake
    )
    r = out.resolutions[0]
    assert (r.status, r.geo_id, r.score) == ("resolved", "g1", 1.0)


def test_strict_same_name_twins_are_ambiguous_never_guessed():
    fake = lambda name, tag: [_node("g1", "Bitcoin"), _node("g2", "bitcoin")]
    strict = {"topic": ResolvePolicy(type_id=TOPIC_TYPE, selection="strict")}
    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")], strict),
        fetch=fake,
    )
    r = out.resolutions[0]
    assert r.status == "ambiguous"
    assert {c.geo_id for c in r.candidates} == {"g1", "g2"}
    assert r.geo_id is None


def test_unique_fuzzy_above_threshold_resolves():
    fake = lambda name, tag: [_node("g1", "US Politic")]  # 1 edit away
    out = resolve_entities(
        _request([ResolveItem(index=0, name="US Politics", policy="topic")]), fetch=fake
    )
    r = out.resolutions[0]
    assert r.status == "resolved" and r.geo_id == "g1" and r.score < 1.0


def test_below_threshold_is_not_found():
    fake = lambda name, tag: [_node("g1", "Completely Different")]
    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")]), fetch=fake
    )
    assert out.resolutions[0].status == "not_found"


def test_wrong_type_candidates_are_gated_out():
    fake = lambda name, tag: [_node("g1", "Bitcoin", type_id=OTHER_TYPE)]
    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")]), fetch=fake
    )
    assert out.resolutions[0].status == "not_found"


def test_threshold_override_per_policy():
    fake = lambda name, tag: [_node("g1", "Bitcoink")]
    strict = {"topic": ResolvePolicy(type_id=TOPIC_TYPE, match_threshold=0.99)}
    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")], strict),
        fetch=fake,
    )
    assert out.resolutions[0].status == "not_found"


# ── batch behaviors ───────────────────────────────────────────────────────


def test_dedupe_one_fetch_per_unique_name_policy():
    calls = []

    def fake(name, tag):
        calls.append(name)
        return [_node("g1", "Bitcoin")]

    items = [
        ResolveItem(index=0, name="Bitcoin", policy="topic"),
        ResolveItem(index=1, name="  bitcoin ", policy="topic"),  # same normalized
        ResolveItem(index=7, name="Bitcoin", policy="topic"),
    ]
    out = resolve_entities(_request(items), fetch=fake)
    assert len(calls) == 1
    assert [r.index for r in out.resolutions] == [0, 1, 7]
    assert all(r.geo_id == "g1" for r in out.resolutions)


def test_per_item_error_isolation():
    def fake(name, tag):
        if name == "Boom":
            raise RuntimeError("indexer sneezed")
        return [_node("g1", name)]

    items = [
        ResolveItem(index=0, name="Boom", policy="topic"),
        ResolveItem(index=1, name="Bitcoin", policy="topic"),
    ]
    out = resolve_entities(_request(items), fetch=fake)
    assert out.resolutions[0].status == "error"
    assert "sneezed" in out.resolutions[0].error
    assert out.resolutions[1].status == "resolved"


def test_tag_restriction_passed_to_fetcher():
    seen = []

    def fake(name, tag):
        seen.append(tag)
        return []

    resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")]), fetch=fake
    )
    assert seen == [CURATED_TAG]


# ── schema constraints ────────────────────────────────────────────────────


def test_schema_rejects_unknown_policy_reference():
    with pytest.raises(ValidationError):
        GeoResolveRequest(
            items=[ResolveItem(index=0, name="X", policy="nope")],
            policies={"topic": ResolvePolicy(type_id=TOPIC_TYPE)},
        )


def test_schema_rejects_malformed_ids():
    with pytest.raises(ValidationError):
        ResolvePolicy(type_id="not-a-geo-id")


# ── team_priority cascade (network-free via injected fetchers) ────────────


from src.api.services.geo_selection_strategies import (  # noqa: E402
    CandidateMeta,
    TeamPriorityStrategy,
)

ROOT = "a19c345ab9866679b001d7d2138d88a1"
CATCHALL = "b5a31f8182b042437ede0f84ee02f104"
DATASET = "5908c73ad336472ccbd983491d2d17e4"


def _strategy(metas, personal=frozenset()):
    return TeamPriorityStrategy(
        meta_fetcher=lambda ids: {m.geo_id: m for m in metas},
        personal_fetcher=lambda space_ids: set(personal),
    )


def _meta(gid, **kw):
    kw.setdefault("found", True)
    kw.setdefault("space_ids", [CATCHALL])
    kw.setdefault("created_at", 100.0)
    return CandidateMeta(geo_id=gid, **kw)


def test_cascade_root_resident_beats_catchall_only():
    s = _strategy([_meta("a"), _meta("b", space_ids=[ROOT, CATCHALL])])
    assert s.select(["a", "b"]) == ("resolved", "b")


def test_cascade_featured_beats_curated():
    s = _strategy([
        _meta("a", space_ids=["x"], has_curated=True),
        _meta("b", space_ids=["x"], has_featured=True),
    ])
    assert s.select(["a", "b"]) == ("resolved", "b")


def test_cascade_two_scored_escalates_to_ambiguous():
    s = _strategy([
        _meta("a", space_ids=["x"], scored=True),
        _meta("b", space_ids=["x"], scored=True),
    ])
    assert s.select(["a", "b"]) == ("ambiguous", None)


def test_cascade_dataset_resident_excluded_unless_root():
    s = _strategy([
        _meta("a", space_ids=[DATASET], backlink_count=999),
        _meta("b", space_ids=["x"]),
    ])
    assert s.select(["a", "b"]) == ("resolved", "b")
    # v3 exception: Root residency shields a dataset resident from exclusion
    s2 = _strategy([
        _meta("a", space_ids=[DATASET, ROOT]),
        _meta("b", space_ids=["x"]),
    ])
    assert s2.select(["a", "b"]) == ("resolved", "a")


def test_cascade_personal_space_excluded():
    s = _strategy(
        [_meta("a", space_ids=["p1"], backlink_count=999), _meta("b", space_ids=["x"])],
        personal={"p1"},
    )
    assert s.select(["a", "b"]) == ("resolved", "b")


def test_cascade_backlinks_then_age_then_id_floor():
    s = _strategy([
        _meta("a", space_ids=["x"], backlink_count=5),
        _meta("b", space_ids=["x"], backlink_count=9),
    ])
    assert s.select(["a", "b"]) == ("resolved", "b")
    s2 = _strategy([
        _meta("a", space_ids=["x"], created_at=200.0),
        _meta("b", space_ids=["x"], created_at=50.0),
    ])
    assert s2.select(["a", "b"]) == ("resolved", "b")
    s3 = _strategy([_meta("b", space_ids=["x"]), _meta("a", space_ids=["x"])])
    assert s3.select(["a", "b"]) == ("resolved", "a")


def test_service_dispatches_team_priority(monkeypatch):
    fake = lambda name, tag: [_node("g1", "Bitcoin"), _node("g2", "bitcoin")]
    picked = {}

    class FakeStrategy:
        def select(self, ids):
            picked["ids"] = ids
            return ("resolved", "g2")

    out = resolve_entities(
        _request([ResolveItem(index=0, name="Bitcoin", policy="topic")]),
        fetch=fake,
        strategy_factory=lambda name: FakeStrategy(),
    )
    assert picked["ids"] == ["g1", "g2"]
    r = out.resolutions[0]
    assert (r.status, r.geo_id) == ("resolved", "g2")


# ── team-wide conformance vectors (shared across all implementations) ─────

import json as _json
import pathlib as _pathlib


def test_team_priority_conformance_vectors():
    """Every implementation of the team cascade must pass these vectors —
    the shared fixture is the cross-repo drift guard (see fixture header)."""
    spec = _json.loads(
        (_pathlib.Path(__file__).parent / "fixtures" / "team_priority_conformance.json").read_text()
    )
    for vec in spec["vectors"]:
        metas, canonical_by_id = [], {}
        for c in vec["candidates"]:
            metas.append(CandidateMeta(
                geo_id=c["id"], found=True,
                space_ids=c.get("space_ids", []),
                represents_spaces=c.get("represents_spaces", []),
                created_at=float(c.get("created_at", 100)),
                backlink_count=c.get("backlink_count", 0),
                data_count=c.get("data_count", 0),
                has_featured=c.get("has_featured", False),
                has_curated=c.get("has_curated", False),
                scored=c.get("scored", False),
            ))
            for cs in c.get("canonical_spaces", []):
                canonical_by_id[cs] = True

        import src.api.services.geo_selection_strategies as strat
        orig = strat.resolve_canonical_space_ids
        strat.resolve_canonical_space_ids = lambda: list(canonical_by_id)
        try:
            s = TeamPriorityStrategy(
                meta_fetcher=lambda ids: {m.geo_id: m for m in metas},
                personal_fetcher=lambda sids: set(vec.get("personal_spaces", [])),
            )
            status, winner = s.select([m.geo_id for m in metas])
        finally:
            strat.resolve_canonical_space_ids = orig

        if vec["expected"].get("escalate"):
            assert status == "ambiguous", f"vector '{vec['name']}': expected escalation, got {status}"
        else:
            assert (status, winner) == ("resolved", vec["expected"]["winner"]), \
                f"vector '{vec['name']}': got ({status}, {winner})"
