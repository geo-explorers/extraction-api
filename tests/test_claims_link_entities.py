"""Unit tests for claims.link_entities (LLM-free parts).

Exercises the facet-parameterized response validator (per-facet index
filtering, dedupe, caps, canonical one-entry-per-claim ordering, unknown-key
handling, malformed-input degradation), the prompt assembly, and the
empty-facets short-circuit — without calling Gemini.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas.claims_link_entities_schema import (
    ClaimLink,
    ClaimsLinkEntitiesRequest,
    Facet,
    LinkClaim,
    VocabularyItem,
)
from src.api.services.claims_link_entities_service import (
    link_claim_entities,
    validate_links_response,
)
from src.config.prompts.claims_link_entities_prompt import build_user_prompt


def _vocab(*names_with_index) -> list:
    return [VocabularyItem(index=i, name=n) for i, n in names_with_index]


def _request(**overrides) -> ClaimsLinkEntitiesRequest:
    defaults = dict(
        claims=[
            LinkClaim(index=0, text="Bitcoin reached a new all-time high."),
            LinkClaim(index=1, text="The senator proposed new election rules."),
        ],
        facets=[
            Facet(key="topics", vocabulary=_vocab((0, "Bitcoin"), (1, "US Politics"))),
            Facet(key="people", vocabulary=_vocab((0, "Cynthia Lummis")), max_per_claim=5),
        ],
    )
    defaults.update(overrides)
    return ClaimsLinkEntitiesRequest(**defaults)


# ── schema constraints ────────────────────────────────────────────────────


def test_schema_rejects_duplicate_facet_keys():
    with pytest.raises(ValidationError):
        _request(facets=[Facet(key="topics"), Facet(key="topics")])


def test_schema_rejects_non_slug_facet_key():
    with pytest.raises(ValidationError):
        _request(facets=[Facet(key="Related People!")])


# ── validate_links_response ───────────────────────────────────────────────


def test_validate_basic_selection():
    req = _request()
    links = validate_links_response(
        {
            "links": [
                {"claim_index": 0, "selections": {"topics": [0], "people": []}},
                {"claim_index": 1, "selections": {"topics": [1], "people": [0]}},
            ]
        },
        req,
    )
    assert links == [
        ClaimLink(claim_index=0, selections={"topics": [0], "people": []}),
        ClaimLink(claim_index=1, selections={"topics": [1], "people": [0]}),
    ]


def test_validate_drops_unknown_indices_dedupes_and_ignores_unknown_keys():
    req = _request()
    links = validate_links_response(
        {
            "links": [
                {
                    "claim_index": 0,
                    "selections": {
                        "topics": [0, 0, 7, "1", None],
                        "people": [3, 0],
                        "hallucinated_facet": [1, 2],
                    },
                }
            ]
        },
        req,
    )
    assert links[0].selections == {"topics": [0, 1], "people": [0]}


def test_validate_enforces_per_facet_caps():
    req = _request(
        facets=[
            Facet(
                key="tags",
                vocabulary=_vocab(*[(i, f"T{i}") for i in range(6)]),
                max_per_claim=2,
            )
        ]
    )
    links = validate_links_response(
        {"links": [{"claim_index": 0, "selections": {"tags": [5, 4, 3, 2]}}]}, req
    )
    assert links[0].selections == {"tags": [5, 4]}


def test_validate_missing_and_duplicate_claims_canonicalized():
    req = _request()
    links = validate_links_response(
        {
            "links": [
                {"claim_index": 1, "selections": {"topics": [1]}},
                {"claim_index": 1, "selections": {"topics": [0]}},  # dup: first wins
                {"claim_index": 9, "selections": {"topics": [0]}},  # unknown: dropped
            ]
        },
        req,
    )
    assert [l.claim_index for l in links] == [0, 1]  # request order restored
    assert links[0].selections == {"topics": [], "people": []}  # missing → empty
    assert links[1].selections == {"topics": [1], "people": []}


def test_validate_sparse_caller_indices_respected():
    # Callers may use their own stable, non-contiguous handles.
    req = _request(
        claims=[LinkClaim(index=42, text="Something happened.")],
        facets=[Facet(key="topics", vocabulary=_vocab((17, "Bitcoin")))],
    )
    links = validate_links_response(
        {"links": [{"claim_index": 42, "selections": {"topics": [17, 0]}}]}, req
    )
    assert links == [ClaimLink(claim_index=42, selections={"topics": [17]})]


@pytest.mark.parametrize(
    "raw", [{}, {"links": "nope"}, {"links": [None, "x", 3]}, [], {"links": [{"claim_index": 0, "selections": "bad"}]}]
)
def test_validate_malformed_degrades_to_empty(raw):
    req = _request()
    links = validate_links_response(raw, req)
    assert [l.claim_index for l in links] == [0, 1]
    assert all(
        l.selections == {"topics": [], "people": []} for l in links
    )


# ── empty-facets short-circuit ────────────────────────────────────────────


def test_all_empty_facets_short_circuit_without_llm():
    req = _request(
        facets=[Facet(key="topics"), Facet(key="people")]
    )
    out = link_claim_entities(req)  # would raise on missing GEMINI_API_KEY if it called out
    assert [l.claim_index for l in out.links] == [0, 1]
    assert out.links[0].selections == {"topics": [], "people": []}
    assert out.model_used == ""


# ── prompt assembly ───────────────────────────────────────────────────────


def test_prompt_renders_facet_blocks_criteria_and_context():
    req = _request(
        facets=[
            Facet(
                key="topics",
                vocabulary=_vocab((0, "Bitcoin")),
                max_per_claim=3,
                criteria="subjects the claim asserts something about",
            ),
            Facet(key="people", vocabulary=_vocab((0, "Cynthia Lummis"))),
        ],
        context="Story: crypto markets",
    )
    prompt = build_user_prompt(claims=req.claims, facets=req.facets, context=req.context)
    assert "[0] Bitcoin reached a new all-time high." in prompt
    assert 'FACET "topics" (select at most 3 per claim; criterion: subjects the claim asserts something about):' in prompt
    assert 'FACET "people" (select at most 3 per claim):' in prompt
    assert "[0] Cynthia Lummis" in prompt
    assert "CONTEXT: Story: crypto markets" in prompt
    assert '"topics": [<int>, ...], "people": [<int>, ...]' in prompt


def test_prompt_no_context_line_when_context_empty():
    req = _request()
    prompt = build_user_prompt(claims=req.claims, facets=req.facets, context="")
    assert "CONTEXT:" not in prompt
