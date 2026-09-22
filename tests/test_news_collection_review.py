"""Collection review (news.review_collections) — the LLM-free parts.

The guards, the plan parser, apply_plan's renumbering and the fail-open
orchestration, with the model call replaced by a seam. Fixtures are real prod
claims from the 2026-09-22 probe: the Greenland "no U.S. adversary" triple
(one sentence cut in three), the Kristersson resignation pair (one event told
twice), and the merges the guards refused on the probe.
"""

import json

import pytest

from src.api.schemas.news_claim_extract_schema import (
    ExtractedClaim,
    ExtractedCollection,
    ExtractedQuote,
    NewsArticleSource,
)
from src.api.services.news_collection_review_service import (
    MAX_CLAIM_WORDS,
    MAX_MERGE_GROWTH,
    Merge,
    ReviewPlan,
    apply_plan,
    build_review_prompt,
    build_source_check_prompt,
    guard_merge,
    guard_plan,
    merged_claim,
    parse_plan,
    parse_source_check,
    review_collections,
)

FRAME = "Under the Greenland security agreement announced by President Trump on September 19, 2026, no U.S. adversary can "
TAIL = " without express written approval from the United States"
ADVERSARY = [
    FRAME + "have a base in Greenland" + TAIL,
    FRAME + "make sensitive investments in Greenland" + TAIL,
    FRAME + "have a military presence in Greenland" + TAIL,
]
ADVERSARY_MERGED = (
    FRAME + "have a base or military presence in Greenland, or make sensitive investments there," + TAIL
)
AGREEMENT = [
    "President Donald Trump announced on September 19, 2026 that the United States entered into an agreement with the Kingdom of Denmark and Greenland that gives the United States permanent control over security in Greenland",
    "The United States will immediately begin developing a large military presence in Greenland following the security agreement announced by President Trump on September 19, 2026",
]


def _claim(text: str, topic: str = "t", conf: float = 0.9, imp=None, src=None) -> ExtractedClaim:
    return ExtractedClaim(text=text, topic=topic, confidence=conf, importance=imp, source_indices=src or [0])


def _story():
    """Two collections: the agreement (2 claims) and the adversary triple (3)."""
    claims = [_claim(t, "Greenland security agreement", imp=0.9) for t in AGREEMENT] + [
        _claim(t, "US adversary restrictions", conf=0.8, imp=0.7, src=[i]) for i, t in enumerate(ADVERSARY)
    ]
    collections = [
        ExtractedCollection(name="Greenland security agreement", type="topic", claim_indices=[0, 1]),
        ExtractedCollection(name="US adversary restrictions", type="topic", claim_indices=[2, 3, 4]),
    ]
    quotes = [ExtractedQuote(text="permanent control", speaker="Donald Trump", claim_index=0),
              ExtractedQuote(text="express written approval", speaker=None, claim_index=4)]
    return claims, quotes, collections, ["Greenland security agreement", "US adversary restrictions"]


# ── guard_merge ────────────────────────────────────────────────────────────

def test_guard_accepts_the_greenland_merge():
    assert guard_merge(ADVERSARY, ADVERSARY_MERGED) is None


def test_guard_refuses_a_merge_that_loses_a_name():
    parts = [
        "Eighteen suspects arrested in the July 2021 assassination of Haitian President Jovenel Moïse were extradited from Haiti to the United States on September 20, 2026, to face trial in Florida",
        "A military plane transported the 18 suspects out of Haiti to the United States on September 20, 2026, as confirmed by Jason Reding Quiñones, the U.S. Attorney for the Southern District of Florida",
    ]
    merged = "Eighteen suspects in the July 2021 assassination of Haitian President Jovenel Moïse were flown from Haiti to the United States on September 20, 2026, to face trial in Florida, U.S. Attorney Jason Reding Quiñones confirmed"
    reason = guard_merge(parts, merged)
    assert reason and reason.startswith("loses") and "southern" in reason


def test_guard_reads_number_words_and_inflections_as_kept():
    # "Eighteen" → "18" and "Haiti" → "Haitian" are the same tokens to the guard.
    assert guard_merge(["Eighteen suspects left Haiti on Monday", "The suspects flew to Miami on Monday"],
                       "18 suspects left Haiti for Miami on Monday") is None


def test_guard_refuses_added_names_and_numbers():
    assert guard_merge(ADVERSARY[:2], ADVERSARY_MERGED.replace("United States", "United States and NATO")).startswith("adds")
    assert guard_merge(ADVERSARY[:2], ADVERSARY_MERGED + " by 2027").startswith("adds")


def test_guard_enforces_the_reading_limits():
    long = ADVERSARY_MERGED + " " + " ".join(["and"] * (MAX_CLAIM_WORDS))
    assert f"at most {MAX_CLAIM_WORDS}" in guard_merge(ADVERSARY, long)
    short_parts = ["Zcash rose 17% on Monday", "Zcash fell 3% on Tuesday"]
    grown = "Zcash rose 17% on Monday and then, after a quiet and uneventful morning session, fell 3% on Tuesday"
    assert word_growth(short_parts, grown) > MAX_MERGE_GROWTH
    assert f"at most {MAX_MERGE_GROWTH} more" in guard_merge(short_parts, grown)


def word_growth(parts, merged):
    return len(merged.split()) - max(len(p.split()) for p in parts)


# ── parse_plan / guard_plan ───────────────────────────────────────────────

def test_parse_plan_refuses_bad_indices_and_keeps_good_ones():
    raw = json.dumps({
        "merges": [
            {"keep": 2, "drop": [3, 4], "text": ADVERSARY_MERGED},
            {"keep": 9, "drop": [1]},              # out of range
            {"keep": 0, "drop": [0]},              # keep in drop
            {"keep": 1, "drop": [], "text": "rewrite"},  # nothing to drop: a single-claim rewrite, not a merge
            {"keep": "x"},                         # malformed → dropped
        ],
        "collections": [{"name": "Greenland security agreement", "claims": [0, 1, 2, "z", 40]}],
    })
    plan = parse_plan(raw, 5)
    assert [m.refused is None for m in plan.merges] == [True, False, False, False]
    assert all(m.refused.startswith("bad indices") for m in plan.merges[1:])
    assert plan.collections == [("Greenland security agreement", [0, 1, 2])]


def test_guard_plan_refuses_overlapping_merges():
    plan = ReviewPlan(merges=[Merge(keep=2, drop=[3], text=None), Merge(keep=3, drop=[4], text=None)])
    guard_plan(plan, AGREEMENT + ADVERSARY)
    assert plan.merges[0].refused is None
    assert plan.merges[1].refused.startswith("overlaps")


# ── merged_claim ──────────────────────────────────────────────────────────

def test_merged_claim_unions_sources_and_takes_min_confidence_max_importance():
    kept = _claim("a", conf=0.9, imp=0.5, src=[0])
    m = merged_claim(kept, [_claim("b", conf=0.7, imp=0.8, src=[1, 2]), _claim("c", conf=0.95, src=[0])], "abc")
    assert (m.text, m.source_indices, m.confidence, m.importance) == ("abc", [0, 1, 2], 0.7, 0.8)
    assert merged_claim(kept, [_claim("b")], None).text == "a"


# ── apply_plan ────────────────────────────────────────────────────────────

def test_apply_merges_the_triple_and_moves_the_survivor_next_door():
    claims, quotes, collections, order = _story()
    plan = ReviewPlan(
        merges=[Merge(keep=2, drop=[3, 4], text=ADVERSARY_MERGED)],
        collections=[("Greenland security agreement", [0, 1, 2]), ("US adversary restrictions", [])],
    )
    result, report = apply_plan(plan, claims, quotes, collections, order)
    assert report.applied and report.merges == 1 and report.merged_claims == 2 and report.moves == 1
    out_claims, out_quotes, out_collections, out_order = result
    assert [c.text for c in out_claims] == AGREEMENT + [ADVERSARY_MERGED]
    assert out_claims[2].source_indices == [0, 1, 2]
    assert [(c.name, c.claim_indices) for c in out_collections] == [("Greenland security agreement", [0, 1, 2])]
    assert out_order == ["Greenland security agreement"]
    # The quote that pointed at a dropped claim now points at the merged one.
    assert [(q.text, q.claim_index) for q in out_quotes] == [("permanent control", 0), ("express written approval", 2)]


def test_a_refused_merge_stays_local_and_its_parts_go_home():
    claims, quotes, collections, order = _story()
    # The model merged the triple AND moved the survivor, but the merge is refused.
    plan = ReviewPlan(
        merges=[Merge(keep=2, drop=[3, 4], text=ADVERSARY_MERGED, refused="merge [2, 3, 4] loses ['x']")],
        collections=[("Greenland security agreement", [0, 1, 2]), ("US adversary restrictions", [])],
    )
    result, report = apply_plan(plan, claims, quotes, collections, order)
    assert report.applied and report.merges == 0 and report.moves == 0
    out_claims, _, out_collections, out_order = result
    assert len(out_claims) == 5
    assert [(c.name, c.claim_indices) for c in out_collections] == [
        ("Greenland security agreement", [0, 1]), ("US adversary restrictions", [2, 3, 4]),
    ]
    assert out_order == order
    assert report.rejected == ["merge [2, 3, 4] loses ['x']"]


def test_a_forgotten_claim_returns_to_its_collection_and_nothing_is_dropped():
    claims, quotes, collections, order = _story()
    plan = ReviewPlan(merges=[], collections=[("Greenland security agreement", [0, 1]), ("US adversary restrictions", [2, 3])])
    result, report = apply_plan(plan, claims, quotes, collections, order)
    assert report.applied
    assert result[2][1].claim_indices == [2, 3, 4]


def test_a_move_into_an_unknown_collection_is_ignored():
    claims, quotes, collections, order = _story()
    plan = ReviewPlan(merges=[], collections=[("Greenland security agreement", [0, 1]), ("Brand new block", [2, 3, 4])])
    result, report = apply_plan(plan, claims, quotes, collections, order)
    assert report.applied and report.moves == 0
    assert [c.name for c in result[2]] == ["Greenland security agreement", "US adversary restrictions"]


def test_a_result_with_a_one_claim_collection_is_discarded():
    claims, quotes, collections, order = _story()
    # Merge the triple but leave the survivor where it is: one-claim block.
    plan = ReviewPlan(
        merges=[Merge(keep=2, drop=[3, 4], text=ADVERSARY_MERGED)],
        collections=[("Greenland security agreement", [0, 1]), ("US adversary restrictions", [2])],
    )
    result, report = apply_plan(plan, claims, quotes, collections, order)
    assert result is None and not report.applied
    assert "one-claim collection" in report.rejected[-1]


# ── prompts ───────────────────────────────────────────────────────────────

def test_review_prompt_carries_claims_collections_and_the_limits():
    p = build_review_prompt("H", ["A", "B"], [("Block", [0, 1])])
    assert "0. A\n1. B" in p and "Block: [0, 1]" in p
    assert f"at most {MAX_CLAIM_WORDS} words and at most {MAX_MERGE_GROWTH} words longer" in p
    assert "never drop a claim" in p


def test_source_check_prompt_and_parser():
    p = build_source_check_prompt(["S0", "S1"], [NewsArticleSource(index=0, url="u", title="T", publisher="P", content="body")])
    assert "0. S0\n1. S1" in p and "[T — P]\nbody" in p
    v = parse_source_check(json.dumps({"checks": [
        {"index": 0, "grade": "supported", "false_link": True},
        {"index": 1, "grade": "unsupported", "false_link": False},
        {"index": 7, "grade": "unsupported", "false_link": False},
    ]}), 3)
    assert v[0] and "links facts" in v[0]
    assert v[1] and "do not support" in v[1]
    assert v[2] is None


# ── review_collections: orchestration through the seam ───────────────────

def test_review_applies_a_good_plan_and_reports(monkeypatch):
    claims, quotes, collections, order = _story()
    calls = []

    def fake(prompt, thinking):
        calls.append(thinking)
        if "SENTENCES:" in prompt:
            return json.dumps({"checks": [{"index": 0, "grade": "supported", "false_link": False}]})
        return json.dumps({"merges": [{"keep": 2, "drop": [3, 4], "text": ADVERSARY_MERGED}],
                           "collections": [{"name": "Greenland security agreement", "claims": [0, 1, 2]}]})

    r = review_collections("H", [NewsArticleSource(index=0, url="u", title="T", content="c")], claims, quotes, collections, order, "medium", call=fake)
    assert r.review.applied and r.review.merges == 1 and r.review.moves == 1
    assert len(r.claims) == 3 and r.claims[2].text == ADVERSARY_MERGED
    assert calls == ["medium", "low"], "review at the requested level, the source check cheap"


def test_review_refuses_a_composed_merge_the_sources_do_not_back(monkeypatch):
    claims, quotes, collections, order = _story()

    def fake(prompt, thinking):
        if "SENTENCES:" in prompt:
            return json.dumps({"checks": [{"index": 0, "grade": "supported", "false_link": True}]})
        return json.dumps({"merges": [{"keep": 2, "drop": [3, 4], "text": ADVERSARY_MERGED}],
                           "collections": [{"name": "Greenland security agreement", "claims": [0, 1, 2]}]})

    r = review_collections("H", [NewsArticleSource(index=0, url="u", title="T", content="c")], claims, quotes, collections, order, call=fake)
    assert r.review.applied and r.review.merges == 0
    assert len(r.claims) == 5 and [c.claim_indices for c in r.collections] == [[0, 1], [2, 3, 4]]
    assert any("links facts" in x for x in r.review.rejected)


def test_review_needs_no_source_check_for_a_keep_verbatim_merge():
    claims, quotes, collections, order = _story()
    calls = []

    def fake(prompt, thinking):
        calls.append(prompt[:9])
        return json.dumps({"merges": [{"keep": 0, "drop": [1], "text": None}],
                           "collections": [{"name": "Greenland security agreement", "claims": [0, 2]},
                                           {"name": "US adversary restrictions", "claims": [3, 4]}]})

    r = review_collections("H", [], claims, quotes, collections, order, call=fake)
    assert r.review.applied and r.review.merges == 1 and r.review.moves == 1
    assert len(calls) == 1
    assert [c.text for c in r.claims][0] == AGREEMENT[0]


def test_review_fails_open_on_a_model_error():
    claims, quotes, collections, order = _story()

    def fake(prompt, thinking):
        raise RuntimeError("quota")

    r = review_collections("H", [], claims, quotes, collections, order, call=fake)
    assert not r.review.applied and r.review.rejected == ["review failed: quota"]
    assert [c.text for c in r.claims] == [c.text for c in claims]
    assert r.collections == collections and r.quotes == quotes


def test_review_fails_open_on_garbage_json():
    claims, quotes, collections, order = _story()
    r = review_collections("H", [], claims, quotes, collections, order, call=lambda p, t: "not json at all")
    assert not r.review.applied and r.review.rejected[0].startswith("review failed")


def test_review_skips_trivial_input():
    r = review_collections("H", [], [_claim("only one")], [], [], [], call=lambda p, t: pytest.fail("must not call"))
    assert not r.review.applied
