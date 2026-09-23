"""Collection review (news.review_collections) — the LLM-free parts.

The guards, the parsers for each step (grouping, rescues, merge plan, check),
the merge/assemble renumbering and the fail-open orchestration, with the
model call replaced by a seam that answers each step by the first sentence of
its prompt. Fixtures are real prod claims from the 2026-09-22 probe: the
Greenland "no U.S. adversary" triple (one sentence cut in three), the
Kristersson resignation pair (one event told twice), the Pew poll that sat
alone, and the merges the guards refused on the probe.
"""

import json
import re

import pytest

from src.api.schemas.news_claim_extract_schema import (
    ExtractedClaim,
    ExtractedCollection,
    ExtractedQuote,
    NewsArticleSource,
)
from src.api.services.news_collection_review_service import (
    CORE_IMPORTANCE,
    DEDUP_JACCARD,
    MAX_CLAIM_WORDS,
    MAX_MERGE_GROWTH,
    MIN_BLOCK,
    Grouping,
    Merge,
    ProseIndex,
    Rescue,
    ReviewPlan,
    accept_order,
    apply_merges,
    assemble,
    build_check_prompt,
    build_group_prompt,
    build_order_prompt,
    build_rescue_prompt,
    build_review_prompt,
    build_source_check_prompt,
    guard_grouping,
    guard_merge,
    guard_plan,
    guard_rescue,
    heading_case,
    jaccard,
    merged_claim,
    opener_fits,
    parse_check,
    parse_grouping,
    parse_order,
    parse_plan,
    parse_rescues,
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
PEW = "A January 2026 Pew Research Center poll found that 58 percent of Americans opposed a U.S. takeover of Greenland, while 21 percent favored it"
NUUK = "Protesters in Nuuk held banners reading 'Make America Go Away' and 'We Are Not Property!' in January 2026 to oppose a U.S. takeover of Greenland"
# The sources state every date and figure the claims above carry — the anchor
# guard (claim_anchor_guard) refuses any rescued or merged sentence otherwise.
SOURCES = [NewsArticleSource(index=0, url="u0", title="T0", publisher="P", content=(
             "body zero. President Trump announced the Greenland security agreement on September 19, 2026. "
             "A January 2026 Pew Research Center poll found 58 percent of Americans opposed a takeover and 21 percent favored it."
           )),
           NewsArticleSource(index=1, url="u1", title="T1", publisher=None, content=(
             "body one. Protesters in Nuuk held banners in January 2026 to oppose a U.S. takeover of Greenland."
           ))]


def _claim(text: str, topic: str = "t", conf: float = 0.9, imp=None, src=None) -> ExtractedClaim:
    return ExtractedClaim(text=text, topic=topic, confidence=conf, importance=imp, source_indices=src or [0])


def _story():
    """The extraction's own grouping: the agreement (2 claims), the adversary
    triple (3) and the Pew poll filed loosely under the agreement — six
    claims, two collections."""
    claims = [_claim(t, "Greenland security agreement", imp=0.9) for t in AGREEMENT] + [
        _claim(t, "US adversary restrictions", conf=0.8, imp=0.7, src=[i]) for i, t in enumerate(ADVERSARY)
    ] + [_claim(PEW, "Greenland security agreement", imp=0.4, src=[1])]
    collections = [
        ExtractedCollection(name="Greenland security agreement", type="topic", claim_indices=[0, 1, 5]),
        ExtractedCollection(name="US adversary restrictions", type="topic", claim_indices=[2, 3, 4]),
    ]
    quotes = [ExtractedQuote(text="permanent control", speaker="Donald Trump", claim_index=0),
              ExtractedQuote(text="express written approval", speaker=None, claim_index=4)]
    return claims, quotes, collections, ["Greenland security agreement", "US adversary restrictions"]


# What a well-behaved model answers at each step for _story(): the agreement
# pair and the adversary triple as blocks, the Pew poll alone; the triple
# merges into one sentence (its block thins to one, the survivor is set
# aside); the Nuuk protest rescues the poll; the check passes every block and
# homes the adversary survivor in the agreement block.
GROUP_OK = json.dumps({"blocks": [{"name": "Details of the security agreement", "claims": [0, 1]},
                                  {"name": "Restrictions on US adversaries", "claims": [2, 3, 4]}], "lone": [5]})
MERGE_OK = json.dumps({"merges": [{"keep": 2, "drop": [3, 4], "text": ADVERSARY_MERGED}]})
RESCUE_OK = json.dumps({"rescues": [{"lone": 5, "name": "Opposition to a US takeover",
                                     "claim": {"text": NUUK, "source_indices": [1], "confidence": 0.9, "importance": 0.5}},
                                    {"lone": 2, "name": "x", "claim": None}]})


def check_json(homes=None, faults=None):
    """A check that passes every block (indices beyond the real ones are
    ignored by the parser) and homes the given lone claims."""
    blocks = [{"index": i, "misfits": [], "purpose": True, "heading_ok": True, "duplicates": []} for i in range(6)]
    for i, f in (faults or {}).items():
        blocks[i].update(f)
    return json.dumps({"blocks": blocks, "lone": [{"index": i, "home": h} for i, h in (homes or {}).items()]})


def identity_order(prompt):
    """The order call's default answer: the blocks as listed."""
    return json.dumps({"order": [int(i) for i in re.findall(r"^\[(\d+)\] ", prompt, re.M)]})


CHECK_OK = check_json(homes={2: 0})
SOURCE_OK = json.dumps({"checks": [{"index": 0, "grade": "supported", "false_link": False}]})


def seam(group=GROUP_OK, rescue=RESCUE_OK, merge=MERGE_OK, check=CHECK_OK, source=SOURCE_OK, order=identity_order, log=None):
    """A model that answers each step by the first sentence of its prompt."""
    def call(prompt, thinking):
        step = prompt.split(" ", 2)[1]
        if log is not None:
            log.append((step, thinking))
        if prompt.startswith("You group"):
            return group(prompt) if callable(group) else group
        if prompt.startswith("You find company"):
            return rescue
        if prompt.startswith("You review"):
            return merge
        if prompt.startswith("You are the final check"):
            return check(prompt) if callable(check) else check
        if prompt.startswith("Check each sentence"):
            return source(prompt) if callable(source) else source
        if prompt.startswith("You order"):
            return order(prompt) if callable(order) else order
        raise AssertionError(f"unknown prompt: {prompt[:40]}")
    return call


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
    })
    plan = parse_plan(raw, 5)
    assert [m.refused is None for m in plan.merges] == [True, False, False, False]
    assert all(m.refused.startswith("bad indices") for m in plan.merges[1:])


BLOCKS = [("Agreement", [0, 1]), ("Adversaries", [2, 3, 4])]


def test_guard_plan_refuses_overlapping_merges():
    plan = ReviewPlan(merges=[Merge(keep=2, drop=[3], text=None), Merge(keep=3, drop=[4], text=None)])
    guard_plan(plan, AGREEMENT + ADVERSARY, BLOCKS)
    assert plan.merges[0].refused is None
    assert plan.merges[1].refused.startswith("overlaps")


def test_guard_plan_refuses_a_merge_across_blocks():
    plan = ReviewPlan(merges=[Merge(keep=1, drop=[2], text=None)])
    guard_plan(plan, AGREEMENT + ADVERSARY, BLOCKS)
    assert plan.merges[0].refused.endswith("crosses blocks")


# ── grouping ─────────────────────────────────────────────────────────────

def test_parse_grouping_reads_blocks_and_lone_and_drops_junk():
    g = parse_grouping(json.dumps({"blocks": [{"name": "A", "claims": [0, 1, "x", 9]}, {"claims": [2]}, "junk"], "lone": [3, 3, 40]}), 5)
    assert g.blocks == [("A", [0, 1])] and g.lone == [3]


def test_guard_grouping_owns_the_invariants():
    # 1 sits in two blocks (first wins), "Thin" has one claim (set aside), ""
    # has no heading (set aside), 6 is nowhere (accounted for as lone).
    g = Grouping(blocks=[("A", [0, 1]), ("B", [1, 2, 3]), ("Thin", [4]), ("", [5, 8])], lone=[7])
    out = guard_grouping(g, 9)
    assert out.blocks == [("A", [0, 1]), ("B", [2, 3])]
    assert out.lone == [4, 5, 6, 7, 8]


# ── rescues ──────────────────────────────────────────────────────────────

def test_parse_rescues_keeps_one_per_lone_claim_and_marks_null_as_refused():
    raw = json.dumps({"rescues": [
        {"lone": 5, "name": "Opposition", "claim": {"text": NUUK, "source_indices": [1, 7], "confidence": 0.9, "importance": 0.5}},
        {"lone": 5, "name": "Again", "claim": {"text": "dup", "source_indices": [0]}},   # second answer for 5 → ignored
        {"lone": 6, "name": "Nothing", "claim": None},
        {"lone": 2, "name": "Not lone", "claim": {"text": "x", "source_indices": [0]}},  # 2 is not lone → ignored
    ]})
    rs = parse_rescues(raw, [5, 6], 2)
    assert [(r.lone, r.name, r.refused) for r in rs] == [(5, "Opposition", None), (6, "Nothing", "the sources carry nothing")]
    assert rs[0].claim.text == NUUK and rs[0].claim.source_indices == [1] and rs[0].claim.topic == "Opposition"


def test_guard_rescue_refuses_what_publish_would_undo():
    ok = Rescue(lone=5, name="Opposition", claim=_claim(NUUK, src=[1]))
    assert guard_rescue(ok, PEW, AGREEMENT) is None
    assert guard_rescue(Rescue(lone=5, name="", claim=_claim(NUUK, src=[1])), PEW, []) == "no heading"
    assert guard_rescue(Rescue(lone=5, name="O", claim=ExtractedClaim(text=NUUK, topic="O", source_indices=[])), PEW, []) == "no source"
    assert "words" in guard_rescue(Rescue(lone=5, name="O", claim=_claim("Too short a claim", src=[1])), PEW, [])
    # A near-restatement of the lone claim: publish's dedup would fold it back, leaving the block at one.
    restated = PEW.replace("found that", "reported that")
    assert jaccard(restated, PEW) >= DEDUP_JACCARD
    assert guard_rescue(Rescue(lone=5, name="O", claim=_claim(restated, src=[1])), PEW, []) == "restates a claim the story already has"
    # ...and of any other existing claim.
    assert guard_rescue(Rescue(lone=5, name="O", claim=_claim(AGREEMENT[0], src=[1])), PEW, AGREEMENT) == "restates a claim the story already has"


# ── check ────────────────────────────────────────────────────────────────

def test_parse_check_faults_from_fields_and_homes_lone_claims():
    blocks = [("A", [0, 1]), ("B", [2, 3])]
    raw = json.dumps({"blocks": [
        {"index": 0, "misfits": [1], "purpose": True, "heading_ok": True, "duplicates": [], "reason": "claim 1 is about the poll"},
        {"index": 1, "misfits": [], "purpose": False, "heading_ok": False, "duplicates": [[2, 3]], "reason": ""},
    ], "lone": [{"index": 5, "home": 1}, {"index": 6, "home": -1}, {"index": 7, "home": 9}]})
    c = parse_check(raw, blocks, [5, 6, 7])
    assert c.reasons == ['block "A": claims [1] do not belong — claim 1 is about the poll',
                         'block "B": no shared subject; heading vague or untrue; same fact twice [[2, 3]]']
    assert c.homes == {5: 1}


def test_parse_check_passes_clean_blocks_and_faults_unanswered_ones():
    blocks = [("A", [0, 1]), ("B", [2, 3])]
    c = parse_check(json.dumps({"blocks": [{"index": 0, "misfits": [], "purpose": True, "heading_ok": True, "duplicates": []}]}), blocks, [])
    assert c.reasons == ['block "B": not checked']
    both = json.dumps({"blocks": [{"index": i, "misfits": [], "purpose": True, "heading_ok": True, "duplicates": []} for i in range(2)]})
    assert parse_check(both, blocks, []).reasons == []


# ── merged_claim ──────────────────────────────────────────────────────────

def test_merged_claim_unions_sources_and_takes_min_confidence_max_importance():
    kept = _claim("a", conf=0.9, imp=0.5, src=[0])
    m = merged_claim(kept, [_claim("b", conf=0.7, imp=0.8, src=[1, 2]), _claim("c", conf=0.95, src=[0])], "abc")
    assert (m.text, m.source_indices, m.confidence, m.importance) == ("abc", [0, 1, 2], 0.7, 0.8)
    assert merged_claim(kept, [_claim("b")], None).text == "a"


# ── apply_merges / assemble ───────────────────────────────────────────────

def test_apply_merges_the_triple_inside_its_block_and_sets_a_thinned_survivor_aside():
    claims, quotes, collections, order = _story()
    blocks = [("Agreement", [0, 1]), ("Adversaries", [2, 3, 4]), ("Poll", [5, 1])]
    plan = ReviewPlan(merges=[Merge(keep=2, drop=[3, 4], text=ADVERSARY_MERGED),
                              Merge(keep=0, drop=[1], text=None, refused="merge [0, 1] loses ['x']")])
    live, dropped, out_blocks, aside = apply_merges(plan, claims, blocks)
    assert live[2].text == ADVERSARY_MERGED and live[2].source_indices == [0, 1, 2]
    assert live[0].text == AGREEMENT[0], "a refused merge changes nothing"
    assert dropped == {3: 2, 4: 2}
    # The adversary block fell to one claim: gone, its survivor set aside.
    assert out_blocks == [("Agreement", [0, 1]), ("Poll", [5, 1])] and aside == [2]


def test_assemble_renumbers_claims_blocks_and_quotes_and_drops_the_discarded():
    claims, quotes, collections, order = _story()
    claims = claims + [_claim(NUUK, "Opposition", src=[1])]
    blocks = [("Agreement", [0, 1]), ("Adversaries", [2]), ("Opposition", [5, 6])]
    # 3 and 4 merged into 2; 1 discarded (a lone claim nothing homed).
    out_claims, out_quotes, out_collections, out_order = assemble(claims, quotes, {3: 2, 4: 2}, blocks, {1})
    assert [c.text for c in out_claims] == [AGREEMENT[0], ADVERSARY[0], PEW, NUUK]
    assert [c.topic for c in out_claims] == ["Agreement", "Adversaries", "Opposition", "Opposition"]
    assert [(c.name, c.type, c.claim_indices) for c in out_collections] == [
        ("Agreement", "topic", [0]), ("Adversaries", "topic", [1]), ("Opposition", "topic", [2, 3]),
    ]
    assert out_order == ["Agreement", "Adversaries", "Opposition"]
    # The quote on a merged-away claim follows it; one on a discarded claim disappears.
    assert [(q.text, q.claim_index) for q in out_quotes] == [("permanent control", 0), ("express written approval", 1)]


# ── prompts ───────────────────────────────────────────────────────────────

def test_review_prompt_carries_claims_collections_and_the_limits():
    p = build_review_prompt("H", ["A", "B"], [("Block", [0, 1])])
    assert "0. A\n1. B" in p and "Block: [0, 1]" in p
    assert f"at most {MAX_CLAIM_WORDS} words and at most {MAX_MERGE_GROWTH} words longer" in p
    assert "Never drop a claim, never move a claim" in p


def test_group_rescue_and_check_prompts_show_what_each_step_needs():
    g = build_group_prompt("H", ["A", "B", "C"], feedback="- block \"X\": no shared subject", indices=[0, 2])
    assert "0. A\n2. C" in g and "1. B" not in g, "a regroup lists only the claims still alive"
    assert "REJECTED by the check" in g and f"at least {MIN_BLOCK} claims" in g
    assert "feedback" not in build_group_prompt("H", ["A"]), "no feedback block on the first pass"
    r = build_rescue_prompt("H", ["A", "B"], [1], SOURCES)
    assert "- A\n- B" in r and "LONE CLAIMS (index. text):\n1. B" in r and "[0: T0 — P]\nbody zero" in r and "[1: T1]\nbody one" in r
    c = build_check_prompt("H", ["A", "B", "C"], [("Block", [0, 1])], [2])
    assert '[0] "Block"\n   0. A\n   1. B' in c and "LONE CLAIMS (no block yet):\n2. C" in c
    assert "LONE CLAIMS" not in build_check_prompt("H", ["A", "B"], [("Block", [0, 1])], [])


def test_source_check_prompt_and_parser():
    p = build_source_check_prompt(["S0", "S1"], [NewsArticleSource(index=0, url="u", title="T", publisher="P", content="body")])
    assert "0. S0\n1. S1" in p and "[T — P]\nbody" in p
    raw = json.dumps({"checks": [
        {"index": 0, "grade": "supported", "false_link": True},
        {"index": 1, "grade": "unsupported", "false_link": False},
        {"index": 2, "grade": "partly", "false_link": False},
        {"index": 7, "grade": "unsupported", "false_link": False},
    ]})
    v = parse_source_check(raw, 4)
    assert v[0] and "links facts" in v[0]
    assert v[1] and "do not support" in v[1]
    assert v[2] is None and v[3] is None, "partly passes for a merge; an unmentioned sentence passes"
    assert parse_source_check(raw, 4, strict=True)[2] is not None, "partly fails for a rescued fact"


# ── review_collections: orchestration through the seam ───────────────────

def test_review_rebuilds_the_blocks_end_to_end():
    claims, quotes, collections, order = _story()
    log = []
    r = review_collections("H", SOURCES, claims, quotes, collections, order, "medium", call=seam(log=log))
    rep = r.review
    assert rep.applied and rep.check == "ok"
    assert (rep.blocks, rep.rescued_claims, rep.merges, rep.merged_claims, rep.moves, rep.dropped_claims) == (2, 1, 1, 2, 1, 0)
    # The triple is one sentence now; its survivor, left alone, was homed in the
    # agreement block; the poll got the Nuuk protest for company.
    assert [c.text for c in r.claims] == AGREEMENT + [ADVERSARY_MERGED, PEW, NUUK]
    assert [(c.name, c.claim_indices) for c in r.collections] == [
        ("Details of the security agreement", [0, 1, 2]), ("Opposition to a US takeover", [3, 4]),
    ]
    assert r.collection_order == ["Details of the security agreement", "Opposition to a US takeover"]
    # The merged claim carries the union of sources, the rescued one its own; every claim's topic is its block.
    assert r.claims[2].source_indices == [0, 1, 2] and r.claims[4].source_indices == [1]
    assert [c.topic for c in r.claims] == ["Details of the security agreement"] * 3 + ["Opposition to a US takeover"] * 2
    assert r.quotes == [ExtractedQuote(text="permanent control", speaker="Donald Trump", claim_index=0),
                        ExtractedQuote(text="express written approval", speaker=None, claim_index=2)]
    # Steps and their thinking: group cheap, merge at the caller's level and its
    # source check cheap, rescue reads the sources and gets its own check, the
    # final check cheap — and the order call beside it (parallel, so either
    # may log first).
    assert log[:5] == [("group", "low"), ("review", "medium"), ("each", "low"), ("find", "medium"), ("each", "low")]
    assert sorted(log[5:]) == [("are", "medium"), ("order", "low")]
    assert list(rep.steps) == ["group", "merge", "rescue", "order", "check"]
    assert rep.order == "same"


def test_review_drops_a_lone_claim_nothing_homes_but_never_a_core_one():
    claims, quotes, collections, order = _story()
    no_rescue = json.dumps({"rescues": [{"lone": 5, "name": "x", "claim": None}]})
    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(rescue=no_rescue, check=check_json(homes={2: 0})))
    assert r.review.applied and r.review.dropped_claims == 1 and r.review.rescued_claims == 0 and r.review.blocks == 1
    assert PEW not in [c.text for c in r.claims] and any("carry nothing" in x for x in r.review.rejected)
    # The same story with the poll graded a core claim: the review is discarded instead.
    claims[5] = _claim(PEW, imp=CORE_IMPORTANCE, src=[1])
    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(rescue=no_rescue, check=check_json(homes={2: 0})))
    assert not r.review.applied and any("core claim" in x for x in r.review.rejected)
    assert r.collections == collections


def test_review_refuses_a_rescue_the_sources_do_not_back():
    claims, quotes, collections, order = _story()
    graded = []

    def source(prompt):
        graded.append(prompt)
        # The merge's check comes first and passes; the rescue's (strict) fails on "partly".
        grade = "partly" if NUUK in prompt else "supported"
        return json.dumps({"checks": [{"index": 0, "grade": grade, "false_link": False}]})

    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(source=source, check=check_json(homes={2: 0, 5: 0})))
    assert r.review.applied and r.review.rescued_claims == 0 and r.review.moves == 2
    assert NUUK not in [c.text for c in r.claims]
    assert any(x.startswith("rescue of 5: the sources do not support") for x in r.review.rejected)
    assert len(graded) == 2


def test_review_refuses_a_rescue_or_merge_carrying_a_date_no_source_states():
    """The anchor guard runs on every sentence the review writes, before the
    model source check: a rescued fact dated by the model's own arithmetic and
    a merge that invents a year are refused in code, with the reason."""
    claims, quotes, collections, order = _story()
    guessed = NUUK.replace("in January 2026", "on January 14, 2026")
    rescue = json.dumps({"rescues": [{"lone": 5, "name": "Opposition to a US takeover",
                                      "claim": {"text": guessed, "source_indices": [1], "confidence": 0.9, "importance": 0.5}}]})
    merge = json.dumps({"merges": [{"keep": 2, "drop": [3, 4], "text": ADVERSARY_MERGED.replace("September 19, 2026", "September 19, 2025")}]})
    graded = []
    r = review_collections("H", SOURCES, claims, quotes, collections, order,
                           call=seam(rescue=rescue, merge=merge, source=lambda p: graded.append(p) or SOURCE_OK,
                                     check=check_json(homes={5: 0})))
    assert r.review.applied and r.review.rescued_claims == 0 and r.review.merges == 0
    assert guessed not in [c.text for c in r.claims] and len(r.claims) == 6
    assert any(x.startswith("rescue of 5: unanchored date 'January 14, 2026': no source states this date") for x in r.review.rejected)
    # The merge's invented year is caught by whichever guard sees it first —
    # guard_merge's "every number of the parts" rule or the anchor guard.
    assert any(x.startswith("merge [2, 3, 4]") for x in r.review.rejected), r.review.rejected
    assert graded == [], "a sentence the guard refused never reaches the model source check"


def test_review_refuses_a_composed_merge_the_sources_do_not_back():
    claims, quotes, collections, order = _story()
    source = lambda prompt: json.dumps({"checks": [{"index": 0, "grade": "supported", "false_link": NUUK not in prompt}]})
    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(source=source))
    assert r.review.applied and r.review.merges == 0
    assert len(r.claims) == 7 and any("links facts" in x for x in r.review.rejected)
    assert [c.claim_indices for c in r.collections] == [[0, 1], [2, 3, 4], [5, 6]]


def test_review_regroups_once_on_a_rejected_check_and_discards_on_a_second():
    claims, quotes, collections, order = _story()
    verdicts = iter([check_json(faults={0: {"misfits": [1], "reason": "off"}}), check_json(homes={2: 0})])
    groups = []

    def group(prompt):
        groups.append(prompt)
        if len(groups) == 1:
            return GROUP_OK
        return json.dumps({"blocks": [{"name": "Details of the security agreement", "claims": [0, 1]},
                                      {"name": "Opposition to a US takeover", "claims": [5, 6]}], "lone": [2]})

    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(group=group, check=lambda p: next(verdicts)))
    assert r.review.applied and r.review.check == "repaired" and r.review.moves == 1
    assert any(x.startswith("first grouping: block") for x in r.review.rejected)
    assert "REJECTED by the check" in groups[1]
    listing = groups[1].split("CLAIMS (index. text):\n", 1)[1].split("\n\n", 1)[0]
    assert [line.split(".")[0] for line in listing.splitlines()] == ["0", "1", "2", "5", "6"], \
        "the regroup lists live claims only, the rescued one included"

    # Rejected twice → the extraction's grouping ships.
    always_bad = check_json(faults={0: {"purpose": False}})
    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(check=always_bad))
    assert not r.review.applied and r.review.check == "rejected"
    assert r.review.rejected[-1] == "grouping rejected twice — review discarded"
    assert r.claims == claims and r.collections == collections


def test_review_discards_when_grouping_produces_no_block():
    claims, quotes, collections, order = _story()
    r = review_collections("H", SOURCES, claims, quotes, collections, order, call=seam(group=json.dumps({"blocks": [], "lone": [0, 1]})))
    assert not r.review.applied and r.review.rejected == ["grouping produced no block — review discarded"]


def test_review_without_sources_skips_the_rescue():
    claims, quotes, collections, order = _story()
    log = []
    r = review_collections("H", [], claims, quotes, collections, order, call=seam(check=check_json(homes={2: 0, 5: 0}), log=log))
    assert r.review.applied and "find" not in [s for s, _ in log] and r.review.moves == 2


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


# ── headings: sentence case from the story's own prose ───────────────────
# Real claims from Armando's injector job 2647 (2026-09-23), whose headings
# came back in Title Case from the group step and sentence case from the
# rescue step, in one story.

ARMANDO_HEADLINE = "Trump approval hits record low as Republicans break with president over Iran war"
ARMANDO_CLAIMS = [
    "The United States House of Representatives voted 220 to 204 on September 15, 2026, to direct President Donald Trump to end the war in Iran or seek explicit congressional authorization.",
    "Seven House Republicans, including Nancy Mace, voted with Democrats on September 15, 2026, to limit President Donald Trump's war powers in Iran.",
    "President Donald Trump's overall approval rating fell to a career low of 32 percent in a Reuters/Ipsos poll completed on September 20, 2026.",
    "Registered voters in a September 2026 Reuters/Ipsos poll favored Democrats over Republicans for the upcoming midterm elections by 43 percent to 35 percent.",
    "A CNN poll conducted September 16-17, 2026, showed Defense Secretary Pete Hegseth's approval rating at 34 percent overall and 72 percent among Republicans.",
    "Israel's National Security Minister Itamar Ben-Gvir called on Prime Minister Benjamin Netanyahu to recognize Argentina's sovereignty over the Falkland Islands.",
]


def test_heading_case_takes_proper_nouns_from_the_story_prose():
    prose = ProseIndex([ARMANDO_HEADLINE, *ARMANDO_CLAIMS])
    cases = {
        "House Vote on Iran War Powers": "House vote on Iran war powers",       # "vote" is only seen as "voted"
        "Trump Approval Ratings": "Trump approval ratings",                     # "ratings" only as "rating"
        "GOP Midterm Election Prospects": "GOP midterm election prospects",     # acronym kept; "prospects" never seen
        "Defense Secretary Pete Hegseth": "Defense Secretary Pete Hegseth",     # the prose capitalises the title
        "Cost of living issues": "Cost of living issues",
        "Diplomatic row over the Falklands": "Diplomatic row over the Falklands",  # "Falklands" only from "Falkland Islands": left as written
        "republican candidates on iran": "Republican candidates on Iran",       # the prose restores capitals too
        "Israel's E1 Tenders And The U.S. Response": "Israel's E1 tenders and the U.S. response",
        "AP-NORC Poll Findings": "AP-NORC poll findings",
    }
    for given, expected in cases.items():
        assert heading_case(given, prose) == expected, given


def test_heading_case_judges_a_word_next_to_its_neighbours_first():
    # Seen live (2026-09-23): "Foreign" kept its capital from "Foreign Minister"
    # in a heading about foreign policy, and "States" lost it to "member
    # states". The pair in the prose decides before the word alone does.
    prose = ProseIndex([
        "Hungary's Foreign Minister Anita Orban announced the foreign policy shift on Monday.",
        "Ukraine's Foreign Minister Andrii Sybiha praised the decision by Hungary's government.",
        "The United States said the member states must decide, and the poll states otherwise; other states agreed.",
        "Taiwan's National Security Council asked for security guarantees at the Rodeo I summit near Al-Aqsa Mosque.",
        "The Senate Foreign Relations Committee questioned Marco Rubio on the secretary-NSA arrangement.",
    ])
    # Seen live: "Relations" inside a committee's name capitalised "relationship";
    # an in-name inflection leaves the model's casing alone. A hyphenated token
    # the prose has only in parts is judged part by part.
    assert heading_case("Trump-Rubio relationship evolution", prose) == "Trump-Rubio relationship evolution"
    assert heading_case("Rubio Dual Secretary-NSA Role", prose) == "Rubio dual secretary-NSA role"
    assert heading_case("Shift in Hungary's Foreign Policy", prose) == "Shift in Hungary's foreign policy"
    assert heading_case("United States Response", prose) == "United States response"
    assert heading_case("Statement by the Foreign Minister", prose) == "Statement by the Foreign Minister"
    # Capitals seen only inside a longer name do not outweigh a lower-case use.
    assert heading_case("Taiwan's Defense And Security", prose) == "Taiwan's defense and security"
    # A roman numeral, and a hyphenated name the prose has as one token.
    assert heading_case("Conditions At Rodeo I Prison", prose) == "Conditions at Rodeo I prison"
    assert heading_case("Incursion At Al-Aqsa Mosque", prose) == "Incursion at Al-Aqsa Mosque"
    # "Taiwan's" is seen only opening a sentence: nothing tells a name from
    # grammar, so in a sentence-case heading it is left as written;
    # "Ministry" is unseen and lower-cased.
    assert heading_case("Response by Taiwan's ministry", prose) == "Response by Taiwan's ministry"
    assert heading_case("Response by Taiwan's Ministry", prose) == "Response by taiwan's ministry", "Title Case: no information"


def test_heading_case_counts_only_words_away_from_a_sentence_start():
    # "Approval" opens a sentence (capitalised for grammar) and is lowercase
    # mid-sentence elsewhere: it is not a proper noun. "Reuters" is.
    prose = ProseIndex(["Approval of the president fell. A Reuters poll found approval at 32 percent."])
    assert heading_case("Approval Figures From Reuters", prose) == "Approval figures from Reuters"
    assert prose.counts("approval") == (0, 0, 1) and prose.counts("reuters") == (1, 0, 0)


def test_heading_case_lowercases_weak_evidence_only_in_a_title_case_heading():
    # Seen live: a byline ("Paul Adams, Diplomatic correspondent") was the only
    # capital "Diplomatic" and no lower-case use existed in eleven sources.
    # Inside a Title Case heading the model's capital means nothing, so the
    # word is lower-cased; in a sentence-case heading it is left alone.
    prose = ProseIndex(["The report by Paul Adams, Diplomatic correspondent, said Israel would retaliate."])
    assert heading_case("Israeli Diplomatic Retaliation Against The UK", prose) == "Israeli diplomatic retaliation against the UK"
    assert heading_case("Israeli Diplomatic retaliation", prose) == "Israeli Diplomatic retaliation"
    assert prose.counts("diplomatic") == (0, 1, 0)


# ── reading order: parsed and guarded ────────────────────────────────────

def test_parse_order_reads_a_permutation_only():
    assert parse_order(json.dumps({"order": [2, 0, 1]}), 3) == [2, 0, 1]
    for bad in ([0, 1], [0, 1, 1], [0, 1, 5], "x", None):
        assert parse_order(json.dumps({"order": bad}), 3) == [], bad
    assert parse_order(json.dumps({}), 3) == []


def test_accept_order_holds_the_opener_to_the_headline():
    blocks = [("House vote on Iran war powers", [0]), ("Trump approval ratings", [1]), ("Outlook on Iran war", [2])]
    assert accept_order([1, 0, 2], blocks, ARMANDO_HEADLINE) == ([1, 0, 2], "changed")
    assert accept_order([0, 1, 2], blocks, ARMANDO_HEADLINE) == ([0, 1, 2], "same")
    assert accept_order([], blocks, ARMANDO_HEADLINE) == ([0, 1, 2], "kept")
    # A new opener that shares no subject word with the headline: the grouped order stands.
    off = [("Trump approval ratings", [0]), ("Cost of living issues", [1])]
    assert accept_order([1, 0], off, ARMANDO_HEADLINE) == ([0, 1], "refused")
    assert opener_fits("UK, France and Canada impose sanctions on Israeli West Bank settlements", "United Kingdom sanctions and arms ban")
    assert not opener_fits("Hungary expels 10 Russian diplomats", "Cost of living issues")


def test_prompts_ask_for_sentence_case_and_the_order_prompt_carries_the_rule():
    g = build_group_prompt("H", ["A", "B"])
    assert "sentence case" in g and "narrative flow" not in g, "one ordering rule, the order call's"
    assert "sentence case" in build_rescue_prompt("H", ["A", "B"], [1], SOURCES)
    assert "order" not in build_check_prompt("H", ["A", "B"], [("Block", [0, 1])], []), "the check is unchanged"
    o = build_order_prompt("H", ["A", "B", "C"], [("X", [0, 1]), ("Y", [2])])
    assert o.startswith("You order") and '[0] "X"\n   0. A\n   1. B\n[1] "Y"\n   2. C' in o
    assert "MAIN clause" in o and "contiguous run" in o


def test_review_applies_the_order_call_and_cases_the_headings():
    claims, quotes, collections, order = _story()
    title_case = GROUP_OK.replace("Details of the security agreement", "Details Of The Security Agreement")
    headline = "Protesters oppose the US takeover of Greenland"
    r = review_collections(headline, SOURCES, claims, quotes, collections, order,
                           call=seam(group=title_case, check=check_json(homes={2: 0}), order=json.dumps({"order": [1, 0]})))
    assert r.review.applied and r.review.order == "changed"
    # The opposition block opens (it shares "takeover" with the headline), the
    # agreement block follows; the heading's Title Case is gone; every claim's
    # topic is its cased heading.
    assert [(c.name, c.claim_indices) for c in r.collections] == [
        ("Opposition to a US takeover", [3, 4]), ("Details of the security agreement", [0, 1, 2]),
    ]
    assert r.collection_order == ["Opposition to a US takeover", "Details of the security agreement"]
    assert [c.topic for c in r.claims] == ["Details of the security agreement"] * 3 + ["Opposition to a US takeover"] * 2
    # The same order with the agreement headline: the opposition block shares
    # no word with it, so the order is refused and the grouping's stands.
    r = review_collections("Greenland security agreement announced", SOURCES, claims, quotes, collections, order,
                           call=seam(check=check_json(homes={2: 0}), order=json.dumps({"order": [1, 0]})))
    assert r.review.order == "refused" and r.collection_order[0] == "Details of the security agreement"
    # An order call that answers garbage or fails: kept, the review itself untouched.
    r = review_collections(headline, SOURCES, claims, quotes, collections, order, call=seam(order="not json"))
    assert r.review.applied and r.review.order == "kept" and any(x.startswith("order:") for x in r.review.rejected)
    assert r.collection_order == ["Details of the security agreement", "Opposition to a US takeover"]


def test_a_discarded_review_still_orders_and_cases_the_extractions_blocks():
    claims, quotes, collections, order = _story()
    always_bad = check_json(faults={0: {"purpose": False}})
    headline = "United States adversaries barred from Greenland under new agreement"
    log = []
    r = review_collections(headline, SOURCES, claims, quotes, collections, order,
                           call=seam(check=always_bad, order=json.dumps({"order": [1, 0]}), log=log))
    assert not r.review.applied and r.review.check == "rejected" and r.review.order == "changed"
    assert ("order", "low") in log and "order" in r.review.steps
    assert [c.name for c in r.collections] == ["US adversary restrictions", "Greenland security agreement"]
    assert r.collection_order == ["US adversary restrictions", "Greenland security agreement"]
    assert [c.claim_indices for c in r.collections] == [[2, 3, 4], [0, 1, 5]], "the extraction's membership, untouched"
    assert [c.text for c in r.claims] == [c.text for c in claims]
    # The order call failing changes nothing but a note.
    r = review_collections(headline, SOURCES, claims, quotes, collections, order,
                           call=seam(check=always_bad, order=lambda p: (_ for _ in ()).throw(RuntimeError("quota"))))
    assert not r.review.applied and r.review.order == "kept" and any(x.startswith("order: ") for x in r.review.rejected)
    assert [c.name for c in r.collections] == ["Greenland security agreement", "US adversary restrictions"]


def test_review_skips_trivial_input():
    r = review_collections("H", [], [_claim("only one")], [], [], [], call=lambda p, t: pytest.fail("must not call"))
    assert not r.review.applied
