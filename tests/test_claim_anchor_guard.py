"""Source-anchored dates and numbers: the guard (pure), the calendar the
prompt gets, and the enforcement step (repair once, then drop) around it.

The fixture tests/fixtures/claim_anchor_cases.json is shared with news-worker's
lib/claim-anchors.ts — both suites run the same cases so the two copies of the
rule cannot drift."""

import json
from datetime import date
from pathlib import Path

import pytest

from src.api.schemas.news_claim_extract_schema import (
  ExtractedClaim,
  ExtractedCollection,
  ExtractedQuote,
  NewsArticleSource,
  NewsClaimExtractResponse,
)
from src.api.services.claim_anchor_enforce import (
  build_anchor_repair_prompt,
  drop_claims,
  enforce_anchors,
  parse_anchor_repairs,
)
from src.api.services.claim_anchor_guard import (
  calendar_block,
  check_claim,
  describe_published,
  find_dates,
  find_numbers,
  parse_published,
  source_anchors,
)
from src.api.services.news_claim_extract_service import _build_prompt

CASES = json.loads((Path(__file__).parent / "fixtures" / "claim_anchor_cases.json").read_text())["cases"]


def anchors_for(sources):
  return [source_anchors(i, s["content"], s.get("published_at")) for i, s in enumerate(sources)]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_case(case):
  problems = check_claim(case["claim"], anchors_for(case["sources"]))
  got = [(p.kind, p.token) for p in problems]
  want = [(e["kind"], e["token"]) for e in case["expect"]]
  assert got == want, [str(p) for p in problems]


# ── Parsing ────────────────────────────────────────────────────────────────


def test_find_dates_takes_the_fullest_match_and_reads_every_common_format():
  found = find_dates("On March 3, 2026 and 4 March, then March 2025, 2026-03-05, and in 2021; not 2,026 nor $2025.")
  assert [(m.text, m.day, m.month, m.year) for m in found] == [
    ("March 3, 2026", 3, 3, 2026),
    ("4 March", 4, 3, None),
    ("March 2025", None, 3, 2025),
    ("2026-03-05", 5, 3, 2026),
    ("2021", None, None, 2021),
  ]


def test_find_numbers_normalises_separators_and_scale_words_and_skips_dates_and_times():
  dates = find_dates("On May 12, 2026 at 10:30 the fund took $85.85 million, 7,281 BTC and 12.1%.")
  nums = find_numbers("On May 12, 2026 at 10:30 the fund took $85.85 million, 7,281 BTC and 12.1%.", [(d.start, d.end) for d in dates])
  assert [(n.digits, n.value) for n in nums] == [("85.85", 85_850_000.0), ("7281", 7281.0), ("12.1", 12.1)]


def test_parse_published_reads_dates_and_timestamps():
  assert parse_published("2026-03-05") == date(2026, 3, 5)
  assert parse_published("2026-05-12T10:05:02.000Z") == date(2026, 5, 12)
  assert parse_published("2026-04-17T02:54:00+00:00") == date(2026, 4, 17)
  assert parse_published(None) is None
  assert parse_published("last week") is None


# ── The calendar the model reads ───────────────────────────────────────────


def test_calendar_block_spells_out_each_weekday_for_each_source():
  block = calendar_block(anchors_for([
    {"content": "x", "published_at": "2026-03-05T14:00:00Z"},
    {"content": "y", "published_at": None},
  ]))
  assert "Source 0: published Thursday, 5 March 2026." in block
  assert "Tuesday = 3 March 2026" in block and "Thursday = 5 March 2026" in block and "Friday = 27 February 2026" in block
  assert "yesterday = 4 March 2026" in block
  assert "Source 1: publication date unknown" in block
  assert "The year of these sources is 2026." in block


def test_describe_published_adds_the_weekday_and_keeps_the_raw_value():
  assert describe_published("2026-03-05T14:00:00Z") == "Thursday, 5 March 2026 (2026-03-05T14:00:00Z)"
  assert describe_published("unknown") == "unknown"
  assert describe_published(None) is None


def test_extraction_prompt_carries_the_weekday_and_the_calendar():
  prompt = _build_prompt("H", [NewsArticleSource(index=0, url="u", title="t", content="On Tuesday it rained.", published_at="2026-03-05")], ["T"])
  assert "'published_at': 'Thursday, 5 March 2026 (2026-03-05)'" in prompt
  assert "CALENDAR (per source" in prompt
  assert "Tuesday = 3 March 2026" in prompt
  assert "never compute it yourself" in prompt


# ── Enforcement: repair once, then drop ────────────────────────────────────


def _result():
  return NewsClaimExtractResponse(
    claims=[
      ExtractedClaim(text="Trump announced on March 4, 2025 that the Navy could escort tankers", topic="Escorts", source_indices=[0]),
      ExtractedClaim(text="The strait carries about 20 million barrels a day", topic="Traffic", source_indices=[0]),
      ExtractedClaim(text="Iran laid mines during the war in 1988", topic="History", source_indices=[0]),
    ],
    quotes=[
      ExtractedQuote(text="escorts if necessary", speaker="Trump", claim_index=0),
      ExtractedQuote(text="twenty million a day", speaker=None, claim_index=1),
    ],
    collections=[
      ExtractedCollection(name="Escorts", type="topic", claim_indices=[0, 1]),
      ExtractedCollection(name="History", type="topic", claim_indices=[2]),
    ],
    collection_order=["Escorts", "History"],
    summary="s",
  )


SOURCES = [NewsArticleSource(
  index=0, url="u", title="Strait", content=(
    "On Tuesday, President Donald Trump announced that the U.S. Navy could provide escorts for tankers if necessary. "
    "About 20 million barrels a day pass the strait. Iran laid mines during the war in 1987 and 1988."
  ), published_at="2026-03-05T14:00:00Z",
)]


def test_enforce_repairs_the_flagged_claim_when_the_rewrite_passes_the_guard():
  seen = {}

  def call(prompt):
    seen["prompt"] = prompt
    return json.dumps({"claims": [{"index": 0, "text": "Trump announced on March 3, 2026 that the Navy could escort tankers"}]})

  out = enforce_anchors(_result(), SOURCES, call)
  assert out.claims[0].text == "Trump announced on March 3, 2026 that the Navy could escort tankers"
  assert len(out.claims) == 3
  assert out.anchor_report.model_dump() == {
    "checked": 3, "flagged": 1, "repaired": 1, "dropped": 0, "dropped_claims": [],
    "problems": ["repaired: date 'March 4, 2025': no source states this date; it says Tuesday and was published Thursday, 5 March 2026"],
  }
  assert "0. Trump announced on March 4, 2025" in seen["prompt"]
  assert "Tuesday = 3 March 2026" in seen["prompt"]
  assert "[0] Strait — published 2026-03-05T14:00:00Z" in seen["prompt"]


def test_enforce_drops_a_claim_the_repair_leaves_unanchored_and_renumbers_the_rest():
  call = lambda p: json.dumps({"claims": [{"index": 0, "text": "Trump announced on March 4, 2026 that the Navy could escort tankers"}]})
  out = enforce_anchors(_result(), SOURCES, call)
  assert [c.text for c in out.claims] == [
    "The strait carries about 20 million barrels a day",
    "Iran laid mines during the war in 1988",
  ]
  assert [(q.text, q.claim_index) for q in out.quotes] == [("twenty million a day", 0)]
  assert [(c.name, c.claim_indices) for c in out.collections] == [("Escorts", [0]), ("History", [1])]
  assert out.collection_order == ["Escorts", "History"]
  r = out.anchor_report
  assert (r.flagged, r.repaired, r.dropped) == (1, 0, 1)
  assert r.dropped_claims == ["Trump announced on March 4, 2025 that the Navy could escort tankers"]
  assert r.problems == ["dropped: date 'March 4, 2025': no source states this date; it says Tuesday and was published Thursday, 5 March 2026"]


def test_enforce_drops_when_the_repair_call_fails_and_when_it_is_absent():
  def boom(_):
    raise RuntimeError("quota")

  assert enforce_anchors(_result(), SOURCES, boom).anchor_report.dropped == 1
  assert enforce_anchors(_result(), SOURCES, None).anchor_report.dropped == 1


def test_enforce_touches_nothing_when_every_claim_is_anchored():
  res = _result()
  res.claims[0].text = "Trump announced on Tuesday that the Navy could escort tankers"
  calls = []
  out = enforce_anchors(res, SOURCES, lambda p: calls.append(p) or "{}")
  assert calls == []
  assert out.anchor_report.model_dump() == {"checked": 3, "flagged": 0, "repaired": 0, "dropped": 0, "dropped_claims": [], "problems": []}


def test_enforce_refuses_a_repair_that_grew_past_the_word_ceiling():
  long = "Trump announced on March 3, 2026 that the Navy could escort tankers " + "again and again " * 12
  out = enforce_anchors(_result(), SOURCES, lambda p: json.dumps({"claims": [{"index": 0, "text": long}]}))
  assert out.anchor_report.dropped == 1


def test_parse_anchor_repairs_ignores_junk_and_unknown_indices():
  raw = '```json\n{"claims": [{"index": "0", "text": " fixed "}, {"index": 5, "text": "x"}, {"index": 1}, "junk"]}\n```'
  assert parse_anchor_repairs(raw, [0, 1]) == {0: "fixed"}
  assert parse_anchor_repairs("not json", [0]) == {}


def test_drop_claims_with_nothing_to_drop_returns_the_input():
  res = _result()
  assert drop_claims(res, []) is res


def test_repair_prompt_names_each_problem_and_the_word_ceiling():
  anchors = anchors_for([{"content": SOURCES[0].content, "published_at": SOURCES[0].published_at}])
  problems = check_claim("Trump announced on March 4, 2025 that the Navy could escort tankers", anchors)
  prompt = build_anchor_repair_prompt([(0, "Trump announced on March 4, 2025 that the Navy could escort tankers", problems)], SOURCES, calendar_block(anchors))
  assert prompt.startswith("You fix dates and figures")
  assert "problem: date 'March 4, 2025': no source states this date; it says Tuesday and was published Thursday, 5 March 2026" in prompt
  assert "Keep each claim at 40 words or fewer" in prompt
