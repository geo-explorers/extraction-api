"""Unit tests for the news overview-topic pass (LLM-free parts).

Exercises the label validators, the response coercion, the adaptive content
pooling, and the too-little-content degrade path — all the logic ported from
news-worker's injection branch — without calling Claude. The single LLM call
(_call_claude_topics) is intentionally not exercised here; it is covered by the
local end-to-end run in a later phase.
"""

import pytest

from src.api.schemas.news_claim_extract_schema import NewsArticleSource
from src.api.services.news_topics_extract_service import (
    LABEL_MAX_WORDS,
    check_label_issues,
    extract_overview_topics,
    find_label_issues,
    pool_source_content,
    validate_overview_topics_response,
)


def _src(url: str, content: str, title: str = "Title") -> NewsArticleSource:
    return NewsArticleSource(index=0, url=url, title=title, content=content)


# ── check_label_issues ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Stablecoin regulation", []),               # 2 words, clean
        ("Japan earthquake response", []),           # 3 words, clean
        ("Gene therapy clinical trial results", []),  # 5 words, clean (at max)
        ("AI", ["1 words (min 2)"]),                 # too few words
        ("one two three four five six", ["6 words (max 5)"]),  # too many
        ("Crypto and regulation", ['uses "and" conjunction']),
        ("", ["empty"]),
        ("   ", ["empty"]),
    ],
)
def test_check_label_issues(label, expected):
    assert check_label_issues(label) == expected


def test_and_conjunction_requires_whitespace_both_sides():
    # Substrings of words must NOT trip the conjunction rule.
    assert check_label_issues("Android phone launch") == []
    assert check_label_issues("England match report") == []
    assert check_label_issues("Sandal trade dispute") == []
    # But a real conjunction does.
    assert 'uses "and" conjunction' in check_label_issues("war and peace")


def test_check_label_issues_can_stack():
    # 6 words AND an "and" conjunction -> both issues reported.
    issues = check_label_issues("alpha and beta gamma delta epsilon")
    assert "6 words (max 5)" in issues
    assert 'uses "and" conjunction' in issues


def test_find_label_issues_returns_only_problematic():
    labels = ["Clean topic", "AI", "X and Y"]
    issues = find_label_issues(labels)
    flagged = {entry["label"] for entry in issues}
    assert flagged == {"AI", "X and Y"}
    # Shape matches what the regenerate feedback block consumes.
    for entry in issues:
        assert set(entry) == {"label", "issues"}
        assert entry["issues"]  # non-empty


def test_label_max_words_constant_matches_source():
    assert LABEL_MAX_WORDS == 5


# ── validate_overview_topics_response ────────────────────────────────────


def test_validate_accepts_bare_array_and_filters_blanks():
    assert validate_overview_topics_response(["a", "", "  ", "b"]) == ["a", "b"]


def test_validate_drops_non_strings():
    assert validate_overview_topics_response(["a", None, 3, {"x": 1}, "b"]) == ["a", "b"]


def test_validate_accepts_topics_wrapper():
    assert validate_overview_topics_response({"topics": ["x", " ", "y"]}) == ["x", "y"]


def test_validate_empty_array_is_empty_list():
    assert validate_overview_topics_response([]) == []


@pytest.mark.parametrize("bad", ["nope", 42, {"foo": "bar"}, {"topics": "x"}, None])
def test_validate_rejects_unexpected_shapes(bad):
    with pytest.raises(ValueError):
        validate_overview_topics_response(bad)


# ── pool_source_content ──────────────────────────────────────────────────


def test_pool_single_long_source_used_alone():
    body = "x" * 2000
    assert pool_source_content("H", [_src("u1", body)]) == body


def test_pool_single_short_source_used_alone():
    body = "short body about a thing"
    assert pool_source_content("H", [_src("u1", body)]) == body


def test_pool_uses_longest_alone_when_above_threshold():
    long_body = "a" * 1600  # >= 1500 threshold
    short_body = "b" * 100
    out = pool_source_content("H", [_src("u1", long_body), _src("u2", short_body)])
    assert out == long_body


def test_pool_combines_thin_multi_sources_in_length_order():
    s_short = _src("u-short", "b" * 100)
    s_long = _src("u-long", "a" * 300)  # both < 1500 -> pool, longest first
    out = pool_source_content("H", [s_short, s_long])
    assert out.startswith("Source 1 (u-long):\n")
    assert "\n\n---\n\nSource 2 (u-short):\n" in out


def test_pool_falls_back_to_title_when_body_empty():
    title = "T" * 60  # > 50-char fallback threshold
    assert pool_source_content("H", [_src("u1", "", title=title)]) == title


def test_pool_ignores_thin_title_and_falls_back_to_headline():
    out = pool_source_content("My headline", [_src("u1", "", title="short")])
    assert out == "My headline\n\n"


def test_pool_no_sources_returns_headline():
    assert pool_source_content("Just a headline", []) == "Just a headline\n\n"


# ── extract_overview_topics degrade path (no LLM call) ───────────────────


def test_extract_topics_degrades_when_too_little_content():
    # Pooled content is well under 100 chars, so it returns the fallback topic
    # WITHOUT ever constructing an Anthropic client / making a network call.
    assert extract_overview_topics("hi", []) == ["Overview"]
    assert extract_overview_topics("tiny", [_src("u1", "few words")]) == ["Overview"]
