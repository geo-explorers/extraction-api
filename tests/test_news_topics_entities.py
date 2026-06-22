"""Unit tests for story-level topic+entity extraction (LLM-free parts).

Exercises the response validator (incl. the legacy curated_topics/free_topics
fallback), the curated-vs-free classification + 0.6 relevance gate, and the
prompt assembly — all ported from news-worker's extractTopicsAndEntities —
without calling Claude.
"""

import pytest

from src.api.services.news_topics_entities_service import (
    CURATED_MIN_RELEVANCE,
    classify_topics,
    validate_topics_entities_response,
)
from src.config.prompts.news_topics_entities_prompt import (
    ENTITY_TYPE_PROMPT,
    build_user_prompt,
)


# ── validate_topics_entities_response ─────────────────────────────────────


def test_validate_basic_topics_and_entities():
    out = validate_topics_entities_response(
        {
            "topics": [{"name": "Regulation", "relevance": 0.9}],
            "entities": [{"name": "Ripple", "type": "Project", "role": "subject"}],
        }
    )
    assert out["topics"] == [{"name": "Regulation", "relevance": 0.9}]
    assert out["entities"] == [{"name": "Ripple", "type": "Project", "role": "subject"}]


def test_validate_relevance_defaults_to_zero_when_missing():
    out = validate_topics_entities_response({"topics": [{"name": "X"}], "entities": []})
    assert out["topics"] == [{"name": "X", "relevance": 0}]


def test_validate_entity_role_defaults_to_empty():
    out = validate_topics_entities_response(
        {"topics": [], "entities": [{"name": "Ripple", "type": "Project"}]}
    )
    assert out["entities"][0]["role"] == ""


def test_validate_legacy_curated_and_free_fallback():
    # No `topics` -> fall back to curated_topics + free_topics, in that order.
    out = validate_topics_entities_response(
        {
            "curated_topics": [{"name": "DeFi", "relevance": 0.8}],
            "free_topics": [{"name": "Hacks", "relevance": 0.5}],
        }
    )
    assert [t["name"] for t in out["topics"]] == ["DeFi", "Hacks"]


def test_validate_prefers_topics_over_legacy_when_present():
    out = validate_topics_entities_response(
        {
            "topics": [{"name": "A", "relevance": 1}],
            "curated_topics": [{"name": "B", "relevance": 1}],
        }
    )
    assert [t["name"] for t in out["topics"]] == ["A"]


@pytest.mark.parametrize("bad", ["notobj", 42, ["a"], None])
def test_validate_rejects_non_object_root(bad):
    with pytest.raises(ValueError):
        validate_topics_entities_response(bad)


def test_validate_rejects_topic_missing_name():
    with pytest.raises(ValueError):
        validate_topics_entities_response({"topics": [{"relevance": 0.5}], "entities": []})


def test_validate_rejects_entity_missing_type():
    with pytest.raises(ValueError):
        validate_topics_entities_response({"topics": [], "entities": [{"name": "X"}]})


def test_validate_rejects_non_number_relevance():
    with pytest.raises(ValueError):
        validate_topics_entities_response(
            {"topics": [{"name": "X", "relevance": "high"}], "entities": []}
        )


# ── classify_topics (curated vs free, 0.6 gate) ──────────────────────────


def test_classify_curated_by_exact_name():
    out = classify_topics([{"name": "AI regulation", "relevance": 0.9}], ["AI regulation"])
    assert out == [{"name": "AI regulation", "relevance": 0.9, "source": "curated"}]


def test_classify_free_when_not_in_curated():
    out = classify_topics([{"name": "Random", "relevance": 0.1}], ["AI regulation"])
    assert out == [{"name": "Random", "relevance": 0.1, "source": "llm"}]


def test_classify_curated_gate_drops_below_threshold():
    out = classify_topics(
        [{"name": "Cur", "relevance": 0.59}, {"name": "Cur2", "relevance": 0.6}],
        ["Cur", "Cur2"],
    )
    assert [t["name"] for t in out] == ["Cur2"]  # 0.59 curated dropped, 0.6 kept


def test_classify_free_kept_regardless_of_low_relevance():
    out = classify_topics([{"name": "Free", "relevance": 0.05}], [])
    assert out == [{"name": "Free", "relevance": 0.05, "source": "llm"}]


def test_classify_drops_zero_relevance():
    assert classify_topics([{"name": "Z", "relevance": 0}], []) == []


def test_classify_is_case_sensitive():
    out = classify_topics([{"name": "ai regulation", "relevance": 0.9}], ["AI regulation"])
    assert out[0]["source"] == "llm"  # case mismatch -> not curated


def test_curated_min_relevance_constant():
    assert CURATED_MIN_RELEVANCE == 0.6


# ── build_user_prompt ─────────────────────────────────────────────────────


def test_prompt_includes_curated_rule_and_entity_types_when_curated():
    p = build_user_prompt("Headline", "Summary", has_curated=True)
    assert "CURATED TOPIC VALIDATION" in p
    assert ENTITY_TYPE_PROMPT in p
    assert '"name": "Regulation"' in p  # JSON example present, braces intact
    assert "Headline" in p and "Summary" in p


def test_prompt_omits_curated_rule_when_no_curated():
    p = build_user_prompt("H", "S", has_curated=False)
    assert "CURATED TOPIC VALIDATION" not in p
    assert "- topics: 3-10 topic labels relevant to the story." in p
