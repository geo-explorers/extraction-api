"""Debate claims (Step 8 of the news prompt): schema contract + deterministic
cardinality enforcement.

The prompt asks for 0 or 2-4 debate claims, but a count is exactly the kind of
instruction a model occasionally ignores — these tests pin the repair layer
that makes the contract hold regardless (normalize_debate_claims, hooked as a
model_validator on both response models). No LLM calls.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas.news_claim_extract_schema import (
    ExtractedDebateClaim,
    NewsClaimExtractResponse,
    normalize_debate_claims,
)
from src.api.schemas.news_topics_and_claims_schema import NewsTopicsAndClaimsResponse
from src.config.prompts.news_claim_extract_prompt import NEWS_CLAIM_EXTRACT_PROMPT


def _claims(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


# ── Schema shape ────────────────────────────────────────────────────────


def test_missing_field_defaults_to_empty():
    resp = NewsClaimExtractResponse.model_validate({"claims": [], "summary": "s"})
    assert resp.debate_claims == []


def test_round_trip_through_both_response_models():
    payload = _claims("EU AI regulation is justified", "Bitcoin will outperform gold by 2030")
    fused = NewsTopicsAndClaimsResponse(
        topics=["t"],
        debate_claims=NewsClaimExtractResponse.model_validate(
            {"debate_claims": payload}
        ).debate_claims,
    )
    dumped = fused.model_dump()["debate_claims"]
    assert [d["text"] for d in dumped] == [c["text"] for c in payload]


def test_confidence_defaults_and_bounds():
    assert ExtractedDebateClaim.model_validate({"text": "t"}).confidence == 0.8
    with pytest.raises(ValidationError):
        ExtractedDebateClaim.model_validate({"text": "t", "confidence": -0.1})
    with pytest.raises(ValidationError):
        ExtractedDebateClaim.model_validate({"text": "t", "confidence": 1.1})


def test_legacy_checkpoint_without_field_still_validates():
    # A run checkpointed before the field existed replays through finalize via
    # cr.get("debate_claims", []) — the model must accept that shape.
    resp = NewsTopicsAndClaimsResponse(topics=["t"], claims=[], debate_claims=[])
    assert resp.debate_claims == []


# ── Cardinality repair (0 or 2-4, never one) ────────────────────────────


def test_exactly_one_becomes_empty():
    resp = NewsClaimExtractResponse.model_validate({"debate_claims": _claims("lonely position")})
    assert resp.debate_claims == []


def test_five_capped_at_first_four():
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("a", "b", "c", "d", "e")}
    )
    assert [c.text for c in resp.debate_claims] == ["a", "b", "c", "d"]


def test_duplicates_collapse_before_the_one_rule():
    # A pair of identical texts is one claim, and one claim is none at all.
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("Same position.", "  same   POSITION. ")}
    )
    assert resp.debate_claims == []


def test_two_through_four_pass_untouched():
    for n in (2, 3, 4):
        texts = [f"distinct position {i}" for i in range(n)]
        resp = NewsClaimExtractResponse.model_validate({"debate_claims": _claims(*texts)})
        assert [c.text for c in resp.debate_claims] == texts


def test_empty_text_entries_are_dropped():
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("", "real position a", "real position b")}
    )
    assert [c.text for c in resp.debate_claims] == ["real position a", "real position b"]


def test_fused_model_enforces_the_same_contract():
    resp = NewsTopicsAndClaimsResponse(
        topics=["t"],
        debate_claims=[ExtractedDebateClaim(text="only one")],
    )
    assert resp.debate_claims == []


def test_normalizer_is_deterministic_and_idempotent():
    once = normalize_debate_claims([ExtractedDebateClaim(text=f"p{i}") for i in range(6)])
    twice = normalize_debate_claims(once)
    assert [c.text for c in once] == [c.text for c in twice]
    assert len(once) == 4


# ── Prompt smoke (markers, not prose) ───────────────────────────────────


def test_prompt_renders_and_carries_the_contract():
    rendered = NEWS_CLAIM_EXTRACT_PROMPT.format(headline="h", sources=[], topics=[])
    assert "debate_claims" in rendered
    assert "2-4 debate claims" in rendered
    assert "Never return exactly one" in rendered
    # Product-shape guardrail (mirrored in news-worker lib/debate-claims.ts):
    # no pro/con pair of one question — each claim is its own debate.
    assert "becomes its OWN debate" in rendered
    assert "reduce each candidate to the neutral question" in rendered
    assert "BAD pair:" in rendered and "GOOD pair:" in rendered
    # The JSON example must not model the forbidden count of one. The example
    # array runs from the debate_claims key to the summary key that follows it.
    example = rendered.split('"debate_claims"')[1].split('"summary"')[0]
    assert example.count('"text"') >= 2
