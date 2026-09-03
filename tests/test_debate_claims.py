"""Debate claims (Step 8 of the news prompt): schema contract + deterministic
cardinality enforcement.

The product requires 0 or 3-5 debate claims, but a count is exactly the kind of
instruction a model occasionally ignores — these tests pin the repair layer
that makes the contract hold regardless (normalize_debate_claims, hooked as a
model_validator on both response models). No LLM calls.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas.news_claim_extract_schema import (
    ExtractedDebateClaim,
    ExtractedClaim,
    NewsArticleSource,
    NewsClaimExtractResponse,
    normalize_debate_claims,
)
from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.services.news_debate_claim_service import (
    _build_prompt as build_debate_prompt,
    filter_grounded_debate_candidates,
    project_debate_candidates,
)
from src.config.prompts.news_debate_claim_prompt import NEWS_DEBATE_CLAIM_PROMPT
from src.api.schemas.news_topics_and_claims_schema import NewsTopicsAndClaimsResponse
from src.config.prompts.news_claim_extract_prompt import NEWS_CLAIM_EXTRACT_PROMPT


def _claims(*texts: str) -> list[dict]:
    return [{"text": t} for t in texts]


# ── Schema shape ────────────────────────────────────────────────────────


def test_missing_field_defaults_to_empty():
    resp = NewsClaimExtractResponse.model_validate({"claims": [], "summary": "s"})
    assert resp.debate_claims == []


def test_round_trip_through_both_response_models():
    payload = _claims("position one", "position two", "position three")
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


# ── Cardinality repair (0 or 3-5) ──────────────────────────────────────


def test_one_or_two_are_omitted_instead_of_published_as_thin_collection():
    resp = NewsClaimExtractResponse.model_validate({"debate_claims": _claims("lonely position")})
    assert resp.debate_claims == []
    resp = NewsClaimExtractResponse.model_validate({"debate_claims": _claims("one", "two")})
    assert resp.debate_claims == []


def test_six_capped_at_first_five():
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("a", "b", "c", "d", "e", "f")}
    )
    assert [c.text for c in resp.debate_claims] == ["a", "b", "c", "d", "e"]


def test_duplicates_that_leave_under_three_omit_the_collection():
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("Same position.", "  same   POSITION. ")}
    )
    assert resp.debate_claims == []


def test_three_through_five_pass_untouched():
    for n in (3, 4, 5):
        texts = [f"distinct position {i}" for i in range(n)]
        resp = NewsClaimExtractResponse.model_validate({"debate_claims": _claims(*texts)})
        assert [c.text for c in resp.debate_claims] == texts


def test_empty_text_entries_are_dropped():
    resp = NewsClaimExtractResponse.model_validate(
        {"debate_claims": _claims("", "real position a", "real position b", "real position c")}
    )
    assert [c.text for c in resp.debate_claims] == ["real position a", "real position b", "real position c"]


def test_fused_model_omits_an_underfilled_collection():
    resp = NewsTopicsAndClaimsResponse(
        topics=["t"],
        debate_claims=[ExtractedDebateClaim(text="only one")],
    )
    assert resp.debate_claims == []


def test_normalizer_is_deterministic_and_idempotent():
    once = normalize_debate_claims([ExtractedDebateClaim(text=f"p{i}") for i in range(6)])
    twice = normalize_debate_claims(once)
    assert [c.text for c in once] == [c.text for c in twice]
    assert len(once) == 5


# ── Prompt smoke (markers, not prose) ───────────────────────────────────


def test_factual_prompt_defers_debates_to_the_dedicated_pass():
    rendered = NEWS_CLAIM_EXTRACT_PROMPT.format(headline="h", sources=[], topics=[])
    assert "STEP 8: DEBATE PLACEHOLDER" in rendered
    assert "Always return an empty `debate_claims` array" in rendered
    assert '"debate_claims": []' in rendered
    assert "Candidate-discovery sweep" not in rendered
    assert "ONE deliberate exception" not in rendered


# ── Dedicated debate pass ─────────────────────────────────────────────


_SOURCE_TEXT = (
    "Canada's government said supply management protects dairy and poultry "
    "farmers while U.S. negotiators demanded broader market access."
)


def _sources() -> list[NewsArticleSource]:
    return [
        NewsArticleSource(
            index=7,
            url="https://example.com/story",
            title="Trade talks",
            content=_SOURCE_TEXT,
        )
    ]


def _factual_claims() -> list[ExtractedClaim]:
    return [
        ExtractedClaim(
            text="U.S. negotiators demanded broader Canadian market access.",
            topic="Trade talks",
            source_indices=[7],
            importance=0.9,
        ),
        ExtractedClaim(
            text="A previous negotiation occurred in 2022.",
            topic="Background",
            source_indices=[7],
            importance=0.4,
        ),
    ]


def _candidate(**overrides) -> GroundedDebateCandidate:
    data = {
        "neutral_question": "Should Canada preserve agricultural supply management?",
        "opposing_positions": [
            "Farm groups: preserve the system",
            "Trade liberals: open the market",
        ],
        "source_indices": [7],
        "text": "Canada should preserve agricultural supply management.",
    }
    data.update(overrides)
    return GroundedDebateCandidate(**data)


def test_dedicated_prompt_carries_the_product_definition():
    rendered = NEWS_DEBATE_CLAIM_PROMPT.format(
        headline="Trade talks",
        central_claims="[]",
        sources="[]",
    )
    # The definition, in the team's own terms.
    assert "sounds like a headline" in rendered
    assert "large or significant groups" in rendered
    assert "3-5 debate claims" in rendered
    # Discovery lenses, including the societal-instance lens.
    assert "Policy or response" in rendered
    assert "Societal instance" in rendered
    # Exclusions.
    assert "business tactic" in rendered
    assert "market-performance forecast" in rendered
    assert "Allocation or strategy advice" in rendered
    # One claim per question.
    assert "Every returned claim becomes its OWN debate" in rendered
    assert "BAD pair" in rendered and "GOOD pair" in rendered
    assert "neutral question" in rendered
    # Card style, including the soft form-variety preference.
    assert "prefer form variety" in rendered
    assert "manufacture variety" in rendered
    assert "Aim for 6-10 words" in rendered
    assert "20 words is the hard maximum" in rendered
    assert "Orion should disclose its automated hiring criteria" in rendered
    assert "Aster's battery design poses unacceptable safety risks" in rendered
    assert "Riverton's housing shortage is driven by zoning restrictions" in rendered
    assert "Mosaic fusion power will be commercially viable by 2040" in rendered
    # Holdout topics must not be echoed from style examples into eval runs.
    assert "Bitcoin" not in rendered
    assert "Open-source AI" not in rendered


def test_dedicated_prompt_receives_full_sources_and_all_factual_claims():
    rendered = build_debate_prompt("Trade talks", _sources(), _factual_claims())
    assert _SOURCE_TEXT in rendered
    assert "U.S. negotiators demanded broader Canadian market access." in rendered
    assert "A previous negotiation occurred in 2022." in rendered


def test_grounded_candidate_is_projected_to_public_contract():
    candidates = [
        _candidate(),
        _candidate(
            neutral_question="Should Canada compensate protected farmers?",
            text="Canada should compensate protected agricultural producers.",
        ),
        _candidate(
            neutral_question="Should Canada reform agricultural market rules?",
            text="Canada should reform agricultural market rules.",
        ),
    ]
    result = project_debate_candidates(
        filter_grounded_debate_candidates(candidates, _sources(), _factual_claims())
    )
    assert [claim.text for claim in result] == [candidate.text for candidate in candidates]
    assert all(claim.source_indices == [7] for claim in result)


def test_public_projection_omits_one_or_two_grounded_candidates():
    assert project_debate_candidates([_candidate()]) == []


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(text="Canada may preserve agricultural supply management."),
        _candidate(text="Canada's negotiations signal financial health."),
        _candidate(text="Should Canada preserve supply management?"),
        _candidate(neutral_question=""),
        _candidate(opposing_positions=["Only one side"]),
        _candidate(opposing_positions=["Side one", "  "]),
        _candidate(text="one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone"),
    ],
)
def test_malformed_candidates_are_rejected(candidate):
    assert filter_grounded_debate_candidates(
        [candidate], _sources(), _factual_claims()
    ) == []


def test_unknown_source_indices_are_dropped_not_fatal():
    candidate = _candidate(source_indices=[7, 99, 7])
    result = filter_grounded_debate_candidates(
        [candidate], _sources(), _factual_claims()
    )
    assert len(result) == 1
    assert result[0].source_indices == [7]


def test_duplicate_questions_collapse_across_typography():
    # Curly vs straight apostrophes render the same question; the dedupe key
    # must fold them so the rescue pass cannot re-ask an attempted question.
    second = _candidate(
        neutral_question="Should Canada preserve agriculture\u2019s supply management?",
        text="Canada should abandon agricultural supply management.",
    )
    first = _candidate(
        neutral_question="Should Canada preserve agriculture's supply management?",
    )
    result = filter_grounded_debate_candidates(
        [first, second], _sources(), _factual_claims()
    )
    assert [candidate.text for candidate in result] == [first.text]


def test_duplicate_neutral_questions_collapse_even_when_text_differs():
    second = _candidate(
        text="Canada should abandon agricultural supply management.",
        neutral_question="Should Canada preserve agricultural supply management?",
    )
    result = filter_grounded_debate_candidates(
        [_candidate(), second], _sources(), _factual_claims()
    )
    assert [candidate.text for candidate in result] == [
        "Canada should preserve agricultural supply management."
    ]


def test_complete_extractor_replaces_untrusted_first_pass_debates(monkeypatch):
    from src.api.services import news_claim_extract_service as service

    first_pass = NewsClaimExtractResponse(
        claims=_factual_claims(),
        debate_claims=[
            ExtractedDebateClaim(text=f"Untrusted provisional debate {index}")
            for index in range(3)
        ],
    )
    grounded = [
        ExtractedDebateClaim(
            text=text,
            source_indices=[7],
        )
        for text in (
            "Canada should preserve agricultural supply management.",
            "Canada should compensate protected agricultural producers.",
            "Canada should reform agricultural market rules.",
        )
    ]
    monkeypatch.setattr(service, "extract_news_claims_factual", lambda *args: first_pass)
    monkeypatch.setattr(service, "extract_news_debate_claims", lambda *args: grounded)

    result = service.extract_news_claims("Trade talks", _sources(), ["Trade talks"])

    assert result.debate_claims == grounded
    assert "Untrusted" not in result.debate_claims[0].text
