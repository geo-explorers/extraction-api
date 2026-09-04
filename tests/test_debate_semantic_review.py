"""Semantic debate review: prompt contract and fail-closed application logic."""

import pytest

from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.schemas.news_debate_semantic_review_schema import DebateSemanticVerdict
from src.api.services.news_debate_semantic_review_service import (
    _build_review_prompt,
    apply_semantic_review,
    failed_semantic_gates,
)
from src.config.prompts.news_debate_semantic_review_prompt import (
    NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT,
)


def _candidate(text: str = "Canada should preserve supply management"):
    return GroundedDebateCandidate(
        text=text,
        neutral_question="Should Canada preserve supply management?",
    )


def _verdict(**overrides) -> DebateSemanticVerdict:
    data = {
        "candidate_index": 0,
        "real_societal_debate": True,
        "raised_by_story": True,
        "invented_facts": [],
        "no_invented_facts": True,
        "distinct_axis": True,
        "duplicate_of": None,
        "failure_codes": [],
    }
    data.update(overrides)
    return DebateSemanticVerdict(**data)


def test_review_prompt_is_reject_only_and_carries_every_gate():
    rendered = NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT.format(
        headline="h", claims="[]", candidates="[]", prior_axes="[]", sources="[]"
    )
    assert "REJECT-ONLY reviewer" in rendered
    # The definition, in the team's own terms.
    assert "sounds like a headline" in rendered
    assert "large or significant groups" in rendered
    # All four gates.
    assert "REAL SOCIETAL DEBATE" in rendered
    assert "RAISED BY THIS STORY" in rendered
    assert "NO INVENTED FACTS" in rendered
    assert "DISTINCT DEBATE AXIS" in rendered
    # The judgment lines that keep the gates honest.
    assert "business tactic" in rendered
    assert "market speculation" in rendered
    assert "allocation or strategy advice" in rendered
    assert "Do not reject a claim" in rendered
    assert "polices facts, not judgments" in rendered
    assert "Do not supply an overall pass field" in rendered
    # The reviewer examples must not repeat subjects/claims from our eval set.
    assert "Canada" not in rendered
    assert "Fauci" not in rendered
    assert "satellite" not in rendered.lower()


def test_completion_review_separates_current_candidates_from_prior_axes():
    current = _candidate("Canada should reform supply management")
    prior = _candidate("Canada should retain supply management")
    prompt = _build_review_prompt(
        "Canada changes dairy policy", [], [], [current], [prior]
    )
    assert '"candidate_index": 0' in prompt
    assert current.text in prompt
    assert prior.text in prompt
    assert "prior axes (context only; do not return verdicts for these)" in prompt
    assert "leave `duplicate_of` null" in prompt


def test_candidate_passes_only_when_every_gate_passes():
    candidate = _candidate()
    assert apply_semantic_review(
        [candidate], [_verdict()], enforce=True
    ) == [candidate]
    assert failed_semantic_gates(_verdict()) == []


@pytest.mark.parametrize(
    "gate",
    [
        "real_societal_debate",
        "raised_by_story",
        "no_invented_facts",
        "distinct_axis",
    ],
)
def test_each_failed_gate_rejects_the_candidate(gate):
    verdict = _verdict(**{gate: False})
    assert apply_semantic_review([_candidate()], [verdict], enforce=True) == []
    assert failed_semantic_gates(verdict)


def test_duplicate_reference_rejects_even_if_distinct_boolean_is_wrong():
    verdict = _verdict(distinct_axis=True, duplicate_of=0)
    assert apply_semantic_review([_candidate()], [verdict], enforce=True) == []
    assert "DUPLICATE_AXIS" in failed_semantic_gates(verdict)


def test_any_invented_fact_rejects_even_when_boolean_is_true():
    verdict = _verdict(invented_facts=["an unreported subsidy program"])
    assert apply_semantic_review([_candidate()], [verdict], enforce=True) == []
    assert "INVENTED_FACTS" in failed_semantic_gates(verdict)


def test_missing_or_duplicate_verdict_fails_closed():
    candidate = _candidate()
    assert apply_semantic_review([candidate], [], enforce=True) == []
    assert apply_semantic_review(
        [candidate], [_verdict(), _verdict()], enforce=True
    ) == []


def test_shadow_mode_records_failure_without_filtering():
    candidate = _candidate()
    assert apply_semantic_review(
        [candidate], [_verdict(real_societal_debate=False)], enforce=False
    ) == [candidate]


def test_application_computes_gates_instead_of_trusting_failure_codes():
    candidate = _candidate()
    passes_with_bad_metadata = _verdict(failure_codes=["NOT_SOCIETAL_DEBATE"])
    assert apply_semantic_review(
        [candidate], [passes_with_bad_metadata], enforce=True
    ) == [candidate]

    fails_without_metadata = _verdict(real_societal_debate=False, failure_codes=[])
    assert apply_semantic_review(
        [candidate], [fails_without_metadata], enforce=True
    ) == []
