"""Semantic debate review: prompt contract, fail-closed application logic,
and the second opinion on the judgment gates."""

import pytest

from src.api.schemas.news_debate_claim_schema import (
    GroundedDebateCandidate,
    graded_strength,
)
from src.api.schemas.news_debate_semantic_review_schema import (
    DebateJudgmentOpinion,
    DebateJudgmentOpinionResponse,
    DebateSemanticReviewResponse,
    DebateSemanticVerdict,
)
import src.api.services.news_debate_semantic_review_service as review_service
from src.api.services.news_debate_semantic_review_service import (
    _build_opinion_prompt,
    _build_review_prompt,
    apply_semantic_review,
    failed_semantic_gates,
    judgment_rejected,
    merge_second_opinion,
    needs_second_opinion,
    review_news_debate_candidates,
)
from src.config.prompts.news_debate_semantic_review_prompt import (
    NEWS_DEBATE_JUDGMENT_OPINION_PROMPT,
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
        "on_headline": True,
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


def _opinion(**overrides) -> DebateJudgmentOpinion:
    data = {
        "candidate_index": 0,
        "on_headline": True,
        "real_societal_debate": True,
        "raised_by_story": True,
        "strength": 0.7,
    }
    data.update(overrides)
    return DebateJudgmentOpinion(**data)


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
    # Gate one carries the burden of proof for a rejection; the strength grade
    # follows the four gates.
    assert "Set true unless you can say who would NOT hold" in rendered
    assert "5. STRENGTH (`strength`)" in rendered
    assert '"strength": 0.8' in rendered
    # The headline question comes before the gates, with Armando's approval
    # story as the example; a judgment card on the central event is a debate
    # (the checker had rejected "the Iran war is the primary driver of Trump's
    # approval drop" as "an analytical question about polling causes").
    assert rendered.index("THE HEADLINE'S DISAGREEMENT (`on_headline`)") < rendered.index(
        "1. REAL SOCIETAL DEBATE"
    )
    assert "a card about the war alone is off it" in rendered
    assert "A judgment about the central event counts" in rendered
    assert '"on_headline": true' in rendered
    # Analyses capped at a sentence: review time tracks output length (30.6 s
    # -> 23.0 s on the same six-candidate sets), not the thinking level.
    assert "one short sentence, at most 15 words" in rendered
    assert "Do not reject a claim" in rendered
    assert "polices facts, not judgments" in rendered
    assert "Do not supply an overall pass field" in rendered
    # The reviewer examples must not repeat subjects/claims from our eval set.
    assert "Canada" not in rendered
    assert "Fauci" not in rendered
    assert "satellite" not in rendered.lower()


def test_review_grades_the_accepted_cards_and_orders_them_strongest_first():
    cards = [
        _candidate("ranked first by the writer"),
        _candidate("rejected"),
        _candidate("graded strongest"),
    ]
    verdicts = [
        _verdict(candidate_index=0, strength=0.5),
        _verdict(candidate_index=1, real_societal_debate=False, strength=0.9),
        _verdict(candidate_index=2, strength=0.8),
    ]
    accepted = apply_semantic_review(cards, verdicts, enforce=True)
    assert [c.text for c in accepted] == ["graded strongest", "ranked first by the writer"]
    # On-headline cards live in the upper half of the grade.
    assert [c.strength for c in accepted] == [0.9, 0.75]
    assert all(c.on_headline for c in accepted)


def test_a_card_on_the_headline_outranks_a_stronger_card_off_it():
    # The approval story: the war card graded 0.95, the approval card 0.6;
    # the published set must still lead with the approval card.
    assert graded_strength(0.95, on_headline=False) == 0.475
    assert graded_strength(0.6, on_headline=True) == 0.8
    cards = [_candidate("end the war"), _candidate("the war drove the approval drop")]
    verdicts = [
        _verdict(candidate_index=0, on_headline=False, strength=0.95),
        _verdict(candidate_index=1, on_headline=True, strength=0.6),
    ]
    accepted = apply_semantic_review(cards, verdicts, enforce=True)
    assert [c.text for c in accepted] == ["the war drove the approval drop", "end the war"]
    assert [c.on_headline for c in accepted] == [True, False]


def test_second_opinion_prompt_reasks_the_same_gates_without_the_sources():
    rendered = _build_opinion_prompt(
        "Council approves traffic plan",
        [],
        [_candidate("Council should widen the road")],
        [_candidate("Council's plan favours drivers over cyclists")],
    )
    assert "second opinion" in rendered
    assert "do not overturn it out of charity" in rendered
    # Shared text, not a paraphrase: the two readings differ in input only.
    for shared in (
        "THE HEADLINE'S DISAGREEMENT (`on_headline`)",
        "Set true unless you can say who would NOT hold",
        "A judgment about the central event counts",
        "2. RAISED BY THIS STORY (`raised_by_story`)",
    ):
        assert shared in rendered and shared in NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT
    assert "Council should widen the road" in rendered
    assert "Council's plan favours drivers over cyclists" in rendered
    assert "accepted cards (context only" in rendered
    # No sources block in its input: the smaller reading is the point.
    assert "{sources}" not in NEWS_DEBATE_JUDGMENT_OPINION_PROMPT
    assert "\nfull sources\n" not in rendered
    assert "invented" not in NEWS_DEBATE_JUDGMENT_OPINION_PROMPT.lower()
    # The reviewer never sees the grade fields the review itself fills in.
    assert "on_headline\": false" not in rendered and "strength\": 0.0" not in rendered


def test_judgment_rejected_names_only_cards_a_second_opinion_may_overturn():
    verdicts = [
        _verdict(candidate_index=0),
        _verdict(candidate_index=1, real_societal_debate=False),
        _verdict(candidate_index=2, raised_by_story=False, real_societal_debate=False),
        _verdict(candidate_index=3, real_societal_debate=False, invented_facts=["x"]),
        _verdict(candidate_index=4, distinct_axis=False, duplicate_of=0),
        _verdict(candidate_index=5, raised_by_story=False),
        _verdict(candidate_index=5, raised_by_story=False),
    ]
    assert [v.candidate_index for v in judgment_rejected(verdicts)] == [1, 2]


def test_second_opinion_is_needed_when_underfilled_or_off_the_headline():
    on = _candidate("on the headline")
    on.on_headline = True
    off = _candidate("off the headline")
    off.on_headline = False
    rejected_on = _verdict(candidate_index=3, on_headline=True, real_societal_debate=False)
    rejected_off = _verdict(candidate_index=4, on_headline=False, real_societal_debate=False)

    assert needs_second_opinion([off], [rejected_off]) is True
    assert needs_second_opinion([off, off], [rejected_on]) is True
    assert needs_second_opinion([off, off], [rejected_off]) is False
    assert needs_second_opinion([on, off], [rejected_on]) is False
    assert needs_second_opinion([], []) is False


def test_merge_passes_a_judgment_gate_either_reading_passed():
    first = _verdict(
        candidate_index=3,
        on_headline=False,
        real_societal_debate=False,
        debate_analysis="an analytical question about polling causes",
        strength=0.4,
        failure_codes=["NOT_SOCIETAL_DEBATE"],
    )
    merged = merge_second_opinion(
        first, _opinion(debate_analysis="partisans split on the cause", strength=0.7)
    )
    assert failed_semantic_gates(merged) == []
    assert merged.failure_codes == []
    assert merged.on_headline is True
    assert merged.strength == 0.7
    assert merged.debate_analysis == "partisans split on the cause"
    assert merged.candidate_index == 3

    still_rejected = merge_second_opinion(
        first, _opinion(real_societal_debate=False, strength=0.9)
    )
    assert failed_semantic_gates(still_rejected) == ["NOT_SOCIETAL_DEBATE"]
    assert still_rejected.strength == 0.4
    assert still_rejected.debate_analysis == first.debate_analysis


def _configured(monkeypatch, *, enabled=True):
    monkeypatch.setattr(review_service.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(review_service.settings, "news_debate_semantic_review_enforced", True)
    monkeypatch.setattr(review_service.settings, "news_debate_second_opinion_enabled", enabled)


def test_review_takes_one_second_opinion_on_the_judgment_rejects(monkeypatch):
    _configured(monkeypatch)
    cards = [_candidate("accepted"), _candidate("wrongly rejected"), _candidate("rightly rejected")]
    first = DebateSemanticReviewResponse(
        verdicts=[
            _verdict(candidate_index=0, strength=0.8),
            _verdict(candidate_index=1, real_societal_debate=False, strength=0.4),
            _verdict(candidate_index=2, raised_by_story=False, strength=0.3),
        ]
    )
    seen = {}

    def opinion(headline, claims, candidates, accepted):
        seen["candidates"] = [c.text for c in candidates]
        seen["accepted"] = [c.text for c in accepted]
        return DebateJudgmentOpinionResponse(
            verdicts=[
                _opinion(candidate_index=0, strength=0.6),
                _opinion(candidate_index=1, raised_by_story=False),
            ]
        )

    monkeypatch.setattr(review_service, "_gemini_review", lambda *a: first)
    monkeypatch.setattr(review_service, "_gemini_opinion", opinion)
    accepted, verdicts = review_news_debate_candidates("h", [], [], cards)

    assert seen == {
        "candidates": ["wrongly rejected", "rightly rejected"],
        "accepted": ["accepted"],
    }
    assert [c.text for c in accepted] == ["accepted", "wrongly rejected"]
    assert [c.strength for c in accepted] == [0.9, 0.8]
    # The audit carries the merged verdicts: one per candidate, gates final.
    assert [failed_semantic_gates(v) for v in verdicts] == [[], [], ["NOT_FROM_STORY"]]


def test_review_spends_no_second_opinion_on_a_full_set_or_when_told_not_to(monkeypatch):
    _configured(monkeypatch)
    cards = [_candidate("a"), _candidate("b"), _candidate("c")]
    full = DebateSemanticReviewResponse(
        verdicts=[
            _verdict(candidate_index=0),
            _verdict(candidate_index=1),
            _verdict(candidate_index=2, real_societal_debate=False),
        ]
    )
    thin = DebateSemanticReviewResponse(
        verdicts=[
            _verdict(candidate_index=0),
            _verdict(candidate_index=1, real_societal_debate=False),
            _verdict(candidate_index=2, real_societal_debate=False),
        ]
    )

    def unexpected(*args):
        raise AssertionError("no second opinion should run")

    monkeypatch.setattr(review_service, "_gemini_opinion", unexpected)
    monkeypatch.setattr(review_service, "_gemini_review", lambda *a: full)
    assert len(review_news_debate_candidates("h", [], [], cards)[0]) == 2

    monkeypatch.setattr(review_service, "_gemini_review", lambda *a: thin)
    assert len(review_news_debate_candidates("h", [], [], cards, second_opinion=False)[0]) == 1
    _configured(monkeypatch, enabled=False)
    assert len(review_news_debate_candidates("h", [], [], cards)[0]) == 1


def test_a_failed_second_opinion_leaves_the_first_verdicts(monkeypatch):
    _configured(monkeypatch)
    cards = [_candidate("a"), _candidate("b")]
    thin = DebateSemanticReviewResponse(
        verdicts=[_verdict(candidate_index=0), _verdict(candidate_index=1, raised_by_story=False)]
    )

    def failing(*args):
        raise Exception("provider down")

    monkeypatch.setattr(review_service, "_gemini_review", lambda *a: thin)
    monkeypatch.setattr(review_service, "_gemini_opinion", failing)
    accepted, verdicts = review_news_debate_candidates("h", [], [], cards)
    assert [c.text for c in accepted] == ["a"]
    assert failed_semantic_gates(verdicts[1]) == ["NOT_FROM_STORY"]


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
