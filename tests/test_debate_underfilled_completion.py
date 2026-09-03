"""Underfilled 3-5 collection completion orchestration; no provider calls."""

import src.api.services.news_debate_claim_service as service
from src.api.schemas.news_claim_extract_schema import ExtractedClaim, NewsArticleSource
from src.api.schemas.news_debate_claim_schema import GroundedDebateCandidate
from src.api.schemas.news_debate_semantic_review_schema import DebateSemanticVerdict


def _candidate(number: int) -> GroundedDebateCandidate:
    return GroundedDebateCandidate(
        neutral_question=f"Should the council take action {number}?",
        opposing_positions=[
            "Residents' association: support it",
            "Business coalition: oppose it",
        ],
        source_indices=[3],
        text=f"The council should take distinct action {number}.",
    )


def _verdict(index: int, **overrides) -> DebateSemanticVerdict:
    data = {
        "candidate_index": index,
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


def _story():
    sources = [
        NewsArticleSource(
            index=3,
            title="Council vote",
            url="https://example.com",
            content="The council approved a downtown traffic plan after a public vote.",
        )
    ]
    claims = [
        ExtractedClaim(
            text="The council approved a downtown traffic plan.",
            topic="Traffic plan",
            source_indices=[3],
            importance=0.9,
        )
    ]
    return sources, claims


def test_completion_prompt_carries_count_survivors_and_rejection_audit():
    sources, claims = _story()
    survivors = [_candidate(0), _candidate(1)]
    rejected = _candidate(2)
    prompt = service._build_underfilled_rescue_prompt(
        "Council approves traffic plan",
        sources,
        claims,
        survivors,
        [*survivors, rejected],
        [
            _verdict(0),
            _verdict(1),
            _verdict(2, invented_facts=["invented mechanism"]),
        ],
    )
    assert "left 2 publishable" in prompt
    assert "1 and at most 3 NEW" in prompt
    assert all(survivor.neutral_question in prompt for survivor in survivors)
    assert rejected.neutral_question in prompt
    assert "INVENTED_FACTS" in prompt
    assert "proposition and its counterclaim" in prompt
    assert "sounds like a headline" in prompt
    assert "societal instance" in prompt
    assert "prefer a form the" in prompt
    assert "market-performance forecast" in prompt
    assert "allocation or strategy advice" in prompt


def test_completion_does_not_run_once_collection_has_three(monkeypatch):
    sources, claims = _story()
    accepted = [_candidate(i) for i in range(3)]

    def unexpected(*args, **kwargs):
        raise AssertionError("completion generation should not run")

    monkeypatch.setattr(service, "generate_news_debate_underfilled_rescue", unexpected)
    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    assert service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, accepted, accepted, []
    ) == (accepted, [])


def test_zero_survivors_with_candidates_gets_one_fresh_redraw(monkeypatch):
    sources, claims = _story()
    redraw = [_candidate(i) for i in range(3)]

    monkeypatch.setattr(service.settings, "news_debate_zero_retry_enabled", True)
    monkeypatch.setattr(
        service, "generate_news_debate_candidates", lambda *args: redraw
    )
    monkeypatch.setattr(
        service,
        "review_news_debate_candidates",
        lambda *args, **kwargs: (redraw, [_verdict(i) for i in range(3)]),
    )
    completed, audit = service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, [_candidate(9)], [], [_verdict(0)]
    )
    assert completed == redraw
    assert len(audit) == 3
    assert len(service.project_debate_candidates(completed)) == 3


def test_zero_survivors_without_candidates_stays_terminal(monkeypatch):
    sources, claims = _story()

    def unexpected(*args, **kwargs):
        raise AssertionError("no generation should run for a candidate-less zero")

    monkeypatch.setattr(service.settings, "news_debate_zero_retry_enabled", True)
    monkeypatch.setattr(service, "generate_news_debate_candidates", unexpected)
    monkeypatch.setattr(service, "generate_news_debate_underfilled_rescue", unexpected)
    assert service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, [], [], []
    ) == ([], [])


def test_zero_retry_respects_its_flag(monkeypatch):
    sources, claims = _story()

    def unexpected(*args, **kwargs):
        raise AssertionError("zero-retry should not run when disabled")

    monkeypatch.setattr(service.settings, "news_debate_zero_retry_enabled", False)
    monkeypatch.setattr(service, "generate_news_debate_candidates", unexpected)
    assert service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, [_candidate(0)], [], [_verdict(0)]
    ) == ([], [])


def test_completion_runs_for_singleton_survivor(monkeypatch):
    sources, claims = _story()
    singleton = [_candidate(0)]
    additions = [_candidate(1), _candidate(2)]

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(
        service, "generate_news_debate_underfilled_rescue", lambda *args: additions
    )
    monkeypatch.setattr(
        service,
        "review_news_debate_candidates",
        lambda *args, **kwargs: (
            additions,
            [_verdict(i) for i in range(len(additions))],
        ),
    )
    completed, audit = service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, singleton, singleton, [_verdict(0)]
    )
    assert completed == [*singleton, *additions]
    assert len(audit) == 2
    assert len(service.project_debate_candidates(completed)) == 3


def test_completion_runs_for_exactly_two_survivors(monkeypatch):
    sources, claims = _story()
    accepted = [_candidate(0), _candidate(1)]
    called = False

    def generate(*args):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(service, "generate_news_debate_underfilled_rescue", generate)
    result, audit = service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, accepted, accepted, []
    )
    assert called is True
    assert result == accepted
    assert audit == []


def test_multiple_valid_axes_complete_collection_without_counterclaim_padding(monkeypatch):
    sources, claims = _story()
    survivors = [_candidate(0), _candidate(1)]
    additions = [_candidate(2), _candidate(3)]

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(
        service,
        "generate_news_debate_underfilled_rescue",
        lambda *args: additions,
    )

    def accept_review(
        headline, review_sources, review_claims, candidates, *, prior_candidates
    ):
        assert candidates == additions
        assert prior_candidates == survivors
        return candidates, [_verdict(i) for i in range(len(candidates))]

    monkeypatch.setattr(service, "review_news_debate_candidates", accept_review)
    completed, audit = service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, survivors, survivors, [_verdict(0), _verdict(1)]
    )
    assert completed == [*survivors, *additions]
    assert len(audit) == 2
    assert len(service.project_debate_candidates(completed)) == 4


def test_still_underfilled_after_review_is_omitted_at_public_projection(monkeypatch):
    sources, claims = _story()
    survivors = [_candidate(0), _candidate(1)]
    rejected_addition = _candidate(2)

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(
        service,
        "generate_news_debate_underfilled_rescue",
        lambda *args: [rejected_addition],
    )
    monkeypatch.setattr(
        service,
        "review_news_debate_candidates",
        lambda *args, **kwargs: (
            [],
            [_verdict(0, distinct_axis=False, failure_codes=["DUPLICATE_AXIS"])],
        ),
    )
    completed, _ = service.complete_underfilled_news_debate_candidates(
        "headline", sources, claims, survivors, survivors, [_verdict(0), _verdict(1)]
    )
    assert len(completed) == 2
    assert service.project_debate_candidates(completed) == []


def test_exact_replay_of_attempted_axis_is_excluded():
    attempted = _candidate(1)
    replay = GroundedDebateCandidate(
        **{
            **attempted.model_dump(),
            "neutral_question": "  SHOULD the COUNCIL take ACTION 1? ",
            "text": attempted.text.rstrip("."),
        }
    )
    assert service._exclude_attempted_exact_axes([replay], [attempted], 4) == []


def test_claude_zero_survivors_with_candidates_gets_one_fresh_redraw(monkeypatch):
    sources, claims = _story()
    redraw = [_candidate(i) for i in range(3)]

    monkeypatch.setattr(service.settings, "news_debate_zero_retry_enabled", True)
    monkeypatch.setattr(
        service, "generate_news_debate_candidates_claude", lambda *args: redraw
    )
    monkeypatch.setattr(
        service,
        "review_news_debate_candidates_claude",
        lambda *args, **kwargs: (redraw, [_verdict(i) for i in range(3)]),
    )
    completed, audit = service.complete_underfilled_news_debate_candidates_claude(
        "headline", sources, claims, [_candidate(9)], [], [_verdict(0)]
    )
    assert completed == redraw
    assert len(audit) == 3


def test_claude_completion_runs_for_singleton_survivor(monkeypatch):
    sources, claims = _story()
    singleton = [_candidate(0)]
    additions = [_candidate(1), _candidate(2)]

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(
        service,
        "generate_news_debate_underfilled_rescue_claude",
        lambda *args: additions,
    )
    monkeypatch.setattr(
        service,
        "review_news_debate_candidates_claude",
        lambda *args, **kwargs: (
            additions,
            [_verdict(i) for i in range(len(additions))],
        ),
    )
    completed, audit = service.complete_underfilled_news_debate_candidates_claude(
        "headline", sources, claims, singleton, singleton, [_verdict(0)]
    )
    assert completed == [*singleton, *additions]
    assert len(audit) == 2


def test_claude_completion_reviews_only_new_axes_against_prior_attempts(monkeypatch):
    sources, claims = _story()
    survivors = [_candidate(0), _candidate(1)]
    additions = [_candidate(2)]

    monkeypatch.setattr(service.settings, "news_debate_underfilled_rescue_enabled", True)
    monkeypatch.setattr(
        service,
        "generate_news_debate_underfilled_rescue_claude",
        lambda *args: additions,
    )

    def accept_review(
        headline, review_sources, review_claims, candidates, *, prior_candidates
    ):
        assert candidates == additions
        assert prior_candidates == survivors
        return candidates, [_verdict(0)]

    monkeypatch.setattr(
        service, "review_news_debate_candidates_claude", accept_review
    )
    completed, audit = service.complete_underfilled_news_debate_candidates_claude(
        "headline", sources, claims, survivors, survivors, [_verdict(0), _verdict(1)]
    )
    assert completed == [*survivors, *additions]
    assert len(audit) == 1
