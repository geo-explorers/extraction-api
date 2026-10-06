"""claims.score_highlights without a provider: the rubric as questions, what the model is
shown, the score arithmetic, the Perplexity wire format, failure handling, and registry wiring."""

import asyncio

import pytest

from src.api.schemas.claims_score_highlights_schema import (
    ClaimsScoreHighlightsInput,
    HighlightClaim,
    HighlightDocument,
)
from src.config.overrides import activate
from src.decisions import Choice, ChoiceAnswer, DecisionError, YesNo, YesNoAnswer, get_decision_model
from src.decisions import perplexity as pplx
from src.pipeline.claims_score_highlights_core import (
    DECISIONS,
    assemble_result,
    build_content,
    build_questions,
    highlight_score,
)


def _input(**overrides) -> ClaimsScoreHighlightsInput:
    base = dict(
        media_type="debate",
        title="A coffee date is an acceptable first date",
        context="Ann argues Agree; Bob argues Disagree.",
        documents=[
            HighlightDocument(speaker="Ann", content="Coffee saves my time. "),
            HighlightDocument(speaker="Bob", content="Even coffee is too much before you know someone."),
            HighlightDocument(content="An unlabelled passage."),
        ],
        claims=[
            HighlightClaim(id="c0", text="A coffee date saves time", document_indices=[0]),
            HighlightClaim(text="Money should stay out of a first meeting", document_indices=[1, 0, 1]),
            HighlightClaim(text="Nobody said this"),
        ],
    )
    return ClaimsScoreHighlightsInput(**{**base, **overrides})


def _answers(**decisions):
    """One YesNoAnswer per decision; an unnamed one is a confident no."""
    return {decision: YesNoAnswer(probability=decisions.get(decision, 0.0)) for decision in DECISIONS}


# ── The rubric as questions ──────────────────────────────────────────────────


def test_question_is_the_one_counterfactual_yes_no_decision():
    questions = build_questions("debate")
    assert list(questions) == list(DECISIONS) == ["essential"]
    question = questions["essential"]
    assert isinstance(question, YesNo) and question.yes and question.no
    assert "removed from the list of extracted claims" in question.instructions
    assert "answer yes only when the loss is major" in question.instructions
    assert "no other extracted claim carries it" in question.yes
    assert "another extracted claim carries the same point" in question.no


def test_rubric_text_is_overridable_per_run():
    with activate(prompt_overrides={"claims_score_highlights.debate.essential.yes": "A changed bar."}) as scope:
        assert build_questions("debate")["essential"].yes == "A changed bar."
    assert scope.unused == set()


def test_yes_no_criteria_come_as_a_pair():
    with pytest.raises(ValueError, match="pair"):
        YesNo(instructions="?", yes="only one side")
    assert YesNo(instructions="?").yes is None


# ── What the model is shown ──────────────────────────────────────────────────


def test_content_shows_the_whole_discussion_every_claim_and_the_claim_under_test():
    content = build_content(_input(), 1)
    assert list(content) == ["title", "context", "transcript", "extracted_claims", "claim_under_test"]
    assert content["transcript"] == (
        "[0] Ann: Coffee saves my time.\n\n"
        "[1] Bob: Even coffee is too much before you know someone.\n\n"
        "[2] Unknown speaker: An unlabelled passage."
    )
    assert content["extracted_claims"] == (
        "1. (Ann) A coffee date saves time\n"
        "2. (Bob and Ann) Money should stay out of a first meeting\n"
        "3. (unattributed) Nobody said this"
    )
    assert content["claim_under_test"] == (
        "Claim 2, attributed to Bob and Ann, drawn from passage [1], [0], [1]: "
        "Money should stay out of a first meeting"
    )


def test_content_omits_empty_title_and_context_and_handles_an_unattributed_claim():
    content = build_content(_input(title=None, context=""), 2)
    assert list(content) == ["transcript", "extracted_claims", "claim_under_test"]
    assert content["claim_under_test"] == "Claim 3, attributed to unattributed: Nobody said this"


# ── The score ────────────────────────────────────────────────────────────────


def test_score_is_the_essential_probability():
    assert highlight_score({"essential": 0.38}) == pytest.approx(0.38)
    assert highlight_score({"essential": 1.0}) == 1.0 and highlight_score({"essential": 0.0}) == 0.0
    # Another decision, if one is asked, rides along but does not move the score.
    assert highlight_score({"essential": 0.5, "other": 1.0}) == pytest.approx(0.5)


def test_score_claim_rejects_an_answer_set_missing_the_decision():
    from src.pipeline.claims_score_highlights_core import score_claim

    with pytest.raises(DecisionError, match="do not match"):
        score_claim(0, HighlightClaim(text="x"), {"other": YesNoAnswer(1.0)})


def test_assemble_keeps_every_claim_in_order_and_refuses_an_incomplete_set():
    inp = _input()
    out = assemble_result(inp, [_answers(essential=0.62), _answers(essential=1.0), _answers()], "fake", "fake-model")
    assert [c.index for c in out.claims] == [0, 1, 2]
    assert out.claims[0].id == "c0" and out.claims[0].score == pytest.approx(0.62)
    assert out.claims[0].decisions == {"essential": 0.62}
    assert out.claims[1].score == 1.0 and out.claims[2].score == 0.0
    assert out.claims_scored == 3 and out.provider == "fake" and out.model_used == "fake-model"
    # Fewer answer sets than claims is a broken contract, not two unscored claims.
    with pytest.raises(DecisionError, match="answer sets"):
        assemble_result(inp, [_answers()], "fake", "fake-model")


# ── Perplexity wire format ───────────────────────────────────────────────────


def test_perplexity_wire_names_yes_no_noul_with_true_false_criteria():
    assert pplx.to_wire(YesNo(instructions="Defect?", yes="Broken.", no="Wear.")) == {
        "type": "noul",
        "instructions": "Defect?",
        "criteria": {"true": "Broken.", "false": "Wear."},
    }
    assert pplx.to_wire(YesNo(instructions="Defect?")) == {"type": "noul", "instructions": "Defect?"}
    assert pplx.to_wire(Choice(instructions="Team?", options={"billing": "Charges", "other": None})) == {
        "type": "choice",
        "instructions": "Team?",
        "criteria": {"billing": "Charges", "other": None},
    }


# The response from the Decisions API quickstart, as documented on 2026-10-02.
DOCS_ANSWERS = {
    "defect": {"type": "noul", "noul": 0.9424522889347015},
    "sentiment": {
        "type": "choice",
        "choice": "mixed",
        "confidence": 0.9255246944002182,
        "probabilities": {"positive": 0.020649883775315993, "mixed": 0.9503497962668123, "negative": 0.02900031995787183},
    },
}
SENTIMENT = Choice(instructions="Sentiment?", options={"positive": None, "mixed": None, "negative": None})


def test_perplexity_answers_parse_into_typed_answers():
    assert pplx.from_wire("defect", YesNo(instructions="Defect?"), DOCS_ANSWERS) == YesNoAnswer(0.9424522889347015)
    sentiment = pplx.from_wire("sentiment", SENTIMENT, DOCS_ANSWERS)
    assert sentiment.choice == "mixed" and list(sentiment.probabilities) == ["positive", "mixed", "negative"]


def test_perplexity_rejects_missing_or_malformed_answers():
    with pytest.raises(DecisionError, match="no answer"):
        pplx.from_wire("absent", YesNo(instructions="?"), DOCS_ANSWERS)
    with pytest.raises(DecisionError, match="noul"):
        pplx.from_wire("defect", YesNo(instructions="?"), {"defect": {"type": "noul", "noul": True}})
    wider = Choice(instructions="?", options={"positive": None, "mixed": None, "negative": None, "none": None})
    with pytest.raises(DecisionError, match="probability per option"):
        pplx.from_wire("sentiment", wider, DOCS_ANSWERS)
    narrower = Choice(instructions="?", options={"positive": None, "negative": None})
    with pytest.raises(DecisionError, match="unknown option"):
        pplx.from_wire("sentiment", narrower, DOCS_ANSWERS)


class _Response:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_http(monkeypatch, responses):
    calls, sleeps = [], []

    def post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return responses.pop(0)

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(pplx.requests, "post", post)
    monkeypatch.setattr(pplx.asyncio, "sleep", sleep)
    return calls, sleeps


@pytest.mark.asyncio
async def test_perplexity_decide_sends_state_and_questions_and_returns_typed_answers(monkeypatch):
    calls, _ = _patch_http(monkeypatch, [_Response(200, {"answers": DOCS_ANSWERS, "usage": {"input_tokens": 367}})])
    model = pplx.PerplexityDecisionModel(api_key="k")
    answers = await model.decide(
        {"title": "T", "transcript": "[0] Ann: hi"},
        {"defect": YesNo(instructions="Defect?"), "sentiment": SENTIMENT},
        model="pplx-decider-v1-27b",
    )
    assert answers["defect"].probability == pytest.approx(0.942, abs=1e-3) and answers["sentiment"].choice == "mixed"
    (call,) = calls
    assert call["url"] == "https://api.perplexity.ai/v1/decisions"
    assert call["headers"] == {"Authorization": "Bearer k"}
    assert call["json"]["model"] == "pplx-decider-v1-27b"
    assert call["json"]["state"] == {"title": "T", "transcript": "[0] Ann: hi"}
    assert call["json"]["questions"]["defect"] == {"type": "noul", "instructions": "Defect?"}
    assert set(call["json"]) == {"model", "state", "questions"}  # an unknown top-level field is a 400


@pytest.mark.asyncio
async def test_perplexity_waits_retry_after_on_429_and_backs_off_on_an_outage(monkeypatch):
    ok = _Response(200, {"answers": DOCS_ANSWERS})
    calls, sleeps = _patch_http(monkeypatch, [_Response(429, headers={"Retry-After": "3"}), _Response(504, text="<html>"), ok])
    model = pplx.PerplexityDecisionModel(api_key="k")
    await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    # Retry-After for the 429; the 504 is the first outage failure, so the first backoff step.
    assert len(calls) == 3 and sleeps == [3.0, 2.0]

    # A Retry-After the server sets to minutes is capped: a run has a budget of its own.
    calls, sleeps = _patch_http(monkeypatch, [_Response(429, headers={"Retry-After": "600"}), ok])
    await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert sleeps == [pplx.MAX_RETRY_AFTER_SECONDS]

    # A missing or HTTP-date Retry-After falls back to the backoff.
    calls, sleeps = _patch_http(monkeypatch, [_Response(429), _Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), ok])
    await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert sleeps == [2.0, 2.0]


@pytest.mark.asyncio
async def test_perplexity_classifies_what_can_be_retried(monkeypatch):
    model = pplx.PerplexityDecisionModel(api_key="k")

    # A refused key is final: no retry, not retryable.
    unauthorized = _Response(401, {"error": {"message": "Invalid API key", "type": "unauthorized"}})
    calls, sleeps = _patch_http(monkeypatch, [unauthorized])
    with pytest.raises(DecisionError, match="HTTP 401: Invalid API key") as refused:
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == 1 and sleeps == [] and refused.value.retryable is False

    # An outage is retried a bounded number of times, then reported as retryable.
    calls, _ = _patch_http(monkeypatch, [_Response(503)] * pplx.MAX_ATTEMPTS)
    with pytest.raises(DecisionError, match="HTTP 503") as outage:
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == pplx.MAX_ATTEMPTS and outage.value.retryable is True

    # Rate limiting has its own, longer budget and does not spend the outage one.
    calls, _ = _patch_http(monkeypatch, [_Response(429, headers={"Retry-After": "1"})] * pplx.MAX_RATE_LIMIT_ATTEMPTS)
    with pytest.raises(DecisionError, match="HTTP 429") as limited:
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == pplx.MAX_RATE_LIMIT_ATTEMPTS and limited.value.retryable is True

    # A dropped connection is retried like an outage.
    def drop(*args, **kwargs):
        raise pplx.requests.ConnectionError("reset")

    monkeypatch.setattr(pplx.requests, "post", drop)
    with pytest.raises(DecisionError, match="request failed") as dropped:
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert dropped.value.retryable is True

    # A 200 that is not JSON, or not an object, is drift: final.
    calls, _ = _patch_http(monkeypatch, [_Response(200, None, text="<html>")])
    with pytest.raises(DecisionError, match="not JSON") as html:
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert html.value.retryable is False
    calls, _ = _patch_http(monkeypatch, [_Response(200, [1, 2])])
    with pytest.raises(DecisionError, match="not a JSON object"):
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")


def test_provider_factory_needs_a_known_provider_and_a_key(monkeypatch):
    with pytest.raises(ValueError, match="unknown decision provider 'nope'"):
        get_decision_model("nope")
    monkeypatch.setattr(pplx.settings, "perplexity_api_key", None)
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        pplx.PerplexityDecisionModel()


# ── The task ─────────────────────────────────────────────────────────────────


class _FakeDecider:
    provider = "fake"

    def __init__(self, fail_on=(), hang_on=(), retryable=False, delay=0.0):
        self.fail_on = set(fail_on)
        self.hang_on = set(hang_on)
        self.retryable = retryable
        self.delay = delay
        self.seen = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def decide(self, content, questions, *, model):
        self.seen.append((content["claim_under_test"], list(questions), model))
        claim_index = int(content["claim_under_test"].split(",")[0].split()[1]) - 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if claim_index in self.hang_on:
                await asyncio.Event().wait()  # until cancelled
            if self.delay:
                await asyncio.sleep(self.delay)
            if claim_index in self.fail_on:
                raise DecisionError(f"claim {claim_index + 1} failed", retryable=self.retryable)
            return _answers(essential=0.5)
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
async def test_task_scores_each_claim_with_the_configured_model(monkeypatch):
    from src.tasks import claims_score_highlights as task

    fake = _FakeDecider()
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    with activate(llm_overrides={"claims_highlights_model": "other-model"}):
        result = await task._handle(_input(), None)
    assert [c.score for c in result.claims] == [0.5, 0.5, 0.5]
    assert result.claims_scored == 3 and result.provider == "fake" and result.model_used == "other-model"
    assert len(fake.seen) == 3
    assert all(q == ["essential"] and m == "other-model" for _, q, m in fake.seen)


@pytest.mark.asyncio
async def test_one_failed_claim_fails_the_run_and_stops_the_others(monkeypatch):
    """All or nothing: a consumer keeps what it gets, so a half-scored list would stay
    half-scored for good. The other requests are cancelled, not awaited — the hanging claims
    here would otherwise never return."""
    from src.tasks import claims_score_highlights as task

    for retryable in (False, True):
        fake = _FakeDecider(fail_on={1}, hang_on={0, 2}, retryable=retryable)
        monkeypatch.setattr(task, "get_decision_model", lambda provider, fake=fake: fake)
        with pytest.raises(DecisionError, match="claim 2 failed") as failed:
            await asyncio.wait_for(task._handle(_input(), None), timeout=2)
        assert failed.value.retryable is retryable
        assert fake.in_flight == 0, "the hanging requests were cancelled"


@pytest.mark.asyncio
async def test_in_flight_requests_are_bounded_and_each_is_counted_by_the_spend_guard(monkeypatch):
    from src.infrastructure import spend_guard as guard_module
    from src.pipeline import claims_score_highlights_core as core

    counted = []
    monkeypatch.setattr(guard_module.spend_guard, "check_and_record", lambda provider: counted.append(provider))
    fake = _FakeDecider(delay=0.01)
    many = _input(claims=[HighlightClaim(text=f"Claim {i}", document_indices=[0]) for i in range(20)])
    answers = await core.decide_claims(many, fake, "m")
    assert len(answers) == 20
    assert fake.max_in_flight <= core.DECISION_CONCURRENCY
    assert fake.max_in_flight > 1, "the bound is a bound, not a serialisation"
    assert counted == ["fake"] * 20


def test_prompt_overrides_for_the_rubric_are_accepted_at_enqueue():
    accepted = _input(prompt_overrides={"claims_score_highlights.debate.essential.yes": "A changed bar."})
    assert accepted.prompt_overrides == {"claims_score_highlights.debate.essential.yes": "A changed bar."}
    with pytest.raises(ValueError, match="unknown prompt key"):
        _input(prompt_overrides={"claims_score_highlights.debate.essential.maybe": "x"})


def test_every_media_type_has_a_registered_rubric_and_nothing_else_is_registered():
    from typing import get_args

    from src.api.schemas.claims_score_highlights_schema import MediaType
    from src.config.prompt_registry import PROMPTS

    media_types = set(get_args(MediaType))
    for media_type in media_types:
        assert list(build_questions(media_type)) == ["essential"], media_type
    registered = {key.split(".")[1] for key in PROMPTS if key.startswith("claims_score_highlights.")}
    assert registered == media_types


def test_registry_has_the_task_with_its_contract():
    from src.api.schemas.claims_score_highlights_schema import ClaimsScoreHighlightsResult
    from src.tasks.claims_score_highlights import CLAIMS_SCORE_HIGHLIGHTS_SPEC
    from src.tasks.registry import get_task

    entry = get_task("claims.score_highlights")
    assert entry is not None
    assert entry.input_model is ClaimsScoreHighlightsInput
    assert entry.output_model is ClaimsScoreHighlightsResult
    assert CLAIMS_SCORE_HIGHLIGHTS_SPEC.rate_limit_key == "decisions_global"


def test_input_bounds():
    with pytest.raises(ValueError):
        _input(media_type="news")  # no rubric for it yet
    with pytest.raises(ValueError):
        _input(claims=[])
    with pytest.raises(ValueError):
        _input(documents=[])
    with pytest.raises(ValueError, match="out of range"):
        _input(claims=[HighlightClaim(text="x", document_indices=[3])])
    with pytest.raises(ValueError, match="maximum"):
        _input(documents=[HighlightDocument(content="x" * 400_001)])
