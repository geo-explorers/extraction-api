"""claims.score_highlights without a provider: the rubric as questions, what the model is
shown, the score arithmetic, the Perplexity wire format, failure handling, and registry wiring."""

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


def test_assemble_keeps_every_claim_in_order_with_nulls_for_failures():
    inp = _input()
    outcomes = [
        _answers(essential=0.62),
        DecisionError("HTTP 504"),
        _answers(),
    ]
    out = assemble_result(inp, outcomes, "fake", "fake-model")
    assert [c.index for c in out.claims] == [0, 1, 2]
    assert out.claims[0].id == "c0" and out.claims[0].score == pytest.approx(0.62)
    assert out.claims[0].decisions == {"essential": 0.62}
    assert out.claims[1].score is None and out.claims[1].error == "HTTP 504"
    assert out.claims[1].decisions == {}
    assert out.claims[2].score == 0.0
    assert out.claims_scored == 2 and out.provider == "fake" and out.model_used == "fake-model"


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
async def test_perplexity_waits_retry_after_on_429_and_gives_up_on_a_client_error(monkeypatch):
    ok = _Response(200, {"answers": DOCS_ANSWERS})
    calls, sleeps = _patch_http(monkeypatch, [_Response(429, headers={"Retry-After": "3"}), _Response(504, text="<html>"), ok])
    model = pplx.PerplexityDecisionModel(api_key="k")
    await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == 3 and sleeps == [3.0, 4.0]  # Retry-After, then exponential backoff

    unauthorized = _Response(401, {"error": {"message": "Invalid API key", "type": "unauthorized"}})
    calls, sleeps = _patch_http(monkeypatch, [unauthorized])
    with pytest.raises(DecisionError, match="HTTP 401: Invalid API key"):
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == 1 and sleeps == []

    # Retries are bounded: the third retryable failure is the error.
    calls, _ = _patch_http(monkeypatch, [_Response(503), _Response(503), _Response(503)])
    with pytest.raises(DecisionError, match="HTTP 503"):
        await model.decide({"t": "x"}, {"defect": YesNo(instructions="?")}, model="m")
    assert len(calls) == 3


def test_provider_factory_needs_a_known_provider_and_a_key(monkeypatch):
    with pytest.raises(ValueError, match="unknown decision provider 'nope'"):
        get_decision_model("nope")
    monkeypatch.setattr(pplx.settings, "perplexity_api_key", None)
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        pplx.PerplexityDecisionModel()


# ── The task ─────────────────────────────────────────────────────────────────


class _FakeDecider:
    provider = "fake"

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.seen = []

    async def decide(self, content, questions, *, model):
        self.seen.append((content["claim_under_test"], list(questions), model))
        claim_number = int(content["claim_under_test"].split(",")[0].split()[1])
        if claim_number - 1 in self.fail_on:
            raise DecisionError(f"claim {claim_number} failed")
        return _answers(essential=0.5)


@pytest.mark.asyncio
async def test_task_scores_each_claim_with_the_configured_model(monkeypatch):
    from src.tasks import claims_score_highlights as task

    fake = _FakeDecider(fail_on={1})
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    with activate(llm_overrides={"claims_highlights_model": "other-model"}):
        result = await task._handle(_input(), None)
    assert [c.score for c in result.claims] == [0.5, None, 0.5]
    assert result.claims[1].error == "claim 2 failed"
    assert result.claims_scored == 2 and result.provider == "fake" and result.model_used == "other-model"
    assert len(fake.seen) == 3
    assert all(q == ["essential"] and m == "other-model" for _, q, m in fake.seen)


@pytest.mark.asyncio
async def test_task_fails_the_run_when_no_claim_could_be_scored(monkeypatch):
    from src.tasks import claims_score_highlights as task

    monkeypatch.setattr(task, "get_decision_model", lambda provider: _FakeDecider(fail_on={0, 1, 2}))
    with pytest.raises(DecisionError, match="failed"):
        await task._handle(_input(), None)


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
