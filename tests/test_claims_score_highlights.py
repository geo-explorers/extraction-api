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
from src.decisions import Choice, ChoiceAnswer, DecisionError, Scale, ScaleAnswer, YesNo, YesNoAnswer, get_decision_model
from src.decisions import perplexity as pplx
from src.pipeline.claims_score_highlights_core import (
    AXES,
    AXIS_LEVELS,
    DECISIONS,
    DebateDecisions,
    assemble_result,
    build_content,
    build_questions,
    claim_decisions,
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


def _scale(score: float) -> ScaleAnswer:
    """A four-level answer at `score`: all the mass on the lowest level for 0, the highest for 1."""
    return ScaleAnswer(score=score, probabilities=[1 - score, 0.0, 0.0, score])


def _answers(**decisions):
    """One answer per decision: yes/no for `essential`, a scale for each axis; an unnamed one is
    a confident no / the lowest level."""
    answers = {"essential": YesNoAnswer(probability=decisions.get("essential", 0.0))}
    answers.update({axis: _scale(decisions.get(axis, 0.0)) for axis in AXES})
    return answers


# ── The rubric as questions ──────────────────────────────────────────────────


def test_questions_are_the_score_decision_and_one_scale_per_axis():
    questions = build_questions("debate")
    assert list(questions) == list(DECISIONS) == ["essential", "relevance", "quality", "controversy"]
    essential = questions["essential"]
    assert isinstance(essential, YesNo) and essential.yes and essential.no
    assert "removed from the list of extracted claims" in essential.instructions
    assert "answer yes only when the loss is major" in essential.instructions
    assert "no other extracted claim carries it" in essential.yes
    assert "another extracted claim carries the same point" in essential.no
    # Each axis is a four-level scale, lowest level first: the rubric module lists the texts in the
    # task's order, and the wire order is the task's — a swap in either would reach the model as
    # a reversed rung with every key still registered.
    from src.config.prompts.claims_score_highlights_prompt import DEBATE_AXES

    for axis in AXES:
        assert isinstance(questions[axis], Scale) and len(questions[axis].levels) == len(AXIS_LEVELS[axis]) == 4
        texts = DEBATE_AXES[axis][1]
        assert list(texts) == list(AXIS_LEVELS[axis]), axis
        assert list(questions[axis].levels) == [texts[level] for level in AXIS_LEVELS[axis]], axis
    relevance, quality, controversy = (questions[axis] for axis in AXES)
    assert "not about the main claim's subject" in relevance.levels[0] and "sees at once" in relevance.levels[-1]
    assert quality.levels[0].startswith("Misrepresents") and quality.levels[-1].startswith("Nothing to fix")
    assert "look for a flaw" in quality.instructions and "lower level is the usual answer" in quality.instructions
    assert "nobody would dispute" in controversy.levels[0] and "Reasonable people split" in controversy.levels[-1]


def test_rubric_text_is_overridable_per_run():
    overrides = {
        "claims_score_highlights.debate.essential.yes": "A changed bar.",
        "claims_score_highlights.debate.quality.clean": "A changed top level.",
    }
    with activate(prompt_overrides=overrides) as scope:
        questions = build_questions("debate")
        assert questions["essential"].yes == "A changed bar."
        assert questions["quality"].levels[-1] == "A changed top level."
    assert scope.unused == set()


def test_yes_no_criteria_come_as_a_pair():
    with pytest.raises(ValueError, match="pair"):
        YesNo(instructions="?", yes="only one side")
    assert YesNo(instructions="?").yes is None


def test_a_scale_needs_two_levels_given_as_a_list():
    with pytest.raises(ValueError, match="two levels"):
        Scale(instructions="?", levels=["only one"])
    # A string is a sequence of characters, which is not a rubric.
    with pytest.raises(ValueError, match="two levels"):
        Scale(instructions="?", levels="low to high")
    assert len(Scale(instructions="?", levels=["low", "high"]).levels) == 2


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


def test_claim_decisions_reject_an_answer_set_missing_or_mistyping_a_decision():
    with pytest.raises(DecisionError, match="do not match"):
        claim_decisions({"other": YesNoAnswer(1.0)})
    without_an_axis = {name: answer for name, answer in _answers().items() if name != "quality"}
    with pytest.raises(DecisionError, match="do not match"):
        claim_decisions(without_an_axis)
    # An axis answered as a yes/no is the wrong shape, however plausible the number.
    with pytest.raises(DecisionError, match="do not match"):
        claim_decisions({**_answers(), "quality": YesNoAnswer(0.9)})


def _decisions(answers, moments=None, groups=None) -> DebateDecisions:
    n = len(answers)
    return DebateDecisions(list(answers), moments or [1 / n] * n, groups or list(range(n)))


def test_assemble_keeps_every_claim_in_order_and_refuses_an_incomplete_set():
    inp = _input()
    answers = [_answers(essential=0.62, relevance=0.75, controversy=0.5), _answers(essential=1.0), _answers()]
    out = assemble_result(inp, _decisions(answers, [0.5, 0.3, 0.2], [0, 1, 0]), "fake", "fake-model")
    assert [c.index for c in out.claims] == [0, 1, 2]
    assert out.claims[0].id == "c0" and out.claims[0].score == pytest.approx(0.62)
    # The axes and the moment ride beside the score, in decision order, and never move it.
    assert list(out.claims[0].decisions) == [*DECISIONS, "moment"]
    assert out.claims[0].decisions == {
        "essential": 0.62, "relevance": 0.75, "quality": 0.0, "controversy": 0.5, "moment": 0.5,
    }
    assert out.claims[1].score == 1.0 and out.claims[2].score == 0.0
    assert [c.repeat_group for c in out.claims] == [0, 1, 0]
    assert out.claims_scored == 3 and out.provider == "fake" and out.model_used == "fake-model"
    # Fewer answer sets than claims is a broken contract, not two unscored claims.
    with pytest.raises(DecisionError, match="answer sets"):
        assemble_result(inp, _decisions([_answers()]), "fake", "fake-model")


def test_moment_score_is_the_weighted_percentiles_within_the_debate():
    # Claim 0 tops the moment question, claim 2 tops controversy, claim 1 is middle on both.
    answers = [_answers(controversy=0.2), _answers(controversy=0.5), _answers(controversy=0.9)]
    out = assemble_result(_input(), _decisions(answers, [0.6, 0.3, 0.1]), "fake", "m")
    assert [c.moment_score for c in out.claims] == pytest.approx([0.6 * 1 + 0.4 * 0, 0.5, 0.6 * 0 + 0.4 * 1])
    # Percentiles, not raw values: a moment probability of 0.31 against 0.30 is the same
    # step as 0.9 against 0.1.
    close = assemble_result(_input(), _decisions(answers, [0.31, 0.30, 0.29]), "fake", "m")
    assert [c.moment_score for c in close.claims] == pytest.approx([0.6, 0.5, 0.4])


def test_percentiles_share_ties_and_a_lone_claim_sits_in_the_middle():
    from src.pipeline.claims_score_highlights_core import percentiles

    assert percentiles([0.1, 0.9, 0.5]) == [0.0, 1.0, 0.5]
    assert percentiles([0.4, 0.4, 0.9]) == [0.25, 0.25, 1.0]
    assert percentiles([0.7]) == [0.5]


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
    # A scale is a `score` question: the level descriptions as an ordered list, lowest first.
    assert pplx.to_wire(SEVERITY) == {
        "type": "score",
        "instructions": "Severity?",
        "criteria": ["Cosmetic", "Inconvenient", "Product unusable"],
    }
    with pytest.raises(DecisionError, match="at most 10 levels"):
        pplx.to_wire(Scale(instructions="?", levels=[str(i) for i in range(11)]))


# The response from the Decisions API quickstart, as documented on 2026-10-02 (noul, choice)
# and the API reference on 2026-10-08 (score).
DOCS_ANSWERS = {
    "defect": {"type": "noul", "noul": 0.9424522889347015},
    "sentiment": {
        "type": "choice",
        "choice": "mixed",
        "confidence": 0.9255246944002182,
        "probabilities": {"positive": 0.020649883775315993, "mixed": 0.9503497962668123, "negative": 0.02900031995787183},
    },
    "severity": {
        "type": "score",
        "score": 1.7838686319784252,
        "confidence": 0.7838686319784252,
        "legend": {"0": "Cosmetic", "1": "Inconvenient", "2": "Product unusable"},
        "probabilities": {"0": 0.008423954913615923, "1": 0.199283458194343, "2": 0.7922925868920411},
    },
}
SENTIMENT = Choice(instructions="Sentiment?", options={"positive": None, "mixed": None, "negative": None})
SEVERITY = Scale(instructions="Severity?", levels=["Cosmetic", "Inconvenient", "Product unusable"])


def test_perplexity_answers_parse_into_typed_answers():
    assert pplx.from_wire("defect", YesNo(instructions="Defect?"), DOCS_ANSWERS) == YesNoAnswer(0.9424522889347015)
    sentiment = pplx.from_wire("sentiment", SENTIMENT, DOCS_ANSWERS)
    assert sentiment.choice == "mixed" and list(sentiment.probabilities) == ["positive", "mixed", "negative"]
    severity = pplx.from_wire("severity", SEVERITY, DOCS_ANSWERS)
    assert isinstance(severity, ScaleAnswer) and len(severity.probabilities) == 3
    # The position is the documented `score` (a weighted level index) scaled to 0..1.
    assert severity.score == pytest.approx(1.7838686319784252 / 2)
    assert severity.probabilities[2] == pytest.approx(0.792, abs=1e-3)


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
    # A scale answered at other levels than asked is a broken contract, whichever way it differs:
    # fewer probabilities than levels, more, a distribution that is not one, or a legend that
    # describes different rungs.
    taller = Scale(instructions="?", levels=["a", "b", "c", "d"])
    with pytest.raises(DecisionError, match="probability per level"):
        pplx.from_wire("severity", taller, DOCS_ANSWERS)
    with pytest.raises(DecisionError, match="probability per level"):
        pplx.from_wire("severity", SEVERITY, {"severity": {"type": "score", "score": 1.5}})
    shorter = Scale(instructions="?", levels=["a", "b"])
    with pytest.raises(DecisionError, match="probability per level"):
        pplx.from_wire("severity", shorter, DOCS_ANSWERS)
    with pytest.raises(DecisionError, match="do not sum to one"):
        pplx.from_wire("severity", SEVERITY, {"severity": {"type": "score", "probabilities": {"0": 0, "1": 0, "2": 0}}})
    reordered = Scale(instructions="?", levels=["Product unusable", "Inconvenient", "Cosmetic"])
    with pytest.raises(DecisionError, match="other levels than asked"):
        pplx.from_wire("severity", reordered, DOCS_ANSWERS)


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
        {"defect": YesNo(instructions="Defect?"), "sentiment": SENTIMENT, "severity": SEVERITY},
        model="pplx-decider-v1-27b",
    )
    assert answers["defect"].probability == pytest.approx(0.942, abs=1e-3) and answers["sentiment"].choice == "mixed"
    assert isinstance(answers["severity"], ScaleAnswer) and answers["severity"].score == pytest.approx(0.892, abs=1e-3)
    (call,) = calls
    assert call["url"] == "https://api.perplexity.ai/v1/decisions"
    assert call["headers"] == {"Authorization": "Bearer k"}
    assert call["json"]["model"] == "pplx-decider-v1-27b"
    assert call["json"]["state"] == {"title": "T", "transcript": "[0] Ann: hi"}
    assert call["json"]["questions"]["defect"] == {"type": "noul", "instructions": "Defect?"}
    assert call["json"]["questions"]["severity"]["criteria"] == ["Cosmetic", "Inconvenient", "Product unusable"]
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


def _varied(claim_index: int):
    """Answers that differ between claims and between axes, as a model that read the content gives."""
    return _answers(
        essential=0.2 + 0.1 * (claim_index % 8),
        relevance=0.9 - 0.1 * (claim_index % 8),
        quality=0.35,
        controversy=0.1 + 0.1 * (claim_index % 8),
    )


class _FakeDecider:
    """Answers all three kinds of request: a claim's questions (varied by claim), the moment
    choice (the lower the claim number, the likelier), and a claim's repeat question (the other
    claim it is paired with in `same` is likely, `none` otherwise)."""

    provider = "fake"

    def __init__(self, fail_on=(), hang_on=(), retryable=False, delay=0.0, same=(), answers=None):
        self.fail_on = set(fail_on)
        self.hang_on = set(hang_on)
        self.retryable = retryable
        self.delay = delay
        self.same = set(same)
        self.answers = answers  # replaces the per-claim answers: claim index -> answer set
        self.seen = []  # (kind, detail, model) per request
        self.contents = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def decide(self, content, questions, *, model):
        self.contents.append((content, questions))
        names = list(questions)
        if names == ["moment"]:
            self.seen.append(("moment", list(questions["moment"].options), model))
            options = questions["moment"].options
            weights = {name: 1.0 / int(name) for name in options}
            total = sum(weights.values())
            return {"moment": ChoiceAnswer(choice=min(options, key=int), probabilities={n: w / total for n, w in weights.items()})}
        if names == ["repeat"]:
            target = int(content["target_claim"].split()[1]) - 1
            self.seen.append(("repeat", target, model))
            weights = {}
            for key, label in questions["repeat"].options.items():
                other = None if key == "none" else int(label.split()[1]) - 1
                weights[key] = 0.1 if other is None else (0.8 if tuple(sorted((target, other))) in self.same else 0.02)
            total = sum(weights.values())
            return {"repeat": ChoiceAnswer(choice=max(weights, key=weights.get), probabilities={k: w / total for k, w in weights.items()})}
        self.seen.append(("claim", content["claim_under_test"], model))
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
            return self.answers[claim_index] if self.answers else _varied(claim_index)
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
async def test_task_scores_each_claim_with_the_configured_model(monkeypatch):
    from src.tasks import claims_score_highlights as task

    fake = _FakeDecider()
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    with activate(llm_overrides={"claims_highlights_model": "other-model"}):
        result = await task._handle(_input(), None)
    assert [c.score for c in result.claims] == pytest.approx([0.2, 0.3, 0.4])
    assert result.claims_scored == 3 and result.provider == "fake" and result.model_used == "other-model"
    kinds = [kind for kind, _, _ in fake.seen]
    # One request per claim, three moment rounds, and one repeat question per claim.
    assert sorted(kinds) == ["claim"] * 3 + ["moment"] * 3 + ["repeat"] * 3
    assert all(m == "other-model" for _, _, m in fake.seen)
    # Every claim is asked the score decision and the three axes in one request.
    assert [list(questions) for content, questions in fake.contents if "claim_under_test" in content] == [list(DECISIONS)] * 3
    # The first claim is the likeliest moment in every round, so it ranks first; no repeats here.
    assert [c.repeat_group for c in result.claims] == [0, 1, 2]
    assert result.claims[0].decisions["moment"] > result.claims[1].decisions["moment"] > result.claims[2].decisions["moment"]
    assert result.claims[0].moment_score == pytest.approx(0.6 * 1 + 0.4 * 0)  # top moment, lowest controversy
    assert result.claims[2].moment_score == pytest.approx(0.6 * 0 + 0.4 * 1)


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
    # A whole run adds the moment rounds and one repeat question per claim.
    counted.clear()
    decided = await core.decide_debate(many, fake, "m")
    assert len(decided.claims) == len(decided.moments) == len(decided.groups) == 20
    assert counted == ["fake"] * (20 + core.MOMENT_ROUNDS + 20)


# ── Moment, repeat groups and the glitch guard ───────────────────────────────


def test_moment_requests_show_the_claims_in_a_different_seeded_order_each_round():
    from src.pipeline.claims_score_highlights_core import MOMENT_ROUNDS, moment_requests, shuffled_order

    many = _input(claims=[HighlightClaim(text=f"Claim {i}", document_indices=[i % 2]) for i in range(12)])
    requests = moment_requests(many)
    assert len(requests) == MOMENT_ROUNDS == 3
    orders = [list(questions["moment"].options) for _, _, questions in requests]
    # Options are named by the claim's number in the input, whatever the order shown.
    assert all(sorted(order, key=int) == [str(i + 1) for i in range(12)] for order in orders)
    assert len({tuple(order) for order in orders}) == 3, "each round shuffles differently"
    assert orders == [list(map(str, (i + 1 for i in shuffled_order(many, r)))) for r in range(3)]
    # Seeded: the same debate is shuffled the same way again.
    assert [list(q["moment"].options) for _, _, q in moment_requests(many)] == orders
    _, content, questions = requests[0]
    assert "transcript" in content and "claim_under_test" not in content
    assert questions["moment"].options["1"] == "(Ann) Claim 0"
    assert "most want to watch" in questions["moment"].instructions


def test_moment_rounds_read_probabilities_back_by_claim_number():
    from src.pipeline.claims_score_highlights_core import moment_rounds

    inp = _input()
    answer = {"moment": ChoiceAnswer(choice="3", probabilities={"3": 0.5, "1": 0.3, "2": 0.2})}
    assert moment_rounds(inp, [answer]) == [[0.3, 0.2, 0.5]]
    with pytest.raises(DecisionError, match="no probability for claim"):
        moment_rounds(inp, [{"moment": ChoiceAnswer(choice="1", probabilities={"1": 1.0})}])
    with pytest.raises(DecisionError, match="moment question"):
        moment_rounds(inp, [{"moment": YesNoAnswer(0.5)}])


def test_repeat_requests_ask_each_claim_which_other_makes_the_same_point_without_the_transcript():
    from src.pipeline.claims_score_highlights_core import repeat_requests

    requests = repeat_requests(_input())
    assert [label for label, _, _ in requests] == ["repeat claim 0", "repeat claim 1", "repeat claim 2"]
    _, content, questions = requests[1]
    assert list(questions) == ["repeat"]
    assert "transcript" not in content and "extracted_claims" not in content
    assert content["title"] == "A coffee date is an acceptable first date"
    assert content["target_claim"] == "Claim 2 (Bob and Ann, passage [1], [0], [1]): Money should stay out of a first meeting"
    # The other claims under neutral keys, in claim order, then `none`. The speaker and passage
    # travel with each claim's text; the key says nothing about it.
    assert questions["repeat"].options == {
        "o01": "Claim 1 (Ann, passage [0]): A coffee date saves time",
        "o02": "Claim 3 (unattributed): Nobody said this",
        "none": "No other claim in this debate makes the same point as the target claim.",
    }
    assert "SAME POINT" in questions["repeat"].instructions
    assert "The speaker does not matter" in questions["repeat"].instructions
    # A lone claim has nothing to be a repeat of.
    assert repeat_requests(_input(claims=[HighlightClaim(text="alone")])) == []


def _repeat_answers(count, directed):
    """One repeat answer per claim from {(target, other): probability}; an unnamed direction
    gets 0.05 and `none` whatever is left."""
    from src.pipeline.claims_score_highlights_core import repeat_options

    answers = []
    for target in range(count):
        probabilities = {key: directed.get((target, j), 0.05) for key, j in repeat_options(count, target)}
        probabilities["none"] = max(0.0, 1 - sum(probabilities.values()))
        answers.append({"repeat": ChoiceAnswer(choice=max(probabilities, key=probabilities.get), probabilities=probabilities)})
    return answers


def test_repeat_scores_average_the_two_directions_of_a_pair():
    from src.pipeline.claims_score_highlights_core import repeat_scores

    scores = repeat_scores(3, _repeat_answers(3, {(0, 1): 0.9, (1, 0): 0.5, (2, 0): 0.45}))
    assert scores[0][1] == scores[1][0] == pytest.approx(0.7)
    assert scores[0][2] == scores[2][0] == pytest.approx((0.45 + 0.05) / 2)
    assert scores[1][2] == scores[2][1] == pytest.approx(0.05)
    assert repeat_scores(1, []) == [[0.0]]
    with pytest.raises(DecisionError, match="3 repeat answers for 4 claims"):
        repeat_scores(4, _repeat_answers(3, {}))
    with pytest.raises(DecisionError, match="repeat question"):
        repeat_scores(2, [{"repeat": YesNoAnswer(0.5)}] * 2)
    with pytest.raises(DecisionError, match="claim 0 has no probability for claim 1"):
        repeat_scores(2, [{"repeat": ChoiceAnswer(choice="none", probabilities={"none": 1.0})}] * 2)


def _matrix(count, pairs):
    scores = [[0.0] * count for _ in range(count)]
    for (i, j), score in pairs.items():
        scores[i][j] = scores[j][i] = score
    return scores


def test_repeat_groups_join_by_average_linkage_under_the_lowest_index():
    from src.pipeline.claims_score_highlights_core import REPEAT_THRESHOLD, repeat_groups

    assert REPEAT_THRESHOLD == 0.3
    assert repeat_groups(_matrix(5, {})) == [0, 1, 2, 3, 4]
    assert repeat_groups(_matrix(5, {(1, 3): 0.9, (3, 4): 0.9, (1, 4): 0.8})) == [0, 1, 2, 1, 1]
    # The bar is inclusive.
    assert repeat_groups(_matrix(2, {(0, 1): 0.3})) == [0, 0]
    assert repeat_groups(_matrix(2, {(0, 1): 0.29})) == [0, 1]
    # No chaining: 3 repeats 1 and half-repeats 4, but 1 and 4 share nothing, so {1, 3} averages
    # (0.5 + 0.0) / 2 = 0.25 with 4 — under the bar. Joining on single pairs would have made one
    # group of three.
    assert repeat_groups(_matrix(5, {(1, 3): 0.9, (3, 4): 0.5})) == [0, 1, 2, 1, 4]
    # When the average holds the group grows, strongest pair first.
    assert repeat_groups(_matrix(4, {(0, 1): 0.4, (1, 2): 0.9, (0, 2): 0.35, (2, 3): 0.1})) == [0, 0, 0, 3]
    assert repeat_groups([[0.0]]) == [0]
    assert repeat_groups([]) == []


def _glitch_inputs():
    """A debate of three claims whose answers differ between claims and axes."""
    return [_varied(i) for i in range(3)], [[0.6, 0.3, 0.1]] * 3


def test_a_healthy_run_passes_the_glitch_check():
    from src.pipeline.claims_score_highlights_core import check_for_glitch

    check_for_glitch(*_glitch_inputs())
    # A saturated claim is not a glitch while its axes still differ.
    saturated = {"essential": YesNoAnswer(1.0), "relevance": _scale(1.0), "quality": _scale(0.0), "controversy": _scale(1.0)}
    check_for_glitch([saturated, *_glitch_inputs()[0][1:]], [[0.6, 0.3, 0.1]])


def test_one_distribution_for_every_axis_question_fails_the_run_and_can_be_retried():
    from src.pipeline.claims_score_highlights_core import check_for_glitch

    claims, rounds = _glitch_inputs()
    same = _scale(0.62)
    stuck = {"essential": YesNoAnswer(0.4), "relevance": same, "quality": same, "controversy": same}
    with pytest.raises(DecisionError, match="claim 1 got one identical distribution") as failed:
        check_for_glitch([claims[0], stuck, claims[2]], rounds)
    assert failed.value.retryable
    # Within a millionth is the same: the provider's own rounding must not hide it.
    nearly = {**stuck, "quality": ScaleAnswer(0.62, [0.38 + 5e-7, 0.0, 0.0, 0.62 - 5e-7])}
    with pytest.raises(DecisionError, match="identical distribution"):
        check_for_glitch([claims[0], nearly, claims[2]], rounds)
    # A hundredth apart is a different answer.
    apart = {**stuck, "quality": ScaleAnswer(0.63, [0.37, 0.0, 0.0, 0.63])}
    check_for_glitch([claims[0], apart, claims[2]], rounds)


def test_the_same_answers_for_every_claim_or_a_flat_moment_or_repeat_answer_fail_the_run():
    from src.pipeline.claims_score_highlights_core import check_for_glitch

    claims, rounds = _glitch_inputs()
    with pytest.raises(DecisionError, match="every claim got the same answers") as failed:
        check_for_glitch([_varied(1)] * 3, rounds)
    assert failed.value.retryable
    # Two claims cannot show it: a pair of similar claims may honestly score alike.
    check_for_glitch([_varied(1)] * 2, [[0.5, 0.5]])
    with pytest.raises(DecisionError, match="moment round 1 gave every claim the same probability"):
        check_for_glitch(claims, [[0.6, 0.3, 0.1], [1 / 3] * 3])
    # A repeat answer that cannot tell two other claims from `none` did not read them.
    repeats = _repeat_answers(3, {(0, 1): 0.6})
    check_for_glitch(claims, rounds, repeats)
    flat = {"repeat": ChoiceAnswer(choice="o01", probabilities={"o01": 1 / 3, "o02": 1 / 3, "none": 1 / 3})}
    with pytest.raises(DecisionError, match="repeat question for claim 1 gave every option the same probability") as failed:
        check_for_glitch(claims, rounds, [repeats[0], flat, repeats[2]])
    assert failed.value.retryable
    # With one other claim the two options may honestly split evenly.
    check_for_glitch(claims, rounds, [{"repeat": ChoiceAnswer(choice="none", probabilities={"o01": 0.5, "none": 0.5})}])


@pytest.mark.asyncio
async def test_a_glitched_run_fails_instead_of_returning_scores(monkeypatch):
    from src.tasks import claims_score_highlights as task

    same = _scale(0.5)
    stuck = {"essential": YesNoAnswer(0.4), "relevance": same, "quality": same, "controversy": same}
    fake = _FakeDecider(answers={0: stuck, 1: _varied(1), 2: _varied(2)})
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    with pytest.raises(DecisionError, match="provider glitch") as failed:
        await task._handle(_input(), None)
    assert failed.value.retryable


@pytest.mark.asyncio
async def test_repeats_share_a_group_id_whichever_debater_said_them(monkeypatch):
    from src.tasks import claims_score_highlights as task

    fake = _FakeDecider(same={(0, 1)})  # Ann's claim and Bob's say the same thing
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    result = await task._handle(_input(), None)
    assert [c.repeat_group for c in result.claims] == [0, 0, 2]


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
        assert list(build_questions(media_type)) == list(DECISIONS), media_type
        # The keys under a media type are exactly the score decision with its two criteria, the
        # moment choice, the repeat choice with its `none` option and each axis with one text per
        # level — nothing stale, nothing missing.
        expected = {f"{media_type}.essential", f"{media_type}.essential.yes", f"{media_type}.essential.no"}
        expected |= {f"{media_type}.moment", f"{media_type}.repeat", f"{media_type}.repeat.none"}
        for axis, levels in AXIS_LEVELS.items():
            expected.add(f"{media_type}.{axis}")
            expected.update(f"{media_type}.{axis}.{level}" for level in levels)
        registered_under = {
            key.removeprefix("claims_score_highlights.")
            for key in PROMPTS
            if key.startswith(f"claims_score_highlights.{media_type}.")
        }
        assert registered_under == expected, media_type
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
