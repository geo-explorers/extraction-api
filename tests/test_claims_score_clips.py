"""claims.score_clips without a provider: the questions, what the model is shown, how answers
become a judgement per clip, failure handling, and registry wiring."""

import asyncio

import pytest

from src.api.schemas.claims_score_clips_schema import ClaimsScoreClipsInput, ScoreClip
from src.api.schemas.claims_score_highlights_schema import HighlightDocument
from src.config.overrides import activate
from src.decisions import Choice, ChoiceAnswer, DecisionError, YesNo, YesNoAnswer
from src.pipeline import claims_score_clips_core as core


def _input(**overrides) -> ClaimsScoreClipsInput:
    base = dict(
        media_type="debate",
        title="It's fine to ghost someone after a few dates",
        context="Urban argues Agree; Susan argues Disagree.",
        documents=[
            HighlightDocument(speaker="Susan", content="The claim is it's fine to ghost. I have to disagree."),
            HighlightDocument(speaker="Urban", content="So is it closure or is it entitlement?"),
            HighlightDocument(speaker="Susan", content="The person who refuses the answer is the one who feels entitled."),
        ],
        clips=[
            ScoreClip(id="a", speaker="Urban", side="Agree", text="So is it closure or is it entitlement?", document_index=1, duration_ms=17000),
            ScoreClip(id="b", speaker="Susan", side="Disagree", text="The person who refuses the answer is the one who feels entitled.", document_index=2),
            ScoreClip(speaker="Susan", text="The claim is it's fine to ghost someone after a few dates."),
        ],
    )
    return ClaimsScoreClipsInput(**{**base, **overrides})


class _FakeDecider:
    """Answers like a model that read the clips: per clip, yes/no probabilities from `yes`
    (clip index -> dict), its reply from `replies` (clip index -> clip index, "none" or
    "elsewhere"); the opener choice favours the lowest clip number, a little differently each
    round so the glitch guard sees real answers."""

    provider = "fake"

    def __init__(self, yes=None, replies=None, fail_on=(), hang_on=(), flat_opener=False, same_answers=False):
        self.yes = yes or {}
        self.replies = replies or {}
        self.fail_on, self.hang_on = set(fail_on), set(hang_on)
        self.flat_opener, self.same_answers = flat_opener, same_answers
        self.requests = []
        self.in_flight = 0

    async def decide(self, content, questions, *, model):
        self.requests.append((content, questions, model))
        if list(questions) == ["opener"]:
            options = questions["opener"].options
            if self.flat_opener:
                weights = {name: 1.0 for name in options}
            else:
                weights = {name: 1.0 / int(name) + 0.001 * len(self.requests) for name in options}
            total = sum(weights.values())
            return {"opener": ChoiceAnswer(choice=min(options, key=int), probabilities={n: w / total for n, w in weights.items()})}
        target = int(content["clip_under_test"].split()[1]) - 1
        self.in_flight += 1
        try:
            if target in self.hang_on:
                await asyncio.Event().wait()
            if target in self.fail_on:
                raise DecisionError(f"clip {target + 1} failed", retryable=True)
            yes = {name: 0.5 if self.same_answers else 0.9 - 0.1 * target for name in core.YES_NO}
            yes.update(self.yes.get(target, {}))
            options = questions["responds_to"].options
            wanted = self.replies.get(target, "none")
            if isinstance(wanted, int):
                wanted = next(key for key, label in options.items() if label and label.startswith(f"Clip {wanted + 1} "))
            reply = {key: (0.8 if key == wanted else 0.2 / (len(options) - 1)) for key in options}
            return {
                **{name: YesNoAnswer(probability=p) for name, p in yes.items()},
                "responds_to": ChoiceAnswer(choice=wanted, probabilities=reply),
            }
        finally:
            self.in_flight -= 1


# ── Questions and content ────────────────────────────────────────────────────


def test_each_clip_is_asked_three_yes_no_questions_and_which_clip_it_answers():
    questions = core.build_questions(_input(), 1)
    assert list(questions) == ["own_side", "standalone", "complete", "responds_to"]
    for name in core.YES_NO:
        assert isinstance(questions[name], YesNo) and questions[name].yes and questions[name].no
    assert "words as transcribed" in questions["own_side"].instructions
    assert "misheard word" in questions["own_side"].instructions
    assert "seen nothing else" in questions["standalone"].instructions
    assert "right after its last word" in questions["complete"].instructions
    reply = questions["responds_to"]
    assert isinstance(reply, Choice)
    # Susan's clip may answer Urban's, never her own other clip.
    labels = [label for key, label in reply.options.items() if key not in ("none", "elsewhere")]
    assert labels == [core.clip_label(_input().clips[0], 0)]
    assert set(reply.options) >= {"none", "elsewhere"}


def test_content_shows_the_debate_every_clip_and_the_clip_under_test():
    content = core.build_content(_input(), 0)
    assert list(content) == ["title", "context", "transcript", "candidate_clips", "clip_under_test"]
    assert content["transcript"].startswith("[0] Susan: The claim is")
    assert content["candidate_clips"].count("\n") == 2
    assert content["clip_under_test"] == 'Clip 1 (Urban, Agree, 17 s, in passage [1]): "So is it closure or is it entitlement?"'
    assert core.clip_label(_input().clips[2], 2) == 'Clip 3 (Susan): "The claim is it\'s fine to ghost someone after a few dates."'


def test_opener_rounds_show_the_clips_in_different_seeded_orders():
    requests = core.opener_requests(_input())
    assert len(requests) == core.OPENER_ROUNDS
    orders = [list(questions["opener"].options) for _, _, questions in requests]
    assert all(sorted(order, key=int) == ["1", "2", "3"] for order in orders)
    assert len({tuple(order) for order in orders}) > 1, "the rounds differ in order"
    assert orders == [list(q["opener"].options) for _, _, q in core.opener_requests(_input())], "seeded"
    assert core.opener_requests(_input(clips=_input().clips[:1])) == []


def test_rubric_text_is_overridable_per_run():
    with activate(prompt_overrides={"claims_score_clips.debate.standalone": "Changed."}):
        assert core.build_questions(_input(), 0)["standalone"].instructions == "Changed."
    accepted = _input(prompt_overrides={"claims_score_clips.debate.opener": "Pick one."})
    assert accepted.prompt_overrides == {"claims_score_clips.debate.opener": "Pick one."}
    with pytest.raises(ValueError, match="unknown prompt key"):
        _input(prompt_overrides={"claims_score_clips.debate.hook": "x"})


# ── A run ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_judges_every_clip_with_the_configured_model(monkeypatch):
    from src.tasks import claims_score_clips as task

    fake = _FakeDecider(yes={2: {"own_side": 0.05}}, replies={1: 0, 2: "elsewhere"})
    monkeypatch.setattr(task, "get_decision_model", lambda provider: fake)
    with activate(llm_overrides={"claims_highlights_model": "other-model"}):
        result = await task._handle(_input(), None)
    assert result.clips_scored == 3 and result.provider == "fake" and result.model_used == "other-model"
    assert {model for _, _, model in fake.requests} == {"other-model"}
    # One request per clip and three opener rounds.
    assert len(fake.requests) == 3 + core.OPENER_ROUNDS
    a, b, c = result.clips
    assert [clip.id for clip in result.clips] == ["a", "b", None]
    # The lowest clip number is the likeliest opener every round, so it ranks first.
    assert (a.hook, b.hook, c.hook) == (1.0, 0.5, 0.0)
    assert a.decisions["opener"] > b.decisions["opener"] > c.decisions["opener"]
    # Susan's reply points at Urban's clip; the motion read aloud answers nothing listed.
    assert (a.responds_to, b.responds_to, c.responds_to) == (None, 0, None)
    assert c.decisions["answers_elsewhere"] == pytest.approx(0.8)
    assert c.decisions["own_side"] == pytest.approx(0.05)
    assert set(a.decisions) == {"own_side", "standalone", "complete", "opener", "answers_elsewhere"}


@pytest.mark.asyncio
async def test_a_lone_clip_is_its_own_opener():
    fake = _FakeDecider()
    one = _input(clips=_input().clips[:1])
    decisions = await core.decide_clips(one, fake, "m")
    result = core.assemble_result(one, decisions, "fake", "m")
    assert len(fake.requests) == 1
    assert result.clips[0].hook == 0.5 and result.clips[0].decisions["opener"] == 1.0


@pytest.mark.asyncio
async def test_one_failed_clip_fails_the_run_and_stops_the_others():
    fake = _FakeDecider(fail_on={1}, hang_on={0, 2})
    with pytest.raises(DecisionError, match="clip 2 failed") as failed:
        await asyncio.wait_for(core.decide_clips(_input(), fake, "m"), timeout=2)
    assert failed.value.retryable
    assert fake.in_flight == 0, "the hanging requests were cancelled"


@pytest.mark.asyncio
async def test_a_glitched_run_fails_and_can_be_retried():
    for fake in (_FakeDecider(flat_opener=True), _FakeDecider(same_answers=True)):
        with pytest.raises(DecisionError, match="provider glitch") as failed:
            await core.decide_clips(_input(), fake, "m")
        assert failed.value.retryable


def test_answers_of_the_wrong_shape_fail_the_run():
    with pytest.raises(DecisionError, match="clip questions"):
        core.clip_decisions(_input(), 0, {"own_side": YesNoAnswer(probability=0.5)})
    answers = {name: YesNoAnswer(probability=0.5) for name in core.YES_NO}
    answers["responds_to"] = ChoiceAnswer(choice="none", probabilities={"none": 1.0})
    with pytest.raises(DecisionError, match="no probability"):
        core.clip_decisions(_input(), 1, answers)


# ── Registry and bounds ──────────────────────────────────────────────────────


def test_the_rubric_keys_are_exactly_registered():
    from src.config.prompt_registry import PROMPTS

    expected = {f"debate.{name}{suffix}" for name in core.YES_NO for suffix in ("", ".yes", ".no")}
    expected |= {"debate.responds_to", "debate.responds_to.none", "debate.responds_to.elsewhere", "debate.opener"}
    registered = {key.removeprefix("claims_score_clips.") for key in PROMPTS if key.startswith("claims_score_clips.")}
    assert registered == expected


def test_registry_has_the_task_with_its_contract():
    from src.api.schemas.claims_score_clips_schema import ClaimsScoreClipsResult
    from src.tasks.claims_score_clips import CLAIMS_SCORE_CLIPS_SPEC
    from src.tasks.registry import get_task

    entry = get_task("claims.score_clips")
    assert entry is not None
    assert entry.input_model is ClaimsScoreClipsInput and entry.output_model is ClaimsScoreClipsResult
    assert CLAIMS_SCORE_CLIPS_SPEC.rate_limit_key == "decisions_global"


def test_input_bounds():
    with pytest.raises(ValueError):
        _input(media_type="news")
    with pytest.raises(ValueError):
        _input(clips=[])
    with pytest.raises(ValueError, match="out of range"):
        _input(clips=[ScoreClip(speaker="A", text="x", document_index=3)])
    with pytest.raises(ValueError):
        _input(clips=[ScoreClip(speaker="A", text="x")] * 25)
