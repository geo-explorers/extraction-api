"""transcript.punctuate and debates.pick_moments (GEO-3638) without a provider: what Claude is
shown, which answers are accepted, how a broken answer is sent back, and registry wiring."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.api.schemas.debates_pick_moments_schema import (
    DebatesPickMomentsInput,
    MomentClaim,
    MomentSentence,
    MomentSpeaker,
)
from src.api.schemas.transcript_punctuate_schema import PunctuateTurn, TranscriptPunctuateInput
from src.decisions.base import DecisionError
from src.pipeline import debates_pick_moments_core as pick
from src.pipeline import transcript_punctuate_core as punct
from src.pipeline.claude_json import parse_json


class _FakeClaude:
    """Answers each request with the next canned reply, and records the conversations."""

    def __init__(self, *replies, stop_reason="end_turn"):
        self.replies = list(replies)
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=self.stop_reason)


# --- transcript.punctuate ---------------------------------------------------------------


def _turns() -> TranscriptPunctuateInput:
    return TranscriptPunctuateInput(
        media_type="debate",
        title="Voter ID should be required nationwide",
        turns=[
            PunctuateTurn(speaker="Nico", words="it is a federal problem so it should be federal is that fair".split()),
            PunctuateTurn(speaker="Dahmmy", words="no".split()),
        ],
    )


def test_punctuate_shows_every_word_numbered_within_its_turn():
    shown = punct.render(_turns())
    assert "Turn 0 (Nico, 13 words, numbered 0-12)" in shown
    assert "0:it 1:is 2:a" in shown and "12:fair" in shown
    assert "Turn 1 (Dahmmy, 1 words, numbered 0-0):\n0:no" in shown


def test_punctuate_accepts_rising_ends_and_closes_a_turn_left_open():
    turns, problems = punct.check(_turns(), {"turns": [{"turn": 0, "ends": [[8, "."], [12, "?"]]}, {"turn": 1, "ends": []}]})
    assert problems == []
    assert [(e.word, e.mark) for e in turns[0].ends] == [(8, "."), (12, "?")]
    # A turn always ends a sentence, so the model need not say so.
    assert [(e.word, e.mark) for e in turns[1].ends] == [(0, ".")]


@pytest.mark.parametrize(
    "ends, problem",
    [
        ([[8, "."], [4, "."]], "out of order"),
        ([[13, "."]], "outside 0-12"),
        ([[8, ","]], "is not a [word number"),
    ],
)
def test_punctuate_refuses_ends_out_of_order_out_of_range_or_badly_marked(ends, problem):
    turns, problems = punct.check(_turns(), {"turns": [{"turn": 0, "ends": ends}, {"turn": 1, "ends": [[0, "."]]}]})
    assert turns is None
    assert any(problem in p for p in problems), problems


def test_punctuate_refuses_a_missing_turn():
    _, problems = punct.check(_turns(), {"turns": [{"turn": 0, "ends": [[12, "."]]}]})
    assert problems == ["Turn 1 is missing."]


def test_a_refused_answer_goes_back_with_its_problems_and_the_fix_is_kept():
    client = _FakeClaude(
        {"turns": [{"turn": 0, "ends": [[20, "."]]}, {"turn": 1, "ends": []}]},
        {"turns": [{"turn": 0, "ends": [[8, "."]]}, {"turn": 1, "ends": []}]},
    )
    result = asyncio.run(punct.punctuate(_turns(), client=client))
    assert [[(e.word, e.mark) for e in t.ends] for t in result.turns] == [[(8, "."), (12, ".")], [(0, ".")]]
    assert result.sentences == 3 and result.provider == "anthropic"
    retry = client.calls[1]["messages"]
    assert retry[1]["role"] == "assistant" and "outside 0-12" in retry[2]["content"]


def test_an_answer_still_wrong_after_the_last_attempt_fails_retryably():
    bad = {"turns": [{"turn": 0, "ends": [[20, "."]]}]}
    with pytest.raises(DecisionError) as raised:
        asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(bad, bad)))
    assert raised.value.retryable


def test_an_answer_cut_off_at_the_token_limit_is_not_trusted():
    good = {"turns": [{"turn": 0, "ends": [[12, "."]]}, {"turn": 1, "ends": []}]}
    with pytest.raises(DecisionError):
        asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(good, good, stop_reason="max_tokens")))


def test_a_provider_failure_is_retryable():
    with pytest.raises(DecisionError) as raised:
        asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(RuntimeError("overloaded"))))
    assert raised.value.retryable


def test_json_is_found_in_a_fence_or_surrounding_text():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here it is: {"a": 1} hope that helps') == {"a": 1}


# --- debates.pick_moments ---------------------------------------------------------------


def _debate() -> DebatesPickMomentsInput:
    """Four turns, alternating, three 5 s sentences each: sentences 0-2 turn 0 (speaker 1),
    3-5 turn 1 (speaker 2), 6-8 turn 2 (speaker 1), 9-11 turn 3 (speaker 2)."""
    sentences = []
    for turn in range(4):
        speaker = 1 if turn % 2 == 0 else 2
        for k in range(3):
            start = (turn * 3 + k) * 5_000
            sentences.append(MomentSentence(speaker=speaker, turn=turn, start_ms=start, end_ms=start + 4_800, text=f"Point {turn}.{k}."))
    return DebatesPickMomentsInput(
        media_type="debate",
        title="Voter ID should be required nationwide",
        speakers=[MomentSpeaker(slot=1, name="Dahmmy", side="Disagree"), MomentSpeaker(slot=2, name="Nico", side="Agree")],
        sentences=sentences,
        claims=[
            MomentClaim(index=0, speaker=1, text="States manage voter ID fine"),
            MomentClaim(index=5, speaker=2, text="Federal elections need federal rules"),
            MomentClaim(index=8, speaker=1, text="This president would abuse it"),
            MomentClaim(index=9, speaker=2, text="Every president tips the scales"),
        ],
    )


def _moment(speaker, start, end, claim, answers=None, role="answer"):
    return {"speaker": speaker, "start_sentence": start, "end_sentence": end, "claim": claim, "claim_fit": 0.8, "role": role, "answers": answers, "why": "x"}


GOOD = {
    "story": "States suffice; elections are federal; this president; every president.",
    "moments": [
        _moment(1, 0, 2, 0, role="opening"),
        _moment(2, 3, 5, 5, answers=0),
        _moment(1, 6, 8, 8, answers=1, role="rebuttal"),
        _moment(2, 9, 11, 9, answers=2, role="closing"),
    ],
}


def test_pick_shows_the_motion_sides_lengths_claims_and_numbered_sentences_by_turn():
    shown = pick.render(_debate())
    assert "Motion: Voter ID should be required nationwide" in shown
    assert "Speaker 2 (Nico): Agree with the motion" in shown
    assert "Target clip length: 10.0-18.0 s. Hard limits: 6.0-25.0 s." in shown
    assert "[9] Speaker 2: Every president tips the scales" in shown
    assert "## Turn 1: Speaker 2 (Nico)\n[3] 15.0-19.8 s: Point 1.0." in shown


def test_pick_accepts_four_alternating_clips_and_times_them_from_their_sentences():
    (story, moments), problems = pick.check(_debate(), GOOD)
    assert problems == []
    assert [(m.speaker, m.start_ms, m.end_ms, m.claim_index) for m in moments] == [
        (1, 0, 14_800, 0),
        (2, 15_000, 29_800, 5),
        (1, 30_000, 44_800, 8),
        (2, 45_000, 59_800, 9),
    ]
    assert moments[3].answers == 2 and moments[0].role == "opening"


def _with(position, **changes):
    answer = json.loads(json.dumps(GOOD))
    answer["moments"][position].update(changes)
    return answer


@pytest.mark.parametrize(
    "answer, problem",
    [
        ({"moments": GOOD["moments"][:3]}, "exactly 4"),
        (_with(1, speaker=1, start_sentence=6, end_sentence=8, claim=8), "alternate"),
        (_with(0, end_sentence=3), "not all Speaker 1's"),
        (_with(0, end_sentence=0), "lasts 4.8 s"),
        (_with(1, claim=0), "is Speaker 1's, not Speaker 2's"),
        (_with(1, claim=42), "no claim [42]"),
        (_with(2, claim=0), "link the same claim"),
        (_with(2, start_sentence=2, end_sentence=2), "lasts 4.8 s"),
        (_with(0, answers=2), "does not play before it"),
        (_with(2, answers=0), "answers its own speaker's clip"),
    ],
)
def test_pick_refuses_a_reel_that_breaks_a_rule(answer, problem):
    accepted, problems = pick.check(_debate(), answer)
    assert accepted is None
    assert any(problem in p for p in problems), problems


def test_pick_refuses_overlapping_clips():
    debate = _debate()
    # Two speaker-1 clips over the same turn-0 sentences.
    answer = {"moments": [_moment(1, 0, 2, 0), _moment(2, 3, 5, 5), _moment(1, 0, 2, 8), _moment(2, 9, 11, 9)]}
    _, problems = pick.check(debate, answer)
    assert any("overlap" in p for p in problems), problems


def test_pick_refuses_a_clip_across_two_turns_of_one_speaker():
    debate = _debate()
    for s in debate.sentences[3:6]:
        s.speaker = 1  # turn 1 is now speaker 1 too, so 2-3 is one speaker over two turns
    _, problems = pick.check(debate, _with(0, start_sentence=2, end_sentence=3))
    assert any("span more than one turn" in p for p in problems), problems


def test_pick_sends_a_broken_reel_back_and_returns_the_fixed_one():
    client = _FakeClaude({"moments": GOOD["moments"][:3]}, GOOD)
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert result.attempts == 2 and len(result.moments) == 4
    assert "exactly 4" in client.calls[1]["messages"][2]["content"]


def test_pick_avoids_refused_stretches():
    from src.api.schemas.debates_pick_moments_schema import AvoidRange

    debate = _debate()
    debate.avoid = [AvoidRange(start_sentence=7, end_sentence=7, reason="argues the other side")]
    assert "- sentences 7-7: argues the other side" in pick.render(debate)
    _, problems = pick.check(debate, GOOD)
    assert any("told to avoid" in p for p in problems), problems


def test_pick_inputs_need_two_speakers_and_known_claim_speakers():
    data = _debate().model_dump()
    data["claims"][0]["speaker"] = 3
    with pytest.raises(ValueError):
        DebatesPickMomentsInput(**data)


def test_both_tasks_are_registered_with_their_contracts_on_the_claude_budget():
    from src.tasks.registry import get_task

    for name in ("transcript.punctuate", "debates.pick_moments"):
        task = get_task(name)
        assert task is not None and task.runnable is not None
    assert get_task("transcript.punctuate").input_model is TranscriptPunctuateInput
    assert get_task("debates.pick_moments").input_model is DebatesPickMomentsInput
    from src.tasks.debates_pick_moments import DEBATES_PICK_MOMENTS_SPEC
    from src.tasks.transcript_punctuate import TRANSCRIPT_PUNCTUATE_SPEC

    for spec in (DEBATES_PICK_MOMENTS_SPEC, TRANSCRIPT_PUNCTUATE_SPEC):
        assert spec.rate_limit_key == "claude_global" and spec.concurrency == 1
