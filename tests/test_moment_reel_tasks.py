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
from src.pipeline.claude_json import FinalDecisionError, parse_json
from hatchet_sdk.exceptions import NonRetryableException


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
    turns, problems, _ = punct.check(_turns(), {"turns": [{"turn": 0, "ends": [[8, "."], [12, "?"]]}, {"turn": 1, "ends": []}]})
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
    turns, problems, _ = punct.check(_turns(), {"turns": [{"turn": 0, "ends": ends}, {"turn": 1, "ends": [[0, "."]]}]})
    assert turns is None
    assert any(problem in p for p in problems), problems


def test_punctuate_refuses_a_missing_turn():
    _, problems, _ = punct.check(_turns(), {"turns": [{"turn": 0, "ends": [[12, "."]]}]})
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


def test_an_answer_still_wrong_after_the_last_attempt_fails_without_an_engine_retry():
    bad = {"turns": [{"turn": 0, "ends": [[20, "."]]}]}
    with pytest.raises(FinalDecisionError) as raised:
        asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(bad, bad)))
    # Hatchet only stops retrying on NonRetryableException: asking again could not help.
    assert isinstance(raised.value, NonRetryableException) and not raised.value.retryable


def test_an_answer_cut_off_at_the_token_limit_is_final_and_not_asked_again():
    good = {"turns": [{"turn": 0, "ends": [[12, "."]]}, {"turn": 1, "ends": []}]}
    client = _FakeClaude(good, good, stop_reason="max_tokens")
    with pytest.raises(FinalDecisionError):
        asyncio.run(punct.punctuate(_turns(), client=client))
    # Every end is required, so a second answer would be as long: it is not asked for.
    assert len(client.calls) == 1


def test_a_provider_failure_is_retryable():
    with pytest.raises(DecisionError) as raised:
        asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(RuntimeError("overloaded"))))
    assert raised.value.retryable and not isinstance(raised.value, NonRetryableException)


def test_claude_calls_count_against_the_spend_budget_news_uses(monkeypatch):
    from src.pipeline import claude_json

    keys = []
    monkeypatch.setattr(claude_json.spend_guard, "check_and_record", keys.append)
    good = {"turns": [{"turn": 0, "ends": [[12, "."]]}, {"turn": 1, "ends": []}]}
    asyncio.run(punct.punctuate(_turns(), client=_FakeClaude(good)))
    assert keys == ["claude"]


def test_the_transcript_is_marked_for_prompt_caching():
    good = {"turns": [{"turn": 0, "ends": [[12, "."]]}, {"turn": 1, "ends": []}]}
    client = _FakeClaude(good)
    asyncio.run(punct.punctuate(_turns(), client=client))
    call = client.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_json_is_read_whole_or_from_a_fence_as_the_news_paths_read_it():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here it is:\n```json\n{"a": 1}\n```') == {"a": 1}


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
    assert "Target clip length: 10.0-18.0 s. Hard limits: 6.0-22.0 s." in shown
    assert "[9] Speaker 2: Every president tips the scales" in shown
    assert "## Turn 1: Speaker 2 (Nico)\n[3] 15.0-19.8 s (4.8 s): Point 1.0." in shown


def test_pick_accepts_four_alternating_clips_and_times_them_from_their_sentences():
    (story, moments), problems, soft = pick.check(_debate(), GOOD)
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
    accepted, problems, _ = pick.check(_debate(), answer)
    assert accepted is None
    assert any(problem in p for p in problems), problems


def test_a_one_based_answers_is_refused_not_a_crash():
    # Claude numbering the clips from 1: the last clip "answers" clip 4, which does not exist.
    accepted, problems, _ = pick.check(_debate(), _with(3, answers=4))
    assert accepted is None
    assert any("answers clip 4" in p for p in problems), problems
    # And sent back like any refused answer, not raised for Hatchet to rerun.
    client = _FakeClaude(_with(3, answers=4), GOOD)
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert result.attempts == 2


def test_a_check_that_cannot_read_the_answer_sends_it_back():
    from src.pipeline.claude_json import ask_json

    def unreadable(answer):
        return answer["missing"], [], []

    client = _FakeClaude({"x": 1}, {"missing": "ok"})
    value, _, attempts = asyncio.run(ask_json("s", "u", unreadable, task="t", attempts=2, client=client))
    assert (value, attempts) == ("ok", 2)


def test_a_clip_too_long_is_trimmed_from_its_start_to_fit():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    for k, sentence in enumerate(debate.sentences[0:3]):
        sentence.start_ms, sentence.end_ms = k * 12_000, k * 12_000 + 11_800  # 0-35.8 s in all
    for k, sentence in enumerate(debate.sentences[3:], start=3):
        sentence.start_ms, sentence.end_ms = 36_000 + (k - 3) * 5_000, 36_000 + (k - 3) * 5_000 + 4_800
    (_, moments), problems, _ = pick.check(debate, GOOD)
    assert problems == []
    # Sentence 0 is dropped: 1-2 is 23.8 s and keeps the clip's ending.
    assert (moments[0].start_sentence, moments[0].end_sentence, moments[0].start_ms) == (1, 2, 12_000)


def _long_first_turn(debate, lengths, texts=None):
    """Re-time turn 0 (sentences 0-2) to `lengths` seconds back to back, then the rest after it."""
    at = 0
    for k, seconds in enumerate(lengths):
        sentence = debate.sentences[k]
        sentence.start_ms, sentence.end_ms = at, at + int(seconds * 1000)
        if texts:
            sentence.text = texts[k]
        at += int(seconds * 1000)
    for k, sentence in enumerate(debate.sentences[3:], start=3):
        sentence.start_ms, sentence.end_ms = at + (k - 3) * 5_000, at + (k - 3) * 5_000 + 4_800


def test_a_trim_is_marked_on_the_clip_and_costs_no_extra_call():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [12, 12, 11.8])  # 35.8 s
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert problems == [] and soft == []
    assert (moments[0].start_sentence, moments[0].trimmed_from_sentence) == (1, 0)
    # A clip that was not trimmed says so.
    assert moments[1].trimmed_from_sentence is None
    # The trimmed reel is the answer: one call, not a send-back for each trim.
    client = _FakeClaude(GOOD, GOOD, GOOD)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 1 and len(client.calls) == 1


def test_a_trim_that_can_only_land_on_a_dangling_opener_is_not_nudged():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 34 s; every start that fits (1 or 2) opens on a dangling word, so the trim takes 1.
    _long_first_turn(debate, [10, 12, 12], ["One.", "And that.", "But so."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert problems == [] and moments[0].start_sentence == 1
    # Starting earlier would only be trimmed back: the model is not told to.
    assert soft == []


def test_a_trim_that_would_go_below_the_minimum_reports_the_real_length():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [27, 2, 3])  # 32 s; dropping sentence 0 leaves 5 s
    _, problems, _ = pick.check(debate, GOOD)
    assert any("Clip 0 lasts 32.0 s" in p for p in problems), problems


def test_a_trim_steps_past_a_dangling_opener_when_a_clean_start_fits():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 34 s. Dropping sentence 0 lands on "And", so the trim steps on to sentence 2.
    _long_first_turn(debate, [10, 12, 12], ["One.", "And that is why.", "It holds."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert problems == []
    assert moments[0].start_sentence == 2
    assert not any('opens on "and"' in p for p in soft), soft


def test_a_soft_answer_between_hard_ones_is_kept_and_its_nudge_sent_once():
    debate = _debate()
    debate.sentences[3].text = "And it is a federal problem."
    broken = {"moments": GOOD["moments"][:3]}
    client = _FakeClaude(broken, GOOD, broken)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 2 and len(result.moments) == 4
    conversation = client.calls[-1]["messages"]
    first, second = conversation[2]["content"], conversation[4]["content"]
    assert "exactly 4" in first and 'opens on "and"' not in first
    assert 'opens on "and"' in second


def test_one_sentence_too_long_is_still_refused():
    debate = _debate()
    debate.sentences[2].end_ms = debate.sentences[2].start_ms + 31_000
    _, problems, _ = pick.check(debate, _with(0, start_sentence=2, end_sentence=2))
    assert any("lasts 31.0 s" in p for p in problems), problems


def test_hard_problems_are_sent_back_without_the_soft_ones():
    debate = _debate()
    debate.sentences[3].text = "And it is a federal problem."
    client = _FakeClaude({"moments": GOOD["moments"][:3]}, GOOD, GOOD)
    asyncio.run(pick.pick_moments(debate, client=client))
    first_send_back = client.calls[1]["messages"][2]["content"]
    assert "exactly 4" in first_send_back and 'opens on "and"' not in first_send_back


def test_pick_refuses_overlapping_clips():
    debate = _debate()
    # Two speaker-1 clips over the same turn-0 sentences.
    answer = {"moments": [_moment(1, 0, 2, 0), _moment(2, 3, 5, 5), _moment(1, 0, 2, 8), _moment(2, 9, 11, 9)]}
    _, problems, _ = pick.check(debate, answer)
    assert any("overlap" in p for p in problems), problems


def test_pick_refuses_a_clip_across_two_turns_of_one_speaker():
    debate = _debate()
    for s in debate.sentences[3:6]:
        s.speaker = 1  # turn 1 is now speaker 1 too, so 2-3 is one speaker over two turns
    _, problems, _ = pick.check(debate, _with(0, start_sentence=2, end_sentence=3))
    assert any("span more than one turn" in p for p in problems), problems


def test_pick_sends_a_broken_reel_back_and_returns_the_fixed_one():
    client = _FakeClaude({"moments": GOOD["moments"][:3]}, GOOD)
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert result.attempts == 2 and len(result.moments) == 4
    assert "exactly 4" in client.calls[1]["messages"][2]["content"]


def test_a_clip_opening_on_a_connective_is_asked_again_but_kept_on_the_last_answer():
    debate = _debate()
    debate.sentences[3].text = "And it is a federal problem."
    accepted, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and accepted is not None
    assert any('opens on "and"' in p for p in soft), soft
    # Asked again while attempts remain; the last answer is kept with it.
    client = _FakeClaude(GOOD, GOOD, GOOD)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 3 and len(client.calls) == 3
    assert 'opens on "and"' in client.calls[1]["messages"][2]["content"]


def test_an_acceptable_answer_is_kept_when_the_ones_sent_back_get_worse():
    debate = _debate()
    debate.sentences[3].text = "Yeah it is a federal problem."
    broken = {"moments": GOOD["moments"][:3]}
    client = _FakeClaude(GOOD, broken, broken)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    # The first answer only opened on "yeah": better than two that break hard rules.
    assert result.attempts == 1 and len(result.moments) == 4 and len(client.calls) == 3


def test_openers_are_matched_on_straight_and_curly_apostrophes_alike():
    debate = _debate()
    debate.sentences[3].text = "That\u2019s why it is federal."
    _, _, soft = pick.check(debate, GOOD)
    assert any("that's" in p for p in soft), soft
    # A sentence that opens on "this" and says what it is about stands alone.
    debate.sentences[3].text = "This is the number that matters."
    _, _, soft = pick.check(debate, GOOD)
    assert soft == []


def test_pick_avoids_refused_stretches():
    from src.api.schemas.debates_pick_moments_schema import AvoidRange

    debate = _debate()
    debate.avoid = [AvoidRange(start_sentence=7, end_sentence=7, reason="argues the other side")]
    assert "- sentences 7-7: argues the other side" in pick.render(debate)
    _, problems, _ = pick.check(debate, GOOD)
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

    from src.pipeline.claude_json import REQUEST_TIMEOUT_S

    for spec, attempts in ((DEBATES_PICK_MOMENTS_SPEC, pick.CALLS), (TRANSCRIPT_PUNCTUATE_SPEC, punct.ATTEMPTS)):
        assert spec.rate_limit_key == "claude_global" and spec.concurrency == 1
        # One rate unit per Claude call the run can make, and time for every one of them.
        assert spec.rate_limit_units == attempts
        assert spec.execution_timeout.total_seconds() >= attempts * REQUEST_TIMEOUT_S


# --- candidate reels and the judge (11 Nov grading) -------------------------------------

# A second valid reel: speaker 2 opens, then the same exchange the other way round.
OTHER = {
    "story": "Speaker 2 opens on federal rules; Speaker 1 answers.",
    "moments": [
        _moment(2, 3, 5, 5, role="opening"),
        _moment(1, 0, 2, 0, answers=0),
        _moment(2, 9, 11, 9, answers=1, role="rebuttal"),
        _moment(1, 6, 8, 8, answers=2, role="closing"),
    ],
}
BROKEN = {"moments": GOOD["moments"][:3]}


def test_candidate_reels_that_pass_are_kept_and_a_broken_one_is_dropped():
    passed, problems, soft = pick.check_reels(_debate(), {"reels": [BROKEN, GOOD, OTHER]})
    assert problems == [] and soft == []
    assert [moments[0].speaker for (_, moments), _ in passed] == [1, 2]


def test_an_answer_is_sent_back_only_when_no_candidate_passes():
    passed, problems, _ = pick.check_reels(_debate(), {"reels": [BROKEN, BROKEN]})
    assert passed is None
    assert any(p.startswith("Reel 0: ") and "exactly 4" in p for p in problems), problems
    assert any(p.startswith("Reel 1: ") for p in problems), problems


def test_the_judge_picks_between_passing_reels_from_the_clips_alone():
    client = _FakeClaude({"reels": [GOOD, OTHER]}, {"best": 1, "why": "Nico opens stronger."})
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1]
    assert (result.candidates, result.judge_why) == (2, "Nico opens stronger.")
    judged = client.calls[1]["messages"][0]["content"][0]["text"]
    # The viewer's view: who speaks and what they say, not the claims or the numbered transcript.
    assert "Clip 1, Nico, 14.8 s: Point 1.0. Point 1.1. Point 1.2." in judged
    assert "Claims:" not in judged and "[3]" not in judged and "Federal elections need" not in judged
    assert "## Reel 1 (59.2 s)" in judged


def test_one_passing_reel_needs_no_judge():
    client = _FakeClaude({"reels": [GOOD, BROKEN]})
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert len(client.calls) == 1 and result.candidates == 1 and result.judge_why is None


def test_a_judge_that_cannot_answer_keeps_the_first_reel():
    client = _FakeClaude({"reels": [GOOD, OTHER]}, {"best": 7}, {"best": "x"})
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert [m.speaker for m in result.moments] == [1, 2, 1, 2] and len(client.calls) == 3


def test_a_judge_request_that_fails_keeps_the_first_reel():
    client = _FakeClaude({"reels": [GOOD, OTHER]}, RuntimeError("overloaded"))
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert [m.speaker for m in result.moments] == [1, 2, 1, 2]


def test_a_clean_reel_is_taken_over_one_with_soft_problems_without_a_judge():
    debate = _debate()
    debate.sentences[0].text = "So it is a state matter."  # GOOD opens on "so"
    clean = json.loads(json.dumps(OTHER))
    clean["moments"][1]["start_sentence"] = 1  # starts past it
    client = _FakeClaude({"reels": [GOOD, clean]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1] and len(client.calls) == 1


def test_a_clip_opening_on_so_is_a_soft_problem():
    debate = _debate()
    debate.sentences[3].text = "So it is a federal problem."
    _, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and any('opens on "so"' in p for p in soft), soft


def test_a_reel_over_seventy_seconds_is_asked_to_shorten():
    debate = _debate()
    debate.clip_max_ms = 30_000
    for k, sentence in enumerate(debate.sentences):
        sentence.start_ms, sentence.end_ms = k * 6_200, k * 6_200 + 6_000  # 18.4 s clips, 73.6 s reel
    _, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and any("The reel runs 73.6 s" in p for p in soft), soft
    debate.sentences[11].end_ms -= 4_000  # 69.6 s
    assert pick.check(debate, GOOD)[2] == []


def test_a_story_calls_the_speakers_by_first_name():
    (story, _), _, _ = pick.check(_debate(), {**GOOD, "story": "Speaker 1 says states; Speaker 2 disagrees."})
    assert story == "Dahmmy says states; Nico disagrees."
    debate = _debate()
    debate.speakers[0].name = "Dahmmy Hilfigger (@nsddahmmy)"
    assert pick.named(debate, "Speaker 1 and Speaker 3") == "Dahmmy and Speaker 3"
