"""transcript.punctuate and debates.pick_moments (GEO-3638) without a provider: what Claude is
shown, which answers are accepted, how a broken answer is sent back, and registry wiring."""

import asyncio
import json
import re
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


class _CutOff:
    """A canned reply that stops at the token limit."""

    def __init__(self, text):
        self.text = text


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
        if isinstance(reply, _CutOff):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply.text)], stop_reason="max_tokens")
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
    # Nor is its answer shaped by a schema: punctuate keeps asking as it did.
    assert "output_config" not in client.calls[0]


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


# --- transcript.punctuate: ends inside a clause (GEO-3694) ---------------------------------


def _timed(text: str, pauses: dict = None, speaker: str = "Nico") -> PunctuateTurn:
    """A turn whose words each last 200 ms with 50 ms between them, or `pauses[i]` ms after word i."""
    starts, ends, at = [], [], 0
    for i, _ in enumerate(text.split()):
        starts.append(at)
        ends.append(at + 200)
        at += 200 + (pauses or {}).get(i, 50)
    return PunctuateTurn(speaker=speaker, words=text.split(), starts_ms=starts, ends_ms=ends)


def _input(*turns: PunctuateTurn) -> TranscriptPunctuateInput:
    return TranscriptPunctuateInput(media_type="debate", title="Illegal immigration is a security risk", turns=list(turns))


def _ends(*turns) -> dict:
    return {"turns": [{"turn": t, "ends": ends} for t, ends in enumerate(turns)]}


SPLIT = "so illegal immigration is a national security risk in my view"


def test_an_end_on_a_word_that_cannot_end_a_sentence_is_dropped_when_the_speaker_went_straight_on():
    # "...is a. National security risk..." with 50 ms between "a" and "national".
    turns, problems, soft = punct.check(_input(_timed(SPLIT)), _ends([[4, "."], [10, "."]]))
    assert problems == [] and soft == []
    assert [(e.word, e.mark) for e in turns[0].ends] == [(10, ".")]


def test_the_end_stays_after_a_pause_long_enough_to_be_a_trail_off():
    # "i think the." then a 900 ms pause and a fresh start: the speaker broke off.
    inp = _input(_timed("i think the the question is how it is framed", pauses={2: punct.JOIN_GAP_MS + 400}))
    turns, problems, soft = punct.check(inp, _ends([[2, "."], [9, "."]]))
    assert problems == [] and soft == []
    assert [e.word for e in turns[0].ends] == [2, 9]


def test_a_turn_ending_on_such_a_word_still_ends_there_and_is_never_joined_to_the_next_turn():
    inp = _input(_timed("we should look at the"), _timed("idea of a border", speaker="Dahmmy"))
    turns, problems, soft = punct.check(inp, _ends([[4, "."]], [[3, "."]]))
    assert problems == [] and soft == []
    assert [[e.word for e in t.ends] for t in turns] == [[4], [3]]


@pytest.mark.parametrize(
    "words, ends",
    [
        # A question or exclamation is the model's call: "...you agree, or?"
        ("you agree or that is what I hear", [[2, "?"], [7, "."]]),
        # Capitalised, so a name or a letter: "We call it Plan A."
        ("we call it Plan A that is the idea", [[4, "."], [8, "."]]),
    ],
)
def test_ends_marked_as_questions_or_on_capitalised_words_are_untouched(words, ends):
    turns, problems, soft = punct.check(_input(_timed(words)), _ends(ends))
    assert problems == [] and soft == []
    assert [[e.word, e.mark] for e in turns[0].ends] == ends


def test_without_word_times_such_an_end_is_kept_and_sent_back_as_a_soft_problem():
    inp = _input(PunctuateTurn(speaker="Nico", words=SPLIT.split()))
    turns, problems, soft = punct.check(inp, _ends([[4, "."], [10, "."]]))
    assert problems == [] and [e.word for e in turns[0].ends] == [4, 10]
    assert len(soft) == 1 and 'ends on "a"' in soft[0] and "word 4" in soft[0]


OPENS = "this is how we got here in the first place where corporations argued for more sugar"


@pytest.mark.parametrize(
    "words, first_end",
    [
        (OPENS, 9),  # "...in the first place. Where corporations argued..."
        ("they worry about the border that the border security can serve a purpose", 4),
        ("i keep coming back to the idea to the idea of a human relationship", 6),
        ("we tried a sugar tax which is why prices went up", 4),
    ],
)
def test_a_sentence_that_opens_like_the_rest_of_the_one_before_is_sent_back_as_a_soft_problem(words, first_end):
    last = len(words.split()) - 1
    turns, problems, soft = punct.check(_input(_timed(words)), _ends([[first_end, "."], [last, "."]]))
    # The answer stands as it is: only the model can tell a clause from a sentence.
    assert problems == [] and [e.word for e in turns[0].ends] == [first_end, last]
    assert len(soft) == 1 and f"starting at word {first_end + 1}" in soft[0] and f"end at word {first_end}" in soft[0]


@pytest.mark.parametrize(
    "words, pauses, ends",
    [
        # "Where I come from..." after a real pause: a sentence of its own.
        ("that is my answer where i come from people vote", {3: punct.HINT_GAP_MS + 500}, [[3, "."], [9, "."]]),
        # "Where in the Constitution...?" asks, so it opens a sentence.
        ("that is my answer where in the constitution is that", {}, [[3, "."], [9, "?"]]),
        # After a question, a new sentence starts whatever it opens with.
        ("is that fair to the first point yes", {}, [[2, "?"], [7, "."]]),
    ],
)
def test_openers_after_a_long_pause_a_question_or_opening_one_are_not_hinted(words, pauses, ends):
    turns, problems, soft = punct.check(_input(_timed(words, pauses=pauses)), _ends(ends))
    assert problems == [] and soft == []
    assert [[e.word, e.mark] for e in turns[0].ends] == ends


def test_a_one_word_sentence_is_not_read_as_opening_with_the_next_sentence():
    # "That." is not "That the...": the opener stops at the sentence's own end.
    words = "so what about that That The point is it fails"
    turns, problems, soft = punct.check(_input(_timed(words)), _ends([[3, "."], [4, "."], [9, "."]]))
    assert problems == [] and soft == []
    assert [e.word for e in turns[0].ends] == [3, 4, 9]


@pytest.mark.parametrize("extra, joined", [(0, True), (1, False)])
def test_an_end_is_joined_up_to_and_including_the_join_gap(extra, joined):
    turns, problems, soft = punct.check(
        _input(_timed(SPLIT, pauses={4: punct.JOIN_GAP_MS + extra})), _ends([[4, "."], [10, "."]])
    )
    assert problems == [] and soft == []
    assert [e.word for e in turns[0].ends] == ([10] if joined else [4, 10])


@pytest.mark.parametrize("extra, hinted", [(0, True), (1, False)])
def test_an_opener_is_hinted_up_to_and_including_the_hint_gap(extra, hinted):
    turns, problems, soft = punct.check(
        _input(_timed(OPENS, pauses={9: punct.HINT_GAP_MS + extra})), _ends([[9, "."], [15, "."]])
    )
    assert problems == [] and [e.word for e in turns[0].ends] == [9, 15]
    assert len(soft) == (1 if hinted else 0)


@pytest.mark.parametrize("joined", [True, False])
def test_a_hint_goes_back_once_and_the_second_answer_is_kept_whether_it_joins_or_not(joined):
    first = _ends([[9, "."], [15, "."]])
    second = _ends([[15, "."]]) if joined else first
    client = _FakeClaude(first, second)
    result = asyncio.run(punct.punctuate(_input(_timed(OPENS)), client=client))
    # Two answers, within the budget the task reserves; the hint went back with the first.
    assert len(client.calls) == punct.ATTEMPTS == 2
    assert "starting at word 10" in client.calls[1]["messages"][2]["content"]
    assert [e.word for e in result.turns[0].ends] == ([15] if joined else [9, 15])


def test_an_answer_with_nothing_to_hint_is_kept_without_asking_again():
    client = _FakeClaude(_ends([[4, "."], [10, "."]]))
    result = asyncio.run(punct.punctuate(_input(_timed(SPLIT)), client=client))
    assert len(client.calls) == 1
    assert [e.word for e in result.turns[0].ends] == [10] and result.sentences == 1


@pytest.mark.parametrize(
    "starts, ends",
    [([0, 1], None), ([0], [1]), ([0, 1], [1, 2, 3])],
)
def test_word_times_come_as_a_pair_with_one_time_per_word(starts, ends):
    with pytest.raises(ValueError):
        PunctuateTurn(speaker="Nico", words=["a", "b"], starts_ms=starts, ends_ms=ends)


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
    assert "Target clip length: 10.0-18.0 s. Hard limits: 6.0-22.0 s. The whole reel: under 72.0 s." in shown
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


def test_a_clip_too_long_is_sent_back_with_its_running_times_and_cut_from_its_end_as_a_last_resort():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    for k, sentence in enumerate(debate.sentences[0:3]):
        sentence.start_ms, sentence.end_ms = k * 12_000, k * 12_000 + 11_800  # 0-35.8 s in all
    for k, sentence in enumerate(debate.sentences[3:], start=3):
        sentence.start_ms, sentence.end_ms = 36_000 + (k - 3) * 5_000, 36_000 + (k - 3) * 5_000 + 4_800
    (_, moments), problems, _ = pick.check(debate, GOOD)
    # Refused, with how long it runs to each sentence so the next pick fits (12 Nov: the trims
    # from the start cut the setup in 6 of 36 clips).
    assert problems == [
        "Clip 0, sentences 0-2, runs 35.8 s, over the 30.0 s limit. From sentence 0 it runs 11.8 s to "
        "sentence 0, 23.8 s to sentence 1, 35.8 s to sentence 2. End it earlier, after a sentence where "
        "its point has landed, or start it later on a sentence that opens a thought, to run 6.0-30.0 s."
    ]
    # As a last resort it is cut from its end: 0-1 is 23.8 s and keeps the clip's setup.
    assert (moments[0].start_sentence, moments[0].end_sentence, moments[0].end_ms) == (0, 1, 23_800)
    assert (moments[0].trimmed_to_sentence, moments[0].trimmed_from_sentence) == (2, None)


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


def test_a_clip_over_the_limit_is_picked_again_and_cut_to_fit_only_as_a_last_resort():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [12, 12, 11.8])  # 35.8 s
    # The model picks it again shorter: that answer wins, nothing cut.
    client = _FakeClaude(GOOD, _with(0, end_sentence=1))
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 2 and (result.moments[0].start_sentence, result.moments[0].end_sentence) == (0, 1)
    assert result.moments[0].trimmed_to_sentence is None
    assert "Clip 0, sentences 0-2, runs 35.8 s, over the 30.0 s limit" in client.calls[1]["messages"][2]["content"]
    # Never picked shorter: the last answer's reel is kept, cut to fit, so the run still has a reel.
    client = _FakeClaude(GOOD, GOOD, GOOD)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 3 and len(client.calls) == 3
    assert (result.moments[0].end_sentence, result.moments[0].trimmed_to_sentence) == (1, 2)
    # A clip that was not cut says so.
    assert result.moments[1].trimmed_to_sentence is None and result.moments[1].trimmed_from_sentence is None


def test_a_start_trim_that_can_only_land_on_a_dangling_opener_is_flagged_as_it_plays():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 55 s and sentence 0 alone is over the limit, so no end fits; every start that fits
    # (1 or 2) opens on a dangling word, so the trim takes 1.
    _long_first_turn(debate, [31, 12, 12], ["One.", "And that.", "But so."])
    # Turn 0 is OTHER's second clip.
    (_, moments), problems, soft = pick.check(debate, OTHER)
    assert moments[1].start_sentence == 1 and moments[1].trimmed_from_sentence == 0
    assert len(problems) == 1 and problems[0].startswith("Clip 1, sentences 0-2, runs 55.0 s")
    # The opener of a clip cut to fit is held against it too: it is what the viewer hears.
    assert soft == ['Clip 1 opens on "and", which leans on what came before or is filler. Start it on a sentence that opens a thought.']
    # As the first clip it is refused all the same: a newcomer meets the debate on "And that."
    _, problems, _ = pick.check(debate, GOOD)
    assert problems[1:] == [
        'Clip 0, the reel\'s first clip, opens on sentence 1 (cut to fit from sentence 0), which opens '
        'on "and", a connective or filler. The first clip is where a newcomer meets the debate: start '
        "it on a sentence that begins a thought, or pick another first clip."
    ]


def test_a_trim_that_would_go_below_the_minimum_reports_the_real_length():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 36 s; sentence 0 alone is over the limit, and dropping it leaves 5 s.
    _long_first_turn(debate, [31, 2, 3])
    value, problems, _ = pick.check(debate, GOOD)
    assert value is None and any("Clip 0 lasts 36.0 s" in p for p in problems), problems


def test_a_trim_steps_past_a_dangling_opener_when_a_clean_start_fits():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 55 s and no end fits. Dropping sentence 0 lands on "But", so the trim steps on to sentence 2.
    _long_first_turn(debate, [31, 12, 12], ["One.", "But that is why.", "We hold."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert len(problems) == 1 and "over the 30.0 s limit" in problems[0]
    assert moments[0].start_sentence == 2
    assert not any('opens on "but"' in p for p in soft), soft


def test_a_soft_answer_between_hard_ones_is_kept_and_its_nudge_sent_once():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."
    broken = {"moments": GOOD["moments"][:3]}
    client = _FakeClaude(broken, GOOD, broken)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 2 and len(result.moments) == 4
    conversation = client.calls[-1]["messages"]
    first, second = conversation[2]["content"], conversation[4]["content"]
    assert "exactly 4" in first and 'opens on "but"' not in first
    assert 'opens on "but"' in second


def test_one_sentence_too_long_is_still_refused():
    debate = _debate()
    debate.sentences[2].end_ms = debate.sentences[2].start_ms + 31_000
    _, problems, _ = pick.check(debate, _with(0, start_sentence=2, end_sentence=2))
    assert any("lasts 31.0 s" in p for p in problems), problems


def test_hard_problems_are_sent_back_without_the_soft_ones():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."
    client = _FakeClaude({"moments": GOOD["moments"][:3]}, GOOD, GOOD)
    asyncio.run(pick.pick_moments(debate, client=client))
    first_send_back = client.calls[1]["messages"][2]["content"]
    assert "exactly 4" in first_send_back and 'opens on "but"' not in first_send_back


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
    debate.sentences[3].text = "But it is a federal problem."
    accepted, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and accepted is not None
    assert any('opens on "but"' in p for p in soft), soft
    # Asked again while attempts remain; the last answer is kept with it.
    client = _FakeClaude(GOOD, GOOD, GOOD)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 3 and len(client.calls) == 3
    assert 'opens on "but"' in client.calls[1]["messages"][2]["content"]


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

    for spec, attempts, concurrency in (
        (DEBATES_PICK_MOMENTS_SPEC, pick.CALLS, 4),
        (TRANSCRIPT_PUNCTUATE_SPEC, punct.ATTEMPTS, 1),
    ):
        assert spec.rate_limit_key == "claude_global" and spec.concurrency == concurrency
        # One rate unit per Claude call the run can make.
        assert spec.rate_limit_units == attempts
    # Punctuate has time for every call; pick is held to its deadline (tests/test_pick_deadline.py).
    assert TRANSCRIPT_PUNCTUATE_SPEC.execution_timeout.total_seconds() >= punct.ATTEMPTS * REQUEST_TIMEOUT_S


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
    # Both calls answer under their JSON schema: the reply is the JSON alone.
    assert client.calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": pick.REELS_SCHEMA}}
    assert client.calls[1]["output_config"] == {"format": {"type": "json_schema", "schema": pick.JUDGE_SCHEMA}}


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
    debate.sentences[6].text = "Because it is a state matter."  # GOOD's third clip opens on "because"
    clean = json.loads(json.dumps(OTHER))
    clean["moments"][3]["start_sentence"] = 7  # starts past it
    client = _FakeClaude({"reels": [GOOD, clean]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1] and len(client.calls) == 1


def test_a_clip_opening_on_because_is_a_soft_problem():
    debate = _debate()
    debate.sentences[3].text = "Because it is a federal problem."
    _, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and any('opens on "because"' in p for p in soft), soft


def _clips_of(debate, seconds):
    """Re-time every sentence so each GOOD clip (three sentences) runs `seconds`."""
    sentence_ms = int(seconds * 1000 / 3)
    for k, sentence in enumerate(debate.sentences):
        sentence.start_ms, sentence.end_ms = k * sentence_ms, (k + 1) * sentence_ms


def test_a_reel_over_four_clips_at_the_target_is_asked_to_shorten():
    debate = _debate()
    debate.clip_max_ms = 30_000
    _clips_of(debate, 18.3)  # 73.2 s, over 4 x 18 s
    _, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and any("keep the four clips under 72.0 s" in p for p in soft), soft
    # Four clips inside the 10-18 s target are never told to shorten: 70.8 s.
    _clips_of(debate, 17.7)
    assert pick.check(debate, GOOD)[2] == []
    # The limit follows the target the caller sends.
    debate.target_max_ms = 17_000
    assert any("under 68.0 s" in p for p in pick.check(debate, GOOD)[2])


def test_reels_that_pass_with_soft_problems_are_sent_back_however_many_pass():
    debate = _debate()
    debate.sentences[6].text = "Because it is a state matter."  # both reels have a later clip opening on it
    passed, problems, soft = pick.check_reels(debate, {"reels": [GOOD, OTHER]})
    assert problems == [] and len(passed) == 2
    # 12 Nov: reels that passed together were never asked again, so later clips kept "But…" openers.
    assert any(p.startswith('Reel 0: Clip 2 opens on "because"') for p in soft), soft
    assert any(p.startswith('Reel 1: Clip 3 opens on "because"') for p in soft), soft
    # Never fixed: the last answer's reels are judged, as they were.
    client = _FakeClaude({"reels": [GOOD, OTHER]}, {"reels": [GOOD, OTHER]}, {"reels": [GOOD, OTHER]}, {"best": 1, "why": "x"})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert (result.attempts, result.candidates, len(client.calls)) == (3, 2, 4)
    # One passing reel with a soft problem is sent back, with the other reels' problems too.
    _, _, soft = pick.check_reels(debate, {"reels": [GOOD, BROKEN]})
    assert any(p.startswith('Reel 0: Clip 2 opens on "because"') for p in soft), soft
    assert any(p.startswith("Reel 1: ") and "exactly 4" in p for p in soft), soft


def test_a_story_calls_the_speakers_by_short_name():
    (story, _), _, _ = pick.check(_debate(), {**GOOD, "story": "Speaker 1 says states; Speaker 2 disagrees."})
    assert story == "Dahmmy says states; Nico disagrees."
    debate = _debate()
    debate.speakers[0].name = "Dahmmy Hilfigger (@nsddahmmy)"
    debate.speakers[1].name = "Dr. Nico Lagan"
    assert pick.named(debate, "Speaker 1, Speaker 2 and Speaker 3") == "Dahmmy, Nico and Speaker 3"


def test_the_story_is_written_with_speaker_numbers_for_the_names_to_be_filled_in():
    from src.config.prompts.debates_pick_moments_prompt import DEBATES_PICK_MOMENTS_SYSTEM

    # named() picks names that tell the speakers apart; a name Claude wrote itself could not.
    assert 'call each speaker "Speaker N" with their number as given above' in DEBATES_PICK_MOMENTS_SYSTEM
    assert "first name" not in DEBATES_PICK_MOMENTS_SYSTEM
    # Not slots 1 and 2 by name: the input takes any two, and a story told "Speaker 2" for a
    # speaker numbered 0 would misname them.
    assert '"Speaker 1" and "Speaker 2"' not in DEBATES_PICK_MOMENTS_SYSTEM


def test_names_are_filled_in_for_whatever_numbers_the_speakers_have():
    data = _debate().model_dump()
    swap = {1: 3, 2: 7}
    for speaker in data["speakers"]:
        speaker["slot"] = swap[speaker["slot"]]
    for row in data["sentences"] + data["claims"]:
        row["speaker"] = swap[row["speaker"]]
    debate = DebatesPickMomentsInput(**data)
    assert "Speaker 7 (Nico): Agree with the motion" in pick.render(debate)
    assert pick.named(debate, "Speaker 3 opens; Speaker 7 answers.") == "Dahmmy opens; Nico answers."


def test_speakers_who_share_a_first_name_keep_their_full_names_in_the_story_and_for_the_judge():
    debate = _debate()
    debate.speakers[0].name, debate.speakers[1].name = "Alex Chen", "Alex Park"
    assert pick.named(debate, "Speaker 1 v Speaker 2") == "Alex Chen v Alex Park"
    judged = pick.render_for_judge(debate, [pick.check(debate, GOOD)[0]])
    assert "Clip 1, Alex Chen," in judged and "Clip 2, Alex Park," in judged


def test_the_judge_reads_each_clip_as_the_viewer_hears_it():
    # geo-chat cuts the filler before the clip plays and capitalises what is left (without_lead),
    # so the judge reads "Voter ID…", as the opener rules do (heard_start).
    debate = _debate()
    debate.sentences[0].text = "Um, so voter ID is a state matter."
    debate.sentences[3].text = "Uh. And federal rules win."
    judged = pick.render_for_judge(debate, [pick.check(debate, GOOD)[0]])
    assert "Clip 1, Dahmmy, 14.8 s: Voter ID is a state matter. Point 0.1. Point 0.2." in judged
    assert "Clip 2, Nico, 14.8 s: Federal rules win. Point 1.1. Point 1.2." in judged
    # Filler geo-chat keeps, to stay over the minimum length, is read as said.
    debate.clip_min_ms = debate.target_min_ms = 14_000
    judged = pick.render_for_judge(debate, [pick.check(debate, GOOD)[0]])
    assert "Clip 1, Dahmmy, 14.8 s: Um, so voter ID is a state matter. Point 0.1." in judged


def test_the_first_clip_is_chosen_for_the_context_not_for_strength():
    from src.config.prompts.debates_pick_moments_prompt import (
        DEBATES_PICK_MOMENTS_JUDGE_SYSTEM as judge,
        DEBATES_PICK_MOMENTS_SYSTEM as picker,
    )

    # 11 Nov, Yaniv: the first clip establishes the debate's context; it isn't picked for being
    # the strongest line, and the strongest moment can come last. No hook anywhere: master's
    # "Clip 1 is the hook" and "does the first clip hook?" pushed reels to open big.
    assert "Clip 1 must provide the context" in picker
    assert "not for being the strongest line" in picker
    assert "The strongest moment doesn't have to open the reel; it often lands best at the end." in picker
    assert "Clip 4 lands the debate: often its strongest moment" in picker
    for prompt in (picker, judge):
        assert "hook" not in prompt.lower()
        assert "grab" not in prompt.lower()

    # The judge decides "in this order", so the context is part of the first test it applies:
    # ranked lower, a reel without it could win on the exchange before the context was weighed.
    follow, exchange = (judge.index(f"\n{n}. ") for n in (1, 2))
    assert follow < judge.index("The first clip provides the context") < exchange
    assert follow < judge.index("Weigh this before anything else") < exchange
    assert "don't favor one for opening on its strongest line" in judge



# --- answers cut off at the token limit (11 Nov: 8 of 20 pick runs) ---------------------


def test_the_reel_schemas_keep_to_what_structured_outputs_accepts():
    # Every object closed and every property required; no numeric or length constraints.
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for key in ("minimum", "maximum", "minLength", "maxLength", "maxItems", "minItems"):
                assert key not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(pick.REELS_SCHEMA)
    walk(pick.JUDGE_SCHEMA)
    # The prompt's example clip has exactly the schema's fields.
    assert set(_moment(1, 0, 2, 0)) == set(pick.REELS_SCHEMA["properties"]["reels"]["items"]["properties"]["moments"]["items"]["properties"])


def test_a_cut_off_reel_answer_is_asked_again_once_shorter_and_the_answer_kept():
    from src.pipeline.claude_json import CUT_OFF_MESSAGE

    prose = "Let me read the transcript closely. " * 50
    client = _FakeClaude(_CutOff(prose), {"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert result.attempts == 2 and len(client.calls) == 2
    conversation = client.calls[1]["messages"]
    assert conversation[1] == {"role": "assistant", "content": prose}
    assert conversation[2] == {"role": "user", "content": CUT_OFF_MESSAGE}
    assert "shorter" in CUT_OFF_MESSAGE and "nothing else" in CUT_OFF_MESSAGE


def test_a_reel_answer_cut_off_twice_fails_without_an_engine_retry():
    client = _FakeClaude(_CutOff("a"), _CutOff("b"), {"reels": [GOOD]})
    with pytest.raises(FinalDecisionError, match="cut off at the token limit"):
        asyncio.run(pick.pick_moments(_debate(), client=client))
    # One send-back for a cut-off, not one per attempt left.
    assert len(client.calls) == 2


def test_a_cut_off_after_an_acceptable_answer_keeps_that_answer():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."  # a soft problem: asked again
    client = _FakeClaude({"reels": [GOOD]}, _CutOff("..."), {"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 1 and len(client.calls) == 2


def test_a_cut_off_answer_logs_its_tail_not_the_whole_answer(monkeypatch):
    from src.pipeline import claude_json

    lines = []
    monkeypatch.setattr(claude_json.logger, "warning", lines.append)
    head = "TRANSCRIPT-HEAD " + "x" * 3000
    client = _FakeClaude(_CutOff(head + "THE TAIL"), {"reels": [GOOD]})
    asyncio.run(pick.pick_moments(_debate(), client=client))
    cut = [line for line in lines if "cut off" in line]
    assert len(cut) == 1 and "THE TAIL" in cut[0] and "TRANSCRIPT-HEAD" not in cut[0]
    assert len(cut[0]) < claude_json.CUT_OFF_TAIL_CHARS + 200


def test_a_cut_off_judge_answer_is_asked_again_too():
    client = _FakeClaude({"reels": [GOOD, OTHER]}, _CutOff("I think"), {"best": 1, "why": "x"})
    result = asyncio.run(pick.pick_moments(_debate(), client=client))
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1] and len(client.calls) == 3


# --- the first clip's opening (11 Nov: 5 of 12 openers weak) -----------------------------


@pytest.mark.parametrize(
    "text, problem",
    [
        # The 11 Nov worst case: Whisper had "…models, and not just get completely overwhelmed…".
        ("and not just get completely overwhelmed by a society.", "starts mid-sentence"),
        ("not just get completely overwhelmed by a society.", "starts mid-sentence"),
        # Filler geo-chat cuts is skipped first; what plays after it is what counts.
        ("And so but it is a state matter.", 'opens on "but"'),
        ("Um, they want you to believe it.", 'opens on "they"'),
        # Too short to cut: geo-chat keeps three words of the sentence, so it plays as said.
        ("So what now?", 'opens on "so"'),
        ("Um. But it is a state matter.", 'opens on "but"'),
        ("But it is a state matter.", 'opens on "but"'),
        ("Because it is a state matter.", 'opens on "because"'),
        ("That's why it is a state matter.", 'opens on "that\'s"'),
        ("Yeah, it is a state matter.", 'opens on "yeah"'),
        ("Okay, it is a state matter.", 'opens on "okay"'),
        ("I mean, it is a state matter.", 'opens on "i mean"'),
        ("Like, it is a state matter.", 'opens on "like"'),
        ("You know, right?", 'opens on "you know"'),
    ],
)
def test_a_first_clip_opening_mid_sentence_or_on_a_connective_is_refused(text, problem):
    debate = _debate()
    debate.sentences[0].text = text
    value, problems, _ = pick.check(debate, GOOD)
    assert any(p.startswith("Clip 0, the reel's first clip, opens on sentence 0") and problem in p for p in problems), problems
    # Refused, but usable as a last resort: its only problems are the first clip's.
    assert value is not None
    # The same sentence opening a later clip is at most a soft problem.
    value, problems, _ = pick.check(debate, OTHER)
    assert problems == [] and value is not None


@pytest.mark.parametrize(
    "text",
    [
        "Voter ID is a state matter.",
        "50 states already run their own rolls.",
        "iPhone voters prove it.",
        "You know nothing about the rolls.",
        "I meant every word of it.",
        "This is the number that matters.",
        # geo-chat cuts the filler first (story_moments::filler_lead): the viewer hears "We need rules."
        "Um, we need rules.",
        "And we need rules.",
        "So, voter ID is a state matter.",
        "Uh, you know, voter ID is a state matter.",
        "Um. Voter ID is a state matter.",
        "Émilie Durand ran the rolls in Ohio.",
    ],
)
def test_a_first_clip_opening_on_a_sentence_that_starts_a_thought_is_accepted(text):
    debate = _debate()
    debate.sentences[0].text = text
    value, problems, soft = pick.check(debate, GOOD)
    assert (problems, soft) == ([], []) and value is not None


def test_a_lowercase_opener_says_nothing_in_a_transcript_without_capitals():
    debate = _debate()
    for sentence in debate.sentences:
        sentence.text = sentence.text.lower()
    debate.sentences[0].text = "voter id is a state matter."
    assert pick.check(debate, GOOD)[1] == []


def _past_sentence_0(reel):
    """`reel` with its clip from turn 0 starting at sentence 1."""
    reel = json.loads(json.dumps(reel))
    for moment in reel["moments"]:
        if moment["start_sentence"] == 0:
            moment["start_sentence"] = 1
    return reel


def test_one_candidate_with_a_good_opener_is_taken_without_a_send_back():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."  # GOOD opens on it; OTHER opens on sentence 3
    client = _FakeClaude({"reels": [GOOD, _past_sentence_0(OTHER)]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 1 and result.candidates == 1
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1] and result.first_clip_problems == []


def test_reels_that_break_only_the_first_clip_rules_are_sent_back_then_kept_as_a_last_resort():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."  # GOOD's opener
    debate.sentences[3].text = "but it is federal."  # OTHER's opener, mid-sentence too
    answer = {"reels": [GOOD, OTHER]}
    client = _FakeClaude(answer, answer, answer, {"best": 1, "why": "x"})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    # Sent back twice with the first clips' problems, then the last answer's reels judged.
    assert len(client.calls) == 4 and result.attempts == 3 and result.candidates == 2
    assert 'Reel 0: Clip 0, the reel\'s first clip, opens on sentence 0, which opens on "but"' in client.calls[1]["messages"][2]["content"]
    assert [m.speaker for m in result.moments] == [2, 1, 2, 1]
    # "but it is federal." is one flaw: mid-sentence, not also its "but".
    assert len(result.first_clip_problems) == 1 and "starts mid-sentence" in result.first_clip_problems[0]


def test_a_reel_fixed_after_a_first_clip_send_back_wins_with_nothing_to_report():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."
    client = _FakeClaude({"reels": [GOOD]}, {"reels": [_past_sentence_0(OTHER)]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 2 and result.first_clip_problems == []


def test_a_reel_with_other_hard_problems_is_no_last_resort():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."
    broken = {"moments": GOOD["moments"][:3]}  # opens on "but", and only three clips
    value, problems, _ = pick.check(debate, broken)
    assert value is None and any("exactly 4" in p for p in problems)
    with pytest.raises(FinalDecisionError):
        asyncio.run(pick.pick_moments(debate, client=_FakeClaude(broken, broken, broken)))


@pytest.mark.parametrize(
    "text, opener",
    [
        ("It's time we stopped pretending states can run federal elections.", "it's"),
        ("Yes, voter ID should be federal.", "yes"),
    ],
)
def test_a_first_clip_opening_on_its_or_yes_is_a_soft_problem_not_refused(text, opener):
    debate = _debate()
    debate.sentences[0].text = text
    value, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and value is not None
    assert any(p.startswith(f'Clip 0 opens on "{opener}"') for p in soft), soft


def test_a_trim_steps_past_a_mid_sentence_start_when_a_clean_start_fits():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 55 s and no end fits. Dropping sentence 0 lands on a run-on, so the trim steps on to sentence 2.
    _long_first_turn(debate, [31, 12, 12], ["One.", "not just get completely overwhelmed.", "We need rules."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert len(problems) == 1 and moments[0].start_sentence == 2


def _status_error(status):
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        f"HTTP {status}", response=httpx.Response(status, request=request), body=None
    )


def test_a_request_claude_rejects_is_final_not_rerun():
    # A schema structured outputs cannot compile is a 400 on every run alike.
    client = _FakeClaude(_status_error(400))
    with pytest.raises(FinalDecisionError, match="rejected the request"):
        asyncio.run(pick.pick_moments(_debate(), client=client))
    assert len(client.calls) == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
def test_a_rate_limit_or_overload_stays_retryable(status):
    client = _FakeClaude(_status_error(status))
    with pytest.raises(DecisionError) as raised:
        asyncio.run(pick.pick_moments(_debate(), client=client))
    assert raised.value.retryable and not isinstance(raised.value, FinalDecisionError)


def test_a_cut_off_after_a_last_resort_still_takes_its_retry():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."  # GOOD's opener: a last resort
    client = _FakeClaude({"reels": [GOOD]}, _CutOff("..."), {"reels": [_past_sentence_0(OTHER)]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 3 and result.attempts == 3 and result.first_clip_problems == []


def test_a_cut_off_with_no_retry_left_keeps_the_last_resort():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."
    client = _FakeClaude({"reels": [GOOD]}, _CutOff("a"), _CutOff("b"))
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 3 and result.attempts == 1 and result.first_clip_problems


# --- round 2: trims, numbers, and answers held when a later call cannot be made ------------


def test_a_trim_with_no_clean_start_takes_a_soft_opener_over_one_clip_1_refuses():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 55 s and no end fits. Sentence 1 runs on and opens on "and"; sentence 2 opens on "It's", soft on clip 1.
    _long_first_turn(debate, [31, 12, 12], ["One.", "and not just get completely overwhelmed.", "It's time we set rules."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert len(problems) == 1 and moments[0].start_sentence == 2 and moments[0].trimmed_from_sentence == 0


@pytest.mark.parametrize("text", ["1960s America had no voter ID laws.", "10x more people vote by mail now.", "2nd, the rolls are clean."])
def test_a_first_clip_opening_on_a_number_with_a_suffix_is_not_mid_sentence(text):
    debate = _debate()
    debate.sentences[0].text = text
    value, problems, _ = pick.check(debate, GOOD)
    assert problems == [] and value is not None


def test_sentences_opening_on_numbers_do_not_make_a_transcript_look_uncased():
    debate = _debate()
    for sentence in debate.sentences[:7]:
        sentence.text = "50 states agree."
    assert pick.cased(debate)


def _guard_trips_after(monkeypatch, calls):
    from src.pipeline import claude_json
    from src.infrastructure.spend_guard import SpendLimitExceeded

    made = []

    def check_and_record(key):
        if len(made) >= calls:
            raise SpendLimitExceeded("hourly budget spent")
        made.append(key)

    monkeypatch.setattr(claude_json.spend_guard, "check_and_record", check_and_record)


def _last_resort_debate():
    debate = _debate()
    debate.sentences[0].text = "But it is a state matter."  # GOOD's opener: a last resort
    return debate


def test_a_spend_guard_trip_on_a_send_back_keeps_the_last_resort(monkeypatch):
    _guard_trips_after(monkeypatch, 1)
    client = _FakeClaude({"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(_last_resort_debate(), client=client))
    assert len(client.calls) == 1 and result.attempts == 1 and result.first_clip_problems


def test_a_spend_guard_trip_before_any_answer_is_still_final(monkeypatch):
    _guard_trips_after(monkeypatch, 0)
    with pytest.raises(FinalDecisionError, match="spend guard"):
        asyncio.run(pick.pick_moments(_debate(), client=_FakeClaude({"reels": [GOOD]})))


@pytest.mark.parametrize("failure", [RuntimeError("overloaded"), _status_error(500), _status_error(400)])
def test_a_send_back_whose_request_fails_keeps_the_last_resort(failure):
    client = _FakeClaude({"reels": [GOOD]}, failure)
    result = asyncio.run(pick.pick_moments(_last_resort_debate(), client=client))
    assert len(client.calls) == 2 and result.attempts == 1 and result.first_clip_problems


def test_a_send_back_whose_request_fails_keeps_an_acceptable_answer():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."  # a soft problem: asked again
    client = _FakeClaude({"reels": [GOOD]}, RuntimeError("overloaded"))
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 2 and result.attempts == 1 and result.first_clip_problems == []


def test_each_answer_logs_its_token_counts(monkeypatch):
    from src.pipeline import claude_json

    lines = []
    monkeypatch.setattr(claude_json.logger, "info", lines.append)
    usage = SimpleNamespace(input_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=None, output_tokens=40)
    message = SimpleNamespace(content=[], usage=usage)
    claude_json._log_usage("t", 2, message)
    assert lines == ["t: answer 2 used input_tokens 10, cache_read_input_tokens 900, output_tokens 40"]


# --- round 3: clip openers as the viewer hears them, past the filler geo-chat cuts ----------


def test_the_filler_list_is_geo_chats():
    # A copy of geo-chat crates/media/src/clip_ranges.rs (FILLER, FILLER_PAIRS, KNOW_OBJECT_NEXT,
    # SO_ADVERB_NEXT) and story_moments.rs (FILLER_KEEP_WORDS). When geo-chat changes them, change
    # the copy and this test together.
    assert pick.FILLER == ("um", "uh", "er", "ah", "hmm", "so", "and")
    assert pick.FILLER_PAIRS == (("you", "know"),)
    assert pick.KNOW_OBJECT_NEXT == (
        "nothing", "what", "how", "why", "who", "where", "when", "everything", "anything", "something",
        "more", "better", "exactly",
    )
    assert pick.SO_ADVERB_NEXT == ("far", "much", "many", "long", "few", "little", "often", "called", "bad", "good")
    assert pick.FILLER_KEEP_WORDS == 3
    # One copy: the opener list takes the filler from it, so kept filler still counts as an opener.
    assert pick.DANGLING_OPENERS[: len(pick.FILLER)] == pick.FILLER


def _heard(*sentences):
    """Sentences `sentences` as one clip's turn: where the viewer starts hearing it, as words."""
    debate = _debate()
    debate.clip_min_ms = 1_000  # long enough that geo-chat cuts the filler (see the minimum test below)
    for k, text in enumerate(sentences):
        debate.sentences[k].text = text
    k, words = pick.heard_start(debate, 0, len(sentences) - 1)
    return k, " ".join(words)


def test_a_clip_is_heard_from_where_geo_chat_cuts_its_filler():
    # geo-chat's own cases (story_moments.rs, a_clip_starts_past_its_filler_but_keeps_a_short_sentence_whole).
    assert _heard("So, people embrace this.") == (0, "people embrace this.")
    assert _heard("And so the point stands.") == (0, "the point stands.")
    assert _heard("Um, so the point stands.") == (0, "the point stands.")
    assert _heard("You know, the point stands.") == (0, "the point stands.")
    assert _heard("So what is it?") == (0, "what is it?")
    assert _heard("So what now?") == (0, "So what now?")
    assert _heard("So far the data agrees.") == (0, "So far the data agrees.")
    assert _heard("You know nothing about it.") == (0, "You know nothing about it.")
    assert _heard("But for that law we'd lose.") == (0, "But for that law we'd lose.")
    assert _heard("Soon we will see.") == (0, "Soon we will see.")
    # A comma ends the phrase (clip_ranges::runs_on): "So, many…" and "You know, what…" are filler.
    assert _heard("So, many people disagree.") == (0, "many people disagree.")
    assert _heard("You know, what matters is turnout.") == (0, "what matters is turnout.")
    # A_first_sentence_of_nothing_but_filler_is_dropped_whole.
    assert _heard("Um.", "The data shows it.") == (1, "The data shows it.")
    assert _heard("Uh.", "Um, so the data shows it.") == (1, "the data shows it.")
    assert _heard("Um.", "So what now?") == (1, "So what now?")


def test_filler_stays_when_cutting_it_would_leave_the_clip_under_the_minimum():
    # geo-chat's story_reel keeps the filler when the clip would run under CLIP_MIN_MS without it,
    # so the viewer hears "Um." and the first clip is refused on it.
    debate = _debate()
    timed = [(0, 1_000, "Um."), (1_100, 3_500, "We need"), (3_600, 6_500, "rules now.")]  # 6.5 s, 5.4 s past "Um."
    for k, (start_ms, end_ms, text) in enumerate(timed):
        debate.sentences[k].start_ms, debate.sentences[k].end_ms, debate.sentences[k].text = start_ms, end_ms, text
    assert pick.heard_start(debate, 0, 2) == (0, ["Um."])
    assert any('opens on "um"' in p for p in pick.check(debate, GOOD)[1])
    # Long enough without it: cut, as before.
    debate.sentences[2].end_ms = 8_000
    assert pick.heard_start(debate, 0, 2) == (1, ["We", "need"])
    # The line itself is geo-chat's "< CLIP_MIN_MS": exactly 6.0 s past "Um." cuts it, a millisecond less keeps it.
    debate.sentences[2].end_ms = 7_100
    assert pick.heard_start(debate, 0, 2) == (1, ["We", "need"])
    debate.sentences[2].end_ms = 7_099
    assert pick.heard_start(debate, 0, 2) == (0, ["Um."])
    # Filler inside a sentence is timed as its share of the words: "And" is a quarter of a 6.4 s
    # sentence, 1.6 s, leaving 4.8 s, so it stays; at 9.6 s it would leave 7.2 s and is cut.
    debate = _debate()
    debate.sentences[0].start_ms, debate.sentences[0].end_ms, debate.sentences[0].text = 0, 6_400, "And we need rules."
    assert pick.heard_start(debate, 0, 0) == (0, ["And", "we", "need", "rules."])
    debate.sentences[0].end_ms = 9_600
    assert pick.heard_start(debate, 0, 0) == (0, ["we", "need", "rules."])


def test_a_filler_sentence_before_a_run_on_still_refuses_the_first_clip():
    debate = _debate()
    debate.sentences[0].text, debate.sentences[1].text = "Um.", "but it is a state matter."
    _, problems, _ = pick.check(debate, GOOD)
    assert len(problems) == 1 and "starts mid-sentence" in problems[0], problems


def test_a_first_clip_past_its_filler_keeps_the_soft_rule_for_what_is_heard():
    debate = _debate()
    debate.sentences[0].text = "Um, it's time we stopped pretending states can run federal elections."
    value, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and value is not None
    assert any(p.startswith('Clip 0 opens on "it\'s"') for p in soft), soft


def test_a_later_clip_is_nudged_on_what_plays_after_its_filler():
    debate = _debate()
    debate.sentences[3].text = "And federal rules win."  # plays as "Federal rules win."
    assert pick.check(debate, GOOD)[1:] == ([], [])
    debate.sentences[3].text = "And they want it federal."  # plays as "They want it federal."
    _, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and any(p.startswith('Clip 1 opens on "they"') for p in soft), soft


def test_a_trim_reads_a_start_past_its_filler_as_clip_1_does():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    # 0-2 is 55 s and no end fits. Sentence 1 plays as "We need rules.", which clip 1 keeps: the trim takes it.
    _long_first_turn(debate, [31, 12, 12], ["One.", "Um, we need rules.", "It holds."])
    (_, moments), problems, soft = pick.check(debate, GOOD)
    assert len(problems) == 1 and moments[0].start_sentence == 1


def test_an_accented_capital_opens_a_sentence_in_a_cased_transcript():
    debate = _debate()
    for sentence in debate.sentences:
        sentence.text = "Élan matters here."
    assert pick.cased(debate)
    debate.sentences[0].text = "élan matters here."  # a lowercase accented letter still runs on
    assert any("starts mid-sentence" in p for p in pick.check(debate, GOOD)[1])


def test_a_spend_guard_trip_on_a_send_back_keeps_an_acceptable_answer(monkeypatch):
    _guard_trips_after(monkeypatch, 1)
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."  # a soft problem: asked again
    client = _FakeClaude({"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 1 and result.attempts == 1 and result.first_clip_problems == []


# --- GEO-3691: every clip's opener, split sentences, over-long clips, problems for the judge ----


@pytest.mark.parametrize(
    "text, opener",
    [
        # What the 12 Nov preview opened later clips on.
        ("Though, you know, I've personally been in this scenario before.", "though"),
        ("For example, my state checks every voter's ID.", "for example"),
        ("They're working against themselves, and most of these gals agree.", "they're"),
        ("It needs to be included in the analysis.", "it"),
        ("That argument falls apart at the border.", "that"),
        ("Agree so i think i think it seems we agree as a blanket statement.", "agree"),
        # An answer word with nothing of its own after it.
        ("Yes.", "yes"),
        ("Yeah, but it is a federal problem.", "yeah"),
    ],
)
def test_a_later_clip_opening_on_a_word_it_leans_on_is_sent_back(text, opener):
    debate = _debate()
    debate.sentences[3].text = text  # GOOD's second clip
    value, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and value is not None
    assert soft == [f'Clip 1 opens on "{opener}", which leans on what came before or is filler. Start it on a sentence that opens a thought.']


@pytest.mark.parametrize(
    "text, hard",
    [
        ("Though, you know, voter ID is a state matter.", True),
        ("For example, my state checks every voter's ID.", True),
        ("They're working against the states.", True),
        ("That argument falls apart at the border.", True),
        ("Agree, voter ID is a state matter.", True),
        # Often states a position ("It is the death of relationships…"): soft, as "It's" is.
        ("It is the death of state elections as we know them.", False),
    ],
)
def test_the_new_openers_keep_the_first_clips_rules(text, hard):
    debate = _debate()
    debate.sentences[0].text = text
    value, problems, soft = pick.check(debate, GOOD)
    assert value is not None
    assert bool(problems) == hard and bool(soft) != hard, (problems, soft)


@pytest.mark.parametrize(
    "text",
    [
        # Natural answers to the clip before: kept on a later clip.
        "Yes, voter ID should be federal.",
        "Yeah, I think that the truth has a threshold and a time limit.",
        "Agreed, the rolls should be federal.",
        # Refers to the clip just before by its own words (01a179d7 clip 3).
        "What... What is problematic to me with that thinking is the road to redemption.",
        "What is problematic to me with that thinking is the road to redemption.",
        # A question opens a thought, and so does "To" before a verb or a person.
        "Where do we draw the line on federal rules?",
        "To be fair, states run their rolls well.",
        "To me, this is a federal question.",
    ],
)
def test_an_answer_to_the_clip_before_is_not_held_against_a_later_clip(text):
    debate = _debate()
    debate.sentences[3].text = text
    assert pick.check(debate, GOOD)[1:] == ([], [])


def test_yes_still_cannot_open_the_first_clip_cleanly():
    # The newcomer heard no question: "Yes, …" on clip 1 stays a soft problem, "Yeah, …" a hard one.
    debate = _debate()
    debate.sentences[0].text = "Yes, voter ID should be federal."
    assert pick.check(debate, GOOD)[2]
    debate.sentences[0].text = "Yeah, I think voter ID should be federal."
    assert pick.check(debate, GOOD)[1]


def test_a_later_clip_starting_mid_sentence_is_sent_back():
    # 12 Nov: two later clips opened on a lowercase run-on, which only the first clip was checked for.
    debate = _debate()
    debate.sentences[3].text = "not just get completely overwhelmed by a society."
    value, problems, soft = pick.check(debate, GOOD)
    assert problems == [] and value is not None
    assert soft == ["Clip 1 starts mid-sentence (its first sentence opens on a lowercase word). Start it on a sentence that opens a thought."]


def _split_soft(position):
    return (
        f'Clip {position} starts on a sentence marked "second half of a split sentence": the transcript may have '
        "broken a sentence in two there. If it did, start the clip where that sentence starts, or pick another stretch."
    )


@pytest.mark.parametrize(
    "before, text",
    [
        # 12 Nov split errors.
        ("Policy is how we got in this situation in the first place.", "Where corporations argued for sugar in soft drinks."),
        # A sentence the transcript ended mid-phrase: the next one is its second half.
        ("I think legal immigration is a.", "National security risk for everyone."),
        ("We should ask whether.", "States can run their own rolls."),
        # A one-word sentence is all hanging, capital or not.
        ("Because.", "States can run their own rolls."),
    ],
)
def test_a_sentence_the_transcript_split_off_the_one_before_is_marked_and_sent_back_on_any_clip(before, text):
    debate = _debate()
    debate.sentences[3].text, debate.sentences[4].text = before, text  # turn 1
    assert pick.continues(debate, 4) and not pick.continues(debate, 3)
    assert f"[4] 20.0-24.8 s (4.8 s, second half of a split sentence): {text}" in pick.render(debate)
    assert "[3] 15.0-19.8 s (4.8 s): " in pick.render(debate)
    # A later clip starting there is sent back, softly.
    value, problems, soft = pick.check(debate, _with(1, start_sentence=4))
    assert problems == [] and value is not None
    assert soft == [_split_soft(1)]
    # So is the first clip: the mark is a guess, so it never refuses a reel.
    debate = _debate()
    for k, sentence in enumerate((before, text)):
        debate.sentences[k].text = sentence
    value, problems, soft = pick.check(debate, _with(0, start_sentence=1))
    assert problems == [] and value is not None
    assert soft == [_split_soft(0)]
    assert pick.first_clip_problems(debate, 1, 2, 1) == []


def test_a_split_half_opening_on_that_is_held_to_the_that_opener_rule_not_the_mark():
    # "And that the border security…" (12 Nov) is marked, but what refuses it as a first clip is
    # its opener "that", a rule of its own; the mark adds no second problem.
    debate = _debate()
    debate.sentences[3].text = "There is a difference between migrants and cartel members."
    debate.sentences[4].text = "And that the border security can serve a useful purpose."
    assert pick.continues(debate, 4)
    value, problems, soft = pick.check(debate, _with(1, start_sentence=4))
    assert problems == [] and soft == [
        'Clip 1 opens on "that", which leans on what came before or is filler. Start it on a sentence that opens a thought.'
    ]
    debate.sentences[0].text, debate.sentences[1].text = debate.sentences[3].text, debate.sentences[4].text
    value, problems, soft = pick.check(debate, _with(0, start_sentence=1))
    assert value is not None and soft == []
    assert len(problems) == 1 and 'which opens on "that", a connective or filler' in problems[0]


@pytest.mark.parametrize(
    "before, text",
    [
        ("Policy got us here.", "Where do we draw the line?"),
        ("Policy got us here.", "That argument falls apart."),
        ("Policy got us here.", "Federal rules win."),
        # No rule for "To …": it opens a sentence of its own as often as not.
        ("Policy got us here.", "To the first point, states run rolls well."),
        ("That is how it feels in my head.", "To the, like, to the idea of a human relationship."),
        # A question is no clause of the sentence before, whatever its second word.
        ("Policy got us here.", "Where in the Constitution does it say that?"),
        ("Policy got us here.", "Where exactly is the line?"),
        # A capital past the first word is a letter or a name, not a hanging article.
        ("We need Plan A.", "Federal rules failed."),
        ("They give the plan an A.", "States run it well."),
        # A question or an exclamation was finished as said, whatever word it ends on.
        ("What if?", "States ran every roll themselves."),
        ("Oh my!", "States ran every roll themselves."),
        # Only within a turn: the other speaker's "the." is not this sentence's first half, and a
        # turn's first sentence is no clause of the other speaker's.
        (None, "Federal rules win."),
        (None, "That the states run their rolls well is plain."),
        (None, "Where corporations argued for sugar, states pushed back."),
    ],
)
def test_a_sentence_that_opens_a_thought_is_not_marked_split(before, text):
    debate = _debate()
    if before is None:
        debate.sentences[2].text = "We need the."  # turn 0, speaker 1
        k = 3  # turn 1's first sentence
    else:
        debate.sentences[3].text = before
        k = 4
    debate.sentences[k].text = text
    assert not pick.continues(debate, k)
    assert "second half of a split sentence" not in pick.render(debate)


def test_a_split_is_found_past_a_sentence_of_nothing_but_filler():
    debate = _debate()
    debate.clip_min_ms = 1_000  # short enough that geo-chat cuts the filler sentence
    debate.sentences[3].text = "I think legal immigration is a."  # turn 1
    debate.sentences[4].text = "Um."
    debate.sentences[5].text = "National security risk for all."
    assert pick.continues(debate, 5)
    assert "[5] 25.0-29.8 s (4.8 s, second half of a split sentence): National security risk for all." in pick.render(debate)
    # A later clip starting on the filler is heard from the half, and sent back.
    value, problems, soft = pick.check(debate, _with(1, start_sentence=4))
    assert problems == [] and value is not None
    assert soft == [_split_soft(1)]
    # A turn of filler before a clause is still a turn start.
    debate.sentences[3].text = "Um."
    debate.sentences[5].text = "That the states run their rolls well is plain."
    assert not pick.continues(debate, 5)


def _split_first_clip(debate):
    """GOOD with its first clip starting on sentence 1, which the transcript marks as a split half."""
    debate.sentences[0].text = "Policy is how we got in this situation in the first place."
    debate.sentences[1].text = "Where corporations argued for sugar in soft drinks."
    return _with(0, start_sentence=1)


def test_a_reel_whose_only_problem_is_a_split_first_clip_is_sent_back_but_still_usable():
    debate = _debate()
    marked = _split_first_clip(debate)
    client = _FakeClaude({"reels": [marked]}, {"reels": [marked]}, {"reels": [marked]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    # Sent back each time with the mark, never refused: the reel is kept, as passing, not as a
    # last resort, so no first-clip problem is reported.
    assert result.attempts == 3 and len(client.calls) == 3
    assert all(_split_soft(0) in call["messages"][-1]["content"] for call in client.calls[1:])
    assert result.moments[0].start_sentence == 1 and result.first_clip_problems == []


def test_a_clean_reel_is_taken_over_one_with_a_split_first_clip():
    debate = _debate()
    marked = _split_first_clip(debate)
    client = _FakeClaude({"reels": [marked, GOOD]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert len(client.calls) == 1 and result.candidates == 1 and result.moments[0].start_sentence == 0
    # And a split first clip beats a first clip that breaks a hard rule.
    debate = _debate()
    marked = _split_first_clip(debate)
    debate.sentences[3].text = "But it is a federal problem."  # OTHER's first clip: hard
    client = _FakeClaude({"reels": [OTHER, marked]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.moments[0].start_sentence == 1 and result.first_clip_problems == []


def test_a_trim_prefers_a_split_start_to_one_the_first_clip_may_not_open_on():
    debate = _debate()
    _long_first_turn(debate, [12, 6, 6], ["Voter ID is a state matter.", "But states differ.", "Where states argued for it."])
    # 0-2 runs 24 s, over the 22 s limit; 1-2 and 2 fit. 1 opens on "but" (hard on the first clip),
    # 2 is a split half (soft): the trim starts at 2.
    assert pick.trim_to_fit(debate, 0, 2) == 2


def test_a_clip_over_the_limit_is_cut_from_its_end_only_where_it_ends_a_whole_sentence():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    texts = ["Voter ID is a state matter.", "Policy got us here.", "Where corporations argued for sugar."]
    # 0-2 is 35.8 s. Ending at 1 would cut a sentence in half (2 continues it); ending at 0 keeps
    # a whole sentence and the setup, so the last resort ends there.
    _long_first_turn(debate, [12, 12, 11.8], texts)
    (_, moments), problems, _ = pick.check(debate, GOOD)
    assert len(problems) == 1 and "over the 30.0 s limit" in problems[0]
    assert (moments[0].start_sentence, moments[0].end_sentence, moments[0].trimmed_to_sentence) == (0, 0, 2)
    # Sentence 0 alone is under the minimum: no clean end fits, so it drops sentence 0, as before.
    _long_first_turn(debate, [5, 14, 11.8], texts)
    (_, moments), _, _ = pick.check(debate, GOOD)
    assert (moments[0].start_sentence, moments[0].end_sentence, moments[0].trimmed_from_sentence) == (1, 2, 0)
    assert moments[0].trimmed_to_sentence is None
    # With no start that fits either, the end is cut anyway: a reel beats none.
    _long_first_turn(debate, [5, 20, 31], texts)
    assert pick.fit_to_limit(debate, 0, 2) == (0, 1)


def test_a_reel_with_only_soft_problems_beats_one_with_a_clip_cut_to_fit():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [12, 12, 11.8])  # GOOD's first clip runs 35.8 s
    debate.sentences[6].text = "Because it is a state matter."  # GOOD's third clip opens on it
    soft = _with(0, end_sentence=1)  # fits; still opens clip 2 on "because"
    # An over-long reel first, then soft ones: the soft one is kept, nothing cut.
    client = _FakeClaude(GOOD, soft, soft)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 3 and all(m.trimmed_to_sentence is None for m in result.moments)
    # A soft one first, then over-long ones: the soft one is still kept.
    client = _FakeClaude(soft, GOOD, GOOD)
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 1 and result.moments[0].end_sentence == 1


def test_a_send_back_that_comes_back_worse_does_not_replace_the_better_answer():
    # Every soft answer is sent back now, so the answer kept is the one whose best reel has the
    # fewest problems, not the latest (review of GEO-3691).
    debate = _debate()
    debate.sentences[6].text = "Because it is a state matter."
    debate.sentences[9].text = "But every president tips it."
    better = _with(2, start_sentence=7)  # only clip 3 opens on "but"
    result = asyncio.run(pick.pick_moments(debate, client=_FakeClaude(better, GOOD, BROKEN)))
    assert result.attempts == 1 and result.moments[2].start_sentence == 7
    # Two reels to judge, then a single worse one: the two still reach the judge.
    other = json.loads(json.dumps(better))
    other["moments"][3].update(start_sentence=10)  # clip 4 at 10-11, clean; clip 3 at 7-8, clean
    other["moments"][1].update(start_sentence=3, end_sentence=4)
    debate.sentences[3].text = "And so but the states differ."  # clip 2 of `other` opens on "but"
    client = _FakeClaude({"reels": [better, other]}, GOOD, BROKEN, {"best": 1, "why": "x"})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 1 and result.candidates == 2 and len(client.calls) == 4
    assert result.moments[3].start_sentence == 10
    # Equal answers: the later one, as before.
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."
    result = asyncio.run(pick.pick_moments(debate, client=_FakeClaude(GOOD, GOOD, GOOD)))
    assert result.attempts == 3


def test_a_last_resort_that_comes_back_worse_does_not_replace_the_better_one():
    debate = _debate()
    debate.sentences[0].text = "but states run it."  # GOOD's clip 1: mid-sentence (fixable)
    debate.sentences[1].text = "But states run it well."  # better's clip 1: on "but" (fixable)
    debate.sentences[9].text = "But every president tips it."  # GOOD's clip 4 also opens on "but" (soft)
    better = _with(0, start_sentence=1)
    better["moments"][3].update(start_sentence=10)
    result = asyncio.run(pick.pick_moments(debate, client=_FakeClaude(better, GOOD, BROKEN)))
    assert result.attempts == 1 and result.moments[0].start_sentence == 1
    assert len(result.first_clip_problems) == 1


def test_a_lowercase_opener_is_one_flaw_so_a_reel_with_two_bad_clips_does_not_win_the_tie():
    # "but we need…" starts mid-sentence and on "but": one flaw, as "But we need…" is. Counted as
    # two, an answer with two bad clips tied it and, being later, was kept (review of GEO-3691).
    debate = _debate()
    debate.sentences[3].text = "but we need federal rules here."
    debate.sentences[6].text = "Because states do it well."
    debate.sentences[9].text = "But every president tips it."
    one_bad = json.loads(json.dumps(GOOD))
    one_bad["moments"][2].update(start_sentence=7)
    one_bad["moments"][3].update(start_sentence=10)
    two_bad = _with(1, start_sentence=4)
    _, _, soft = pick.check(debate, one_bad)
    assert len(soft) == 1 and soft[0].startswith("Clip 1 starts mid-sentence"), soft
    result = asyncio.run(pick.pick_moments(debate, client=_FakeClaude(one_bad, two_bad, BROKEN)))
    assert result.attempts == 1 and result.moments[1].start_sentence == 3
    # The judge reads one line for the clip, and the first clip one fixable problem.
    judged = pick.render_for_judge(debate, [pick.check(debate, one_bad)[0]])
    assert "- Clip 2 starts mid-sentence." in judged and 'Clip 2 opens on "but"' not in judged
    debate.sentences[0].text = "but states run it."
    assert len(pick.first_clip_problems(debate, 0, 2, 0)) == 1


def test_the_last_resort_log_names_the_problems_of_the_answer_it_keeps(monkeypatch):
    # Ranked, the last resort kept can be an earlier answer than the last: the log gives its own
    # problems, not the last answer's (review of GEO-3691).
    from src.pipeline import claude_json

    lines = []
    monkeypatch.setattr(claude_json.logger, "warning", lines.append)
    debate = _debate()
    debate.sentences[0].text = "but states run it."
    debate.sentences[1].text = "But states run it well."
    debate.sentences[9].text = "But every president tips it."
    better = _with(0, start_sentence=1)
    better["moments"][3].update(start_sentence=10)
    asyncio.run(pick.pick_moments(debate, client=_FakeClaude(better, GOOD, BROKEN)))
    kept = [line for line in lines if "as a last resort" in line]
    assert len(kept) == 1 and "keeping answer 1" in kept[0]
    assert 'opens on sentence 1, which opens on "but"' in kept[0] and "exactly 4" not in kept[0]


def test_a_reel_whose_only_problems_are_later_openers_is_kept_when_never_fixed():
    debate = _debate()
    debate.sentences[3].text = "But it is a federal problem."
    client = _FakeClaude({"reels": [GOOD]}, {"reels": [GOOD]}, {"reels": [GOOD]})
    result = asyncio.run(pick.pick_moments(debate, client=client))
    assert result.attempts == 3 and len(result.moments) == 4 and result.first_clip_problems == []
    # Each send-back carried the opener.
    assert all('Clip 1 opens on "but"' in call["messages"][-1]["content"] for call in client.calls[1:])


def test_the_judge_sees_each_reels_problems_by_clip_as_the_viewer_hears_them():
    debate = _debate()
    debate.sentences[6].text = "Because it is a state matter."  # GOOD's clip 3, OTHER's clip 4
    debate.sentences[9].text = "not just get completely overwhelmed."  # OTHER's clip 3
    reels = [pick.check(debate, GOOD)[0], pick.check(debate, OTHER)[0]]
    judged = pick.render_for_judge(debate, reels)
    good, other = judged.split("## Reel 1")
    assert 'Problems found:\n- Clip 3 opens on "because", which leans on something the viewer did not hear or is filler.' in good
    assert "- Clip 3 starts mid-sentence." in other and '- Clip 4 opens on "because"' in other
    # Still the viewer's view: no claims, no story, no sentence numbers.
    assert "Federal elections need" not in judged and "States suffice" not in judged
    assert not re.search(r"sentences? \d", judged)
    # A clean reel has nothing listed.
    assert "Problems found" not in pick.render_for_judge(_debate(), [pick.check(_debate(), GOOD)[0]])


def test_the_judge_sees_a_clip_cut_to_fit_and_a_reel_over_its_length():
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [12, 12, 11.8])
    judged = pick.render_for_judge(debate, [pick.check(debate, GOOD)[0]])
    assert "- Clip 1 its ending was cut to fit the length limit, so it may stop before its point lands." in judged
    debate = _debate()
    debate.clip_max_ms = 30_000
    _clips_of(debate, 18.3)  # 73.2 s, over 4 x 18 s
    assert "- The reel runs over 72.0 s." in pick.render_for_judge(debate, [pick.check(debate, GOOD)[0]])


def test_flawed_reels_reach_the_judge_with_their_problems_when_none_is_clean():
    debate = _debate()
    debate.sentences[6].text = "Because it is a state matter."  # in both reels
    answer = {"reels": [GOOD, OTHER]}
    client = _FakeClaude(answer, answer, answer, {"best": 0, "why": "x"})
    asyncio.run(pick.pick_moments(debate, client=client))
    assert "Problems found:" in client.calls[3]["messages"][0]["content"][0]["text"]


def test_the_prompts_tell_of_the_split_mark_and_the_problems_listed():
    from src.config.prompts.debates_pick_moments_prompt import (
        DEBATES_PICK_MOMENTS_JUDGE_SYSTEM as judge,
        DEBATES_PICK_MOMENTS_SYSTEM as picker,
    )

    assert f'marked "{pick.SPLIT_MARK}"' in picker
    assert "A clip over the hard limit is sent back, not cut for you." in picker
    assert '"Problems found"' in judge
    # The picker's example answer opener must itself pass the check.
    assert "Yes, the numbers show it" in picker
    debate = _debate()
    debate.sentences[3].text = "Yes, the numbers show it clearly."
    assert pick.check(debate, GOOD)[2] == []


def test_a_last_resort_with_a_soft_flaw_does_not_replace_a_cleaner_one():
    # Two last resorts with the same clip cut to fit: the one whose clip 2 also opens on "But…"
    # ranks worse, though it came later (review of GEO-3691).
    debate = _debate()
    debate.clip_max_ms = 30_000  # timed against a 30 s limit
    _long_first_turn(debate, [12, 12, 11.8])  # GOOD's first clip runs 35.8 s
    debate.sentences[3].text = "But it is a federal problem."
    cleaner = _with(1, start_sentence=4)  # clip 2 at 4-5, clean
    result = asyncio.run(pick.pick_moments(debate, client=_FakeClaude(cleaner, GOOD, GOOD)))
    assert result.attempts == 1 and result.moments[1].start_sentence == 4
    # The soft problem is weighed, not sent back: the send-backs carry the hard problem alone.
    client = _FakeClaude(GOOD, GOOD, GOOD)
    asyncio.run(pick.pick_moments(debate, client=client))
    assert all('opens on "but"' not in call["messages"][-1]["content"] for call in client.calls[1:])


def test_a_lowercase_split_half_is_one_flaw_not_two():
    debate = _debate()
    debate.sentences[3].text = "Policy got us here."
    debate.sentences[4].text = "where corporations argued for sugar."
    (_, moments), problems, soft = pick.check(debate, _with(1, start_sentence=4))
    assert problems == []
    assert soft == ["Clip 1 starts mid-sentence (its first sentence opens on a lowercase word). Start it on a sentence that opens a thought."]
    judged = pick.render_for_judge(debate, [(None, moments)])
    assert "- Clip 2 starts mid-sentence." in judged and "halfway" not in judged


def test_the_judge_is_told_a_split_start_may_be_one_and_to_check_the_words():
    from src.config.prompts.debates_pick_moments_prompt import DEBATES_PICK_MOMENTS_JUDGE_SYSTEM as judge

    debate = _debate()
    debate.sentences[3].text = "Policy is how we got in this situation in the first place."
    debate.sentences[4].text = "Where corporations argued for sugar in soft drinks."
    reel = pick.check(debate, _with(1, start_sentence=4))[0]
    assert "- Clip 2 may start halfway through a sentence." in pick.render_for_judge(debate, [reel])
    assert '"may" start halfway through a sentence is a guess' in judge
    assert "set it aside when the clip plainly starts a thought" in judge
