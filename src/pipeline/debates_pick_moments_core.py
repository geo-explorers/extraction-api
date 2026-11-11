"""DB-free core of debates.pick_moments (GEO-3638): what Claude reads, the rules
its reels must keep, how the best reel is judged, and how it becomes four moments.

Claude offers up to three candidate reels in one answer. Each is checked on its own
(`check`); the answer is sent back only when none passes the hard rules. When more
than one passes, a second call judges them as a viewer meets them, the clips' words
alone, and its pick is returned (`judge`).

Every rule the prompt states that can be checked mechanically is checked here:
four clips, alternating speakers, whole sentences from one turn, length bounds,
no overlap, each clip linked to a different claim of its own speaker, and an
answer playing after what it answers. A broken answer is sent back with its
problems (claude_json.ask_json), so Claude fixes it rather than the run failing.
A clip over the hard limit is first shortened from its start to fit (trim_to_fit) and
marked trimmed_from_sentence, rather than refused.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.api.schemas.debates_pick_moments_schema import (
    MOMENTS,
    DebatesPickMomentsInput,
    DebatesPickMomentsResult,
    PickedMoment,
)
from src.config.overrides import prompts
from src.decisions.base import DecisionError
from src.infrastructure.logger import get_logger
from src.pipeline.claude_json import PROVIDER, ask_json

logger = get_logger(__name__)

TASK = "debates.pick_moments"
# Answers asked for per run: the first, and two sent back with their problems.
ATTEMPTS = 3
# Judge answers per run, when more than one candidate reel passes.
JUDGE_ATTEMPTS = 2
# Claude calls one run can make: what it reserves of the shared budget, and times out by.
CALLS = ATTEMPTS + JUDGE_ATTEMPTS
# Candidate reels asked for in one answer; more are ignored.
CANDIDATES = 3
# Titles a speaker's short name skips: "Dr. Jane Smith" is Jane.
TITLES = {"dr", "mr", "mrs", "ms", "mx", "prof", "sir"}
ROLES = ("opening", "answer", "rebuttal", "closing")
# A clip opening on one of these leans on something the viewer did not hear, or on filler. The
# prompt forbids it; the check asks again (a soft rule: the last answer is kept regardless).
DANGLING_OPENERS = (
    "and", "but", "so", "also", "because", "yeah", "yes", "okay", "ok", "um", "uh", "like",
    "that's", "they", "it's", "which", "then", "plus", "or",
)


def _seconds(ms: int) -> str:
    return f"{ms / 1000:.1f}"


def reel_ms(moments: List[PickedMoment]) -> int:
    """How long a reel plays: its clips back to back."""
    return sum(m.end_ms - m.start_ms for m in moments)


def reel_max_ms(input: DebatesPickMomentsInput) -> int:
    """The longest a reel should run: every clip at the top of its target (a soft rule). Reels
    past 80 s graded worst on 11 Nov."""
    return MOMENTS * input.target_max_ms


def _side(speaker) -> str:
    return f": {speaker.side} with the motion" if speaker.side else ""


def short_names(input: DebatesPickMomentsInput) -> Dict[int, str]:
    """What a story and the judge call each speaker: the first word of their name past any title,
    or their full name when that would not tell the two apart."""

    def short(name: str) -> str:
        words = [w for w in name.split() if w.lower().rstrip(".") not in TITLES]
        return (words or name.split() or [name])[0]

    shorts = {s.slot: short(s.name) for s in input.speakers}
    if len(set(shorts.values())) < len(shorts):
        return {s.slot: s.name for s in input.speakers}
    return shorts


def render(input: DebatesPickMomentsInput) -> str:
    names = {s.slot: s.name for s in input.speakers}
    lines = [f"Motion: {input.title}", ""]
    lines += [f"Speaker {s.slot} ({s.name}){_side(s)}" for s in input.speakers]
    lines += [
        "",
        f"Target clip length: {_seconds(input.target_min_ms)}-{_seconds(input.target_max_ms)} s. "
        f"Hard limits: {_seconds(input.clip_min_ms)}-{_seconds(input.clip_max_ms)} s. "
        f"The whole reel: under {_seconds(reel_max_ms(input))} s.",
        "",
        "Claims:",
    ]
    lines += [f"[{c.index}] Speaker {c.speaker}: {c.text}" for c in input.claims]
    if input.avoid:
        lines += ["", "Do not use these stretches; an earlier reel used them and they were refused:"]
        lines += [f"- sentences {a.start_sentence}-{a.end_sentence}: {a.reason}" for a in input.avoid]
    lines += ["", "Transcript:"]
    turn = None
    for i, s in enumerate(input.sentences):
        if s.turn != turn:
            turn = s.turn
            lines.append(f"\n## Turn {s.turn}: Speaker {s.speaker} ({names[s.speaker]})")
        lines.append(f"[{i}] {_seconds(s.start_ms)}-{_seconds(s.end_ms)} s ({_seconds(s.end_ms - s.start_ms)} s): {s.text.strip()}")
    return "\n".join(lines)


def _first_word(sentence) -> str:
    """A sentence's first word, lowercased, letters and apostrophes only (curly ones straightened)."""
    return re.sub(r"[^a-z']", "", (sentence.text.split() or [""])[0].lower().replace("’", "'"))


def trim_to_fit(input: DebatesPickMomentsInput, start: int, end: int) -> Optional[int]:
    """Where a clip over the hard limit should start so it fits: sentences dropped from its start
    (the payoff at its end stays), and past any that open on a dangling word, so the trim never
    lands on "And ..." when a cleaner start fits. `None` when no start leaves it within the
    limits, so the clip is reported at the length the model picked. A clip within the limit is
    left where it starts."""
    sentences = input.sentences
    length = lambda first: sentences[end].end_ms - sentences[first].start_ms
    if length(start) <= input.clip_max_ms:
        return start
    fitting = [
        first
        for first in range(start + 1, end + 1)
        if input.clip_min_ms <= length(first) <= input.clip_max_ms
    ]
    if not fitting:
        return None
    clean = [first for first in fitting if _first_word(sentences[first]) not in DANGLING_OPENERS]
    return (clean or fitting)[0]


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def check(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[Tuple[str, List[PickedMoment]]], List[str], List[str]]:
    """The answer as (story, moments), its problems, and its soft problems."""
    if not isinstance(answer, dict) or not isinstance(answer.get("moments"), list):
        return None, ['The answer needs a "moments" list.'], []
    rows = answer["moments"]
    problems: List[str] = []
    soft: List[str] = []
    if len(rows) != MOMENTS:
        problems.append(f"There are {len(rows)} clips; there must be exactly {MOMENTS}.")
    claims = {c.index: c for c in input.claims}
    sentences = input.sentences
    moments: List[PickedMoment] = []
    for position, row in enumerate(rows):
        name = f"Clip {position}"
        if not isinstance(row, dict):
            problems.append(f"{name} is not an object.")
            continue
        speaker, start, end = _int(row.get("speaker")), _int(row.get("start_sentence")), _int(row.get("end_sentence"))
        claim = _int(row.get("claim"))
        if speaker is None or start is None or end is None or claim is None:
            problems.append(f"{name} needs integer speaker, start_sentence, end_sentence and claim.")
            continue
        if not 0 <= start <= end < len(sentences):
            problems.append(f"{name}: sentences {start}-{end} are not a range inside 0-{len(sentences) - 1}.")
            continue
        run = sentences[start : end + 1]
        if {s.speaker for s in run} != {speaker}:
            problems.append(f"{name}: sentences {start}-{end} are not all Speaker {speaker}'s.")
            continue
        if len({s.turn for s in run}) != 1:
            problems.append(f"{name}: sentences {start}-{end} span more than one turn.")
            continue
        # A trim is final: it is recorded on the moment (trimmed_from_sentence), not sent back,
        # since a model content with it would send the same pick and be trimmed again. Nor is
        # the opener of a trimmed clip held against it: the model did not choose that start,
        # and starting earlier only makes the clip too long to keep.
        picked_start = start
        trimmed = trim_to_fit(input, start, end)
        start = start if trimmed is None else trimmed
        run = sentences[start : end + 1]
        first_word = _first_word(run[0])
        if start == picked_start and first_word in DANGLING_OPENERS:
            soft.append(
                f'{name} opens on "{first_word}", which leans on what came before or is filler. '
                "Start it on a sentence that opens a thought."
            )
        start_ms, end_ms = run[0].start_ms, run[-1].end_ms
        length = end_ms - start_ms
        if not input.clip_min_ms <= length <= input.clip_max_ms:
            problems.append(
                f"{name} lasts {_seconds(length)} s; it must be {_seconds(input.clip_min_ms)}-"
                f"{_seconds(input.clip_max_ms)} s (aim for {_seconds(input.target_min_ms)}-{_seconds(input.target_max_ms)} s)."
            )
        linked = claims.get(claim)
        if linked is None:
            problems.append(f"{name}: there is no claim [{claim}].")
        elif linked.speaker != speaker:
            problems.append(f"{name}: claim [{claim}] is Speaker {linked.speaker}'s, not Speaker {speaker}'s.")
        fit = row.get("claim_fit")
        fit = min(1.0, max(0.0, float(fit))) if isinstance(fit, (int, float)) and not isinstance(fit, bool) else 0.5
        role = row.get("role") if row.get("role") in ROLES else ("opening" if position == 0 else "answer")
        answers = _int(row.get("answers"))
        if answers is not None and not 0 <= answers < position:
            problems.append(f"{name} answers clip {answers}, which does not play before it.")
        moments.append(
            PickedMoment(
                speaker=speaker,
                start_sentence=start,
                end_sentence=end,
                start_ms=start_ms,
                end_ms=end_ms,
                claim_index=claim,
                claim_fit=fit,
                role=role,
                answers=answers,
                why=str(row.get("why") or "")[:500],
                trimmed_from_sentence=picked_start if start != picked_start else None,
            )
        )
    if len(moments) == len(rows) == MOMENTS:
        for a, b in zip(moments, moments[1:]):
            if a.speaker == b.speaker:
                problems.append("The clips must alternate speakers: A, B, A, B.")
                break
        for i, a in enumerate(moments):
            for j in range(i + 1, len(moments)):
                b = moments[j]
                if a.start_sentence <= b.end_sentence and b.start_sentence <= a.end_sentence:
                    problems.append(f"Clips {i} and {j} overlap.")
                elif a.speaker == b.speaker and a.claim_index == b.claim_index:
                    problems.append(f"Clips {i} and {j} are the same speaker's and link the same claim.")
        for i, m in enumerate(moments):
            # An out-of-range `answers` is already a problem above; only an earlier clip is read.
            if m.answers is not None and 0 <= m.answers < i and moments[m.answers].speaker == m.speaker:
                problems.append(f"Clip {i} answers its own speaker's clip; it can only answer the other speaker.")
    for i, m in enumerate(moments):
        for a in input.avoid:
            if m.start_sentence <= a.end_sentence and a.start_sentence <= m.end_sentence:
                problems.append(f"Clip {i} uses sentences {a.start_sentence}-{a.end_sentence}, which you were told to avoid ({a.reason}).")
    if not problems and reel_ms(moments) > reel_max_ms(input):
        soft.append(
            f"The reel runs {_seconds(reel_ms(moments))} s; keep the four clips under "
            f"{_seconds(reel_max_ms(input))} s together. Shorten the longest clips."
        )
    story = answer.get("story") if isinstance(answer.get("story"), str) else ""
    return ((named(input, story), moments) if not problems else None), problems, soft


def named(input: DebatesPickMomentsInput, story: str) -> str:
    """`story` with any "Speaker N" left in it replaced by that speaker's short name."""
    names = short_names(input)
    return re.sub(
        r"\bSpeaker (\d+)\b",
        lambda m: names.get(int(m.group(1)), m.group(0)),
        story,
    )


Reel = Tuple[str, List[PickedMoment]]


def check_reels(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[List[Tuple[Reel, List[str]]]], List[str], List[str]]:
    """The candidate reels that keep every hard rule, each with its soft problems; the answer's
    problems when none does; and its soft problems when only one reel passed and it has some.
    With two or more passing, the judge weighs them as they are: sending them back for a soft
    nudge could return fewer.

    A single reel (an answer in the old shape, {"moments": ...}) is one candidate."""
    if isinstance(answer, dict) and isinstance(answer.get("reels"), list):
        reels = answer["reels"][:CANDIDATES]
    elif isinstance(answer, dict) and "moments" in answer:
        reels = [answer]
    else:
        return None, ['The answer needs a "reels" list.'], []
    if not reels:
        return None, ['The "reels" list is empty.'], []
    passed: List[Tuple[Reel, List[str]]] = []
    problems: List[str] = []
    softs: List[str] = []
    for number, reel in enumerate(reels):
        value, hard, soft = check(input, reel)
        label = f"Reel {number}: " if len(reels) > 1 else ""
        if hard:
            problems += [label + p for p in hard]
            continue
        passed.append((value, soft))
        softs += [label + p for p in soft]
    if not passed:
        return None, problems, []
    return passed, [], (softs if len(passed) == 1 else [])


def render_for_judge(input: DebatesPickMomentsInput, reels: List[Reel]) -> str:
    """The reels as a viewer meets them: the motion, who speaks for which side, and each clip's
    words and length. No transcript, no claims, no story: the judge sees what the viewer sees."""
    names = short_names(input)
    lines = [f"Motion: {input.title}", ""]
    lines += [f"{names[s.slot]}{_side(s)}" for s in input.speakers]
    for number, (_, moments) in enumerate(reels):
        lines += ["", f"## Reel {number} ({_seconds(reel_ms(moments))} s)"]
        for position, m in enumerate(moments):
            words = " ".join(s.text.strip() for s in input.sentences[m.start_sentence : m.end_sentence + 1])
            lines.append(f"Clip {position + 1}, {names[m.speaker]}, {_seconds(m.end_ms - m.start_ms)} s: {words}")
    return "\n".join(lines)


def judge_check(count: int):
    def check_pick(answer: Any) -> Tuple[Optional[Tuple[int, str]], List[str], List[str]]:
        best = _int(answer.get("best")) if isinstance(answer, dict) else None
        if best is None or not 0 <= best < count:
            return None, [f'The answer needs "best": a reel number from 0 to {count - 1}.'], []
        why = answer.get("why") if isinstance(answer.get("why"), str) else ""
        return (best, why[:500]), [], []

    return check_pick


async def judge(input: DebatesPickMomentsInput, reels: List[Reel], *, client: Any = None) -> Tuple[int, str]:
    """The reel a viewer would watch through and understand, by a second call that sees only the
    clips. When the judge cannot answer, the first reel: the picker put its best first."""
    try:
        (best, why), _, _ = await ask_json(
            prompts.get("debates_pick_moments.judge_system"),
            render_for_judge(input, reels),
            judge_check(len(reels)),
            task=f"{TASK} judge",
            attempts=JUDGE_ATTEMPTS,
            client=client,
        )
        return best, why
    except DecisionError as e:
        logger.warning(f"{TASK}: the judge gave no pick; keeping the first reel: {e}")
        return 0, ""


async def pick_moments(input: DebatesPickMomentsInput, *, client: Any = None) -> DebatesPickMomentsResult:
    passed, model, attempts = await ask_json(
        prompts.get("debates_pick_moments.system"),
        render(input),
        lambda answer: check_reels(input, answer),
        task=TASK,
        attempts=ATTEMPTS,
        client=client,
    )
    # Judged among the reels with no soft problem when there are any, else among all that passed.
    clean = [reel for reel, soft in passed if not soft]
    reels = clean or [reel for reel, _ in passed]
    best, why = (0, "")
    if len(reels) > 1:
        best, why = await judge(input, reels, client=client)
    story, moments = reels[best]
    return DebatesPickMomentsResult(
        story=story,
        moments=moments,
        provider=PROVIDER,
        model_used=model,
        attempts=attempts,
        candidates=len(reels),
        judge_why=why or None,
    )


__all__ = ["CALLS", "TASK", "check", "check_reels", "judge", "pick_moments", "reel_ms", "render", "render_for_judge"]
