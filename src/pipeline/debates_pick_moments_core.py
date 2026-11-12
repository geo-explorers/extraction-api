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

The first clip has hard rules of its own (`first_clip_problems`): it starts a sentence
and a thought, not mid-sentence, on a connective or on filler. A reel that breaks only
those is sent back like any other, but kept as a last resort when no answer has a reel
that keeps every rule, with the rules it breaks on the result (`first_clip_problems`).

Both calls answer under a JSON schema (structured outputs), so the reply is the JSON
alone (`REELS_SCHEMA`, `JUDGE_SCHEMA`).
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
# prompt forbids it; the check asks again (a soft rule: the last answer is kept regardless). For
# the first clip it is a hard rule (first_clip_problems), except FIRST_CLIP_SOFT_OPENERS.
DANGLING_OPENERS = (
    "and", "but", "so", "also", "because", "yeah", "yes", "okay", "ok", "um", "uh", "like",
    "that's", "they", "it's", "which", "then", "plus", "or",
)
# Openers that often state a position rather than lean on what came before ("It's time we…",
# "Yes, voter ID should be federal"): soft on the first clip too, as on later ones.
FIRST_CLIP_SOFT_OPENERS = ("it's", "yes")
# Two-word filler openers, each with whether it counts only when set off by a comma, as geo-chat's
# clip_ranges::filler_lead reads "you know": "You know, the data…" is filler, "you know nothing" a verb.
DANGLING_PAIRS = {("i", "mean"): False, ("you", "know"): True}
# A transcript is cased (Whisper's is) when at least this share of its sentences opens on a capital.
# Only then does a sentence opening on a lowercase letter say it starts mid-sentence.
CASED_SHARE = 0.5

_ANSWERS = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
_MOMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "speaker": {"type": "integer"},
        "start_sentence": {"type": "integer"},
        "end_sentence": {"type": "integer"},
        "claim": {"type": "integer"},
        "claim_fit": {"type": "number"},
        "role": {"type": "string", "enum": list(ROLES)},
        "answers": _ANSWERS,
        "why": {"type": "string"},
    },
    "required": ["speaker", "start_sentence", "end_sentence", "claim", "claim_fit", "role", "answers", "why"],
    "additionalProperties": False,
}
# The picker's answer, as the prompt asks for it. Counts (three reels, four clips) and ranges are
# not expressible in structured outputs' schemas; `check` enforces them.
REELS_SCHEMA = {
    "type": "object",
    "properties": {
        "reels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "story": {"type": "string"},
                    "moments": {"type": "array", "items": _MOMENT_SCHEMA},
                },
                "required": ["story", "moments"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reels"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"best": {"type": "integer"}, "why": {"type": "string"}},
    "required": ["best", "why"],
    "additionalProperties": False,
}
# Cut-off answers sent back to be answered shorter, per call.
CUT_OFF_RETRIES = 1


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


def _bare(word: str) -> str:
    """`word` lowercased, letters and apostrophes only (curly ones straightened)."""
    return re.sub(r"[^a-z']", "", word.lower().replace("’", "'"))


def _head(sentence) -> str:
    """A sentence's first word, as written."""
    return (sentence.text.split() or [""])[0]


def _first_word(sentence) -> str:
    """A sentence's first word, bare."""
    return _bare(_head(sentence))


def dangling_opener(sentence) -> Optional[str]:
    """The dangling word or filler `sentence` opens on (DANGLING_OPENERS, DANGLING_PAIRS), or None."""
    words = sentence.text.split()
    pair = tuple(_bare(w) for w in words[:2])
    if pair in DANGLING_PAIRS and (not DANGLING_PAIRS[pair] or words[1].endswith(",")):
        return " ".join(pair)
    first = _first_word(sentence)
    return first if first in DANGLING_OPENERS else None


def _opens_on_a_letter(text: str) -> bool:
    """Whether `text`'s first letter or digit is a letter. A sentence opening on a number ("50",
    "1960s", "10x") says nothing about case."""
    lead = re.search(r"[A-Za-z0-9]", text)
    return bool(lead) and lead.group(0).isalpha()


def cased(input: DebatesPickMomentsInput) -> bool:
    """Whether the transcript keeps capitals at sentence starts (CASED_SHARE), among the sentences
    that open on a letter."""
    firsts = [re.search(r"[A-Za-z]", s.text).group(0) for s in input.sentences if _opens_on_a_letter(s.text)]
    return bool(firsts) and sum(c.isupper() for c in firsts) >= CASED_SHARE * len(firsts)


def opens_mid_sentence(sentence) -> bool:
    """Whether `sentence` opens on a lowercase word ("and not just get…"): in a cased transcript, the
    speech recogniser heard it run on from the words before, wherever the punctuation put its start.
    A word cased inside ("iPhone") is a name, not a run-on; one opening on a number ("50", "1960s")
    says nothing."""
    head = _head(sentence)
    letters = re.sub(r"[^A-Za-z]", "", head)
    return _opens_on_a_letter(head) and letters.islower()


def _hard_start(sentence, is_cased: bool) -> Tuple[bool, Optional[str]]:
    """What the first clip may not open on, for `sentence`: whether it starts mid-sentence (in a cased
    transcript), and the connective or filler it opens on, if that is hard on the first clip."""
    opener = dangling_opener(sentence)
    return is_cased and opens_mid_sentence(sentence), (opener if opener not in FIRST_CLIP_SOFT_OPENERS else None)


def clean_start(sentence, is_cased: bool) -> bool:
    """Whether `sentence` can open a clip: no dangling opener, and in a cased transcript not
    mid-sentence. What a trim looks for first."""
    return dangling_opener(sentence) is None and not (is_cased and opens_mid_sentence(sentence))


def _start_rank(sentence, is_cased: bool) -> int:
    """How well `sentence` opens a clip, best first: clean (0); only a soft opener, which the first
    clip may keep (1); breaking the first clip's rules (2)."""
    if clean_start(sentence, is_cased):
        return 0
    mid, opener = _hard_start(sentence, is_cased)
    return 2 if mid or opener else 1


def first_clip_problems(input: DebatesPickMomentsInput, start: int, picked_start: int) -> List[str]:
    """Why a first clip starting at sentence `start` cannot open the reel: it starts mid-sentence,
    or on a connective or filler. A newcomer meets the debate there. `picked_start` is where the
    model started it, when a trim moved the start."""
    sentence = input.sentences[start]
    where = (
        f"Clip 0, the reel's first clip, opens on sentence {start}"
        + (f" (cut to fit from sentence {picked_start})" if start != picked_start else "")
    )
    fix = (
        "The first clip is where a newcomer meets the debate: start it on a sentence that begins a "
        "thought, or pick another first clip."
    )
    problems = []
    mid, opener = _hard_start(sentence, cased(input))
    if mid:
        problems.append(f"{where}, which starts mid-sentence (it opens on a lowercase word). {fix}")
    if opener:
        problems.append(f'{where}, which opens on "{opener}", a connective or filler. {fix}')
    return problems


def trim_to_fit(input: DebatesPickMomentsInput, start: int, end: int) -> Optional[int]:
    """Where a clip over the hard limit should start so it fits: sentences dropped from its start
    (the payoff at its end stays), and past any that open on a dangling word or mid-sentence
    (clean_start), so the trim never lands on "And ..." when a cleaner start fits. With no clean
    start, one with only a soft opener ("It's ...") beats one the first clip may not open on
    (_start_rank). `None` when no start leaves it within the
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
    is_cased = cased(input)
    # The earliest of the best-ranked starts: min keeps the first among equals.
    return min(fitting, key=lambda first: _start_rank(sentences[first], is_cased))


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def check(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[Tuple[str, List[PickedMoment]]], List[str], List[str]]:
    """The answer as (story, moments), its problems, and its soft problems. A reel whose only
    problems are its first clip's (first_clip_problems) comes back with them: refused, but usable
    as a last resort."""
    if not isinstance(answer, dict) or not isinstance(answer.get("moments"), list):
        return None, ['The answer needs a "moments" list.'], []
    rows = answer["moments"]
    problems: List[str] = []
    opening: List[str] = []
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
        opener = dangling_opener(run[0])
        if position == 0:
            # Hard, trimmed or not: whoever chose the start, the newcomer hears it first.
            opening = first_clip_problems(input, start, picked_start)
        if start == picked_start and opener and (position > 0 or opener in FIRST_CLIP_SOFT_OPENERS):
            soft.append(
                f'{name} opens on "{opener}", which leans on what came before or is filler. '
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
    if problems:
        return None, problems + opening, soft
    return (named(input, story), moments), opening, soft


def named(input: DebatesPickMomentsInput, story: str) -> str:
    """`story` with any "Speaker N" left in it replaced by that speaker's short name."""
    names = short_names(input)
    return re.sub(
        r"\bSpeaker (\d+)\b",
        lambda m: names.get(int(m.group(1)), m.group(0)),
        story,
    )


Reel = Tuple[str, List[PickedMoment]]
# A candidate reel that passed, with its soft problems; or, as a last resort, one that broke only
# its first clip's rules, with those.
Candidate = Tuple[Reel, List[str]]


def check_reels(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[List[Candidate]], List[str], List[str]]:
    """The candidate reels that keep every hard rule, each with its soft problems; the answer's
    problems when none does; and, when only one reel passed and it has soft problems, those and
    the other reels' problems.
    With two or more passing, the judge weighs them as they are: sending them back for a soft
    nudge could return fewer.

    When none passes, the reels that broke only their first clip's rules come back with the
    problems, each with its first-clip problems: the answer is sent back, but they are a last
    resort (claude_json.ask_json).

    A single reel (an answer in the old shape, {"moments": ...}) is one candidate."""
    if isinstance(answer, dict) and isinstance(answer.get("reels"), list):
        reels = answer["reels"][:CANDIDATES]
    elif isinstance(answer, dict) and "moments" in answer:
        reels = [answer]
    else:
        return None, ['The answer needs a "reels" list.'], []
    if not reels:
        return None, ['The "reels" list is empty.'], []
    passed: List[Candidate] = []
    last_resort: List[Candidate] = []
    problems: List[str] = []
    softs: List[str] = []
    for number, reel in enumerate(reels):
        value, hard, soft = check(input, reel)
        label = f"Reel {number}: " if len(reels) > 1 else ""
        if hard:
            problems += [label + p for p in hard]
            if value is not None:
                last_resort.append((value, hard))
            continue
        passed.append((value, soft))
        softs += [label + p for p in soft]
    if not passed:
        return (last_resort or None), problems, []
    # One reel passed with a soft problem: it goes back with the others' problems too, so the next
    # answer can fix them as well as improve it.
    return passed, [], (softs + problems if len(passed) == 1 and softs else [])


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
            schema=JUDGE_SCHEMA,
            cut_off_retries=CUT_OFF_RETRIES,
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
        schema=REELS_SCHEMA,
        cut_off_retries=CUT_OFF_RETRIES,
    )
    # Judged among the reels with no soft problem when there are any, else among all that passed.
    # (Last-resort reels all carry their first-clip problems, so they are judged among themselves.)
    clean = [(reel, soft) for reel, soft in passed if not soft]
    candidates = clean or passed
    best, why = (0, "")
    if len(candidates) > 1:
        best, why = await judge(input, [reel for reel, _ in candidates], client=client)
    (story, moments), _ = candidates[best]
    # A kept reel's first clip either keeps its rules or this is a last resort that breaks them.
    first = moments[0]
    picked = first.start_sentence if first.trimmed_from_sentence is None else first.trimmed_from_sentence
    broken = first_clip_problems(input, first.start_sentence, picked)
    if broken:
        logger.warning(f"{TASK}: no reel kept the first clip's rules; kept one that breaks them: {broken}")
    return DebatesPickMomentsResult(
        story=story,
        moments=moments,
        provider=PROVIDER,
        model_used=model,
        attempts=attempts,
        candidates=len(candidates),
        judge_why=why or None,
        first_clip_problems=broken,
    )


__all__ = [
    "CALLS",
    "JUDGE_SCHEMA",
    "REELS_SCHEMA",
    "TASK",
    "check",
    "check_reels",
    "dangling_opener",
    "first_clip_problems",
    "judge",
    "pick_moments",
    "reel_ms",
    "render",
    "render_for_judge",
]
