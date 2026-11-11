"""DB-free core of debates.pick_moments (GEO-3638): what Claude reads, the rules
its reel must keep, and how an answer becomes four moments.

Every rule the prompt states that can be checked mechanically is checked here:
four clips, alternating speakers, whole sentences from one turn, length bounds,
no overlap, each clip linked to a different claim of its own speaker, and an
answer playing after what it answers. A broken answer is sent back with its
problems (claude_json.ask_json), so Claude fixes it rather than the run failing.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from src.api.schemas.debates_pick_moments_schema import (
    MOMENTS,
    DebatesPickMomentsInput,
    DebatesPickMomentsResult,
    PickedMoment,
)
from src.config.overrides import prompts
from src.pipeline.claude_json import PROVIDER, ask_json

TASK = "debates.pick_moments"
ROLES = ("opening", "answer", "rebuttal", "closing")


def _seconds(ms: int) -> str:
    return f"{ms / 1000:.1f}"


def render(input: DebatesPickMomentsInput) -> str:
    names = {s.slot: s.name for s in input.speakers}
    lines = [f"Motion: {input.title}", ""]
    for s in input.speakers:
        side = f": {s.side} with the motion" if s.side else ""
        lines.append(f"Speaker {s.slot} ({s.name}){side}")
    lines += [
        "",
        f"Target clip length: {_seconds(input.target_min_ms)}-{_seconds(input.target_max_ms)} s. "
        f"Hard limits: {_seconds(input.clip_min_ms)}-{_seconds(input.clip_max_ms)} s.",
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
        lines.append(f"[{i}] {_seconds(s.start_ms)}-{_seconds(s.end_ms)} s: {s.text.strip()}")
    return "\n".join(lines)


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def check(input: DebatesPickMomentsInput, answer: Any) -> Tuple[Optional[Tuple[str, List[PickedMoment]]], List[str]]:
    """The answer as (story, moments), or every problem with it."""
    if not isinstance(answer, dict) or not isinstance(answer.get("moments"), list):
        return None, ['The answer needs a "moments" list.']
    rows = answer["moments"]
    problems: List[str] = []
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
            if m.answers is not None and moments[m.answers].speaker == m.speaker:
                problems.append(f"Clip {i} answers its own speaker's clip; it can only answer the other speaker.")
    for i, m in enumerate(moments):
        for a in input.avoid:
            if m.start_sentence <= a.end_sentence and a.start_sentence <= m.end_sentence:
                problems.append(f"Clip {i} uses sentences {a.start_sentence}-{a.end_sentence}, which you were told to avoid ({a.reason}).")
    story = answer.get("story") if isinstance(answer.get("story"), str) else ""
    return ((story, moments) if not problems else None), problems


async def pick_moments(input: DebatesPickMomentsInput, *, client: Any = None) -> DebatesPickMomentsResult:
    (story, moments), model, attempts = await ask_json(
        prompts.get("debates_pick_moments.system"),
        render(input),
        lambda answer: check(input, answer),
        task=TASK,
        attempts=3,
        client=client,
    )
    return DebatesPickMomentsResult(
        story=story, moments=moments, provider=PROVIDER, model_used=model, attempts=attempts
    )


__all__ = ["TASK", "check", "pick_moments", "render"]
