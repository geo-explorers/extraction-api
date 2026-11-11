"""DB-free core of transcript.punctuate (GEO-3638): what Claude reads, and how
its answer becomes each turn's sentence ends.

The model sees each turn's words numbered and answers with word numbers and
marks. An answer is accepted only if every turn is present, its ends rise
strictly and stay inside the turn. A turn whose last word is not listed gets a
final ".", since a turn always ends a sentence.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from src.api.schemas.transcript_punctuate_schema import (
    PunctuatedTurn,
    SentenceEnd,
    TranscriptPunctuateInput,
    TranscriptPunctuateResult,
)
from src.config.overrides import prompts
from src.pipeline.claude_json import PROVIDER, ask_json

TASK = "transcript.punctuate"
MARKS = (".", "?", "!")


def render(input: TranscriptPunctuateInput) -> str:
    lines = []
    if input.title:
        lines.append(f"Debate motion: {input.title}\n")
    for t, turn in enumerate(input.turns):
        words = " ".join(f"{i}:{w}" for i, w in enumerate(turn.words))
        lines.append(f"Turn {t} ({turn.speaker}, {len(turn.words)} words, numbered 0-{len(turn.words) - 1}):\n{words}\n")
    return "\n".join(lines)


def check(input: TranscriptPunctuateInput, answer: Any) -> Tuple[Optional[List[PunctuatedTurn]], List[str]]:
    """The answer as each turn's ends, or the problems with it."""
    problems: List[str] = []
    rows = answer.get("turns") if isinstance(answer, dict) else None
    if not isinstance(rows, list):
        return None, ['The answer needs a "turns" list.']
    by_turn = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("turn"), int):
            by_turn[row["turn"]] = row.get("ends")
    out: List[PunctuatedTurn] = []
    for t, turn in enumerate(input.turns):
        ends = by_turn.get(t)
        if not isinstance(ends, list):
            problems.append(f"Turn {t} is missing.")
            continue
        last = len(turn.words) - 1
        parsed: List[SentenceEnd] = []
        previous = -1
        for end in ends:
            if not (isinstance(end, list) and len(end) == 2 and isinstance(end[0], int) and end[1] in MARKS):
                problems.append(f'Turn {t}: {end!r} is not a [word number, "." | "?" | "!"] pair.')
                break
            word, mark = end
            if not previous < word <= last:
                problems.append(
                    f"Turn {t}: word {word} is out of order or outside 0-{last}."
                )
                break
            parsed.append(SentenceEnd(word=word, mark=mark))
            previous = word
        else:
            if not parsed or parsed[-1].word != last:
                parsed.append(SentenceEnd(word=last, mark="."))
            out.append(PunctuatedTurn(ends=parsed))
    return (out if not problems else None), problems


async def punctuate(input: TranscriptPunctuateInput, *, client: Any = None) -> TranscriptPunctuateResult:
    turns, model, _ = await ask_json(
        prompts.get("transcript_punctuate.system"),
        render(input),
        lambda answer: check(input, answer),
        task=TASK,
        client=client,
    )
    return TranscriptPunctuateResult(
        turns=turns,
        sentences=sum(len(t.ends) for t in turns),
        provider=PROVIDER,
        model_used=model,
    )


__all__ = ["TASK", "check", "punctuate", "render"]
