"""DB-free core of transcript.punctuate (GEO-3638): what Claude reads, and how
its answer becomes each turn's sentence ends.

The model sees each turn's words numbered and answers with word numbers and
marks. An answer is accepted only if every turn is present, its ends rise
strictly and stay inside the turn. A turn whose last word is not listed gets a
final ".", since a turn always ends a sentence.

The model sometimes ends a sentence inside a clause ("...is a. National security risk",
"...in the first place. Where corporations argued...") and everything downstream then treats the
halves as two sentences (GEO-3694). Two guards, both inside a turn, never across turns:

- An end on a word that can't end a sentence (`CANNOT_END`) is dropped when the next word follows
  within `JOIN_GAP_MS`, so the halves are one sentence again. After a longer pause the speaker
  trailed off ("i think the. The question is..."), so it stays. Without word times the pause is
  unknown, and the end is sent back as a soft problem instead.
- A sentence that opens like the rest of the one before ("That the...", "Where..." not asking,
  "To the...", "Which..." not asking), within `HINT_GAP_MS` of it, is sent back as a soft problem:
  join them if it continues, keep it if not. Soft problems go back once (`ATTEMPTS`), and an answer
  with only soft ones is accepted, so a wrong hint costs one more answer and nothing else.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from src.api.schemas.transcript_punctuate_schema import (
    PunctuatedTurn,
    PunctuateTurn,
    SentenceEnd,
    TranscriptPunctuateInput,
    TranscriptPunctuateResult,
)
from src.config.overrides import prompts
from src.pipeline.claude_json import PROVIDER, PUNCTUATE, ask_json

TASK = "transcript.punctuate"
# Answers asked for per run: the first, and one sent back with its problems.
ATTEMPTS = 2
MARKS = (".", "?", "!")

# Words no sentence ends on: articles, conjunctions, possessives. Stranded prepositions ("to",
# "of", "with", "in") are left out: spoken English ends on them ("you don't need to.", "that I
# agree with.", "the world they grow up in."), and all 4 such ends in the 12 Nov preview's
# punctuation were sentences like these. Matched case and all, so "Plan A." keeps its end.
CANNOT_END = frozenset(
    {"a", "an", "the", "and", "or", "but", "nor", "because", "although", "whereas", "unless",
     "my", "your", "our", "their", "its"}
)
# Pauses after those words inside a sentence, in the 12 Nov preview's 12 debates with word times
# (991 of them): median 90 ms, 90th percentile 350, 95th 460. 500 ms joins all but the longest
# 5%; the one trail-off there ("i think the. The question") paused 900 ms.
JOIN_GAP_MS = 500
# The opener hint costs at most one more answer, so it reaches further: the 99th percentile of
# those pauses was 890 ms, and "...in the first place. Where corporations argued" paused 640 ms.
HINT_GAP_MS = 1000
# A sentence opening like this usually continues the one before.
_CONTINUES = re.compile(r"(that (the|this|these|those)|where|to (the|a|an)|which)\b")
# Openers that start a question as often as a clause: hinted only when the sentence isn't one.
_ASKS = ("where", "which")
_TRIM = ",.;:…\"'“”"


def _bare(word: str) -> str:
    return word.strip().strip(_TRIM)


def _gap_ms(turn: PunctuateTurn, word: int) -> Optional[int]:
    """The pause after `word`, when the caller sent word times."""
    if turn.starts_ms is None or turn.ends_ms is None:
        return None
    return turn.starts_ms[word + 1] - turn.ends_ms[word]


def _cannot_end(turn: PunctuateTurn, end: SentenceEnd) -> bool:
    word = turn.words[end.word].strip()
    return end.mark == "." and not word.endswith(("?", "!")) and _bare(word) in CANNOT_END


def _said(turn: PunctuateTurn, first: int, last: int) -> str:
    return " ".join(turn.words[first : last + 1])


def repair(t: int, turn: PunctuateTurn, ends: List[SentenceEnd]) -> Tuple[List[SentenceEnd], List[str]]:
    """`ends` without the ones inside a clause the word times show, and soft problems for the
    rest: an end on a word that can't end a sentence with no times to judge it by, and a sentence
    that opens like the rest of the one before it."""
    last = len(turn.words) - 1
    kept: List[SentenceEnd] = []
    soft: List[str] = []
    for end in ends:
        if end.word < last and _cannot_end(turn, end):
            gap = _gap_ms(turn, end.word)
            if gap is None:
                soft.append(
                    f'Turn {t}: the sentence ending at word {end.word} ends on "{_bare(turn.words[end.word])}", '
                    "which can't end a sentence. Join it to the next one, unless the speaker broke off there."
                )
            elif gap <= JOIN_GAP_MS:
                continue
        kept.append(end)
    for k, end in enumerate(kept[:-1]):
        nxt = kept[k + 1]
        start = end.word + 1
        # Its first two words, never past its own end: "That. The point..." doesn't open "that the".
        opener = " ".join(_bare(w).lower() for w in turn.words[start : min(start + 2, nxt.word + 1)])
        match = _CONTINUES.match(opener)
        if end.mark != "." or not match or (match.group(1) in _ASKS and nxt.mark == "?"):
            continue
        gap = _gap_ms(turn, end.word)
        if gap is not None and gap > HINT_GAP_MS:
            continue
        soft.append(
            f'Turn {t}: the sentence starting at word {start} ("{_said(turn, start, min(start + 4, nxt.word))}...") '
            f'may be the rest of the one before ("...{_said(turn, max(end.word - 4, 0), end.word)}"). '
            f"If it continues it, join them by dropping the end at word {end.word}; if it stands on its own, keep it."
        )
    return kept, soft


def render(input: TranscriptPunctuateInput) -> str:
    lines = []
    if input.title:
        lines.append(f"Debate motion: {input.title}\n")
    for t, turn in enumerate(input.turns):
        words = " ".join(f"{i}:{w}" for i, w in enumerate(turn.words))
        lines.append(f"Turn {t} ({turn.speaker}, {len(turn.words)} words, numbered 0-{len(turn.words) - 1}):\n{words}\n")
    return "\n".join(lines)


def check(input: TranscriptPunctuateInput, answer: Any) -> Tuple[Optional[List[PunctuatedTurn]], List[str], List[str]]:
    """The answer as each turn's ends, or the problems with it."""
    problems: List[str] = []
    rows = answer.get("turns") if isinstance(answer, dict) else None
    if not isinstance(rows, list):
        return None, ['The answer needs a "turns" list.'], []
    by_turn = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("turn"), int):
            by_turn[row["turn"]] = row.get("ends")
    out: List[PunctuatedTurn] = []
    soft: List[str] = []
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
            parsed, hints = repair(t, turn, parsed)
            soft += hints
            out.append(PunctuatedTurn(ends=parsed))
    if problems:
        return None, problems, []
    return out, [], soft


async def punctuate(input: TranscriptPunctuateInput, *, client: Any = None) -> TranscriptPunctuateResult:
    turns, model, _ = await ask_json(
        prompts.get("transcript_punctuate.system"),
        render(input),
        lambda answer: check(input, answer),
        task=TASK,
        attempts=ATTEMPTS,
        config=PUNCTUATE,
        client=client,
    )
    return TranscriptPunctuateResult(
        turns=turns,
        sentences=sum(len(t.ends) for t in turns),
        provider=PROVIDER,
        model_used=model,
    )


__all__ = ["CANNOT_END", "HINT_GAP_MS", "JOIN_GAP_MS", "TASK", "check", "punctuate", "render", "repair"]
