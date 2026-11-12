"""DB-free core of debates.pick_moments (GEO-3638): what Claude reads, the rules
its reels must keep, how the best reel is judged, and how it becomes four moments.

Claude offers up to three candidate reels in one answer. Each is checked on its own
(`check`); the answer is sent back when none passes the hard rules, or when every reel
that passes has a soft problem (`check_reels`). When more than one passes, a second call
judges them as a viewer meets them, the clips' words alone with the problems found in
how they play, and its pick is returned (`judge`).

Every rule the prompt states that can be checked mechanically is checked here:
four clips, alternating speakers, whole sentences from one turn, length bounds,
no overlap, each clip linked to a different claim of its own speaker, and an
answer playing after what it answers. A broken answer is sent back with its
problems (claude_json.ask_json), so Claude fixes it rather than the run failing.
A clip over the hard limit is sent back with how long it runs to each sentence
(over_limit_problem); cut to fit, from its end first (fit_to_limit), its reel is a
last resort.

The first clip has hard rules of its own (`first_clip_problems`): it starts a sentence
and a thought, not mid-sentence, on a connective or on filler. Later clips keep the same
rules softly (`later_clip_problems`): a reel that breaks them is sent back, but kept over
no reel; an answer that goes on to a thought ("Yes, the rolls are clean.") is fine there.
A clip that starts on a sentence marked as the second half of a split sentence
(`split_start`, a guess from how it opens) is a soft problem on every clip, the first
included: the guess is wrong often enough that it never refuses a reel. Every opener
rule reads a clip as the viewer hears it, past the filler geo-chat cuts before it plays
(`heard_start`), and whether or not it was cut to fit. A reel that breaks only the first
clip's rules or ran a clip over the limit is sent back like any other, but kept as a last
resort when no answer has a reel that keeps every hard rule, with the rules its first clip
breaks on the result (`first_clip_problems`) and its cut clips marked.

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
# The filler geo-chat cuts from the start of every clip before it plays, so a clip's opener is
# judged on what is left (heard_start). A copy of geo-chat's crates/media/src/clip_ranges.rs
# (FILLER, FILLER_PAIRS, KNOW_OBJECT_NEXT, SO_ADVERB_NEXT, filler_lead) and
# crates/media/src/story_moments.rs (FILLER_KEEP_WORDS, filler_lead): change them together, and
# test_the_filler_list_is_geo_chats will fail until the copy matches again.
FILLER = ("um", "uh", "er", "ah", "hmm", "so", "and")
# Two-word filler, checked before FILLER.
FILLER_PAIRS = (("you", "know"),)
# "you know" before one of these is a verb with its object, not filler: "you know nothing".
KNOW_OBJECT_NEXT = (
    "nothing", "what", "how", "why", "who", "where", "when", "everything", "anything", "something",
    "more", "better", "exactly",
)
# "so" before one of these is an adverb, not filler: "so far the data shows" keeps its "so".
SO_ADVERB_NEXT = ("far", "much", "many", "long", "few", "little", "often", "called", "bad", "good")
# The filler stays when fewer than this many words of its sentence would be left: "So what now?".
FILLER_KEEP_WORDS = 3
# A clip whose heard words open on one of these leans on something the viewer did not hear, or is
# filler geo-chat kept (FILLER_KEEP_WORDS). The prompt forbids it; the check sends it back (a soft
# rule: a reel without it is taken first, and the last answer is kept regardless). For the first
# clip it is a hard rule (first_clip_problems), except FIRST_CLIP_SOFT_OPENERS. The second line is
# what the 12 Nov preview (GEO-3691) opened later clips on: "Though, you know…", a pronoun the viewer
# never heard introduced ("It needs to be included…", "they're working against themselves"), "That…".
DANGLING_OPENERS = FILLER + (
    "but", "also", "because", "yeah", "yes", "okay", "ok", "like",
    "that's", "they", "it's", "which", "then", "plus", "or",
    "though", "they're", "it", "that", "agree", "agreed",
)
# Openers that often state a position rather than lean on what came before ("It's time we…",
# "It is the death of relationships…", "Yes, voter ID should be federal"): soft on the first clip too,
# as on later ones.
FIRST_CLIP_SOFT_OPENERS = ("it's", "it", "yes")
# Words that answer the clip before ("Yes, the rolls are clean."): on a later clip they open a
# thought when the sentence goes on to one of its own (_answers_on). "Agree so i think…" does not.
ANSWER_OPENERS = ("yes", "yeah", "agree", "agreed")
# Two-word openers, each with whether it counts only when set off by a comma: "I mean, …" leans on
# what came before; "You know, …" is filler when geo-chat keeps it ("You know, right?"); an example
# comes before its point.
DANGLING_PAIRS = {("i", "mean"): False, ("you", "know"): True, ("for", "example"): False, ("for", "instance"): False}
# The transcript sometimes breaks one sentence in two and capitalises the second half: Whisper put a
# full stop mid-sentence ("…in the first place. Where corporations argued…"). split_start guesses
# at such a half by how it opens: one of these words, followed by the words that make it a clause
# of the sentence before rather than a sentence of its own. A guess: "Where I come from, …" opens a
# sentence too. So the mark is a hint to the picker and a soft problem, never a refusal; the real
# fix is in the punctuation step (GEO-3694).
# "Where corporations argued…", but not a question: "Where do we draw the line?".
WHERE_QUESTION_NEXT = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should", "will", "has", "have", "had", "else")
# "That the border security can serve…": "that" before its clause's subject, not "That argument…".
THAT_CLAUSE_NEXT = (
    "the", "a", "an", "we", "they", "he", "she", "i", "you", "there", "this", "these", "those",
    "our", "their", "his", "my", "your", "it", "if", "when", "people",
)
# No rule for "To the …" ("To the, like, to the idea of…"): it opens a sentence of its own as often
# as not ("To the extent…", "To the first point, …"), and an idiom list chasing it grew each review.
# The other half of the same mistake: a sentence that ends on one of these words stopped mid-phrase
# ("…legal immigration is a."), so the next sentence of its turn is its second half. Not one ending on
# a question or exclamation mark: "What if?" and "Oh my!" were finished as said.
OPEN_ENDS = ("a", "an", "the", "and", "or", "but", "my", "your", "our", "their", "because", "if", "than", "whether")
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
        split = f", {SPLIT_MARK}" if continues(input, i) else ""
        lines.append(f"[{i}] {_seconds(s.start_ms)}-{_seconds(s.end_ms)} s ({_seconds(s.end_ms - s.start_ms)} s{split}): {s.text.strip()}")
    return "\n".join(lines)


# How the picker's transcript marks a sentence that continues the one before (continues).
SPLIT_MARK = "second half of a split sentence"


def _bare(word: str) -> str:
    """`word` lowercased, letters, digits and apostrophes only (curly ones straightened), as
    geo-chat's clip_ranges::bare reads it."""
    return "".join(c for c in word.lower().replace("’", "'") if c.isalnum() or c == "'")


def _runs_on(word: str) -> bool:
    """Whether no punctuation ends `word`, so it and the next can be one phrase ("so far", "you know
    nothing"); "So, many people" is filler before a new clause. geo-chat's clip_ranges::runs_on."""
    return not word.rstrip().endswith((",", ".", ";", ":", "!", "?", "…"))


def filler_lead(words: List[str]) -> int:
    """How many of `words`' first words are filler a clip starts past: FILLER and FILLER_PAIRS,
    except "so" as an adverb (SO_ADVERB_NEXT) and "you know" with its object (KNOW_OBJECT_NEXT),
    each only when the words run on. geo-chat's clip_ranges::filler_lead."""
    at = lambda i: _bare(words[i]) if i < len(words) else None
    runs_on_at = lambda i: i < len(words) and _runs_on(words[i])
    first = 0
    while (head := at(first)) is not None:
        following = at(first + 1)
        if (
            runs_on_at(first)
            and (head, following) in FILLER_PAIRS
            and not (runs_on_at(first + 1) and at(first + 2) in KNOW_OBJECT_NEXT)
        ):
            first += 2
        elif head in FILLER and not (head == "so" and runs_on_at(first) and following in SO_ADVERB_NEXT):
            first += 1
        else:
            break
    return first


def heard_start(input: DebatesPickMomentsInput, start: int, end: int) -> Tuple[int, List[str]]:
    """Where the viewer starts hearing a clip of sentences `start`-`end`, once geo-chat has cut its
    filler: the sentence, and its words from the first one heard. Sentences that are nothing but
    filler ("Um.") are skipped, then the filler the next one opens on, unless that leaves fewer than
    FILLER_KEEP_WORDS of it. geo-chat's story_moments::filler_lead.

    geo-chat keeps all of it when the clip would run under the minimum length without it
    (story_reel, CLIP_MIN_MS, sent as clip_min_ms): "Um. We need rules." at 6.5 s, "Um." taking
    1 s, plays as said. A skipped sentence's time is known; filler inside a sentence is timed as its
    share of the sentence's words, since sentences carry no word times."""
    sentences = input.sentences
    for k in range(start, end + 1):
        words = sentences[k].text.split()
        lead = filler_lead(words)
        if lead == len(words) and k < end:
            continue
        if len(words) - lead < FILLER_KEEP_WORDS:
            lead = 0
        heard_ms = sentences[k].start_ms + (sentences[k].end_ms - sentences[k].start_ms) * lead // max(len(words), 1)
        if (k > start or lead) and sentences[end].end_ms - heard_ms < input.clip_min_ms:
            return start, sentences[start].text.split()
        return k, words[lead:]
    return end, sentences[end].text.split()


def heard_text(input: DebatesPickMomentsInput, start: int, end: int) -> str:
    """The words of a clip of sentences `start`-`end` as the viewer hears them (heard_start), the
    first heard word capitalised once filler is cut: geo-chat's story_moments::without_lead."""
    k, words = heard_start(input, start, end)
    text = " ".join(words)
    if k != start or len(words) < len(input.sentences[k].text.split()):
        text = text.lstrip(", ")
        text = text[:1].upper() + text[1:]
    return " ".join([text] + [s.text.strip() for s in input.sentences[k + 1 : end + 1]])


def _opener(words: List[str]) -> Optional[str]:
    """The dangling word or filler `words` open on (DANGLING_OPENERS, DANGLING_PAIRS), or None."""
    pair = tuple(_bare(w) for w in words[:2])
    if pair in DANGLING_PAIRS and (not DANGLING_PAIRS[pair] or words[1].endswith(",")):
        return " ".join(pair)
    first = _bare(words[0]) if words else ""
    return first if first in DANGLING_OPENERS else None


def _lead_char(text: str) -> str:
    """`text`'s first letter or digit, any script ("É" as much as "E"); "" if it has none."""
    return next((c for c in text if c.isalnum()), "")


def cased(input: DebatesPickMomentsInput) -> bool:
    """Whether the transcript keeps capitals at sentence starts (CASED_SHARE), among the sentences
    that open on a letter. A sentence opening on a number ("50", "1960s") says nothing about case."""
    firsts = [c for c in (_lead_char(s.text) for s in input.sentences) if c.isalpha()]
    return bool(firsts) and sum(c.isupper() for c in firsts) >= CASED_SHARE * len(firsts)


def opens_mid_sentence(sentence) -> bool:
    """Whether `sentence` opens on a lowercase word ("and not just get…"): in a cased transcript, the
    speech recogniser heard it run on from the words before, wherever the punctuation put its start.
    Read on the sentence's first word as written, filler and all: a capital on "And we need rules."
    is where the recogniser started a sentence, so what plays after the cut "And" starts one too,
    while "and not just…" plays as a run-on. A word cased inside ("iPhone") is a name, not a run-on;
    one opening on a number ("50", "1960s") says nothing (cased)."""
    word = next((w for w in sentence.text.split() if _lead_char(w)), "")
    letters = "".join(c for c in word if c.isalpha())
    return _lead_char(word).isalpha() and letters.islower()


def _ends_on(text: str, marks: Tuple[str, ...]) -> bool:
    """Whether `text` ends on one of `marks`, past any closing quote or bracket."""
    return text.rstrip("\"'”’)").endswith(marks)


def split_start(input: DebatesPickMomentsInput, k: int, words: List[str]) -> bool:
    """Whether sentence `k`, opening on `words`, looks like the second half of a sentence the
    transcript broke in two: the sentence before it in its turn ends mid-phrase (OPEN_ENDS), or
    `words` open as a clause of the sentence before ("Where corporations argued…", "That the border
    security can serve…"). A clip that starts there likely starts mid-sentence, though the
    transcript gave it a capital. A guess, so only ever a soft problem. A turn's first sentence is never one: it cannot be the second half
    of the other speaker's sentence. A sentence of nothing but filler ("Um.") between the halves is
    looked past, as heard_start skips it. A sentence that ends on "?" is no clause of the one before
    ("Where in the Constitution does it say that?")."""
    sentences = input.sentences
    j = k - 1
    while j >= 0 and sentences[j].turn == sentences[k].turn:
        before = sentences[j].text.split()
        if filler_lead(before) < len(before):
            break
        j -= 1
    if j < 0 or sentences[j].turn != sentences[k].turn:
        return False
    last = before[-1]
    if (
        _bare(last) in OPEN_ENDS
        and not _ends_on(last, ("?", "!"))
        and (len(before) == 1 or not any(c.isupper() for c in last))
    ):
        return True
    if _ends_on(sentences[k].text.strip(), ("?",)):
        return False
    first, following = (_bare(w) for w in (words + ["", ""])[:2])
    if first == "where":
        return following not in WHERE_QUESTION_NEXT
    return first == "that" and following in THAT_CLAUSE_NEXT


def continues(input: DebatesPickMomentsInput, k: int) -> bool:
    """Whether sentence `k` is the second half of a split sentence (split_start), read past the
    filler it opens on as geo-chat would cut it: what the picker's transcript marks (SPLIT_MARK)."""
    words = input.sentences[k].text.split()
    return split_start(input, k, words[filler_lead(words):])


def _answers_on(words: List[str]) -> bool:
    """Whether `words`, opening on an answer word ("Yes, …"), go on in the same sentence to a
    thought of their own: at least FILLER_KEEP_WORDS more words that open on no dangling word or
    filler. "Yes, the rolls are clean." does; "Yes.", "Yeah, but…" and "Agree so i think…" don't."""
    rest = words[1:]
    return len(rest) >= FILLER_KEEP_WORDS and _opener(rest) is None


def _opening(
    input: DebatesPickMomentsInput, start: int, end: int, is_cased: bool, later: bool = False
) -> Tuple[bool, bool, Optional[str]]:
    """How a clip of sentences `start`-`end` opens, as the viewer hears it (heard_start): whether it
    starts mid-sentence (in a cased transcript), the dangling word or filler it opens on, and
    otherwise whether it starts on what looks like the second half of a split sentence
    (split_start). Mid-sentence and the opener win over the split mark, which is only a guess, so
    one flaw is not counted twice: "That the border security…" opens on "that", and "where
    corporations argued…" starts mid-sentence, whether or not either is a split half. A `later` clip (not the first)
    may answer the one before it: "Yes, the rolls are clean." (ANSWER_OPENERS)."""
    k, words = heard_start(input, start, end)
    mid = is_cased and opens_mid_sentence(input.sentences[k])
    opener = _opener(words)
    if later and opener in ANSWER_OPENERS and _answers_on(words):
        opener = None
    return mid, not mid and opener is None and split_start(input, k, words), opener


def _start_rank(input: DebatesPickMomentsInput, start: int, end: int, is_cased: bool) -> int:
    """How well sentence `start` opens a clip ending at `end`, best first: clean (0); only a soft
    problem, which the first clip may keep (a soft opener or a split mark, 1); breaking the first
    clip's rules (2)."""
    mid, split, opener = _opening(input, start, end, is_cased)
    if mid or (opener and opener not in FIRST_CLIP_SOFT_OPENERS):
        return 2
    return 1 if opener or split else 0


def first_clip_problems(input: DebatesPickMomentsInput, start: int, end: int, picked_start: int) -> List[str]:
    """Why a first clip of sentences `start`-`end` cannot open the reel: what the viewer hears of it
    (heard_start) starts mid-sentence or on a connective or filler. A newcomer meets the debate
    there. A split mark is not one of these: it is a guess, soft on every clip (later_clip_problems).
    `picked_start` is where the model started it, when a trim moved the start."""
    where = (
        f"Clip 0, the reel's first clip, opens on sentence {start}"
        + (f" (cut to fit from sentence {picked_start})" if start != picked_start else "")
    )
    fix = (
        "The first clip is where a newcomer meets the debate: start it on a sentence that begins a "
        "thought, or pick another first clip."
    )
    problems = []
    mid, _, opener = _opening(input, start, end, cased(input))
    if mid:
        problems.append(f"{where}, which starts mid-sentence (it opens on a lowercase word). {fix}")
    if opener and opener not in FIRST_CLIP_SOFT_OPENERS:
        problems.append(f'{where}, which opens on "{opener}", a connective or filler. {fix}')
    return problems


def later_clip_problems(input: DebatesPickMomentsInput, position: int, start: int, end: int, is_cased: bool) -> List[str]:
    """How a clip of sentences `start`-`end` at `position` opens badly, short of the first clip's
    hard rules: on a sentence marked as the second half of a split sentence (any clip, the first
    included); on a soft opener for the first clip; for a later one, mid-sentence or on a dangling
    word or filler, except an answer that goes on to a thought ("Yes, the rolls are clean."). Soft:
    sent back for a fix, and a reel without them is taken first, but a reel with them still beats
    none."""
    name = f"Clip {position}"
    fix = "Start it on a sentence that opens a thought."
    mid, split, opener = _opening(input, start, end, is_cased, later=position > 0)
    if position == 0:
        mid = False
        opener = opener if opener in FIRST_CLIP_SOFT_OPENERS else None
    problems = []
    if mid:
        problems.append(f"{name} starts mid-sentence (its first sentence opens on a lowercase word). {fix}")
    if split:
        problems.append(
            f'{name} starts on a sentence marked "{SPLIT_MARK}": the transcript may have broken a sentence in two '
            "there. If it did, start the clip where that sentence starts, or pick another stretch."
        )
    if opener:
        problems.append(f'{name} opens on "{opener}", which leans on what came before or is filler. {fix}')
    return problems


def trim_to_fit(input: DebatesPickMomentsInput, start: int, end: int) -> Optional[int]:
    """Where a clip over the hard limit should start so it fits: sentences dropped from its start
    (the payoff at its end stays), and past any whose heard words open on a dangling word or
    mid-sentence, so the trim never lands on "But ..." when a cleaner start fits. With no clean
    start, one with only a soft opener ("It's ...") beats one the first clip may not open on
    (_start_rank, read as first_clip_problems reads it). `None` when no start leaves it within the
    limits. A clip within the limit is left where it starts. The last way to fit a clip
    (fit_to_limit): it drops the setup the clip's story leans on."""
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
    return min(fitting, key=lambda first: _start_rank(input, first, end, is_cased))


def fit_to_limit(input: DebatesPickMomentsInput, start: int, end: int) -> Optional[Tuple[int, int]]:
    """The sentences a clip of `start`-`end` over the hard limit keeps once cut to fit, for a reel
    kept as a last resort: the latest end that fits where the clip still ends on a whole sentence
    (the next one is not its second half, split_start), so its setup stays; else the start
    trim_to_fit picks; else the latest end that fits. `None` when nothing fits; a clip within the
    limit keeps every sentence. Cutting the start lost the setup in 6 of 36 checkable clips of the
    12 Nov preview (GEO-3691), so it comes after a clean end."""
    sentences = input.sentences
    length = lambda first, last: sentences[last].end_ms - sentences[first].start_ms
    if length(start, end) <= input.clip_max_ms:
        return start, end
    ends = [last for last in range(end - 1, start - 1, -1) if input.clip_min_ms <= length(start, last) <= input.clip_max_ms]
    clean = [last for last in ends if not continues(input, last + 1)]
    if clean:
        return start, clean[0]
    first = trim_to_fit(input, start, end)
    if first is not None:
        return first, end
    return (start, ends[0]) if ends else None


def over_limit_problem(input: DebatesPickMomentsInput, name: str, start: int, end: int) -> str:
    """What a clip over the hard limit is sent back with: how long it runs to each of its
    sentences, so the next pick can end (or start) where it fits."""
    sentences = input.sentences
    runs = ", ".join(
        f"{_seconds(sentences[last].end_ms - sentences[start].start_ms)} s to sentence {last}"
        for last in range(start, end + 1)
    )
    return (
        f"{name}, sentences {start}-{end}, runs {_seconds(sentences[end].end_ms - sentences[start].start_ms)} s, "
        f"over the {_seconds(input.clip_max_ms)} s limit. From sentence {start} it runs {runs}. End it "
        "earlier, after a sentence where its point has landed, or start it later on a sentence that "
        f"opens a thought, to run {_seconds(input.clip_min_ms)}-{_seconds(input.clip_max_ms)} s."
    )


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def check(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[Tuple[str, List[PickedMoment]]], List[str], List[str]]:
    """The answer as (story, moments), its problems, and its soft problems. A reel whose only
    problems are its first clip's opening (first_clip_problems) or clips over the hard limit
    (over_limit_problem) comes back with them, those clips cut to fit (fit_to_limit): refused, but
    usable as a last resort."""
    if not isinstance(answer, dict) or not isinstance(answer.get("moments"), list):
        return None, ['The answer needs a "moments" list.'], []
    rows = answer["moments"]
    problems: List[str] = []
    # Hard, but the reel is still a last resort with them.
    fixable: List[str] = []
    soft: List[str] = []
    is_cased = cased(input)
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
        # A clip over the limit is sent back with how long it runs to each sentence, so the model
        # picks the stretch itself; the 12 Nov preview's trims from the start lost the setup. Cut
        # to fit (fit_to_limit, recorded on the moment), the reel is a last resort. One that
        # cannot be cut to fit is reported at the length the model picked.
        picked_start, picked_end = start, end
        fit = fit_to_limit(input, start, end)
        if fit is not None and fit != (start, end):
            fixable.append(over_limit_problem(input, name, start, end))
            start, end = fit
        run = sentences[start : end + 1]
        # Every opener rule reads the clip as it plays, cut to fit or not, past the filler geo-chat
        # cuts: the first clip's are hard (a newcomer meets the debate there), the rest soft, and
        # a split mark soft on every clip.
        if position == 0:
            fixable += first_clip_problems(input, start, end, picked_start)
        soft += later_clip_problems(input, position, start, end, is_cased)
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
                trimmed_to_sentence=picked_end if end != picked_end else None,
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
        return None, problems + fixable, soft
    return (named(input, story), moments), fixable, soft


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
# its first clip's opening rules or ran a clip over the limit (cut to fit), with those and its soft
# problems.
Candidate = Tuple[Reel, List[str]]


def check_reels(
    input: DebatesPickMomentsInput, answer: Any
) -> Tuple[Optional[List[Candidate]], List[str], List[str]]:
    """The candidate reels that keep every hard rule, each with its soft problems; the answer's
    problems when none does; and, when every reel that passed has a soft problem, all of those and
    the other reels' problems.

    A reel with no soft problem is taken without a send-back (pick_moments judges only those). When
    every passing reel has one (a later clip opening on "But…", a reel over its length), the answer
    is sent back to fix them, however many passed: in the 12 Nov preview about 17 clips past the
    first opened mid-thought, and reels that passed together were never asked again. ask_json
    keeps whichever answer's best reel has the fewest problems (answer_rank), so a send-back that
    comes back worse does not replace a better answer.

    When none passes, the reels whose only problems are fixable (their first clip's opening, a clip
    cut to fit) come back with the problems, each with its own and its soft ones: the answer is
    sent back with the hard problems alone, but they are a last resort (claude_json.ask_json).

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
                # With its soft problems too, so answer_rank weighs a later "But…" or a split
                # mark between two last resorts; only the hard ones are sent back.
                last_resort.append((value, hard + soft))
            continue
        passed.append((value, soft))
        softs += [label + p for p in soft]
    if not passed:
        return (last_resort or None), problems, []
    if any(not soft for _, soft in passed):
        return passed, [], []
    # Every reel that passed has a soft problem: they go back with the others' problems too, so the
    # next answer can fix them as well as improve them.
    return passed, [], softs + problems


def answer_rank(candidates: List[Candidate]) -> int:
    """How an answer with no clean reel ranks against another (claude_json.ask_json's `rank`, lower
    is better): the fewest problems of any of its reels, soft ones for reels that passed, fixable
    and soft ones for last resorts (a soft flaw counts between two last resorts too). Before, the latest answer always won, and once every soft answer was
    sent back (GEO-3691) a worse second answer replaced a better first one."""
    return min(len(problems) for _, problems in candidates)


def clip_flaws(input: DebatesPickMomentsInput, position: int, moment: PickedMoment, is_cased: bool) -> List[str]:
    """What the checks found wrong with a clip at `position`, as the viewer hears it: the soft
    problems (later_clip_problems) and the hard ones a last resort keeps (first_clip_problems, a
    cut to fit), in the judge's words. Each is how the clip plays, not what the picker meant."""
    mid, split, opener = _opening(input, moment.start_sentence, moment.end_sentence, is_cased, later=position > 0)
    flaws = []
    if mid:
        flaws.append("starts mid-sentence")
    if split:
        flaws.append("may start halfway through a sentence")
    if opener:
        flaws.append(f'opens on "{opener}", which leans on something the viewer did not hear or is filler')
    if moment.trimmed_from_sentence is not None:
        flaws.append("its opening was cut to fit the length limit, so it may lack its setup")
    if moment.trimmed_to_sentence is not None:
        flaws.append("its ending was cut to fit the length limit, so it may stop before its point lands")
    return flaws


def render_for_judge(input: DebatesPickMomentsInput, reels: List[Reel]) -> str:
    """The reels as a viewer meets them: the motion, who speaks for which side, and each clip's
    words as heard, past the filler geo-chat cuts (heard_text), and its length as picked. Under a
    reel, the problems the checks found in how its clips play (clip_flaws), so a clean reel is
    preferred over one that reads as well but plays worse. No transcript, no claims, no story: the
    judge sees what the viewer sees."""
    names = short_names(input)
    is_cased = cased(input)
    lines = [f"Motion: {input.title}", ""]
    lines += [f"{names[s.slot]}{_side(s)}" for s in input.speakers]
    for number, (_, moments) in enumerate(reels):
        lines += ["", f"## Reel {number} ({_seconds(reel_ms(moments))} s)"]
        flaws = []
        for position, m in enumerate(moments):
            words = heard_text(input, m.start_sentence, m.end_sentence)
            lines.append(f"Clip {position + 1}, {names[m.speaker]}, {_seconds(m.end_ms - m.start_ms)} s: {words}")
            flaws += [f"- Clip {position + 1} {flaw}." for flaw in clip_flaws(input, position, m, is_cased)]
        if reel_ms(moments) > reel_max_ms(input):
            flaws.append(f"- The reel runs over {_seconds(reel_max_ms(input))} s.")
        if flaws:
            lines += ["Problems found:"] + flaws
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
        rank=answer_rank,
    )
    # Judged among the reels with no soft problem when there are any, else among all that passed,
    # with their problems shown (render_for_judge). (Last-resort reels all carry their fixable
    # problems, so they are judged among themselves.)
    clean = [(reel, soft) for reel, soft in passed if not soft]
    candidates = clean or passed
    best, why = (0, "")
    if len(candidates) > 1:
        best, why = await judge(input, [reel for reel, _ in candidates], client=client)
    (story, moments), _ = candidates[best]
    # A kept reel's first clip either keeps its rules or this is a last resort that breaks them.
    first = moments[0]
    picked = first.start_sentence if first.trimmed_from_sentence is None else first.trimmed_from_sentence
    broken = first_clip_problems(input, first.start_sentence, first.end_sentence, picked)
    if broken:
        logger.warning(f"{TASK}: no reel kept the first clip's rules; kept one that breaks them: {broken}")
    cut = [i for i, m in enumerate(moments) if m.trimmed_from_sentence is not None or m.trimmed_to_sentence is not None]
    if cut:
        logger.warning(f"{TASK}: no reel kept every clip within the limit; kept one with clips {cut} cut to fit")
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
    "answer_rank",
    "check",
    "check_reels",
    "clip_flaws",
    "continues",
    "filler_lead",
    "first_clip_problems",
    "fit_to_limit",
    "heard_start",
    "heard_text",
    "judge",
    "later_clip_problems",
    "pick_moments",
    "reel_ms",
    "render",
    "render_for_judge",
    "split_start",
]
