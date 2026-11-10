"""DB-free core of claims.score_clips: what the decision model is shown, what it
is asked, and how its answers become one judgement per clip.

Built on claims_score_highlights_core: the same request runner (bounded
concurrency, all or nothing, spend guard), transcript rendering, percentiles and
glitch tests, so the two tasks behave the same under load and on a bad answer.

Per clip, one request with the whole transcript and every candidate clip in
view: three yes/no questions (own_side, standalone, complete) and a choice of
which other speaker's clip it replies to (or none, or elsewhere). Then the
opener question: a choice over all the clips, asked OPENER_ROUNDS times in
shuffled orders and averaged, so a clip's probability does not depend on where
it sat in the list. A clip's `hook` is its percentile on that average.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from src.api.schemas.claims_score_clips_schema import (
    ClaimsScoreClipsInput,
    ClaimsScoreClipsResult,
    ScoreClip,
    ScoredClip,
)
from src.config.overrides import prompts
from src.decisions.base import Answer, Choice, ChoiceAnswer, DecisionError, DecisionModel, Question, YesNo, YesNoAnswer
from src.pipeline.claims_score_highlights_core import (
    Request,
    _flat,
    _header,
    _same,
    percentiles,
    render_transcript,
    run_requests,
)

TASK = "claims.score_clips"
_P = "claims_score_clips."

OWN_SIDE = "own_side"
STANDALONE = "standalone"
COMPLETE = "complete"
YES_NO = (OWN_SIDE, STANDALONE, COMPLETE)
RESPONDS_TO = "responds_to"
RESPONDS_NONE = "none"
RESPONDS_ELSEWHERE = "elsewhere"
OPENER = "opener"
ANSWERS_ELSEWHERE = "answers_elsewhere"
# Like claims.score_highlights' moment question: three orders average out most
# of a model's lean toward a position in the list.
OPENER_ROUNDS = 3


def clip_label(clip: ScoreClip, index: int) -> str:
    """A clip as the model sees it in a list: its number, speaker, side, length, then its words."""
    who = clip.speaker + (f", {clip.side}" if clip.side else "")
    length = f", {clip.duration_ms / 1000:.0f} s" if clip.duration_ms is not None else ""
    where = f", in passage [{clip.document_index}]" if clip.document_index is not None else ""
    return f"Clip {index + 1} ({who}{length}{where}): \"{clip.text.strip()}\""


def render_clips(input: ClaimsScoreClipsInput) -> str:
    return "\n".join(clip_label(clip, i) for i, clip in enumerate(input.clips))


def respond_options(input: ClaimsScoreClipsInput, target: int) -> List[tuple[str, int]]:
    """(option key, clip index) for the clips the reply question offers clip `target`: the
    other speaker's clips, in clip order, under neutral keys. A debater never replies to
    themselves, so their own clips are not offered."""
    speaker = input.clips[target].speaker
    others = [j for j, clip in enumerate(input.clips) if j != target and clip.speaker != speaker]
    return [(f"c{k + 1:02d}", j) for k, j in enumerate(others)]


def build_questions(input: ClaimsScoreClipsInput, target: int) -> Dict[str, Question]:
    """The per-clip questions for clip `target`."""
    p = f"{_P}{input.media_type}."
    questions: Dict[str, Question] = {
        name: YesNo(
            instructions=prompts.get(p + name),
            yes=prompts.get(f"{p}{name}.yes"),
            no=prompts.get(f"{p}{name}.no"),
        )
        for name in YES_NO
    }
    options: Dict[str, Optional[str]] = {
        key: clip_label(input.clips[j], j) for key, j in respond_options(input, target)
    }
    options[RESPONDS_NONE] = prompts.get(f"{p}{RESPONDS_TO}.{RESPONDS_NONE}")
    options[RESPONDS_ELSEWHERE] = prompts.get(f"{p}{RESPONDS_TO}.{RESPONDS_ELSEWHERE}")
    questions[RESPONDS_TO] = Choice(instructions=prompts.get(p + RESPONDS_TO), options=options)
    return questions


def build_content(input: ClaimsScoreClipsInput, target: int) -> Dict[str, str]:
    """What the model reads for one clip: the debate, every candidate clip, and the clip under
    test. Only the last section differs between the clips of a run."""
    return {
        **_header(input),
        "transcript": render_transcript(input.documents),
        "candidate_clips": render_clips(input),
        "clip_under_test": clip_label(input.clips[target], target),
    }


def clip_requests(input: ClaimsScoreClipsInput) -> List[Request]:
    return [(f"clip {i}", build_content(input, i), build_questions(input, i)) for i in range(len(input.clips))]


def shuffled_order(input: ClaimsScoreClipsInput, round: int) -> List[int]:
    """The clip indices in the order round `round` shows them, seeded by the round and the clip
    words, so a re-score differs from the first only when the model does."""
    digest = hashlib.sha256("\n".join([str(round), *(c.text for c in input.clips)]).encode()).digest()
    order = list(range(len(input.clips)))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


def opener_requests(input: ClaimsScoreClipsInput) -> List[Request]:
    """OPENER_ROUNDS requests, the same question over the clips in different orders. Options are
    keyed by the clip's input number, so answers map back whatever the order shown. A run of one
    clip has nothing to compare."""
    if len(input.clips) < 2:
        return []
    instructions = prompts.get(f"{_P}{input.media_type}.{OPENER}")
    content = {**_header(input), "transcript": render_transcript(input.documents)}
    requests: List[Request] = []
    for round in range(OPENER_ROUNDS):
        options = {str(i + 1): clip_label(input.clips[i], i) for i in shuffled_order(input, round)}
        requests.append((f"opener round {round}", content, {OPENER: Choice(instructions=instructions, options=options)}))
    return requests


def opener_rounds(input: ClaimsScoreClipsInput, answers: Sequence[Mapping[str, Answer]]) -> List[List[float]]:
    rounds: List[List[float]] = []
    for answer_set in answers:
        answer = answer_set.get(OPENER)
        if not isinstance(answer, ChoiceAnswer):
            raise DecisionError("decision answers do not match the opener question")
        try:
            rounds.append([answer.probabilities[str(i + 1)] for i in range(len(input.clips))])
        except KeyError as e:
            raise DecisionError(f"the opener answer has no probability for clip {e}") from e
    return rounds


@dataclass(frozen=True)
class ClipDecisions:
    yes_no: Dict[str, float]
    responds_to: Optional[int]
    answers_elsewhere: float


def clip_decisions(input: ClaimsScoreClipsInput, target: int, answers: Mapping[str, Answer]) -> ClipDecisions:
    """One clip's answers as numbers. The reply is the most probable option: another clip, or
    null for none and elsewhere."""
    yes_no: Dict[str, float] = {}
    for name in YES_NO:
        answer = answers.get(name)
        if not isinstance(answer, YesNoAnswer):
            raise DecisionError("decision answers do not match the clip questions")
        yes_no[name] = answer.probability
    reply = answers.get(RESPONDS_TO)
    if not isinstance(reply, ChoiceAnswer):
        raise DecisionError("decision answers do not match the reply question")
    keys = dict(respond_options(input, target))
    missing = [key for key in [*keys, RESPONDS_NONE, RESPONDS_ELSEWHERE] if key not in reply.probabilities]
    if missing:
        raise DecisionError(f"the reply answer for clip {target} has no probability for {missing}")
    best = max(reply.probabilities, key=lambda key: reply.probabilities[key])
    return ClipDecisions(
        yes_no=yes_no,
        responds_to=keys.get(best),
        answers_elsewhere=reply.probabilities[RESPONDS_ELSEWHERE],
    )


def check_for_glitch(clip_answers: Sequence[Mapping[str, Answer]], rounds: Sequence[Sequence[float]]) -> None:
    """Fail the run on the provider's identical-distribution glitch, as claims.score_highlights
    does: three or more clips with the same answers to every question, or an opener round that
    gives every clip the same probability. Retryable, so the engine asks again."""
    if len(clip_answers) >= 3:
        signatures = [
            [answers[name].probability for name in YES_NO if isinstance(answers.get(name), YesNoAnswer)]
            for answers in clip_answers
        ]
        if all(len(s) == len(YES_NO) for s in signatures) and all(_same(signatures[0], s) for s in signatures[1:]):
            raise DecisionError("every clip got the same answers: provider glitch", retryable=True)
    for round, probabilities in enumerate(rounds):
        if _flat(probabilities):
            raise DecisionError(f"opener round {round} gave every clip the same probability: provider glitch", retryable=True)


@dataclass(frozen=True)
class ClipsDecisions:
    clips: List[ClipDecisions]
    openers: List[float]  # per clip: mean probability over the opener rounds


async def decide_clips(input: ClaimsScoreClipsInput, decider: DecisionModel, model: str) -> ClipsDecisions:
    """Every request of a run in one all-or-nothing batch, checked for the provider glitch."""
    per_clip, opener = clip_requests(input), opener_requests(input)
    answers = await run_requests([*per_clip, *opener], decider, model, task=TASK)
    clip_answers = answers[: len(per_clip)]
    rounds = opener_rounds(input, answers[len(per_clip) :])
    check_for_glitch(clip_answers, rounds)
    decided = [clip_decisions(input, i, a) for i, a in enumerate(clip_answers)]
    # A lone clip is the only possible opener.
    openers = [sum(r[i] for r in rounds) / len(rounds) for i in range(len(input.clips))] if rounds else [1.0]
    return ClipsDecisions(decided, openers)


def assemble_result(
    input: ClaimsScoreClipsInput,
    decisions: ClipsDecisions,
    provider: str,
    model_used: str,
) -> ClaimsScoreClipsResult:
    if not (len(decisions.clips) == len(decisions.openers) == len(input.clips)):
        raise DecisionError(f"{len(decisions.clips)} answer sets for {len(input.clips)} clips")
    hooks = percentiles(decisions.openers)
    scored = [
        ScoredClip(
            index=index,
            id=clip.id,
            hook=min(1.0, max(0.0, hook)),
            decisions={
                **decided.yes_no,
                OPENER: opener,
                ANSWERS_ELSEWHERE: decided.answers_elsewhere,
            },
            responds_to=decided.responds_to,
        )
        for index, (clip, decided, opener, hook) in enumerate(
            zip(input.clips, decisions.clips, decisions.openers, hooks)
        )
    ]
    return ClaimsScoreClipsResult(clips=scored, clips_scored=len(scored), provider=provider, model_used=model_used)


__all__ = [
    "ANSWERS_ELSEWHERE",
    "COMPLETE",
    "OPENER",
    "OWN_SIDE",
    "RESPONDS_TO",
    "STANDALONE",
    "assemble_result",
    "build_content",
    "build_questions",
    "clip_requests",
    "decide_clips",
    "opener_requests",
    "respond_options",
]
