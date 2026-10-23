"""DB-free core of claims.score_highlights: what the decision model is shown,
what it is asked, and how its answers become one score.

No Hatchet dependency and no provider knowledge — the model arrives as a
DecisionModel, so the task and scripts/eval_claim_highlights.py run the same
code, and tests drive it with a fake.

The score: one yes/no decision — would the list misrepresent the discussion
without this claim? — and its probability of yes is the score. Three more
decisions ride in the same request (the content is shared; only their own
wording is added, about half again the input tokens): relevance, quality
and controversy, each a four-level scale read as a position from 0 to 1.
They are returned beside the score in `decisions`; `highlight_score` says
how the score is read off them, and leaves them alone.

Two more questions are asked once per debate, not once per claim, because
they compare claims with each other. The moment question — which claim would a
viewer most want to watch? — is a choice over the whole claim list, asked
MOMENT_ROUNDS times with the list shuffled and averaged, so one claim's
probability does not depend on where it sat in the list. The claim's moment
score blends its percentile on that with its percentile on controversy within
the debate (MOMENT_WEIGHT / CONTROVERSY_WEIGHT). The repeat question — do these
two claims make the same point? — is a yes/no per pair, asked without the
transcript; claims joined by a yes share a repeat-group id.

A run whose answers show the provider's identical-distribution glitch fails
(check_for_glitch) instead of publishing scores that carry no information.

A run scores every claim or fails. A consumer stores what it gets and does
not ask again, so a partially scored result would leave claims unscored for
good; failing the run instead lets the engine retry the whole thing.
"""

import asyncio
import hashlib
import random
import weakref
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from src.api.schemas.claims_score_highlights_schema import (
    ClaimsScoreHighlightsInput,
    ClaimsScoreHighlightsResult,
    HighlightClaim,
    HighlightDocument,
    ScoredClaim,
)
from src.config.overrides import prompts
from src.decisions.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionError,
    DecisionModel,
    Question,
    Scale,
    ScaleAnswer,
    YesNo,
    YesNoAnswer,
)
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import SpendLimitExceeded, spend_guard

logger = get_logger(__name__)

_P = "claims_score_highlights."

# The decisions asked of every claim. The score: whether the list would
# misrepresent the discussion without this claim. Chosen over three role
# decisions (position / clash / turn) and a holistic "is this pivotal?", which
# it outranked alone and in every combination (2026-10-05, see the rubric
# module).
ESSENTIAL = "essential"
# The axes: scales reported beside the score. Their level names, lowest first,
# key the rubric texts in the prompt registry (`<media_type>.<axis>.<level>`);
# the model sees the texts in this order and never the names.
RELEVANCE = "relevance"
QUALITY = "quality"
CONTROVERSY = "controversy"
AXIS_LEVELS: Dict[str, Sequence[str]] = {
    RELEVANCE: ("unrelated", "tangential", "connected", "direct"),
    QUALITY: ("misleading", "unusable", "flawed", "clean"),
    CONTROVERSY: ("none", "mild", "debatable", "divisive"),
}
AXES = tuple(AXIS_LEVELS)
DECISIONS = (ESSENTIAL, *AXES)

# The moment question: how many times the claim list is shown, each time in a
# different order, and the answers averaged. One ordering favours whatever the
# model leans toward at a given position; three orderings average most of that out.
MOMENT_ROUNDS = 3
MOMENT = "moment"
# The moment score: a claim's percentile on the moment question and on the
# controversy axis within its debate. 0.6 was the weight chosen in 30 of 30
# leave-one-debate-out folds of the 21 October proposal.
MOMENT_WEIGHT = 0.6
CONTROVERSY_WEIGHT = 0.4

# The repeat question: a pair is the same point at or above this probability of
# yes. Not tuned: the labelled set has no repeat labels (see the docs).
REPEAT_THRESHOLD = 0.5
# Questions per request the provider accepts is 128; stay a little under.
PAIRS_PER_REQUEST = 120

# Two probabilities closer than this are the same number for the glitch check.
GLITCH_TOLERANCE = 1e-6

# Requests in flight per worker process, across every run it is executing at
# once — a per-run bound would multiply by the worker's slots and a burst of runs
# would turn the provider's 10 requests/second into rate-limit failures. Six in
# flight at the ~1 s a request takes stays under that limit on its own; the
# task's engine-level `concurrency` caps how many runs share the bound.
DECISION_CONCURRENCY = 6

# One semaphore per event loop (asyncio primitives bind to the loop they are
# first used on): the worker has one loop, tests have one per test.
_slots_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _slots_by_loop.get(loop)
    if slots is None:
        slots = asyncio.Semaphore(DECISION_CONCURRENCY)
        _slots_by_loop[loop] = slots
    return slots

UNKNOWN_SPEAKER = "Unknown speaker"


def build_questions(media_type: str) -> Dict[str, Question]:
    """The rubric for `media_type` as decision questions: the yes/no score
    decision, then one scale per axis. The same questions are asked of every
    claim."""
    p = f"{_P}{media_type}."
    questions: Dict[str, Question] = {
        ESSENTIAL: YesNo(
            instructions=prompts.get(p + ESSENTIAL),
            yes=prompts.get(f"{p}{ESSENTIAL}.yes"),
            no=prompts.get(f"{p}{ESSENTIAL}.no"),
        )
    }
    for axis, levels in AXIS_LEVELS.items():
        questions[axis] = Scale(
            instructions=prompts.get(p + axis),
            levels=[prompts.get(f"{p}{axis}.{level}") for level in levels],
        )
    return questions


def _speaker(document: HighlightDocument) -> str:
    return (document.speaker or "").strip() or UNKNOWN_SPEAKER


def attribution(claim: HighlightClaim, documents: Sequence[HighlightDocument]) -> str:
    """Who the claim is attributed to: the speakers of its documents, in order
    of first appearance. A claim citing no documents is unattributed."""
    speakers: List[str] = []
    for index in claim.document_indices:
        speaker = _speaker(documents[index])
        if speaker not in speakers:
            speakers.append(speaker)
    return " and ".join(speakers) if speakers else "unattributed"


def render_transcript(documents: Sequence[HighlightDocument]) -> str:
    return "\n\n".join(f"[{i}] {_speaker(d)}: {d.content.strip()}" for i, d in enumerate(documents))


def render_claims(input: ClaimsScoreHighlightsInput) -> str:
    return "\n".join(
        f"{i + 1}. ({attribution(claim, input.documents)}) {claim.text}" for i, claim in enumerate(input.claims)
    )


def build_content(input: ClaimsScoreHighlightsInput, claim_index: int) -> Dict[str, str]:
    """What the model reads for one claim: the whole discussion, every
    extracted claim, and the claim under test with its attribution. Only the
    last section differs between the claims of a run."""
    claim = input.claims[claim_index]
    content: Dict[str, str] = {}
    if input.title:
        content["title"] = input.title
    if input.context:
        content["context"] = input.context
    content["transcript"] = render_transcript(input.documents)
    content["extracted_claims"] = render_claims(input)
    passages = ", ".join(f"[{i}]" for i in claim.document_indices)
    drawn_from = f", drawn from passage {passages}" if passages else ""
    content["claim_under_test"] = (
        f"Claim {claim_index + 1}, attributed to {attribution(claim, input.documents)}{drawn_from}: {claim.text}"
    )
    return content


# One request to the decision model: a label for the log, what it reads, what it is asked.
Request = tuple[str, Mapping[str, str], Mapping[str, Question]]


async def run_requests(requests: Sequence[Request], decider: DecisionModel, model: str) -> List[Mapping[str, Answer]]:
    """The answers to `requests`, a few at a time, in request order. All or
    nothing: the first request that fails stops the rest (a refused key or an
    unknown model would fail every request the same way, and a rate limit that
    outlasted its retries is a reason to run again later), and its
    DecisionError is raised for the run."""
    slots = _slots()

    async def ask(label: str, content: Mapping[str, str], questions: Mapping[str, Question]) -> Mapping[str, Answer]:
        async with slots:
            spend_guard.check_and_record(decider.provider)
            try:
                return await decider.decide(content, questions, model=model)
            except DecisionError as e:
                logger.warning(
                    f"claims.score_highlights: {label} failed "
                    f"({'retryable' if e.retryable else 'not retryable'}): {e}"
                )
                raise

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(ask(*request)) for request in requests]
    except* (DecisionError, SpendLimitExceeded) as failures:
        # The group cancelled the other requests; the run fails on the first failure.
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


def claim_requests(input: ClaimsScoreHighlightsInput) -> List[Request]:
    """One request per claim: the score decision and the three axes."""
    questions = build_questions(input.media_type)
    return [(f"claim {i}", build_content(input, i), questions) for i in range(len(input.claims))]


async def decide_claims(
    input: ClaimsScoreHighlightsInput,
    decider: DecisionModel,
    model: str,
) -> List[Mapping[str, Answer]]:
    """One decision request per claim, with the answers returned in claim order."""
    return await run_requests(claim_requests(input), decider, model)


def _header(input: ClaimsScoreHighlightsInput) -> Dict[str, str]:
    content: Dict[str, str] = {}
    if input.title:
        content["title"] = input.title
    if input.context:
        content["context"] = input.context
    return content


def shuffled_order(input: ClaimsScoreHighlightsInput, round: int) -> List[int]:
    """The claim indices in the order round `round` shows them. Seeded by the
    round and the claim texts, so the same debate is shuffled the same way every
    time it is scored: a re-score differs from the first only when the model
    does."""
    digest = hashlib.sha256("\n".join([str(round), *(c.text for c in input.claims)]).encode()).digest()
    order = list(range(len(input.claims)))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


def moment_requests(input: ClaimsScoreHighlightsInput) -> List[Request]:
    """MOMENT_ROUNDS requests, each the same question over the claims in a
    different order. The options are named by the claim's number in the input
    list, so the answers map back whatever the order shown."""
    instructions = prompts.get(f"{_P}{input.media_type}.{MOMENT}")
    content = {**_header(input), "transcript": render_transcript(input.documents)}
    requests: List[Request] = []
    for round in range(MOMENT_ROUNDS):
        options = {
            str(i + 1): f"({attribution(input.claims[i], input.documents)}) {input.claims[i].text}"
            for i in shuffled_order(input, round)
        }
        requests.append((f"moment round {round}", content, {MOMENT: Choice(instructions=instructions, options=options)}))
    return requests


def moment_rounds(input: ClaimsScoreHighlightsInput, answers: Sequence[Mapping[str, Answer]]) -> List[List[float]]:
    """Each round's probability per claim, in claim order."""
    rounds: List[List[float]] = []
    for answer_set in answers:
        answer = answer_set.get(MOMENT)
        if not isinstance(answer, ChoiceAnswer):
            raise DecisionError("decision answers do not match the moment question")
        try:
            rounds.append([answer.probabilities[str(i + 1)] for i in range(len(input.claims))])
        except KeyError as e:
            raise DecisionError(f"the moment answer has no probability for claim {e}") from e
    return rounds


def pairs(count: int) -> List[tuple[int, int]]:
    return [(i, j) for i in range(count) for j in range(i + 1, count)]


def repeat_requests(input: ClaimsScoreHighlightsInput) -> List[Request]:
    """The repeat question for every pair of claims, PAIRS_PER_REQUEST to a
    request. The transcript is left out on purpose: the question is about two
    texts, and with the transcript every request would cost as much as a
    claim's."""
    p = f"{_P}{input.media_type}.repeat"
    content = {**_header(input), "extracted_claims": render_claims(input)}
    instructions, yes, no = prompts.get(p), prompts.get(f"{p}.yes"), prompts.get(f"{p}.no")
    everything = pairs(len(input.claims))
    requests: List[Request] = []
    for start in range(0, len(everything), PAIRS_PER_REQUEST):
        batch = everything[start : start + PAIRS_PER_REQUEST]
        questions: Dict[str, Question] = {
            f"{i}_{j}": YesNo(
                instructions=f"{instructions}\n\nClaim A: {input.claims[i].text}\nClaim B: {input.claims[j].text}",
                yes=yes,
                no=no,
            )
            for i, j in batch
        }
        requests.append((f"repeat pairs {start}-{start + len(batch) - 1}", content, questions))
    return requests


def repeat_groups(count: int, answers: Sequence[Mapping[str, Answer]]) -> List[int]:
    """The repeat-group id of each claim: the lowest claim index among the
    claims joined to it by a pair answered yes, directly or through others. A
    claim with no repeat is its own group."""
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = set()
    for answer_set in answers:
        for name, answer in answer_set.items():
            if not isinstance(answer, YesNoAnswer):
                raise DecisionError("decision answers do not match the repeat question")
            i, j = (int(part) for part in name.split("_"))
            seen.add((i, j))
            if answer.probability >= REPEAT_THRESHOLD:
                low, high = sorted((find(i), find(j)))
                parent[high] = low
    if seen != set(pairs(count)):
        raise DecisionError("the repeat answers do not cover every pair of claims")
    return [find(i) for i in range(count)]


def percentiles(values: Sequence[float]) -> List[float]:
    """Each value's place among the others, 0 for the lowest to 1 for the
    highest, ties sharing the middle of their places. A lone value is 0.5."""
    if len(values) == 1:
        return [0.5]
    return [
        (sum(other < value for other in values) + 0.5 * (sum(other == value for other in values) - 1))
        / (len(values) - 1)
        for value in values
    ]


def moment_scores(moments: Sequence[float], controversies: Sequence[float]) -> List[float]:
    """The ranking score: MOMENT_WEIGHT x the moment percentile +
    CONTROVERSY_WEIGHT x the controversy percentile, within the debate."""
    return [
        MOMENT_WEIGHT * m + CONTROVERSY_WEIGHT * c
        for m, c in zip(percentiles(moments), percentiles(controversies))
    ]


def _same(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= GLITCH_TOLERANCE for x, y in zip(a, b))


def check_for_glitch(
    claim_answers: Sequence[Mapping[str, Answer]],
    rounds: Sequence[Sequence[float]],
) -> None:
    """Fail the run on the provider's identical-distribution glitch (three live
    debates on 12 October got one distribution back for all three score
    questions). Three signs, each impossible in an answer that read the content:

    - a claim whose axis scales all have the same distribution over their
      levels, within GLITCH_TOLERANCE — three differently worded rubrics do not
      agree to six decimals;
    - in a debate of three or more claims, every decision identical across all
      the claims — the answers do not depend on the claim;
    - a moment round that gives every claim the same probability.

    The error is retryable: the engine runs the task again and a later request
    usually answers properly. Nothing is returned, so nothing is published."""
    scales = [[a.probabilities for axis in AXES if isinstance(a := answers.get(axis), ScaleAnswer)] for answers in claim_answers]
    for index, vectors in enumerate(scales):
        if len(vectors) == len(AXES) and all(_same(vectors[0], v) for v in vectors[1:]):
            raise DecisionError(
                f"claim {index} got one identical distribution for every axis question: provider glitch",
                retryable=True,
            )
    if len(claim_answers) >= 3:
        signatures = [
            [answers[ESSENTIAL].probability, *(p for v in vectors for p in v)]
            for answers, vectors in zip(claim_answers, scales)
            if isinstance(answers.get(ESSENTIAL), YesNoAnswer)
        ]
        if len(signatures) == len(claim_answers) and all(_same(signatures[0], s) for s in signatures[1:]):
            raise DecisionError("every claim got the same answers: provider glitch", retryable=True)
    for round, probabilities in enumerate(rounds):
        if len(probabilities) >= 3 and max(probabilities) - min(probabilities) <= GLITCH_TOLERANCE:
            raise DecisionError(f"moment round {round} gave every claim the same probability: provider glitch", retryable=True)


@dataclass(frozen=True)
class DebateDecisions:
    """Every answer a run collected."""

    claims: List[Mapping[str, Answer]]  # per claim, in claim order
    moments: List[float]  # per claim: its mean probability over the moment rounds
    groups: List[int]  # per claim: its repeat-group id


async def decide_debate(input: ClaimsScoreHighlightsInput, decider: DecisionModel, model: str) -> DebateDecisions:
    """Every request of a run — the claims, the moment rounds, the repeat
    pairs — in one all-or-nothing batch, checked for the provider glitch."""
    per_claim, moment, repeat = claim_requests(input), moment_requests(input), repeat_requests(input)
    answers = await run_requests([*per_claim, *moment, *repeat], decider, model)
    claim_answers = answers[: len(per_claim)]
    rounds = moment_rounds(input, answers[len(per_claim) : len(per_claim) + len(moment)])
    check_for_glitch(claim_answers, rounds)
    groups = repeat_groups(len(input.claims), answers[len(per_claim) + len(moment) :])
    moments = [sum(r[i] for r in rounds) / len(rounds) for i in range(len(input.claims))]
    return DebateDecisions(claim_answers, moments, groups)


def highlight_score(decisions: Mapping[str, float]) -> float:
    """The probability that the list would misrepresent the discussion without
    this claim. Any other decision present is reported, not aggregated."""
    return min(1.0, max(0.0, decisions[ESSENTIAL]))


def claim_decisions(answers: Mapping[str, Answer]) -> Dict[str, float]:
    """The per-claim decisions as numbers, 0 to 1: the score decision's
    probability of yes, then each axis's position."""
    essential = answers.get(ESSENTIAL)
    axes = {axis: answer for axis in AXES if isinstance(answer := answers.get(axis), ScaleAnswer)}
    if not isinstance(essential, YesNoAnswer) or len(axes) != len(AXES):
        raise DecisionError("decision answers do not match the highlight questions")
    return {ESSENTIAL: essential.probability, **{axis: answer.score for axis, answer in axes.items()}}


def assemble_result(
    input: ClaimsScoreHighlightsInput,
    decisions: DebateDecisions,
    provider: str,
    model_used: str,
) -> ClaimsScoreHighlightsResult:
    """One ScoredClaim per input claim, in input order. An answer set of the
    wrong shape raises DecisionError and fails the run: it means the provider
    broke the DecisionModel contract, which no retry fixes and no claim should
    paper over with a null."""
    answers = decisions.claims
    if not (len(answers) == len(decisions.moments) == len(decisions.groups) == len(input.claims)):
        raise DecisionError(f"{len(answers)} answer sets for {len(input.claims)} claims")
    per_claim = [claim_decisions(answer) for answer in answers]
    ranking = moment_scores(decisions.moments, [d[CONTROVERSY] for d in per_claim])
    scored = [
        ScoredClaim(
            index=index,
            id=claim.id,
            text=claim.text,
            score=highlight_score(numbers),
            decisions={**numbers, MOMENT: moment},
            moment_score=rank,
            repeat_group=group,
        )
        for index, (claim, numbers, moment, rank, group) in enumerate(
            zip(input.claims, per_claim, decisions.moments, ranking, decisions.groups)
        )
    ]
    return ClaimsScoreHighlightsResult(
        claims=scored,
        claims_scored=len(scored),
        provider=provider,
        model_used=model_used,
    )
