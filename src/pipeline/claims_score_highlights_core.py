"""DB-free core of claims.score_highlights: what the decision model is shown,
what it is asked, and how its answers become one score.

No Hatchet dependency and no provider knowledge — the model arrives as a
DecisionModel, so the task and scripts/eval_claim_highlights.py run the same
code, and tests drive it with a fake.

The score: one yes/no decision — would the list misrepresent the discussion
without this claim? — and its probability of yes is the score. `DECISIONS`
is the extension point: a second decision asked here is returned beside it
in `decisions`, and `highlight_score` says how the score is read off them.

A run scores every claim or fails. A consumer stores what it gets and does
not ask again, so a partially scored result would leave claims unscored for
good; failing the run instead lets the engine retry the whole thing.
"""

import asyncio
import weakref
from typing import Dict, List, Mapping, Sequence

from src.api.schemas.claims_score_highlights_schema import (
    ClaimsScoreHighlightsInput,
    ClaimsScoreHighlightsResult,
    HighlightClaim,
    HighlightDocument,
    ScoredClaim,
)
from src.config.overrides import prompts
from src.decisions.base import Answer, DecisionError, DecisionModel, Question, YesNo, YesNoAnswer
from src.infrastructure.logger import get_logger
from src.infrastructure.spend_guard import SpendLimitExceeded, spend_guard

logger = get_logger(__name__)

_P = "claims_score_highlights."

# The decisions asked of every claim. One: whether the list would misrepresent
# the discussion without this claim. Chosen over three role decisions
# (position / clash / turn) and a holistic "is this pivotal?", which it
# outranked alone and in every combination (2026-10-05, see the rubric module).
ESSENTIAL = "essential"
DECISIONS = (ESSENTIAL,)

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
    """The rubric for `media_type` as decision questions, one yes/no each. The
    same questions are asked of every claim."""
    p = f"{_P}{media_type}."
    return {
        decision: YesNo(
            instructions=prompts.get(p + decision),
            yes=prompts.get(f"{p}{decision}.yes"),
            no=prompts.get(f"{p}{decision}.no"),
        )
        for decision in DECISIONS
    }


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


async def decide_claims(
    input: ClaimsScoreHighlightsInput,
    decider: DecisionModel,
    model: str,
) -> List[Mapping[str, Answer]]:
    """One decision request per claim, a few at a time, with the answers
    returned in claim order. All or nothing: the first claim that fails stops
    the rest (a refused key or an unknown model would fail every claim the
    same way, and a rate limit that outlasted its retries is a reason to run
    again later), and its DecisionError is raised for the run."""
    questions = build_questions(input.media_type)
    slots = _slots()

    async def decide(claim_index: int) -> Mapping[str, Answer]:
        async with slots:
            spend_guard.check_and_record(decider.provider)
            try:
                return await decider.decide(build_content(input, claim_index), questions, model=model)
            except DecisionError as e:
                logger.warning(
                    f"claims.score_highlights: claim {claim_index} failed "
                    f"({'retryable' if e.retryable else 'not retryable'}): {e}"
                )
                raise

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(decide(i)) for i in range(len(input.claims))]
    except* (DecisionError, SpendLimitExceeded) as failures:
        # The group cancelled the other requests; the run fails on the first failure.
        raise failures.exceptions[0] from None
    return [task.result() for task in tasks]


def highlight_score(decisions: Mapping[str, float]) -> float:
    """The probability that the list would misrepresent the discussion without
    this claim. Any other decision present is reported, not aggregated."""
    return min(1.0, max(0.0, decisions[ESSENTIAL]))


def score_claim(index: int, claim: HighlightClaim, answers: Mapping[str, Answer]) -> ScoredClaim:
    if not all(isinstance(answers.get(decision), YesNoAnswer) for decision in DECISIONS):
        raise DecisionError("decision answers do not match the highlight questions")
    decisions = {decision: answers[decision].probability for decision in DECISIONS}
    return ScoredClaim(
        index=index,
        id=claim.id,
        text=claim.text,
        score=highlight_score(decisions),
        decisions=decisions,
    )


def assemble_result(
    input: ClaimsScoreHighlightsInput,
    answers: Sequence[Mapping[str, Answer]],
    provider: str,
    model_used: str,
) -> ClaimsScoreHighlightsResult:
    """One ScoredClaim per input claim, in input order. An answer set of the
    wrong shape raises DecisionError and fails the run: it means the provider
    broke the DecisionModel contract, which no retry fixes and no claim should
    paper over with a null."""
    if len(answers) != len(input.claims):
        raise DecisionError(f"{len(answers)} answer sets for {len(input.claims)} claims")
    scored = [score_claim(index, claim, answer) for index, (claim, answer) in enumerate(zip(input.claims, answers))]
    return ClaimsScoreHighlightsResult(
        claims=scored,
        claims_scored=len(scored),
        provider=provider,
        model_used=model_used,
    )
