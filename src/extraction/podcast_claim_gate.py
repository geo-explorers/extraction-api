"""Context-dependence gate for podcast claims (podcast.extract_claims, gate step).

Two model views, because they catch different things:

* DETECT reads each claim ALONE (batches of ten, no transcript) — the way a
  reader of the knowledge graph meets it — and flags pronouns, bare definite
  references ("the deal"), unanchored relative time ("currently"), impersonal
  frames ("is considered") and pundit attribution frames.
* REPAIR reads the flagged claims WITH the transcript and the episode metadata
  in one call and rewrites each minimally, keeps it (the detector was wrong) or
  withholds it. Code then re-checks every rewrite: no lost name or number, a
  word budget, and every added identifier must occur in the transcript.

The gate is fail-open: a failed detect batch leaves its claims unflagged and a
failed repair call keeps the flagged claims as extracted, so a Gemini outage
degrades to today's behaviour instead of blocking the episode. The report says
how often that happened.

Pure with respect to storage: claims in, claims out, plus a report. Measured
numbers and the design history live in the prompt module's docstring.
"""

from __future__ import annotations

import asyncio
import datetime
import re
from typing import Callable, Dict, List, Literal, Optional, Tuple

from google.genai import types
from pydantic import BaseModel, Field

from src.config.overrides import llm, prompts
from src.extraction.premium_claim_extractor import PremiumClaimExtractor
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

DETECT_BATCH_SIZE = 10
DETECT_CONCURRENCY = 8
MAX_ADDED_WORDS = 10
MAX_CLAIM_WORDS = 40
# The repair prompt carries the transcript once; cap it so a pathological
# transcript cannot blow the request (clean transcripts are 20-120k chars).
TRANSCRIPT_CHAR_BUDGET = 400_000

Issue = Literal[
    "context_dependent",
    "relative_time",
    "impersonal_frame",
    "attribution_frame",
    "conversational_order",
    "none",
]


# ── Contracts ───────────────────────────────────────────────────────────────


class GateEpisodeMeta(BaseModel):
    """What the repair call knows about the episode besides the transcript."""

    podcast_name: str = ""
    title: str = ""
    description: str = ""
    air_date: Optional[str] = None  # ISO date or datetime; None = unknown
    people: List[str] = Field(default_factory=list)  # "Name (host)", "Name (guest)"

    def render(self) -> str:
        day = (self.air_date or "")[:10]
        try:
            pretty = datetime.date.fromisoformat(day).strftime("%B %-d, %Y")
        except ValueError:
            pretty = "unknown"
        return (
            f"PODCAST: {self.podcast_name or 'unknown'}\n"
            f"EPISODE: {self.title}\n"
            f"AIR DATE: {pretty}{f' ({day})' if day else ''}\n"
            f"PEOPLE: {', '.join(self.people) or 'unknown'}\n"
            f"DESCRIPTION: {self.description[:1500]}"
        )


class _DetectVerdict(BaseModel):
    index: int
    fires: bool
    issue: Issue = "none"
    rationale: str = ""


class _DetectResponse(BaseModel):
    verdicts: List[_DetectVerdict]


class _RepairDecision(BaseModel):
    index: int
    action: Literal["keep", "rewrite", "withhold"]
    issue: Issue = "none"
    text: str = ""
    reason: str = ""


class _RepairResponse(BaseModel):
    decisions: List[_RepairDecision]


class WithheldClaim(BaseModel):
    text: str
    topic: str
    issue: str
    reason: str


class GateReport(BaseModel):
    enabled: bool = True
    claims_in: int = 0
    flagged: int = 0
    repaired: int = 0
    kept_flagged: int = 0  # flagged, then kept: repair said keep, or its rewrite was rejected
    withheld: int = 0
    rewrites_rejected: int = 0
    errors: int = 0  # detect batches or repair calls that failed (fail-open)
    # Claims the thin-topic filter removed BEFORE the gate (set by the DAG step):
    # extraction output = claims_in + dropped_by_topic_filter.
    dropped_by_topic_filter: int = 0
    withheld_claims: List[WithheldClaim] = Field(default_factory=list)
    detect_model: str = ""
    repair_model: str = ""


# ── Rewrite acceptance (code, no model) ─────────────────────────────────────

_STOP = set(
    "a an the of to in on at for and or but with by from as is are was were be been being that this these "
    "those it its his her their they he she we you i not no than then so such into over under about after "
    "before during while which who whom whose what when where why how".split()
)
_TIME_WORDS = set(
    "monday tuesday wednesday thursday friday saturday sunday january february march april may june july "
    "august september october november december today yesterday tomorrow".split()
)
_CAP_RE = re.compile(r"\b[A-Z][\w'-]*")
_NUM_RE = re.compile(r"\d[\d,.%]*")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    """Straighten curly apostrophes so "Trump’s" and "Trump's" compare equal."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _base(token: str) -> str:
    """A capitalised token without its possessive and trailing punctuation: "Trump's," -> "Trump"."""
    return re.sub(r"'s$", "", token.strip(".,;:'\"")).strip(".,;:'\"")


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _cap_tokens(text: str) -> List[str]:
    """Every capitalised token, possessive stripped ("Trump's" -> "Trump"), sentence-initial included."""
    return [b for b in (_base(w) for w in _CAP_RE.findall(_norm(text))) if b]


def _num_tokens(text: str) -> List[str]:
    return [x.rstrip(",.") for x in _NUM_RE.findall(text)]


def _word_count(text: str) -> int:
    return len(text.split())


def material_words(material: str) -> set:
    """Lower-cased word set of the transcript + metadata: the only identifiers a rewrite may add."""
    return set(_WORD_RE.findall(_norm(material).lower()))


def accept_rewrite(
    original: str,
    rewritten: str,
    issue: str,
    words: set,
) -> Tuple[bool, str]:
    """Decide whether a rewrite may replace the original.

    Ported from news-worker's acceptRepair: the rewrite must keep the fact
    (content-word overlap), lose no capitalised name or number, stay inside the
    reading budget, and every name, year or figure it ADDS must be a word of
    the material (`material_words`). Comparisons ignore possessives and curly
    apostrophes, so "Trump's deal" -> "Trump announced the deal" keeps Trump.
    Two issue-specific allowances: an attribution repair may drop a speaker
    named in the material, a relative-time repair may drop a weekday or month.
    """
    original_n, text = _norm(original), _norm(rewritten or "").strip()
    if not text or text == original_n:
        return False, "unchanged"
    before, after = _content_words(original_n), _content_words(text)
    if before and len(before & after) / len(before) < 0.4:
        return False, "low overlap"
    orig_caps, new_caps = _cap_tokens(original_n), _cap_tokens(text)
    new_caps_lower = {w.lower() for w in new_caps}
    lost_caps = [w for w in orig_caps if w.lower() not in new_caps_lower]
    if issue == "attribution_frame":
        lost_caps = [w for w in lost_caps if w.lower() not in words]
    if issue == "relative_time":
        lost_caps = [w for w in lost_caps if w.lower() not in _TIME_WORDS]
    # Sentence-initial common words ("The", "In") are capitalised by position;
    # losing one is not losing a name.
    lost_caps = [w for w in lost_caps if w.lower() not in _STOP]
    lost_nums = [x for x in _num_tokens(original_n) if x not in text]
    if lost_caps or lost_nums:
        return False, "lost " + ", ".join((lost_caps + lost_nums)[:3])
    added = _word_count(text) - _word_count(original_n)
    if added > MAX_ADDED_WORDS:
        return False, f"adds {added} words, at most {MAX_ADDED_WORDS}"
    limit = max(MAX_CLAIM_WORDS, _word_count(original_n) + MAX_ADDED_WORDS)
    if _word_count(text) > limit:
        return False, f"{_word_count(text)} words, at most {limit}"
    had_caps = {w.lower() for w in orig_caps}
    had_nums = set(_num_tokens(original_n))
    new_names = [
        w for w in new_caps if w.lower() not in had_caps and w.lower() not in _STOP and w.lower() not in words
    ]
    new_nums = [
        x for x in _num_tokens(text)
        if x not in had_nums and not all(part in words for part in _WORD_RE.findall(x.lower()))
    ]
    if new_names or new_nums:
        return False, "not in material: " + ", ".join((new_names + new_nums)[:3])
    return True, ""


# ── The gate ────────────────────────────────────────────────────────────────


class PodcastClaimGate:
    """Detect -> repair -> accept, over one episode's claims_with_topics map."""

    def __init__(self, extractor: PremiumClaimExtractor, on_call: Optional[Callable[[], None]] = None):
        # Shares the extractor's Gemini client, timeout and app-level retries.
        # `on_call` runs before every model request (the task passes the spend
        # guard); if it raises, that request is skipped and the gate fails open.
        self._extractor = extractor
        self._on_call = on_call or (lambda: None)

    @property
    def detect_model(self) -> str:
        return llm.get("gemini_gate_detect_model")

    @property
    def repair_model(self) -> str:
        return llm.get("gemini_gate_repair_model")

    @property
    def temperature(self) -> float:
        return llm.get("gemini_gate_temperature")

    @staticmethod
    def _by_index(indices_items: List[Tuple[int, object]], expected: set, what: str) -> Dict[int, object]:
        """Map model output by index; repair a 1-based reply and warn on anything else off."""
        by = {i: v for i, v in indices_items}
        if set(by) == expected:
            return by
        if set(by) == {i + 1 for i in expected}:
            logger.warning(f"claim gate {what}: indices came back 1-based, shifting")
            return {i - 1: v for i, v in by.items()}
        logger.warning(
            f"claim gate {what}: expected indices {sorted(expected)[:5]}... got {sorted(by)[:5]}... "
            f"(missing {len(expected - set(by))}, unexpected {len(set(by) - expected)})"
        )
        return by

    async def _detect_batch(self, batch: List[str], offset: int) -> List[Optional[_DetectVerdict]]:
        prompt = prompts.get("podcast_gate.detect").format(
            claims="\n".join(f"{i}. {text}" for i, text in enumerate(batch))
        )
        self._on_call()
        raw = await self._extractor._call_gemini(
            prompt=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=_DetectResponse,
                # Recall collapses without thinking (7.4% vs 3.7% residual);
                # pin dynamic thinking so a model override cannot turn it off.
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            ),
            step_name="claim gate detect",
            model=self.detect_model,
        )
        parsed = _DetectResponse.model_validate_json(raw).verdicts
        by_index = self._by_index([(v.index, v) for v in parsed], set(range(len(batch))), "detect")
        out: List[Optional[_DetectVerdict]] = []
        for i in range(len(batch)):
            v = by_index.get(i)
            out.append(
                _DetectVerdict(index=offset + i, fires=v.fires, issue=v.issue, rationale=v.rationale)
                if v
                else _DetectVerdict(index=offset + i, fires=False)
            )
        return out

    async def detect(self, texts: List[str]) -> Tuple[List[Optional[_DetectVerdict]], int]:
        """Verdict per claim (None when its batch failed) and the failed-batch count."""
        batches = [(i, texts[i : i + DETECT_BATCH_SIZE]) for i in range(0, len(texts), DETECT_BATCH_SIZE)]
        sem = asyncio.Semaphore(DETECT_CONCURRENCY)

        async def one(offset: int, batch: List[str]):
            async with sem:
                try:
                    return await self._detect_batch(batch, offset)
                except Exception as e:  # fail-open per batch
                    logger.warning(f"claim gate detect batch at {offset} failed: {type(e).__name__}: {e}")
                    return None

        results = await asyncio.gather(*(one(o, b) for o, b in batches))
        verdicts: List[Optional[_DetectVerdict]] = []
        failed = 0
        for (_, batch), res in zip(batches, results):
            if res is None:
                failed += 1
                verdicts.extend([None] * len(batch))
            else:
                verdicts.extend(res)
        return verdicts, failed

    async def repair(
        self,
        flagged: List[Tuple[int, str, _DetectVerdict]],
        transcript: str,
        meta: GateEpisodeMeta,
    ) -> Dict[int, _RepairDecision]:
        """One grounded call for all flagged claims of the episode; {} when it failed."""
        try:
            prompt = prompts.get("podcast_gate.repair").format(
                max_added_words=MAX_ADDED_WORDS,
                max_claim_words=MAX_CLAIM_WORDS,
                metadata=meta.render(),
                claims="\n".join(
                    f"{i}. {text}\n   verifier ({v.issue}): {v.rationale}" for i, text, v in flagged
                ),
                transcript=transcript[:TRANSCRIPT_CHAR_BUDGET],
            )
            self._on_call()
            raw = await self._extractor._call_gemini(
                prompt=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json",
                    response_schema=_RepairResponse,
                ),
                step_name="claim gate repair",
                model=self.repair_model,
            )
            parsed = _RepairResponse.model_validate_json(raw).decisions
            return self._by_index([(d.index, d) for d in parsed], {i for i, _, _ in flagged}, "repair")
        except Exception as e:  # fail-open: flagged claims stay as extracted
            logger.warning(f"claim gate repair failed, keeping flagged claims: {type(e).__name__}: {e}")
            return {}

    async def apply(
        self,
        claims_with_topics: Dict[str, List[str]],
        transcript: str,
        meta: GateEpisodeMeta,
    ) -> Tuple[Dict[str, List[str]], GateReport]:
        """Gate one episode. Returns the new topic -> claims map (same topic order,
        same claim order, withheld claims removed) and the report."""
        items: List[Tuple[str, str]] = [(t, c) for t, cs in claims_with_topics.items() for c in cs]
        report = GateReport(
            claims_in=len(items), detect_model=self.detect_model, repair_model=self.repair_model
        )
        if not items:
            return dict(claims_with_topics), report

        verdicts, failed_batches = await self.detect([c for _, c in items])
        report.errors += failed_batches
        flagged = [(i, items[i][1], v) for i, v in enumerate(verdicts) if v is not None and v.fires]
        report.flagged = len(flagged)

        decisions: Dict[int, _RepairDecision] = {}
        if flagged:
            decisions = await self.repair(flagged, transcript, meta)
            if not decisions:
                report.errors += 1
        words = material_words(meta.render() + "\n" + transcript)

        out: Dict[str, List[str]] = {t: [] for t in claims_with_topics}
        flagged_idx = {i for i, _, _ in flagged}
        for i, (topic, text) in enumerate(items):
            if i not in flagged_idx:
                out[topic].append(text)
                continue
            d = decisions.get(i)
            if d is None or d.action == "keep":
                report.kept_flagged += 1
                out[topic].append(text)
            elif d.action == "withhold":
                report.withheld += 1
                report.withheld_claims.append(
                    WithheldClaim(text=text, topic=topic, issue=d.issue, reason=d.reason)
                )
            else:
                ok, why = accept_rewrite(text, d.text, d.issue, words)
                if ok:
                    report.repaired += 1
                    out[topic].append(d.text.strip())
                else:
                    report.rewrites_rejected += 1
                    report.kept_flagged += 1
                    out[topic].append(text)
                    logger.info(f"claim gate rejected rewrite ({why}): {text[:80]!r}")
        if not any(out.values()) and report.withheld:
            # Every claim withheld: the episode publishes nothing. Strictness is the
            # point; the caller marks the episode as extracted so it is not retried.
            logger.warning(f"claim gate: all {report.withheld} claims withheld; episode publishes no claims")
        out = {t: cs for t, cs in out.items() if cs}
        logger.info(
            f"claim gate: {report.claims_in} in, {report.flagged} flagged, {report.repaired} repaired, "
            f"{report.withheld} withheld, {report.kept_flagged} kept after flag, "
            f"{report.rewrites_rejected} rewrites rejected, {report.errors} errors"
        )
        return out, report
