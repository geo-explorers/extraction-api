"""Collection review — merge claims that say the same thing, keep every block
at two or more claims, drop nothing.

Why a separate call and not a prompt rule: the extraction prompt already
carries an anti-filler rule and a pairwise paraphrase test, and the model
skips them. Measured on 30 prod stories (2026-09-22, independent Opus judge),
8% of extracted claims restated or split another claim in their collection
and 9-12% of two-claim collections held such a pair; a prompt rewrite alone
took the claims to 5.4% and left the two-claim figure at 9%. One narrow call
that only looks for pairs, plus code that refuses anything unsafe, took the
two-claim figure to ~3% (thinking high) / ~5% (medium).

The model proposes; the code decides. Every merge must keep every name and
number of the claims it replaces and add none, stay within MAX_CLAIM_WORDS and
grow by at most MAX_MERGE_GROWTH over its longest part; a composed sentence
must also pass a source check (a merge of two facts can imply a relation the
sources never state — 11% of composed merges did in the probe). A refused
merge stays local: its parts go back where extraction put them. A claim is
never dropped, only merged or moved. If the result would still leave a
one-claim collection (the team rule), the whole review is discarded and the
input returned as it was — a skipped review is today's behaviour.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from google import genai
from google.genai import types

from src.api.schemas.news_claim_extract_schema import (
  ExtractedClaim,
  ExtractedCollection,
  ExtractedQuote,
  NewsArticleSource,
)
from src.api.schemas.news_collection_review_schema import (
  CollectionReviewReport,
  NewsCollectionReviewResponse,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Pinned in code, not env: the numbers in the module docstring were measured
# on this model, and this task runs on the WORKER service, whose env is not
# the API service's (GEMINI_NEWS_CLAIM_MODEL there never reached the worker).
REVIEW_MODEL = "gemini-3.5-flash"
REVIEW_TEMPERATURE = 0.1
_REQUEST_TIMEOUT_MS = 180_000

# The reading limits every published claim already lives under (news-worker's
# repair guard uses the same 40 / +8): a merged claim is a claim like any other.
MAX_CLAIM_WORDS = 40
MAX_MERGE_GROWTH = 8

# Source bodies the source check sees, split across sources.
SOURCE_CHECK_BUDGET = 60_000

REVIEW_PROMPT = """You review the claims of one news story after extraction. Each claim is published ALONE on a knowledge graph, and claims are grouped into named collections shown as blocks. A reader who sees two claims saying the same thing in one block concludes the second was written to fill it. Your only job is to remove that.

HEADLINE: {headline}

CLAIMS (index. text):
{claims}

COLLECTIONS (name: claim indices):
{collections}

Inside each collection, compare every pair of claims and ask: does the reader learn a NEW fact from the second after reading the first?
- Same fact in other words, or with one detail moved → keep the more specific claim; the other is merged into it (no new text needed).
- One fact cut into pieces that share a frame (clauses of one rule, figures of one poll, one measurement at two time points) → write ONE claim carrying every piece. It must be at most {max_words} words and at most {max_growth} words longer than the longest piece. If not everything fits, merge only the two most alike; if even two cannot fit, they are different facts — leave them.
- A merged claim keeps every name, number and date of the claims it replaces and adds nothing.
- Claims that report DIFFERENT facts are never merged, however similar their wording.
- If a collection is left with one claim, move that claim into the existing collection whose name best covers it. First check that collection does not already state the fact; if it does, merge the moved claim into that claim instead. Never leave a one-claim collection and never drop a claim.
- Collections keep their names. Do not add collections.

Return JSON only:
{{"merges": [{{"keep": <index>, "drop": [<indices>], "text": "<merged text, or null to keep the kept claim verbatim>"}}],
  "collections": [{{"name": "<existing name>", "claims": [<final indices, using the kept index for merged claims>]}}]}}"""

SOURCE_CHECK_PROMPT = """Check each sentence below against the story's source articles. Each sentence was produced by MERGING two or three extracted claims into one.

SENTENCES:
{sentences}

SOURCES:
{sources}

For each sentence:
- grade: "supported" (every fact, name, number and date is stated in the sources; paraphrase is fine), "partly" (the core is there but a detail is not, or is stated more strongly), "unsupported" (not stated or contradicted).
- false_link: true if the sentence joins facts in a way that implies a relation (cause, sequence, same event, same speaker) that the sources do not state.

Return JSON only: {{"checks": [{{"index": <sentence number>, "grade": "supported"|"partly"|"unsupported", "false_link": true|false}}]}}"""


# ── Word-level guards (pure) ───────────────────────────────────────────────

_CAP_STOP = {
  "The", "A", "An", "In", "On", "At", "By", "For", "When", "After", "Before",
  "Both", "As", "During", "Under", "Over", "While", "Following",
}
_NUMBER_WORDS = {
  "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
  "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
  "twelve": "12", "fifteen": "15", "eighteen": "18", "twenty": "20",
  "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100",
}


def word_count(text: str) -> int:
  return len(text.split())


def name_tokens(text: str) -> Set[str]:
  """Proper-name tokens, lower-cased with the possessive off: capitalised
  words that are not the sentence opener (capitalised for grammar) and not a
  number word."""
  out: Set[str] = set()
  for i, w in enumerate(text.split()):
    w = re.sub(r"^[(\"'“‘\[]+|[)\"'”’\],.;:]+$", "", w)
    if i == 0 or w.lower() in _NUMBER_WORDS:
      continue
    if re.match(r"^[A-Z][\w'’.-]*$", w) and w not in _CAP_STOP:
      out.add(re.sub(r"['’]s$", "", w.lower()))
  return out


def number_tokens(text: str) -> Set[str]:
  out = {n.rstrip(",.") for n in re.findall(r"\d[\d,.%]*", text)}
  out |= {
    _NUMBER_WORDS[w.lower().strip(",.")]
    for w in text.split()
    if w.lower().strip(",.") in _NUMBER_WORDS
  }
  return out


def guard_merge(parts: List[str], merged: str) -> Optional[str]:
  """Why a composed merge of `parts` into `merged` must be refused, or None.
  A token counts as kept when the merged text contains it, case-insensitive,
  so "Haiti" is found in "Haitian"."""
  low = merged.lower()
  needed = set().union(*(name_tokens(p) | number_tokens(p) for p in parts))
  lost = sorted(t for t in needed if t not in low)
  if lost:
    return f"loses {lost[:3]}"
  # A token of the merged text is new unless the parts carry it: as a token
  # ("Eighteen" and "18" are one token), as a substring, or as an inflection
  # of one of their names ("Haitian" for "Haiti").
  source_text = " ".join(parts).lower()

  def known(t: str) -> bool:
    return t in needed or t in source_text or any(
      len(p) >= 4 and (t.startswith(p) or p.startswith(t)) for p in needed
    )

  added = sorted(t for t in name_tokens(merged) | number_tokens(merged) if not known(t))
  if added:
    return f"adds {added[:3]}"
  n = word_count(merged)
  if n > MAX_CLAIM_WORDS:
    return f"{n} words, at most {MAX_CLAIM_WORDS}"
  longest = max(word_count(p) for p in parts)
  if n > longest + MAX_MERGE_GROWTH:
    return f"{n} words, at most {MAX_MERGE_GROWTH} more than the longest part ({longest})"
  return None


# ── The plan the model returns, validated (pure) ───────────────────────────

@dataclass
class Merge:
  keep: int
  drop: List[int]
  text: Optional[str]          # None → keep the kept claim verbatim
  refused: Optional[str] = None

  @property
  def parts(self) -> List[int]:
    return [self.keep, *self.drop]


@dataclass
class ReviewPlan:
  merges: List[Merge] = field(default_factory=list)
  # Model's final grouping: collection name → claim indices (pre-merge numbering).
  collections: List[Tuple[str, List[int]]] = field(default_factory=list)


def parse_plan(raw: str, n_claims: int) -> ReviewPlan:
  """Read the model's JSON into a plan; malformed entries become refusals or
  are dropped, never exceptions — the caller decides what a bad plan means."""
  obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
  plan = ReviewPlan()
  for m in obj.get("merges", []) or []:
    try:
      keep = int(m["keep"])
      drop = [int(d) for d in (m.get("drop") or [])]
    except (KeyError, TypeError, ValueError):
      continue
    text = m.get("text")
    text = text.strip() if isinstance(text, str) and text.strip() else None
    merge = Merge(keep=keep, drop=drop, text=text)
    if not drop or any(p < 0 or p >= n_claims for p in merge.parts) or keep in drop or len(set(merge.parts)) != len(merge.parts):
      merge.refused = f"bad indices {merge.parts}"
    plan.merges.append(merge)
  for c in obj.get("collections", []) or []:
    name = c.get("name") if isinstance(c, dict) else None
    if not isinstance(name, str):
      continue
    idx = []
    for i in c.get("claims") or []:
      try:
        i = int(i)
      except (TypeError, ValueError):
        continue
      if 0 <= i < n_claims:
        idx.append(i)
    plan.collections.append((name, idx))
  return plan


def guard_plan(plan: ReviewPlan, claims: List[str]) -> None:
  """Refuse merges that overlap an earlier merge or fail the word guards."""
  taken: Set[int] = set()
  for m in plan.merges:
    if m.refused:
      continue
    if any(p in taken for p in m.parts):
      m.refused = f"overlaps another merge {m.parts}"
      continue
    if m.text is not None:
      reason = guard_merge([claims[p] for p in m.parts], m.text)
      if reason:
        m.refused = f"merge {m.parts} {reason}"
        continue
    taken.update(m.parts)


def merged_claim(kept: ExtractedClaim, dropped: List[ExtractedClaim], text: Optional[str]) -> ExtractedClaim:
  """The claim that stands for a merge: the kept claim's topic, the union of
  sources, the lowest confidence and the highest importance of its parts."""
  parts = [kept, *dropped]
  importances = [c.importance for c in parts if c.importance is not None]
  return ExtractedClaim(
    text=text or kept.text,
    topic=kept.topic,
    source_indices=sorted({i for c in parts for i in c.source_indices}),
    confidence=min(c.confidence for c in parts),
    importance=max(importances) if importances else None,
  )


def apply_plan(
  plan: ReviewPlan,
  claims: List[ExtractedClaim],
  quotes: List[ExtractedQuote],
  collections: List[ExtractedCollection],
  collection_order: List[str],
) -> Tuple[Optional[Tuple[List[ExtractedClaim], List[ExtractedQuote], List[ExtractedCollection], List[str]]], CollectionReviewReport]:
  """Apply the accepted merges and the model's grouping; renumber everything.
  Returns (None, report) when the result would leave a one-claim collection —
  the caller then keeps the input untouched."""
  report = CollectionReviewReport(applied=False)
  report.rejected = [m.refused for m in plan.merges if m.refused]

  texts: List[ExtractedClaim] = list(claims)
  dropped: Dict[int, int] = {}       # dropped index → kept index
  pinned: Set[int] = set()           # parts of a refused merge stay put
  for m in plan.merges:
    if m.refused:
      pinned.update(m.parts)
      continue
    texts[m.keep] = merged_claim(claims[m.keep], [claims[d] for d in m.drop], m.text)
    for d in m.drop:
      dropped[d] = m.keep
    report.merges += 1
    report.merged_claims += len(m.drop)

  original_home: Dict[int, str] = {}
  for c in collections:
    for i in c.claim_indices:
      original_home.setdefault(i, c.name)
  by_name = {c.name: c for c in collections}

  # The model's grouping, restricted to existing collections and live claims.
  grouped: Dict[str, List[int]] = {c.name: [] for c in collections}
  seen: Set[int] = set()
  for name, idx in plan.collections:
    if name not in grouped:
      continue
    for i in idx:
      i = dropped.get(i, i)
      if i in dropped or i in seen or i in pinned or i not in original_home:
        continue
      seen.add(i)
      grouped[name].append(i)
      if original_home[i] != name:
        report.moves += 1
  # A claim the model left out (its merge was refused, or it was forgotten)
  # goes back where extraction had it — a refusal stays local.
  for i, home in original_home.items():
    if i in seen or i in dropped:
      continue
    grouped[home].append(i)
    seen.add(i)

  thin = [name for name, idx in grouped.items() if 0 < len(idx) < 2]
  if thin:
    report.rejected.append(f"would leave a one-claim collection: {thin[:3]} — review discarded")
    return None, report
  if not any(grouped.values()):
    report.rejected.append("no collections survived — review discarded")
    return None, report

  # Renumber: surviving claims keep their original relative order.
  survivors = [i for i in range(len(claims)) if i not in dropped]
  new_index = {old: new for new, old in enumerate(survivors)}
  out_claims = [texts[i] for i in survivors]
  out_collections: List[ExtractedCollection] = []
  for c in collections:
    idx = grouped[c.name]
    if not idx:
      continue
    out_collections.append(ExtractedCollection(
      name=c.name, type=c.type, summary=c.summary,
      claim_indices=[new_index[i] for i in idx],
    ))
  surviving_names = {c.name for c in out_collections}
  out_order = [n for n in collection_order if n in surviving_names]
  out_order += [c.name for c in out_collections if c.name not in out_order]
  out_quotes: List[ExtractedQuote] = []
  for q in quotes:
    i = dropped.get(q.claim_index, q.claim_index)
    if i in new_index:
      out_quotes.append(ExtractedQuote(text=q.text, speaker=q.speaker, claim_index=new_index[i]))
  report.applied = True
  return (out_claims, out_quotes, out_collections, out_order), report


# ── Prompts ────────────────────────────────────────────────────────────────

def build_review_prompt(headline: str, claims: List[str], collections: List[Tuple[str, List[int]]]) -> str:
  return REVIEW_PROMPT.format(
    headline=headline,
    max_words=MAX_CLAIM_WORDS,
    max_growth=MAX_MERGE_GROWTH,
    claims="\n".join(f"{i}. {t}" for i, t in enumerate(claims)),
    collections="\n".join(f"{name}: {idx}" for name, idx in collections),
  )


def build_source_check_prompt(sentences: List[str], sources: List[NewsArticleSource]) -> str:
  budget = max(2_000, SOURCE_CHECK_BUDGET // max(1, len(sources)))
  src = "\n\n".join(
    f"[{s.title}{f' — {s.publisher}' if s.publisher else ''}]\n{s.content[:budget]}" for s in sources
  )
  return SOURCE_CHECK_PROMPT.format(
    sentences="\n".join(f"{i}. {t}" for i, t in enumerate(sentences)),
    sources=src,
  )


def parse_source_check(raw: str, n: int) -> List[Optional[str]]:
  """Per sentence: a refusal reason, or None when it passed. A sentence the
  reply does not mention passes — the check is a filter, not a gate."""
  obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
  verdict: List[Optional[str]] = [None] * n
  for c in obj.get("checks", []) or []:
    try:
      i = int(c["index"])
    except (KeyError, TypeError, ValueError):
      continue
    if not 0 <= i < n:
      continue
    if c.get("grade") == "unsupported":
      verdict[i] = "the sources do not support the merged sentence"
    elif c.get("false_link") is True:
      verdict[i] = "the merged sentence links facts the sources do not link"
  return verdict


# ── Orchestration ──────────────────────────────────────────────────────────

def _gemini(prompt: str, thinking_level: Optional[str]) -> str:
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  kwargs: dict = {"temperature": REVIEW_TEMPERATURE, "response_mime_type": "application/json"}
  if thinking_level:
    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
  return client.models.generate_content(
    model=REVIEW_MODEL, contents=prompt, config=types.GenerateContentConfig(**kwargs),
  ).text or ""


def review_collections(
  headline: str,
  sources: List[NewsArticleSource],
  claims: List[ExtractedClaim],
  quotes: List[ExtractedQuote],
  collections: List[ExtractedCollection],
  collection_order: List[str],
  thinking_level: str = "high",
  call=None,
) -> NewsCollectionReviewResponse:
  """One review call, guards, one source check for composed merges, apply.
  Fails open: on any error the input comes back untouched with applied=False.
  `call` is the LLM seam for tests (prompt, thinking_level) -> raw text."""
  if not settings.gemini_api_key and call is None:
    raise Exception("GEMINI_API_KEY not configured for collection review")
  call = call or _gemini
  untouched = NewsCollectionReviewResponse(
    claims=claims, quotes=quotes, collections=collections,
    collection_order=collection_order, review=CollectionReviewReport(applied=False),
  )
  if len(claims) < 2 or not collections:
    return untouched
  t0 = time.time()
  try:
    texts = [c.text for c in claims]
    prompt = build_review_prompt(headline, texts, [(c.name, list(c.claim_indices)) for c in collections])
    plan = parse_plan(call(prompt, thinking_level), len(claims))
    guard_plan(plan, texts)

    composed = [m for m in plan.merges if not m.refused and m.text is not None]
    if composed:
      verdicts = parse_source_check(
        call(build_source_check_prompt([m.text for m in composed], sources), "low"),
        len(composed),
      )
      for m, reason in zip(composed, verdicts):
        if reason:
          m.refused = f"merge {m.parts}: {reason}"

    result, report = apply_plan(plan, claims, quotes, collections, collection_order)
    report.seconds = round(time.time() - t0, 1)
    if result is None:
      logger.warning(f"collection review discarded for '{headline[:60]}': {report.rejected[-1]}")
      untouched.review = report
      return untouched
    out_claims, out_quotes, out_collections, out_order = result
    logger.info(
      f"collection review '{headline[:60]}': {report.merges} merges (-{report.merged_claims} claims), "
      f"{report.moves} moves, {len(report.rejected)} refused, {report.seconds}s"
    )
    return NewsCollectionReviewResponse(
      claims=out_claims, quotes=out_quotes, collections=out_collections,
      collection_order=out_order, review=report,
    )
  except Exception as e:  # noqa: BLE001 — fail open by design
    logger.warning(f"collection review failed for '{headline[:60]}', returning input: {e}")
    untouched.review = CollectionReviewReport(
      applied=False, rejected=[f"review failed: {str(e)[:120]}"], seconds=round(time.time() - t0, 1),
    )
    return untouched
