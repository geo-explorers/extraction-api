"""Enforce source-anchored dates and numbers on an extracted claim set.

The guard (claim_anchor_guard) says which dates, years and numbers no source
vouches for. This module decides what happens next, the same shape as the
review's reading budget: the model gets ONE chance to fix exactly the
offending tokens from the sources, code re-checks the rewrite, and a claim
that still carries an unanchored token is dropped — never published with a
guessed date. Dropped texts travel in the report so the drop can be audited.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.api.schemas.news_claim_extract_schema import (
  AnchorReport,
  ExtractedClaim,
  NewsArticleSource,
  NewsClaimExtractResponse,
)
from src.api.services.claim_anchor_guard import (
  MAX_CLAIM_WORDS,
  Problem,
  SourceAnchors,
  calendar_block,
  check_claim,
  source_anchors,
)

logger = logging.getLogger(__name__)

# The repair sees the sources again, trimmed to this many characters in total.
REPAIR_SOURCE_BUDGET = 60_000

REPAIR_PROMPT = """You fix dates and figures in extracted news claims so that every one of them is stated by the sources.

Each claim below carries a date, year or number that NO source states. For each claim, return a corrected claim that keeps every other word as it is and changes ONLY the offending token(s):
- If a source states the correct date or figure, use it exactly.
- If a source gives the day only as a weekday or "yesterday", use the CALENDAR below to write the absolute date — never compute weekdays yourself.
- If the sources give no date or figure at all for that fact, remove the offending token and keep the source's own wording (e.g. "in March" stays "in March", "on Tuesday" stays "on Tuesday"; a figure nobody states is dropped with its unit).
- Never add a year, month, day or figure the sources do not state or the calendar does not give. Never add other facts. Keep each claim at {max_words} words or fewer.

CALENDAR
{calendar}

CLAIMS TO FIX
{claims}

SOURCES
{sources}

Return ONLY JSON: {{"claims": [{{"index": <index>, "text": "<corrected claim>"}}]}}"""


def _json_object(raw: str) -> dict:
  s = (raw or "").strip()
  m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
  if m:
    s = m.group(1)
  try:
    v = json.loads(s)
  except (json.JSONDecodeError, TypeError):
    return {}
  return v if isinstance(v, dict) else {}


def _sources_text(sources: Sequence[NewsArticleSource], budget: int) -> str:
  per = max(2_000, budget // max(1, len(sources)))
  parts = []
  for s in sources:
    head = f"[{s.index}] {s.title}" + (f" — {s.publisher}" if s.publisher else "") + (
      f" — published {s.published_at}" if s.published_at else ""
    )
    parts.append(f"{head}\n{(s.content or '')[:per]}")
  return "\n\n".join(parts)


def build_anchor_repair_prompt(
  failing: Sequence[Tuple[int, str, Sequence[Problem]]],
  sources: Sequence[NewsArticleSource],
  calendar: str,
) -> str:
  lines = []
  for index, text, problems in failing:
    why = "; ".join(str(p) for p in problems)
    lines.append(f"{index}. {text}\n   problem: {why}")
  return REPAIR_PROMPT.format(
    max_words=MAX_CLAIM_WORDS,
    calendar=calendar or "(no publication dates known — resolve nothing)",
    claims="\n".join(lines),
    sources=_sources_text(sources, REPAIR_SOURCE_BUDGET),
  )


def parse_anchor_repairs(raw: str, allowed: Sequence[int]) -> Dict[int, str]:
  """index → corrected text, for the indices we asked about; junk is ignored."""
  out: Dict[int, str] = {}
  ok = set(allowed)
  for item in _json_object(raw).get("claims") or []:
    if not isinstance(item, dict):
      continue
    try:
      i = int(item.get("index"))
    except (TypeError, ValueError):
      continue
    text = str(item.get("text") or "").strip()
    if i in ok and text:
      out[i] = text
  return out


def drop_claims(result: NewsClaimExtractResponse, drop: Sequence[int]) -> NewsClaimExtractResponse:
  """The response without the claims at `drop`: quotes of a dropped claim go
  with it, every other claim index is renumbered, collections lose the
  dropped members and an emptied collection disappears."""
  gone = set(drop)
  if not gone:
    return result
  remap: Dict[int, int] = {}
  claims: List[ExtractedClaim] = []
  for i, c in enumerate(result.claims):
    if i in gone:
      continue
    remap[i] = len(claims)
    claims.append(c)
  quotes = [q.model_copy(update={"claim_index": remap[q.claim_index]}) for q in result.quotes if q.claim_index in remap]
  collections = []
  for col in result.collections:
    kept = [remap[i] for i in col.claim_indices if i in remap]
    if kept:
      collections.append(col.model_copy(update={"claim_indices": kept}))
  names = {c.name for c in collections}
  order = [n for n in result.collection_order if n in names]
  return result.model_copy(update={"claims": claims, "quotes": quotes, "collections": collections, "collection_order": order})


def _words(text: str) -> int:
  return len(re.findall(r"[\w'’-]+", text))


def enforce_anchors(
  result: NewsClaimExtractResponse,
  sources: Sequence[NewsArticleSource],
  call: Optional[Callable[[str], str]],
  anchors: Optional[Sequence[SourceAnchors]] = None,
) -> NewsClaimExtractResponse:
  """Guard every claim; repair the failing ones once through `call`; drop
  what still fails. Fails open on a broken repair call: the failing claims
  are then dropped, since the alternative is publishing a guessed date."""
  anchors = list(anchors) if anchors is not None else [source_anchors(s.index, s.content, s.published_at) for s in sources]
  report = AnchorReport(checked=len(result.claims))
  failing: List[Tuple[int, str, List[Problem]]] = []
  for i, c in enumerate(result.claims):
    problems = check_claim(c.text, anchors)
    if problems:
      failing.append((i, c.text, problems))
  report.flagged = len(failing)
  if not failing:
    result.anchor_report = report
    return result

  repairs: Dict[int, str] = {}
  if call is not None:
    try:
      raw = call(build_anchor_repair_prompt(failing, sources, calendar_block(anchors)))
      repairs = parse_anchor_repairs(raw, [i for i, _, _ in failing])
    except Exception as e:  # noqa: BLE001 — a failed repair drops, never publishes
      logger.warning(f"anchor repair call failed: {e}")

  drop: List[int] = []
  for i, text, problems in failing:
    fixed = repairs.get(i)
    if fixed and _words(fixed) <= MAX_CLAIM_WORDS and not check_claim(fixed, anchors):
      result.claims[i] = result.claims[i].model_copy(update={"text": fixed})
      report.repaired += 1
      report.problems.append(f"repaired: {'; '.join(str(p) for p in problems)}")
      continue
    drop.append(i)
    report.dropped_claims.append(text)
    report.problems.append(f"dropped: {'; '.join(str(p) for p in problems)}")
  report.dropped = len(drop)
  out = drop_claims(result, drop)
  out.anchor_report = report
  for text, why in zip(report.dropped_claims, [p for p in report.problems if p.startswith("dropped")]):
    logger.info(f"anchor guard dropped claim — {why} — {text!r}")
  return out
