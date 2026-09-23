"""Collection review — rebuild a story's blocks from its claims: group,
rescue, merge, check.

Why not prompt rules: the extraction prompt decides the collections BEFORE the
claims exist (one collection per topic label, at least two claims each, every
paragraph filed under the nearest label "even if the fit is loose"), so the
claims get bent to the labels. Measured on 30 prod stories (2026-09-22,
independent Opus judge with a high bar): 22% of claims sat loosely under their
heading, 48% of blocks held one such claim, 21% of headings did not describe
their claims, and 8% of claims restated another in the same block. Stricter
filing rules in the same call changed nothing (21.8%); more thinking made it
worse (30%). Rebuilding the blocks from the claims afterwards, in small calls
the code checks, took loose to 7.6%, wrong headings to 3.6%, incoherent blocks
15% → 3.6%, with the block count kept (5.6 vs 5.5 per story) and more claims
in blocks (14.0 vs 13.1).

The chain, each step fail-open:
1. GROUP — one call sees only the claim texts (never the topic labels), sorts
   them by subject, membership first, then writes each heading from its
   members; 2-4 claims per block; a claim with no partner is set aside.
2. MERGE — the same-fact review: claims saying the same thing inside a block
   are merged. Every merge must keep every name and number of the claims it
   replaces and add none, stay within MAX_CLAIM_WORDS and grow by at most
   MAX_MERGE_GROWTH over its longest part; a composed sentence must also
   pass the source check. A block a merge thins to one claim sets that claim
   aside, so the rescue sees it.
3. RESCUE — for each claim set aside, one call goes back to the sources for
   ONE more fact on the same subject (never a definition, a gloss, or a piece
   of the same fact); the new claim must pass the word guards, a duplicate
   check at publish's own Jaccard threshold, and the source check. 79% of
   lone claims were rescued in the probe, 0 unsupported, 0 glosses.
4. CHECK — one call judges every block both ways (each claim belongs under
   the heading; the heading is specific and true of every member; no
   same-fact pair), and names a block whose heading is true of each claim
   still aside, if any. Code verifies the references and that every claim is
   accounted for. A rejected grouping gets ONE regroup with the reasons; a
   second rejection discards the review.
A claim still aside after all that folds into the block the check named, or
is dropped — unless it is a core claim (importance ≥ CORE_IMPORTANCE), in
which case the review is discarded rather than lose it. A discarded review
returns the input untouched — the extraction's own grouping, today's
behaviour.
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
from src.api.services.claim_anchor_guard import calendar_block, check_claim, source_anchors
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
# repair guard uses the same 40 / +8): a merged or rescued claim is a claim
# like any other.
MAX_CLAIM_WORDS = 40
MIN_CLAIM_WORDS = 8
MAX_MERGE_GROWTH = 8

# The team rule (no one-claim block) and the size the probe found keeps the
# block count without loose filing: a bigger group splits by subject.
MIN_BLOCK = 2
MAX_BLOCK = 4

# A rescued claim that publish's dedup (Jaccard ≥ 0.6) would fold into its
# lone claim leaves the block at one again — refuse it here instead.
DEDUP_JACCARD = 0.6

# A claim this central is never dropped: a review that would drop one is
# discarded and the extraction's grouping ships. 0.9 is where the extraction
# rubric puts "the core event itself"; official responses grade 0.7-0.85, and
# a "did not immediately respond" at 0.8 must not veto a whole review.
CORE_IMPORTANCE = 0.9

# Thinking per step: grouping and checking see ~15 short claims (cheap), the
# rescue reads the sources; the merge review runs at the caller's level.
GROUP_THINKING = "low"
RESCUE_THINKING = "medium"
CHECK_THINKING = "medium"

# Source bodies the source check and the rescue see, split across sources.
SOURCE_CHECK_BUDGET = 60_000
RESCUE_SOURCE_BUDGET = 90_000

GROUP_PROMPT = """You group the claims of one news story into blocks. Each claim is published ALONE on a knowledge graph, and claims are shown to readers in blocks under a heading. Your only job is to decide which claims go together and what each block is called. Do not rewrite, drop, or add any claim.

HEADLINE: {headline}

CLAIMS (index. text):
{claims}
{feedback}
A block is a set of claims a reader would expect under one heading because they are about the same specific subject — the same event, decision, actor's conduct, dispute, mechanism or consequence — not merely the same story.

Procedure:
1. For each claim, note (silently) the one specific subject it is about.
2. Group the claims that share a subject — membership FIRST. Two claims belong together only when a reader who has read the block's heading would expect BOTH of them there. Being about the same story, or sharing one keyword with the group, is not belonging. Do not file a claim under the nearest group.
3. Only then name each block from its members: a plain, specific heading of 2-6 words that is true of every claim in it, as an editor would head that section. "These all mention X" is not a heading. If it is true of only some members, the block is two blocks, or the odd claim is out.
4. Every block holds at least {min_block} claims. A claim that shares its subject with no other claim goes into "lone" — never file it under a heading that does not describe it.
5. A block holds {min_block} to {max_block} claims. A group of more must be split into specific blocks when its claims cleanly separate by subject; only if no clean split exists may it stay larger. Never split a group just to reach a count.
6. Order the blocks by narrative flow: event → causes → consequences → responses → context.

Return JSON only:
{{"blocks": [{{"name": "<heading>", "claims": [<indices>]}}], "lone": [<indices>]}}"""

RESCUE_PROMPT = """You find company for a lone claim from the story's sources. Claims were extracted from the sources of one news story and grouped into blocks shown to readers under a heading. The claims under LONE CLAIMS share their subject with no other claim, so they have no block, and every published claim must sit in a named block of at least two.

For EACH lone claim: search the SOURCES for ONE more substantive fact about the SAME specific subject that is not already among the claims — a different event, figure, named actor's action, decision, or consequence — so that the lone claim and the new fact form a block a reader would expect under one heading. Then name that block.

HEADLINE: {headline}

EXISTING CLAIMS (all of them, so you never restate one):
{claims}

LONE CLAIMS (index. text):
{lone}

SOURCES:
{sources}

CALENDAR (read weekdays and "yesterday" off this; never compute them):
{calendar}

Rules for the new claim:
- Every name, number, date and fact must be traceable to a specific sentence in the SOURCES. Nothing from prior knowledge. If in doubt, leave it out. A weekday or "yesterday" in a source becomes the CALENDAR's date for that source; a year is written only when a source states it or the calendar gives it — code refuses the claim otherwise.
- It must state a DIFFERENT fact from the lone claim and from every existing claim. A restatement, a gloss, or a piece cut from the lone claim's own fact is forbidden — an empty result is better.
- NOT a fact for this purpose, even when the sources state it: what an organisation, product, person or term IS (a definition or profile); how something works in general; a commentator's characterisation of the lone claim's event; a detail of the lone claim's own event (its location, its wallet address, its exact time). The reader must learn a second thing that HAPPENED, was DECIDED, or was MEASURED about the same subject.
- At most ONE new claim per lone claim. If the sources carry nothing that qualifies, return no claim for it.
- Self-contained: full proper names, no pronoun before its referent, no "the company"/"the deal"; name the event inside the claim; absolute dates; {min_words}-35 words.
- source_indices: the indices of the sources that state the fact. confidence 0.9+ when explicit. importance for this story (0.3-1.0).
- The heading: 2-6 plain words, true of the lone claim and the new claim.

Return JSON only:
{{"rescues": [{{"lone": <index>, "name": "<heading>", "claim": {{"text": "...", "source_indices": [0], "confidence": 0.9, "importance": 0.6}}}}]}}
Include every lone index; one whose sources carry nothing gets "claim": null."""

CHECK_PROMPT = """You are the final check on how a news story's claims were grouped into blocks. Each block is shown to readers under its heading. Judge with a HIGH bar and reject anything a careful editor would not publish as is.

HEADLINE: {headline}

BLOCKS:
{blocks}
{lone}
For every block:
- "misfits": the indices of claims that do NOT belong under the heading — a reader who opened that heading would not expect exactly that claim there.
- "purpose": true when the block has one clear shared subject every member contributes to; false for a catch-all or "these all mention X".
- "heading_ok": true when the heading is specific and true of every member; false if it is vague, a catch-all ("Other developments", "Context and reactions"), or promises something the claims do not deliver.
- "duplicates": pairs of claim indices inside the block that state the same fact.
- "reason": under 20 words, only when something is wrong.
For every lone claim: "home" = the index of the block whose heading is true of it as written, or -1 if none is.

Return JSON only:
{{"blocks": [{{"index": 0, "misfits": [], "purpose": true, "heading_ok": true, "duplicates": [], "reason": ""}}], "lone": [{{"index": 0, "home": -1}}]}}"""

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
- Never drop a claim, never move a claim to another collection, never rename or add a collection. If a merge leaves a collection with one claim, leave it — that is handled after you.

Return JSON only:
{{"merges": [{{"keep": <index>, "drop": [<indices>], "text": "<merged text, or null to keep the kept claim verbatim>"}}]}}"""

SOURCE_CHECK_PROMPT = """Check each sentence below against the story's source articles. Each sentence was written by the system rather than extracted verbatim: a merge of two or three extracted claims into one, or a fact added from the sources to accompany another claim.

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


def jaccard(a: str, b: str) -> float:
  """Token Jaccard on lower-cased words, the measure publish's dedup uses."""
  ta = set(re.findall(r"[a-z0-9]+", a.lower()))
  tb = set(re.findall(r"[a-z0-9]+", b.lower()))
  if not ta or not tb:
    return 0.0
  return len(ta & tb) / len(ta | tb)


def _json_object(raw: str) -> dict:
  return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def _int_list(values, n: int) -> List[int]:
  """Valid, distinct indices below n, in the order given."""
  out: List[int] = []
  for v in values or []:
    try:
      i = int(v)
    except (TypeError, ValueError):
      continue
    if 0 <= i < n and i not in out:
      out.append(i)
  return out


# ── Step 1: the grouping the model proposes, validated (pure) ──────────────

@dataclass
class Grouping:
  blocks: List[Tuple[str, List[int]]] = field(default_factory=list)
  lone: List[int] = field(default_factory=list)


def parse_grouping(raw: str, n_claims: int) -> Grouping:
  """Read the model's blocks; bad entries are dropped, never exceptions."""
  obj = _json_object(raw)
  g = Grouping()
  for b in obj.get("blocks", []) or []:
    if not isinstance(b, dict) or not isinstance(b.get("name"), str):
      continue
    g.blocks.append((b["name"].strip(), _int_list(b.get("claims"), n_claims)))
  g.lone = _int_list(obj.get("lone"), n_claims)
  return g


def guard_grouping(g: Grouping, n_claims: int) -> Grouping:
  """Code owns the invariants the prompt only asks for: a claim sits in one
  block at most, a block has a heading and at least MIN_BLOCK claims (a
  thinner one sets its claims aside), and every claim is accounted for."""
  seen: Set[int] = set()
  blocks: List[Tuple[str, List[int]]] = []
  aside: List[int] = []
  for name, idx in g.blocks:
    idx = [i for i in idx if i not in seen]
    seen.update(idx)
    if name and len(idx) >= MIN_BLOCK:
      blocks.append((name, idx))
    else:
      aside.extend(idx)
  lone = [i for i in g.lone if i not in seen] + aside
  lone += [i for i in range(n_claims) if i not in seen and i not in lone]
  return Grouping(blocks=blocks, lone=sorted(set(lone)))


# ── Step 2: rescues, validated (pure) ──────────────────────────────────────

@dataclass
class Rescue:
  lone: int
  name: str
  claim: Optional[ExtractedClaim]
  refused: Optional[str] = None


def parse_rescues(raw: str, lone: List[int], n_sources: int) -> List[Rescue]:
  """One Rescue per lone index the model answered for; a null claim is a
  refusal ("the sources carry nothing"), a malformed one too."""
  obj = _json_object(raw)
  out: List[Rescue] = []
  for r in obj.get("rescues", []) or []:
    if not isinstance(r, dict):
      continue
    try:
      i = int(r.get("lone"))
    except (TypeError, ValueError):
      continue
    if i not in lone or any(x.lone == i for x in out):
      continue
    name = r.get("name") if isinstance(r.get("name"), str) else ""
    c = r.get("claim")
    if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip():
      out.append(Rescue(lone=i, name=name.strip(), claim=None, refused="the sources carry nothing"))
      continue
    try:
      claim = ExtractedClaim(
        text=c["text"].strip(), topic=name.strip(),
        source_indices=_int_list(c.get("source_indices"), n_sources),
        confidence=float(c.get("confidence", 0.8)), importance=c.get("importance"),
      )
    except Exception as e:  # noqa: BLE001 — pydantic bounds
      out.append(Rescue(lone=i, name=name.strip(), claim=None, refused=f"malformed claim: {str(e)[:60]}"))
      continue
    out.append(Rescue(lone=i, name=name.strip(), claim=claim))
  return out


def guard_rescue(r: Rescue, lone_text: str, existing: List[str]) -> Optional[str]:
  """Why a rescue must be refused, or None: no heading, no source, outside
  the reading limits, or a claim publish would dedup into an existing one
  (the lone claim included — that would put the block back at one)."""
  if r.claim is None:
    return r.refused or "no claim"
  if not r.name:
    return "no heading"
  if not r.claim.source_indices:
    return "no source"
  n = word_count(r.claim.text)
  if n > MAX_CLAIM_WORDS or n < MIN_CLAIM_WORDS:
    return f"{n} words, {MIN_CLAIM_WORDS}-{MAX_CLAIM_WORDS} allowed"
  for t in [lone_text, *existing]:
    if jaccard(r.claim.text, t) >= DEDUP_JACCARD:
      return "restates a claim the story already has"
  return None


# ── Step 4: the check, parsed (pure) ───────────────────────────────────────

@dataclass
class Check:
  reasons: List[str] = field(default_factory=list)   # empty → every block passed
  homes: Dict[int, int] = field(default_factory=dict)  # lone claim → block index


def parse_check(raw: str, blocks: List[Tuple[str, List[int]]], lone: List[int]) -> Check:
  """A block fails on any misfit, a missing purpose, a bad heading or a
  same-fact pair — decided from the fields, not from a verdict flag."""
  obj = _json_object(raw)
  check = Check()
  answered: Set[int] = set()
  for v in obj.get("blocks", []) or []:
    if not isinstance(v, dict):
      continue
    try:
      b = int(v.get("index"))
    except (TypeError, ValueError):
      continue
    if not 0 <= b < len(blocks) or b in answered:
      continue
    answered.add(b)
    name, idx = blocks[b]
    misfits = [i for i in _int_list(v.get("misfits"), 10**9) if i in idx]
    dups = [d for d in (v.get("duplicates") or []) if isinstance(d, list) and len(d) == 2]
    faults = []
    if misfits:
      faults.append(f"claims {misfits} do not belong")
    if v.get("purpose") is False:
      faults.append("no shared subject")
    if v.get("heading_ok") is False:
      faults.append("heading vague or untrue")
    if dups:
      faults.append(f"same fact twice {dups[:2]}")
    if faults:
      reason = v.get("reason") if isinstance(v.get("reason"), str) else ""
      check.reasons.append(f"block \"{name}\": " + "; ".join(faults) + (f" — {reason.strip()}" if reason.strip() else ""))
  for b in range(len(blocks)):
    if b not in answered:
      check.reasons.append(f"block \"{blocks[b][0]}\": not checked")
  for v in obj.get("lone", []) or []:
    if not isinstance(v, dict):
      continue
    try:
      i, home = int(v.get("index")), int(v.get("home", -1))
    except (TypeError, ValueError):
      continue
    if i in lone and 0 <= home < len(blocks):
      check.homes[i] = home
  return check


# ── Step 3: the merge plan the model returns, validated (pure) ─────────────

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


def parse_plan(raw: str, n_claims: int) -> ReviewPlan:
  """Read the model's JSON into a plan; malformed entries become refusals or
  are dropped, never exceptions — the caller decides what a bad plan means."""
  obj = _json_object(raw)
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
  return plan


def guard_plan(plan: ReviewPlan, claims: List[str], blocks: List[Tuple[str, List[int]]]) -> None:
  """Refuse merges that cross blocks, overlap an earlier merge or fail the
  word guards — a merge only ever joins claims that sit in one block."""
  block_of: Dict[int, int] = {i: b for b, (_, idx) in enumerate(blocks) for i in idx}
  taken: Set[int] = set()
  for m in plan.merges:
    if m.refused:
      continue
    if len({block_of.get(p, -1) for p in m.parts}) != 1 or block_of.get(m.keep) is None:
      m.refused = f"merge {m.parts} crosses blocks"
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


def apply_merges(
  plan: ReviewPlan,
  claims: List[ExtractedClaim],
  blocks: List[Tuple[str, List[int]]],
) -> Tuple[List[ExtractedClaim], Dict[int, int], List[Tuple[str, List[int]]], List[int]]:
  """Apply the accepted merges inside the blocks. Returns the claims with
  merged texts in place, dropped index → kept index, the blocks without the
  dropped claims, and the claims a merge left alone in their block (the
  block is gone; they are set aside for the check to home or drop)."""
  texts: List[ExtractedClaim] = list(claims)
  dropped: Dict[int, int] = {}
  for m in plan.merges:
    if m.refused:
      continue
    texts[m.keep] = merged_claim(claims[m.keep], [claims[d] for d in m.drop], m.text)
    for d in m.drop:
      dropped[d] = m.keep
  out_blocks: List[Tuple[str, List[int]]] = []
  aside: List[int] = []
  for name, idx in blocks:
    live = [i for i in idx if i not in dropped]
    if len(live) >= MIN_BLOCK:
      out_blocks.append((name, live))
    else:
      aside.extend(live)
  return texts, dropped, out_blocks, aside


def assemble(
  claims: List[ExtractedClaim],
  quotes: List[ExtractedQuote],
  dropped: Dict[int, int],
  blocks: List[Tuple[str, List[int]]],
  discarded: Set[int],
) -> Tuple[List[ExtractedClaim], List[ExtractedQuote], List[ExtractedCollection], List[str]]:
  """Renumber for the response: surviving claims keep their relative order
  (rescued claims sit after the extracted ones, where they were appended),
  every block becomes a topic collection in block order, a quote follows its
  claim into a merge and disappears with a discarded claim."""
  survivors = [i for i in range(len(claims)) if i not in dropped and i not in discarded]
  new_index = {old: new for new, old in enumerate(survivors)}
  out_claims = [
    ExtractedClaim(text=claims[i].text, topic=next((n for n, idx in blocks if i in idx), claims[i].topic),
                   source_indices=claims[i].source_indices, confidence=claims[i].confidence, importance=claims[i].importance)
    for i in survivors
  ]
  out_collections = [
    ExtractedCollection(name=name, type="topic", summary="", claim_indices=[new_index[i] for i in idx if i in new_index])
    for name, idx in blocks
  ]
  out_quotes: List[ExtractedQuote] = []
  for q in quotes:
    i = dropped.get(q.claim_index, q.claim_index)
    if i in new_index:
      out_quotes.append(ExtractedQuote(text=q.text, speaker=q.speaker, claim_index=new_index[i]))
  return out_claims, out_quotes, out_collections, [c.name for c in out_collections]


# ── Prompts ────────────────────────────────────────────────────────────────

def _numbered(texts: List[str], indices: Optional[List[int]] = None) -> str:
  idx = indices if indices is not None else list(range(len(texts)))
  return "\n".join(f"{i}. {texts[i]}" for i in idx)


def _sources_block(sources: List[NewsArticleSource], budget_total: int, indexed: bool = False) -> str:
  budget = max(2_000, budget_total // max(1, len(sources)))
  return "\n\n".join(
    f"[{f'{i}: ' if indexed else ''}{s.title}{f' — {s.publisher}' if s.publisher else ''}]\n{s.content[:budget]}"
    for i, s in enumerate(sources)
  )


def build_group_prompt(headline: str, claims: List[str], feedback: str = "", indices: Optional[List[int]] = None) -> str:
  """`indices` restricts the listing to the claims still alive (a regroup
  after merges); the numbering stays the claims' own."""
  fb = ""
  if feedback:
    fb = f"\nA previous grouping was REJECTED by the check for these reasons — fix every one of them:\n{feedback}\n"
  return GROUP_PROMPT.format(
    headline=headline, claims=_numbered(claims, indices), feedback=fb, min_block=MIN_BLOCK, max_block=MAX_BLOCK,
  )


def build_rescue_prompt(
  headline: str, claims: List[str], lone: List[int], sources: List[NewsArticleSource], alive: Optional[List[int]] = None,
) -> str:
  """`alive` restricts the existing-claims listing to the claims still live
  after merges; the lone claims keep their own numbering."""
  existing = [claims[i] for i in alive] if alive is not None else claims
  return RESCUE_PROMPT.format(
    headline=headline,
    claims="\n".join(f"- {t}" for t in existing),
    lone=_numbered(claims, lone),
    sources=_sources_block(sources, RESCUE_SOURCE_BUDGET, indexed=True),
    calendar=calendar_block([source_anchors(s.index, s.content, s.published_at) for s in sources])
    or "(no publication dates known — resolve no relative date)",
    min_words=MIN_CLAIM_WORDS,
  )


def build_review_prompt(headline: str, claims: List[str], collections: List[Tuple[str, List[int]]]) -> str:
  return REVIEW_PROMPT.format(
    headline=headline,
    max_words=MAX_CLAIM_WORDS,
    max_growth=MAX_MERGE_GROWTH,
    claims=_numbered(claims),
    collections="\n".join(f"{name}: {idx}" for name, idx in collections),
  )


def build_check_prompt(headline: str, claims: List[str], blocks: List[Tuple[str, List[int]]], lone: List[int]) -> str:
  block_text = "\n".join(
    f"[{b}] \"{name}\"\n" + "\n".join(f"   {i}. {claims[i]}" for i in idx) for b, (name, idx) in enumerate(blocks)
  )
  lone_text = f"\nLONE CLAIMS (no block yet):\n{_numbered(claims, lone)}\n" if lone else ""
  return CHECK_PROMPT.format(headline=headline, blocks=block_text, lone=lone_text)


def build_source_check_prompt(sentences: List[str], sources: List[NewsArticleSource]) -> str:
  return SOURCE_CHECK_PROMPT.format(
    sentences=_numbered(sentences),
    sources=_sources_block(sources, SOURCE_CHECK_BUDGET),
  )


def parse_source_check(raw: str, n: int, strict: bool = False) -> List[Optional[str]]:
  """Per sentence: a refusal reason, or None when it passed. A sentence the
  reply does not mention passes — the check is a filter, not a gate. Strict
  (for rescued claims, which are new facts) also refuses "partly": a detail
  the sources do not state is not a fact to add."""
  obj = _json_object(raw)
  verdict: List[Optional[str]] = [None] * n
  for c in obj.get("checks", []) or []:
    try:
      i = int(c["index"])
    except (KeyError, TypeError, ValueError):
      continue
    if not 0 <= i < n:
      continue
    if c.get("grade") == "unsupported" or (strict and c.get("grade") == "partly"):
      verdict[i] = "the sources do not support the sentence"
    elif c.get("false_link") is True:
      verdict[i] = "the sentence links facts the sources do not link"
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
  """Group → rescue → merge → check, then assemble. The extraction's own
  collections are the fallback only: they come back untouched whenever a
  step fails or the check rejects twice. `call` is the LLM seam for tests
  (prompt, thinking_level) -> raw text; each prompt opens with its own
  first sentence so a seam can tell the steps apart."""
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
  report = CollectionReviewReport(applied=False)
  last = [t0]

  def stamp(step: str) -> None:
    now = time.time()
    report.steps[step] = round(report.steps.get(step, 0.0) + now - last[0], 1)
    last[0] = now

  def discard(reason: str) -> NewsCollectionReviewResponse:
    report.rejected.append(f"{reason} — review discarded")
    report.seconds = round(time.time() - t0, 1)
    logger.warning(f"collection review discarded for '{headline[:60]}': {reason}")
    untouched.review = report
    return untouched

  try:
    live: List[ExtractedClaim] = list(claims)
    texts = [c.text for c in live]

    # 1. Group from the claims alone.
    grouping = guard_grouping(parse_grouping(call(build_group_prompt(headline, texts), GROUP_THINKING), len(live)), len(live))
    stamp("group")
    if not grouping.blocks:
      return discard("grouping produced no block")
    blocks, lone = grouping.blocks, grouping.lone

    # 2. Merge claims that say the same thing inside a block. A block a merge
    # thins to one claim sets its survivor aside, so the rescue sees it too.
    plan = parse_plan(call(build_review_prompt(headline, texts, blocks), thinking_level), len(live))
    guard_plan(plan, texts, blocks)
    # Every sentence the review writes obeys the same rule as extraction: a
    # date, year or figure must be one a source states or code resolved.
    anchors = [source_anchors(s.index, s.content, s.published_at) for s in sources]
    for m in plan.merges:
      if not m.refused and m.text is not None:
        problems = check_claim(m.text, anchors)
        if problems:
          m.refused = f"merge {m.parts}: unanchored {'; '.join(str(p) for p in problems)}"
    composed = [m for m in plan.merges if not m.refused and m.text is not None]
    if composed:
      verdicts = parse_source_check(
        call(build_source_check_prompt([m.text for m in composed], sources), "low"), len(composed),
      )
      for m, reason in zip(composed, verdicts):
        if reason:
          m.refused = f"merge {m.parts}: {reason}"
    report.rejected += [m.refused for m in plan.merges if m.refused]
    live, dropped, blocks, thinned = apply_merges(plan, live, blocks)
    texts = [c.text for c in live]
    report.merges = sum(1 for m in plan.merges if not m.refused)
    report.merged_claims = len(dropped)
    lone = sorted(set(lone) | set(thinned))
    alive = [i for i in range(len(live)) if i not in dropped]
    stamp("merge")

    # 3. Rescue each lone claim with one more fact from the sources.
    if lone and sources:
      rescues = parse_rescues(
        call(build_rescue_prompt(headline, texts, lone, sources, alive), RESCUE_THINKING), lone, len(sources),
      )
      for r in rescues:
        r.refused = guard_rescue(r, texts[r.lone], [texts[i] for i in alive])
        if not r.refused:
          problems = check_claim(r.claim.text, anchors)
          if problems:
            r.refused = "unanchored " + "; ".join(str(p) for p in problems)
      candidates = [r for r in rescues if not r.refused]
      if candidates:
        verdicts = parse_source_check(
          call(build_source_check_prompt([r.claim.text for r in candidates], sources), "low"),
          len(candidates), strict=True,
        )
        for r, reason in zip(candidates, verdicts):
          if reason:
            r.refused = reason
      for r in rescues:
        if r.refused:
          report.rejected.append(f"rescue of {r.lone}: {r.refused}")
          continue
        live.append(r.claim)
        texts.append(r.claim.text)
        alive.append(len(live) - 1)
        blocks.append((r.name, [r.lone, len(live) - 1]))
        lone.remove(r.lone)
        report.rescued_claims += 1
      stamp("rescue")

    # 4. Check both ways; one regroup with the reasons; a second rejection discards.
    check = parse_check(call(build_check_prompt(headline, texts, blocks, lone), CHECK_THINKING), blocks, lone)
    stamp("check")
    report.check = "ok"
    if check.reasons:
      retry = guard_grouping(
        parse_grouping(
          call(build_group_prompt(headline, texts, "\n".join(f"- {r}" for r in check.reasons), indices=alive), GROUP_THINKING),
          len(live),
        ),
        len(live),
      )
      retry_blocks = [(n, [i for i in idx if i in alive]) for n, idx in retry.blocks]
      retry_blocks = [(n, idx) for n, idx in retry_blocks if len(idx) >= MIN_BLOCK]
      retry_lone = [i for i in alive if not any(i in idx for _, idx in retry_blocks)]
      if not retry_blocks:
        return discard("regroup produced no block")
      check2 = parse_check(call(build_check_prompt(headline, texts, retry_blocks, retry_lone), CHECK_THINKING), retry_blocks, retry_lone)
      stamp("repair")
      if check2.reasons:
        report.check = "rejected"
        report.rejected += check2.reasons
        return discard("grouping rejected twice")
      report.check = "repaired"
      report.rejected += [f"first grouping: {r}" for r in check.reasons]
      blocks, lone, check = retry_blocks, retry_lone, check2

    # A claim still alone folds into the block the check named, or is dropped —
    # never a core claim.
    discarded: Set[int] = set()
    for i in lone:
      home = check.homes.get(i)
      if home is not None:
        blocks[home][1].append(i)
        report.moves += 1
      elif (live[i].importance or 0) >= CORE_IMPORTANCE:
        return discard(f"would drop a core claim ({live[i].text[:60]!r})")
      else:
        discarded.add(i)
        report.dropped_claims += 1

    out_claims, out_quotes, out_collections, out_order = assemble(live, quotes, dropped, blocks, discarded)
    report.applied = True
    report.blocks = len(out_collections)
    report.seconds = round(time.time() - t0, 1)
    logger.info(
      f"collection review '{headline[:60]}': {report.blocks} blocks, {report.rescued_claims} rescued, "
      f"{report.merges} merges (-{report.merged_claims}), {report.moves} folded, {report.dropped_claims} dropped, "
      f"check {report.check}, {len(report.rejected)} refused, {report.seconds}s"
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
