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
5. ORDER — alongside the check, on the same finished blocks, one cheap call
   returns the reading order (in parallel, so the story waits no longer than
   for the check: 4.3s at low thinking against the check's 7.3s, measured on
   15 injector stories). Code accepts it only as a permutation whose opener
   shares a word with the headline.
A claim still aside after all that folds into the block the check named, or
is dropped — unless it is a core claim (importance ≥ CORE_IMPORTANCE), in
which case the review is discarded rather than lose it. A discarded review
returns the input untouched — the extraction's own grouping, today's
behaviour — ordered and cased like any other.

Order and case (2026-09-23, Armando's two asks on the blocks): the order is
decided ONCE, LAST, on every finished block — earlier the group step ordered
by "event first" before the rescues existed, so the earliest event opened a
story about something else and rescued blocks trailed in whatever order the
lone claims had. The rule is the story first: the block stating the headline's
main clause opens, blocks on the same subject follow it, each side thread runs
contiguously, background last. Headings are sentence-cased in code from the
story's own prose (the claims and sources): a word the prose capitalises
mid-sentence keeps its capital, one it writes lowercase loses it — the prompts
ask for sentence case, the code guarantees it.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
from src.config.overrides import bind_context, llm, prompts
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Model and temperature: settings.news_collection_review_model / _temperature
# (defaults are the values the module docstring's numbers were measured on;
# this task runs on the WORKER service, whose env is separate from the API's),
# read per call through llm.get so a run may override them. The prompts live
# in src/config/prompts/news_collection_review_prompt.py and are fetched
# through prompts.get for the same reason.
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
# The order call sees headings and short claims — nothing to weigh, and it
# must finish inside the check's own time.
ORDER_THINKING = "low"

# Source bodies the source check and the rescue see, split across sources.
SOURCE_CHECK_BUDGET = 60_000
RESCUE_SOURCE_BUDGET = 90_000

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


# ── Headings: sentence case from the story's own prose (pure) ──────────────

_WORD = re.compile(r"[A-Za-z][A-Za-z'’.-]*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Words of 4+ letters that name no subject, for the opener guard.
_STOP = {
  "with", "over", "from", "that", "this", "after", "amid", "into", "than", "their", "about", "says", "said",
  "will", "have", "been", "were", "also", "more", "most", "other", "some", "such", "what", "when", "where",
  "which", "while", "would", "could", "should", "being", "during", "before", "under", "between", "against",
  "among", "around", "through", "because", "following", "despite", "without", "within", "them", "they",
}


def _key(word: str) -> str:
  """A word's dictionary key: lower-cased, possessive and edge punctuation off."""
  return re.sub(r"['’]s$", "", word.strip("'’.-").lower())


def _inflection(a: str, b: str) -> bool:
  """"vote"/"voted", "Falklands"/"Falkland": one is a 4+ letter prefix of the other."""
  return len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))


class ProseIndex:
  """How the story's own prose — headline, claims, sources — capitalises each
  word away from a sentence start. That is the dictionary of proper nouns
  for the headings: "Iran" and "Hegseth" keep their capital, "war powers"
  and "midterm" lose it, and no model or word list has to know which is
  which."""

  def __init__(self, texts: List[str]):
    # Capitalised mid-sentence with no capitalised neighbour ("Iran"), or
    # inside a run of capitals ("Security" in "National Security Minister"),
    # or lower-cased; and words seen only opening a sentence.
    self.alone: Dict[str, int] = {}
    self.in_name: Dict[str, int] = {}
    self.lows: Dict[str, int] = {}
    self.starts: Set[str] = set()
    # The same per word pair (both away from the sentence start), keyed by
    # the pair and the position of the word judged: "States" in "United
    # States" is a proper noun however often "member states" occurs.
    self.pair_caps: Dict[Tuple[str, str, int], int] = {}
    self.pair_lows: Dict[Tuple[str, str, int], int] = {}
    for text in texts:
      for sentence in _SENTENCE.split(text or ""):
        words = _WORD.findall(sentence)
        if words:
          self.starts.add(_key(words[0]))
        words = words[1:]
        keys = [_key(w) for w in words]
        for i, (w, k) in enumerate(zip(words, keys)):
          if len(k) < 2:
            continue
          if w[0].isupper():
            beside = (i > 0 and words[i - 1][0].isupper()) or (i + 1 < len(words) and words[i + 1][0].isupper())
            side, pairs = (self.in_name if beside else self.alone), self.pair_caps
          else:
            side, pairs = self.lows, self.pair_lows
          side[k] = side.get(k, 0) + 1
          if i > 0:
            pairs[(keys[i - 1], k, 1)] = pairs.get((keys[i - 1], k, 1), 0) + 1
          if i + 1 < len(keys):
            pairs[(k, keys[i + 1], 0)] = pairs.get((k, keys[i + 1], 0), 0) + 1

  def counts(self, word: str, before: Optional[str] = None, after: Optional[str] = None) -> Tuple[int, int, int]:
    """(capitalised on its own, capitalised only inside a longer name,
    lower-cased) sightings of the word mid-sentence: next to the same
    neighbours when the prose has the pair (then all its capitals count as
    the word's own); else the word itself; else its inflections."""
    k = _key(word)
    pairs = [(_key(before), k, 1)] if before else []
    pairs += [(k, _key(after), 0)] if after else []
    caps = sum(self.pair_caps.get(p, 0) for p in pairs)
    lows = sum(self.pair_lows.get(p, 0) for p in pairs)
    if caps or lows:
      return caps, 0, lows
    alone, in_name, lows = self.alone.get(k, 0), self.in_name.get(k, 0), self.lows.get(k, 0)
    if not (alone or in_name or lows):
      alone = sum(n for key, n in self.alone.items() if _inflection(k, key))
      in_name = sum(n for key, n in self.in_name.items() if _inflection(k, key))
      lows = sum(n for key, n in self.lows.items() if _inflection(k, key))
    return alone, in_name, lows

  def opens_sentences(self, word: str) -> bool:
    return _key(word) in self.starts

  def seen(self, word: str) -> bool:
    """The word itself, mid-sentence, in any case."""
    k = _key(word)
    return k in self.alone or k in self.in_name or k in self.lows


def _fixed_case(core: str) -> bool:
  """Written as-is whatever the prose says: acronyms (GOP, US), single
  capitals (Rodeo I), tokens with digits (E1, G7), dotted initialisms
  (U.S.), internal capitals (McDonald)."""
  letters = [ch for ch in core if ch.isalpha()]
  return (
    (len(letters) >= 2 and all(ch.isupper() for ch in letters))
    or (len(core) == 1 and core.isupper())
    or any(ch.isdigit() for ch in core)
    or "." in core.rstrip(".")
    or any(ch.isupper() for ch in core[1:])
  )


def _decide(core: str, prose: ProseIndex, before: Optional[str], after: Optional[str]) -> Optional[bool]:
  """True → capitalise, False → lower-case, None → the evidence is too weak
  to say: the word is capitalised only inside longer names ("Diplomatic"
  in a byline, "Falkland" in "Falkland Islands"), or only opens sentences
  (nothing tells a proper noun from grammar)."""
  own, in_name, lows = prose.counts(core, before, after)
  if own > lows:
    return True
  if lows > own:
    return False
  if own == 0:
    return None if in_name or prose.opens_sentences(core) else False
  return None


# Words a Title Case heading capitalises or not by convention, so they say
# nothing about whether the rest was meant as names.
_SMALL = {
  "a", "an", "the", "of", "on", "in", "at", "to", "for", "by", "over", "with", "and", "or", "vs", "from", "as",
  "into", "about", "after", "amid", "against", "under", "between", "without", "through", "during", "before",
}


def _title_cased(words: List[str]) -> bool:
  """Every substantive word after the first capitalised — the model wrote
  Title Case, so its capitals carry no information about names."""
  cores = [re.sub(r"^[^A-Za-z]+", "", w) for w in words[1:]]
  substantive = [c for c in cores if c and c.lower() not in _SMALL and not _fixed_case(c)]
  return len(substantive) >= 2 and all(c[0].isupper() for c in substantive)


def _case_word(word: str, prose: ProseIndex, first: bool, before: Optional[str], after: Optional[str], title: bool) -> str:
  if "-" in word.strip("-"):
    # A hyphenated name the prose has as one token ("Al-Aqsa") is judged
    # whole; anything else part by part ("secretary-NSA").
    parts = word.split("-")
    if not first and prose.seen(word) and _decide(word, prose, before, after) is True:
      return "-".join(p if not p or _fixed_case(p) else p[0].upper() + p[1:] for p in parts)
    return "-".join(
      _case_word(p, prose, first and j == 0, parts[j - 1] if j else before, parts[j + 1] if j + 1 < len(parts) else after, title)
      if p else p for j, p in enumerate(parts)
    )
  m = re.match(r"^([^A-Za-z]*)([A-Za-z][A-Za-z0-9'’.]*)(.*)$", word)
  if not m:
    return word
  lead, core, tail = m.groups()
  if first:
    if core[0].islower() and core[1:].islower():
      core = core[0].upper() + core[1:]
    return lead + core + tail
  if _fixed_case(core):
    return word
  verdict = _decide(core, prose, before, after)
  if verdict is None:
    # Weak evidence: the model's own casing stands, unless it wrote the
    # whole heading in Title Case, which says nothing — then lower-case.
    verdict = False if title else None
  if verdict is True:
    core = core[0].upper() + core[1:]
  elif verdict is False:
    core = core.lower()
  return lead + core + tail


def heading_case(name: str, prose: ProseIndex) -> str:
  """The heading in sentence case: the first word capitalised, every other
  word as the story's prose writes it mid-sentence, next to the same
  neighbour when the prose has that pair ("Foreign" in "Foreign Minister"
  but "foreign policy"). A word the prose never uses is lower-cased —
  headings are made of their claims' words, so a proper noun is always
  seen; a common word may only be seen inflected — and acronyms and the
  like stay as written."""
  words = name.split()
  title = _title_cased(words)
  return " ".join(
    _case_word(w, prose, i == 0, words[i - 1] if i else None, words[i + 1] if i + 1 < len(words) else None, title)
    for i, w in enumerate(words)
  )


# ── Reading order, guarded (pure) ──────────────────────────────────────────

def _permutation(values, n: int) -> List[int]:
  """`values` as a permutation of range(n), else []."""
  order = _int_list(values, n)
  return order if len(order) == n else []


def _stem(k: str) -> str:
  """"adversaries"/"adversary", "sanctions"/"sanction": the plural and the
  common verb endings off, when a 4+ letter stem remains."""
  for suffix, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ing", "")):
    if k.endswith(suffix) and len(k) - len(suffix) >= 4:
      return k[: -len(suffix)] + rep
  return k


def _subject_words(text: str) -> Set[str]:
  return {_stem(k) for k in (_key(w) for w in _WORD.findall(text)) if len(k) >= 4 and k not in _STOP}


def opener_fits(headline: str, heading: str) -> bool:
  """The one thing code can check about an order: the block put first must
  share a subject word with the headline (a stem or an inflection counts)."""
  hw, bw = _subject_words(headline), _subject_words(heading)
  return any(a == b or _inflection(a, b) for a in bw for b in hw)


def accept_order(order: List[int], blocks: List[Tuple[str, List[int]]], headline: str) -> Tuple[List[int], str]:
  """The order to apply and the report word: "same", "changed", "kept" (no
  usable order came back) or "refused" (its opener is off the headline)."""
  identity = list(range(len(blocks)))
  if len(order) != len(blocks):
    return identity, "kept"
  if order == identity:
    return identity, "same"
  if not opener_fits(headline, blocks[order[0]][0]):
    return identity, "refused"
  return order, "changed"


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
  return prompts.get("news_collection_review.group").format(
    headline=headline, claims=_numbered(claims, indices), feedback=fb, min_block=MIN_BLOCK, max_block=MAX_BLOCK,
  )


def build_rescue_prompt(
  headline: str, claims: List[str], lone: List[int], sources: List[NewsArticleSource], alive: Optional[List[int]] = None,
) -> str:
  """`alive` restricts the existing-claims listing to the claims still live
  after merges; the lone claims keep their own numbering."""
  existing = [claims[i] for i in alive] if alive is not None else claims
  return prompts.get("news_collection_review.rescue").format(
    headline=headline,
    claims="\n".join(f"- {t}" for t in existing),
    lone=_numbered(claims, lone),
    sources=_sources_block(sources, RESCUE_SOURCE_BUDGET, indexed=True),
    calendar=calendar_block([source_anchors(s.index, s.content, s.published_at) for s in sources])
    or "(no publication dates known — resolve no relative date)",
    min_words=MIN_CLAIM_WORDS,
  )


def build_review_prompt(headline: str, claims: List[str], collections: List[Tuple[str, List[int]]]) -> str:
  return prompts.get("news_collection_review.review").format(
    headline=headline,
    max_words=MAX_CLAIM_WORDS,
    max_growth=MAX_MERGE_GROWTH,
    claims=_numbered(claims),
    collections="\n".join(f"{name}: {idx}" for name, idx in collections),
  )


def _blocks_listing(claims: List[str], blocks: List[Tuple[str, List[int]]]) -> str:
  return "\n".join(
    f"[{b}] \"{name}\"\n" + "\n".join(f"   {i}. {claims[i]}" for i in idx) for b, (name, idx) in enumerate(blocks)
  )


def build_check_prompt(headline: str, claims: List[str], blocks: List[Tuple[str, List[int]]], lone: List[int]) -> str:
  lone_text = f"\nLONE CLAIMS (no block yet):\n{_numbered(claims, lone)}\n" if lone else ""
  return prompts.get("news_collection_review.check").format(headline=headline, blocks=_blocks_listing(claims, blocks), lone=lone_text)


def build_order_prompt(headline: str, claims: List[str], blocks: List[Tuple[str, List[int]]]) -> str:
  return prompts.get("news_collection_review.order").format(
    headline=headline, blocks=_blocks_listing(claims, blocks),
    order_rule=prompts.get("news_collection_review.order_rule"),
  )


def parse_order(raw: str, n_blocks: int) -> List[int]:
  """The order the model returned as a permutation of the blocks, or []."""
  return _permutation(_json_object(raw).get("order"), n_blocks)


def build_source_check_prompt(sentences: List[str], sources: List[NewsArticleSource]) -> str:
  return prompts.get("news_collection_review.source_check").format(
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

def order_call(call, headline: str, claims: List[str], blocks: List[Tuple[str, List[int]]], report: CollectionReviewReport) -> List[int]:
  """The reading order of `blocks` from one cheap call, or [] — the order is
  never worth failing the story for, so a failed call is a note in the
  report and the blocks stay as grouped. Its seconds land in the report."""
  t0 = time.time()
  try:
    return parse_order(call(build_order_prompt(headline, claims, blocks), ORDER_THINKING), len(blocks))
  except Exception as e:  # noqa: BLE001
    report.rejected.append(f"order: {str(e)[:80]}")
    return []
  finally:
    report.steps["order"] = round(report.steps.get("order", 0.0) + time.time() - t0, 1)


def check_and_order(
  call, headline: str, claims: List[str], blocks: List[Tuple[str, List[int]]], lone: List[int], report: CollectionReviewReport,
) -> Tuple[Check, List[int]]:
  """The check and the order call side by side on the same blocks: the
  order is shorter than the check, so the story waits for the check alone."""
  with ThreadPoolExecutor(max_workers=2) as pool:
    # bind_context: a pool thread starts with an empty context, so the run's
    # override scope would not reach the calls without it.
    checking = pool.submit(bind_context(call), build_check_prompt(headline, claims, blocks, lone), CHECK_THINKING)
    ordering = pool.submit(bind_context(order_call), call, headline, claims, blocks, report)
    return parse_check(checking.result(), blocks, lone), ordering.result()


def order_and_case(
  resp: NewsCollectionReviewResponse, headline: str, prose: ProseIndex, call, report: CollectionReviewReport,
) -> None:
  """A discarded review ships the extraction's own blocks; they still get the
  reading order and the headings' case, so the reader sees them like any
  other story's. Collections, their order and the claims' topics are
  renamed together."""
  blocks = [(c.name, list(c.claim_indices)) for c in resp.collections]
  order, report.order = accept_order(order_call(call, headline, [c.text for c in resp.claims], blocks, report), blocks, headline)
  cased = {c.name: heading_case(c.name, prose) for c in resp.collections}
  resp.collections = [resp.collections[i].model_copy(update={"name": cased[resp.collections[i].name]}) for i in order]
  resp.collection_order = [c.name for c in resp.collections]
  resp.claims = [c.model_copy(update={"topic": cased.get(c.topic, c.topic)}) for c in resp.claims]


def _gemini(prompt: str, thinking_level: Optional[str]) -> str:
  client = genai.Client(
    api_key=settings.gemini_api_key,
    http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
  )
  kwargs: dict = {"temperature": llm.get("news_collection_review_temperature"), "response_mime_type": "application/json"}
  if thinking_level:
    kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
  return client.models.generate_content(
    model=llm.get("news_collection_review_model"), contents=prompt, config=types.GenerateContentConfig(**kwargs),
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
  prose = ProseIndex([headline, *(c.text for c in claims), *(s.content for s in sources)])

  def stamp(step: str) -> None:
    now = time.time()
    report.steps[step] = round(report.steps.get(step, 0.0) + now - last[0], 1)
    last[0] = now

  def discard(reason: str) -> NewsCollectionReviewResponse:
    report.rejected.append(f"{reason} — review discarded")
    logger.warning(f"collection review discarded for '{headline[:60]}': {reason}")
    # The extraction's own blocks ship, but the reader still gets the order
    # and the case: one cheap call, fail-open.
    order_and_case(untouched, headline, prose, call, report)
    report.seconds = round(time.time() - t0, 1)
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

    # 4 + 5. Check both ways, the reading order alongside; one regroup with
    # the reasons (and its own order); a second rejection discards.
    check, ordering = check_and_order(call, headline, texts, blocks, lone, report)
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
      check2, ordering = check_and_order(call, headline, texts, retry_blocks, retry_lone, report)
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

    # What the reader sees is decided once, last, on the finished blocks:
    # the reading order, guarded, then the headings' case.
    order, report.order = accept_order(ordering, blocks, headline)
    blocks = [(heading_case(name, prose), idx) for name, idx in (blocks[i] for i in order)]

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
