"""Source-anchored dates and numbers for extracted claims.

A claim may carry a date or a number only when code can point at where it
came from. Nothing else earns one:

- LITERAL: a source states it — the same day and month (any common format),
  the same year, the same number (after separators and scale words are
  normalised: "7 million" = "7m" = "7,000,000").
- RESOLVED BY CODE: a source says a weekday ("on Tuesday") or a relative word
  ("yesterday", "this week") and the claim's date is the one that phrase
  denotes against that source's publication date (or a full date the source
  itself states). The weekday arithmetic is done here, never by the model —
  on 476 curator corrections (2026-09-23) the model's own resolution was off
  by a day or a year in the great majority.
- PUBLICATION YEAR: a day-and-month may carry the publication year; a year
  alone may be the publication year ("this year"), the year before ("last
  year") or the year after ("next year") when a source uses that phrase.

Everything else is refused with a reason a repair prompt can act on. The
guard is pure: no model, no clock — a source without a parseable publication
date simply anchors nothing beyond its literal text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

MONTHS: Dict[str, int] = {
  "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
  "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
  "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
  "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# How far a weekday or relative word may reach from its anchor, either way.
RESOLVE_WINDOW_DAYS = 7
# A repaired claim may not grow past this (the same ceiling the review uses).
MAX_CLAIM_WORDS = 40

_MONTH_RE = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"

# Order matters: fuller patterns first so a bare year inside "March 3, 2026"
# is claimed by the full date, not counted twice.
_DATE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
  ("iso", re.compile(rf"\b({_YEAR})-(\d{{2}})-(\d{{2}})\b")),
  ("mdy", re.compile(rf"\b({_MONTH_RE})\.? (\d{{1,2}}){_ORD}(?:,? ({_YEAR}))?\b(?!\s*(?:%|percent))", re.I)),
  ("dmy", re.compile(rf"\b(\d{{1,2}}){_ORD} ({_MONTH_RE})\.?(?:,? ({_YEAR}))?\b", re.I)),
  ("my", re.compile(rf"\b({_MONTH_RE})\.?,? ({_YEAR})\b", re.I)),
  ("y", re.compile(rf"(?<![\d$€£¥.,-])({_YEAR})(?![\d,.]\d|\s*(?:%|percent))\b")),
]

_NUMBER_RE = re.compile(
  r"(?<![\w.,/-])([$€£¥]\s?)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|percent|per cent|trillion|billion|million|thousand|bn|mn|tn|m|k)?(?![\w/-])",
  re.I,
)
_SCALE = {"trillion": 1e12, "tn": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6,
          "thousand": 1e3, "k": 1e3}
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")

_RELATIVE_DAY: Dict[str, Tuple[int, int]] = {
  # phrase → (offset from anchor in days, tolerance either way)
  "yesterday": (-1, 1),
  "today": (0, 1),
  "tonight": (0, 1),
  "this morning": (0, 1),
  "this evening": (0, 1),
  "last night": (-1, 1),
  "overnight": (-1, 1),
  "earlier today": (0, 1),
}
_RELATIVE_SPAN: Dict[str, Tuple[int, int]] = {
  # phrase → (earliest offset, latest offset) from anchor
  "this week": (-7, 1),
  "earlier this week": (-7, 0),
  "last week": (-14, -5),
  "over the weekend": (-4, 0),
  "this weekend": (-3, 3),
  "last weekend": (-9, -2),
  "in recent days": (-10, 0),
}
_RELATIVE_YEAR: Dict[str, int] = {"this year": 0, "earlier this year": 0, "later this year": 0,
                                  "last year": -1, "next year": 1, "a year ago": -1}


@dataclass
class DateMention:
  text: str
  day: Optional[int]
  month: Optional[int]
  year: Optional[int]
  start: int
  end: int

  @property
  def full(self) -> bool:
    return self.day is not None and self.month is not None


@dataclass
class NumberMention:
  text: str
  value: float
  digits: str
  start: int
  end: int


@dataclass
class Problem:
  kind: str  # "date" | "year" | "number"
  token: str
  reason: str

  def __str__(self) -> str:
    return f"{self.kind} '{self.token}': {self.reason}"


@dataclass
class SourceAnchors:
  """Everything one source can vouch for, computed once."""
  index: int
  published: Optional[date]
  text_lower: str
  dates: List[DateMention]
  numbers: List[NumberMention]
  weekdays: Set[int] = field(default_factory=set)      # weekday indices the text mentions
  relative_days: Set[str] = field(default_factory=set)  # keys of _RELATIVE_DAY present
  relative_spans: Set[str] = field(default_factory=set)
  relative_years: Set[str] = field(default_factory=set)

  @property
  def anchor_dates(self) -> List[date]:
    """Publication date plus every full date the source states — a weekday
    can be resolved against any of them."""
    out: List[date] = [self.published] if self.published else []
    for m in self.dates:
      if m.full and m.year:
        try:
          out.append(date(m.year, m.month, m.day))  # type: ignore[arg-type]
        except ValueError:
          pass
    return out

  @property
  def years(self) -> Set[int]:
    return {m.year for m in self.dates if m.year} | ({self.published.year} if self.published else set())


# ── Parsing ──────────────────────────────────────────────────────────────


def _month(token: str) -> int:
  return MONTHS[token.lower().rstrip(".")]


def find_dates(text: str) -> List[DateMention]:
  """Every date mention in `text`, fullest match first, no overlaps."""
  taken: List[Tuple[int, int]] = []
  found: List[DateMention] = []

  def free(a: int, b: int) -> bool:
    return all(b <= s or a >= e for s, e in taken)

  for kind, pat in _DATE_PATTERNS:
    for m in pat.finditer(text):
      a, b = m.span()
      if not free(a, b):
        continue
      try:
        if kind == "iso":
          y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
          date(y, mo, d)
          found.append(DateMention(m.group(0), d, mo, y, a, b))
        elif kind == "mdy":
          d = int(m.group(2))
          if not 1 <= d <= 31:
            continue
          found.append(DateMention(m.group(0), d, _month(m.group(1)), int(m.group(3)) if m.group(3) else None, a, b))
        elif kind == "dmy":
          d = int(m.group(1))
          if not 1 <= d <= 31:
            continue
          found.append(DateMention(m.group(0), d, _month(m.group(2)), int(m.group(3)) if m.group(3) else None, a, b))
        elif kind == "my":
          found.append(DateMention(m.group(0), None, _month(m.group(1)), int(m.group(2)), a, b))
        else:
          found.append(DateMention(m.group(0), None, None, int(m.group(1)), a, b))
      except (ValueError, KeyError):
        continue
      taken.append((a, b))
  found.sort(key=lambda x: x.start)
  return found


def find_numbers(text: str, skip: Iterable[Tuple[int, int]] = ()) -> List[NumberMention]:
  """Every numeric token with its value after separators and scale words;
  spans in `skip` (dates) and clock times are left alone."""
  skip = list(skip) + [m.span() for m in _TIME_RE.finditer(text)]
  out: List[NumberMention] = []
  for m in _NUMBER_RE.finditer(text):
    a, b = m.span()
    if any(not (b <= s or a >= e) for s, e in skip):
      continue
    digits = (m.group(2) or "").replace(",", "") + (m.group(3) or "")
    try:
      value = float(digits)
    except ValueError:
      continue
    scale = (m.group(4) or "").lower()
    if scale in _SCALE:
      value *= _SCALE[scale]
    out.append(NumberMention(m.group(0).strip(), value, digits, a, b))
  return out


def parse_published(value: Optional[str]) -> Optional[date]:
  """The calendar date of an ISO-ish publication timestamp, or None."""
  if not value:
    return None
  s = value.strip()
  for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
    try:
      return datetime.strptime(s[: len(datetime.now().strftime(fmt))] if fmt != "%Y-%m-%dT%H:%M:%S.%f" else s.rstrip("Z").split("+")[0], fmt).date()
    except ValueError:
      continue
  try:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
  except ValueError:
    pass
  m = re.match(rf"({_YEAR})-(\d{{2}})-(\d{{2}})", s)
  if m:
    try:
      return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
      return None
  return None


def source_anchors(index: int, content: str, published_at: Optional[str]) -> SourceAnchors:
  text = content or ""
  lower = text.lower()
  dates = find_dates(text)
  anchors = SourceAnchors(
    index=index,
    published=parse_published(published_at),
    text_lower=lower,
    dates=dates,
    numbers=find_numbers(text, [(d.start, d.end) for d in dates]),
  )
  for i, w in enumerate(WEEKDAYS):
    if re.search(rf"\b{w}\b", lower):
      anchors.weekdays.add(i)
  anchors.relative_days = {k for k in _RELATIVE_DAY if k in lower}
  anchors.relative_spans = {k for k in _RELATIVE_SPAN if k in lower}
  anchors.relative_years = {k for k in _RELATIVE_YEAR if k in lower}
  return anchors


# ── Rules ────────────────────────────────────────────────────────────────


def _nearest_weekday(anchor: date, weekday: int) -> List[date]:
  """The most recent and the next occurrence of `weekday` around `anchor`
  (anchor itself when it is that weekday)."""
  back = anchor - timedelta(days=(anchor.weekday() - weekday) % 7)
  ahead = anchor + timedelta(days=(weekday - anchor.weekday()) % 7)
  return [back] if back == ahead else [back, ahead]


def _resolved_days(src: SourceAnchors) -> Set[date]:
  """Every calendar day a source's weekday and relative phrases can denote."""
  days: Set[date] = set()
  for anchor in src.anchor_dates:
    for wd in src.weekdays:
      for d in _nearest_weekday(anchor, wd):
        if abs((d - anchor).days) <= RESOLVE_WINDOW_DAYS:
          days.add(d)
    for key in src.relative_days:
      off, tol = _RELATIVE_DAY[key]
      for k in range(-tol, tol + 1):
        days.add(anchor + timedelta(days=off + k))
    for key in src.relative_spans:
      lo, hi = _RELATIVE_SPAN[key]
      for k in range(lo, hi + 1):
        days.add(anchor + timedelta(days=k))
  return days


def _year_from_publication(year: int, src: SourceAnchors) -> bool:
  """The year a source's own calendar gives: its publication year, or the
  year before / after when the text says "last year" / "next year". This is
  the only way a year gets ADDED to a day-and-month or a month — a year
  merely mentioned elsewhere in the article ("a 2025 report") licenses
  nothing, or "became mayor in February" plus a stray 2025 would pass as
  "February 2025"."""
  if not src.published:
    return False
  if src.published.year == year:
    return True
  return any(src.published.year + _RELATIVE_YEAR[k] == year for k in src.relative_years)


def _year_literal(year: int, src: SourceAnchors) -> bool:
  return any(s.year == year for s in src.dates)


def _date_supported(m: DateMention, src: SourceAnchors) -> Tuple[bool, str]:
  """Whether one source vouches for a date mention, and why not if not."""
  if m.full:
    literal = [s for s in src.dates if s.full and s.day == m.day and s.month == m.month]
    if literal:
      if m.year is None or any(s.year == m.year for s in literal) or (
        all(s.year is None for s in literal) and _year_from_publication(m.year, src)
      ):
        return True, ""
      return False, f"the source dates it {literal[0].text}, not {m.year}"
    try:
      wanted = {date(y, m.month, m.day) for y in ([m.year] if m.year else sorted(src.years) or [])}  # type: ignore[arg-type]
    except ValueError:
      return False, "not a real calendar date"
    if wanted & _resolved_days(src):
      return True, ""
    if src.weekdays or src.relative_days or src.relative_spans:
      names = ", ".join(WEEKDAY_NAMES[w] for w in sorted(src.weekdays))
      hint = f"; it says {names}" if names else ""
      pub = f" and was published {src.published.strftime('%A, %-d %B %Y')}" if src.published else ""
      return False, f"no source states this date{hint}{pub}"
    return False, "no source states this date"
  if m.month is not None:  # month + year
    assert m.year is not None
    if any(s.month == m.month and s.year == m.year for s in src.dates):
      return True, ""
    month_named = re.search(rf"\b{MONTH_NAMES[m.month - 1].lower()}\b|\b{MONTH_NAMES[m.month - 1][:3].lower()}\b", src.text_lower)
    if not month_named:
      return False, f"no source mentions {MONTH_NAMES[m.month - 1]}"
    if _year_from_publication(m.year, src):
      return True, ""
    return False, f"no source dates {MONTH_NAMES[m.month - 1]} to {m.year}"
  # bare year
  assert m.year is not None
  if _year_literal(m.year, src) or _year_from_publication(m.year, src):
    return True, ""
  return False, f"no source states the year {m.year}"


def _number_supported(n: NumberMention, src: SourceAnchors) -> bool:
  if n.digits and n.digits in src.text_lower:
    return True
  for s in src.numbers:
    if s.value == n.value or (n.value and abs(s.value - n.value) / n.value < 1e-6):
      return True
  return False


def check_claim(text: str, sources: Sequence[SourceAnchors], numbers: bool = True) -> List[Problem]:
  """Every date, year and number in `text` that no source vouches for.
  With no sources there is nothing to hold a claim to, so nothing is flagged
  — the sweep's verify-only nights must not write anyone off for a date they
  could not check."""
  problems: List[Problem] = []
  if not sources:
    return problems
  dates = find_dates(text)
  for m in dates:
    verdicts = [_date_supported(m, s) for s in sources]
    if any(ok for ok, _ in verdicts):
      continue
    reasons = [why for ok, why in verdicts if why]
    # Prefer the most specific explanation (a literal near-miss over a blanket "no source").
    reason = next((r for r in reasons if "not " in r and "no source" not in r), reasons[0] if reasons else "no sources")
    problems.append(Problem("year" if not m.full and m.month is None else "date", m.text, reason))
  if numbers:
    for n in find_numbers(text, [(d.start, d.end) for d in dates]):
      if not any(_number_supported(n, s) for s in sources):
        problems.append(Problem("number", n.text, "no source states this figure"))
  return problems


# ── Prompt help ──────────────────────────────────────────────────────────


def calendar_block(sources: Sequence[SourceAnchors]) -> str:
  """What the model must not compute itself: each source's publication day
  with its weekday, and what each weekday and 'yesterday' mean for it."""
  lines: List[str] = []
  years: Set[int] = set()
  for src in sources:
    if not src.published:
      lines.append(f"Source {src.index}: publication date unknown — resolve no relative date from it.")
      continue
    p = src.published
    years.add(p.year)
    days = ", ".join(
      f"{WEEKDAY_NAMES[w]} = {_nearest_weekday(p, w)[0].strftime('%-d %B %Y')}" for w in range(7)
    )
    lines.append(
      f"Source {src.index}: published {p.strftime('%A, %-d %B %Y')}. In this source: {days}; "
      f"yesterday = {(p - timedelta(days=1)).strftime('%-d %B %Y')}."
    )
  if years:
    lines.append(f"The year of these sources is {', '.join(str(y) for y in sorted(years))}. Write no other year unless a source states it.")
  return "\n".join(lines)


def describe_published(published_at: Optional[str]) -> Optional[str]:
  """'Thursday, 5 March 2026 (2026-03-05T…)' — the weekday spelled out so the
  model reads it instead of computing it."""
  d = parse_published(published_at)
  if not d:
    return published_at
  return f"{d.strftime('%A, %-d %B %Y')} ({published_at})"
