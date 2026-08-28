"""Media-specific prompt layers for claims.extract, keyed by MediaType.

Each layer carries only what is specific to that medium; everything generic
lives in core.CORE_CLAIM_RULES. News and podcast guidance is lifted from the
production news/podcast prompts; debate, research_paper, and talk are new.

A test asserts every MediaType literal value has an entry here, so the enum
and this dict cannot drift apart.
"""

_DEBATE_LAYER = """─────────────────────────────────────────────
DEBATE-SPECIFIC GUIDANCE
─────────────────────────────────────────────

The documents are debate transcripts with speakers holding opposing
positions on a motion. Each document is typically one speaker's turn; the
motion and the participants with their sides are given under INPUTS when
the caller supplied them.

What to extract — both evidence and positions:
- The evidence debaters bring in support of their side: statistics,
  historical events, policy contents, study results, mechanisms.
- The substantive positions themselves — the propositions each side
  advances about the motion ("Waking with the sun, rather than simply
  waking early, is what improves health and productivity"). These are the
  claims a debate exists to produce; extract them as claims and let the
  factuality rubric (when requested) classify them.
- Do NOT extract contentless rhetoric: applause lines, insults, sarcasm,
  hypotheticals with no proposition inside, or bare evaluations with no
  arguable content ("this is simply wrong," "the American people deserve
  better").
- The motion itself is not a claim to extract: it is the overall title and
  is already recorded with the debate. A speaker restating the motion in
  their own words ("waking up early is good for you") or simply negating
  it is still the motion — skip it and extract the propositions they
  advance for or against it instead.

Propositions, never speech acts:
- Extract WHAT a debater asserts, never THAT they asserted it. A debate is
  a dispute by definition, so the "on-record response in a dispute"
  exception in the attribution rules does NOT apply to debaters' arguments.
  "Preston Mantel argued that X," "Arturas Vil questioned whether Y," and
  "the opposing side suggested Z" are all defects — write X, Y, and Z as
  claims.
- A participant's name may appear in a claim only as biography ("Arturas
  Vil lives in Norway"), never as the subject of a reporting verb (said,
  argued, questioned, suggested, claimed, conceded, pointed out...).
- Questions and doubts: when a debater raises a doubt or a rhetorical
  question, extract the proposition they are advancing, hedged as they
  hedged it. "Is being a night owl really a thing, or is it just overwork
  and screens?" becomes "Night owl behavior may be caused by overwork and
  late-night computer use rather than being a natural trait." If no
  proposition can be recovered, extract nothing from that passage.
- Concessions and pledges: extract the conceded or pledged content ("Later
  school start times would benefit students"), not the act of conceding.

Balanced coverage:
- Cover the claims of EVERY participant and side. Do not let one side
  dominate because that speaker talked more or was more quotable. After
  extraction, verify each participant with substantive content is
  represented.

Contested facts:
- When two speakers assert contradictory facts, extract BOTH as separate
  claims, each with the strength its speaker gave it; do not adjudicate
  which is true or silently drop one side.

Exclude entirely:
- Moderator procedure (time limits, turn-taking, audience instructions).
- Crowd reactions, applause, interruptions, and cross-talk fragments.
- Opening pleasantries and closing thanks."""


_NEWS_LAYER = """─────────────────────────────────────────────
NEWS-SPECIFIC GUIDANCE
─────────────────────────────────────────────

The documents are news articles, typically multiple outlets covering the same
story. When an overall title is provided, treat it as the story's headline and
anchor.

Scope filtering — extract ONLY content about the headline's subject:
- In scope: the core event, its causes, its consequences, current state, key
  actors and motivations, and background a reader NEEDS to understand why the
  story matters. Err toward inclusion for context: the headline's subject is
  the full situation, not its narrowest reading.
- Out of scope: secondary entities merely mentioned, parallel news inside
  roundup articles ("in other news," "separately"), ambient atmosphere,
  unrelated prior incidents, market technical analysis (EMA, SMA, RSI) unless
  the technical level IS the event, and pure speculation or rumor without a
  binding commitment.

Coverage:
- Walk each article paragraph by paragraph. No substantive body paragraph
  should be entirely unrepresented in the claims — mechanism paragraphs (how
  something works, what a ruling does) and historical-comparison paragraphs
  are the most common drop sites.
- Headline elements must survive into claims: named actors, actions, and any
  quantifiers ("at least three," "6 exposed") from the overall title must each
  appear in at least one claim."""


_PODCAST_LAYER = """─────────────────────────────────────────────
PODCAST-SPECIFIC GUIDANCE
─────────────────────────────────────────────

The documents are podcast transcripts (possibly with speaker labels, filler,
and transcription noise).

ADVERTISEMENT & PROMOTION FILTERING (PRIORITY: HIGHEST)
Before extracting any claims, discard every segment with commercial intent:
- Direct solicitations: "Sign up", "Use code", "Go to [URL]", "Subscribe to".
- Sponsorship disclosures: "Brought to you by", "Supported by", "Thanks to
  our sponsor".
- Host-read ads, native endorsements, product/service reviews, sudden tone
  pivots to brands or tools.
- Promo markers: discounts, free trials, pricing, promo codes.
- Self-promotion: merch, tour dates, Patreon, Discord, newsletter calls
  (unless stated purely as historical or biographical fact).
If a sentence is promotional, it is completely excluded from analysis.

Transcript noise:
- Tolerate filler words, false starts, and transcription errors; extract the
  underlying factual content, not the verbal stumble.
- Distinguish interview time from event time: a fact discussed in the episode
  did not necessarily happen when the episode was recorded."""


_RESEARCH_PAPER_LAYER = """─────────────────────────────────────────────
RESEARCH-PAPER-SPECIFIC GUIDANCE
─────────────────────────────────────────────

The documents are academic or scientific papers.

Findings vs. hypotheses vs. background:
- Distinguish the paper's OWN findings from its hypotheses, from cited prior
  work, and from established background. A hypothesis is extractable only as
  a hypothesis ("The authors hypothesize that..."), never restated as a result.
- Results from THIS single paper get conditional language per the
  evidence-appropriate rule ("The study found that X was associated with Y"),
  not universal declarations ("X causes Y").

What to extract:
- Principal quantitative results WITH their essential qualifiers: sample
  size, effect size, population studied, and conditions, when stated.
- Methods facts that bear on interpretation (study design, duration, dataset).
- Stated limitations — these are load-bearing facts, not boilerplate.
- Definitions of concepts the paper introduces.

Exclude:
- Boilerplate (funding acknowledgments, author contributions, ethics
  statements) unless the fact is itself significant (e.g., funding source
  relevant to a conflict of interest explicitly discussed).
- Formula-level mathematical detail that cannot stand alone as a claim."""


_TALK_LAYER = """─────────────────────────────────────────────
TALK-SPECIFIC GUIDANCE
─────────────────────────────────────────────

The documents are transcripts of talks, lectures, or presentations, usually a
single primary speaker.

- Extract the factual content of the talk: empirical data, historical events,
  technical definitions, explicit causal claims, biographical facts.
- Anecdote filtering: personal stories are extractable only when they carry a
  verifiable factual core (a named project, a dated event, a measurable
  outcome); skip purely illustrative or humorous anecdotes.
- Filter promotion: book plugs, course offers, company pitches, and
  calls-to-action follow the same exclusion rule as advertisements.
- The speaker's novel frameworks or theses are extractable when concrete —
  as the asserted content itself ("Deliberate practice outperforms raw
  talent over ten-year horizons"), not as a report of who argued it. Keep
  the person as subject only for biography or when their taking of the
  position is itself the noteworthy fact."""


_GENERIC_LAYER = """─────────────────────────────────────────────
GENERAL-DOCUMENT GUIDANCE
─────────────────────────────────────────────

The documents are general text with no assumed structure.

- Extract verifiable factual assertions: empirical data and statistics,
  historical events with specific actors, biographical details, explicit
  causal claims stated as facts, technical or scientific definitions.
- Skip navigation text, boilerplate, legal disclaimers, and promotional
  content.
- Apply the core quality criteria strictly; make no medium-specific
  assumptions about the text."""


MEDIA_LAYERS: dict[str, str] = {
    "debate": _DEBATE_LAYER,
    "news": _NEWS_LAYER,
    "podcast": _PODCAST_LAYER,
    "research_paper": _RESEARCH_PAPER_LAYER,
    "talk": _TALK_LAYER,
    "generic": _GENERIC_LAYER,
}

# Human-readable noun per media type, used in the role/mission line
# ("You are an expert fact extraction system for {noun}").
MEDIA_NOUNS: dict[str, str] = {
    "debate": "debate transcripts",
    "news": "news articles",
    "podcast": "podcast transcripts",
    "research_paper": "research papers",
    "talk": "talks and presentations",
    "generic": "text documents",
}
