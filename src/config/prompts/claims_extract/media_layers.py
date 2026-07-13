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

The documents are debate transcripts with multiple speakers holding opposing
positions.

Factual assertions vs. rhetoric:
- Extract the verifiable factual assertions debaters make in support of their
  positions (statistics, historical events, policy contents, study results).
- Do NOT extract rhetorical flourishes, applause lines, insults, hypotheticals,
  or pure value judgments ("this is simply wrong," "the American people
  deserve better").
- A debater's stated position or commitment IS itself an extractable fact when
  concrete ("Candidate X pledged to veto any bill that...") — this is the
  attribution-keeping exception: the act of committing is the fact.

Balanced coverage:
- Cover the factual assertions of EVERY participant and side. Do not let one
  side's claims dominate because that speaker talked more or was more
  quotable. After extraction, verify each named participant with substantive
  factual content is represented.

Contested facts:
- When two speakers assert contradictory facts, extract BOTH as separate
  claims with evidence-appropriate hedging where warranted; do not adjudicate
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
- The speaker's novel frameworks or theses are extractable as attributed
  positions when concrete ("Jane Smith argues that...") — the biography/
  position exception to attribution stripping applies."""


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
