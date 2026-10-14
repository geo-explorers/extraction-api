"""Media-specific prompt layers for claims.extract, keyed by MediaType.

Each layer carries only what is specific to that medium; everything generic
lives in core.CORE_CLAIM_RULES. News guidance is lifted from the production
news prompt; debate, research_paper, and talk are new.

The podcast layer (revised 2026-10-14) trades recall for precision: podcast
speech is mostly opinion, anecdote and advice around a few checkable facts,
and this layer is the only place the generalized pipeline can say so. It
states an explicit precedence over the core's coverage, per-topic-minimum
and rewrite-not-delete rules, admits a claim only when it is specific,
checkable, record-worthy and self-contained, lists the podcast failure
classes to delete on sight, and carries the canonical-naming, role, legal-
matter and relative-time rules from the podcast.extract_claims prompt.
Measured on 6 episodes against the previous layer (per-claim judges,
gemini-2.5-pro, two runs of this layer averaged; extraction itself varies
+/-3 points run to run): claims 215 -> 164, not self-contained 20.0% ->
8.9% (6.1% / 11.7%), weak 11.2% -> 3.4%, claims passing both 68.8% -> 84%,
production facts retained 70% -> 55%.

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
  classification rubrics (when requested) label them.

Prefer the arguable proposition over the incidental fact:
- A debate's value is in what the two sides actually disagree about. When a
  passage contains both a broad position and the narrow detail offered in
  support of it, extract the position; extract the supporting detail as its
  own claim only when it carries independent weight (a named study, a
  statistic, a dated event).
- This is a matter of what to select, never of how to phrase. Represent each
  debater's position exactly as strongly as they put it: do not sharpen a
  hedged position into a flat assertion, and do not soften a flat assertion
  into a hedge. "Chatbots may increase dependence" and "chatbots increase
  dependence" are different claims and the speaker chose one of them. A
  claim that overstates a debater's position misrepresents them, which is a
  worse failure than a claim that is narrow.
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
- Rebuttals: when a speaker restates an opponent's point in order to rebut
  it ("My opponent says X, but..."), extract the speaker's OWN proposition —
  the rebuttal — not X. X belongs to the speaker who advanced it: extract it
  only from that speaker's own words, and never cite the rebutting
  speaker's document for it in document_indices. If the opponent's own
  words are not among the documents, do not extract X at all.

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
and transcription noise). Podcast speech is long, conversational, and mostly
opinion, anecdote and advice wrapped around a small number of checkable
facts. The job here is to find those facts and nothing else. Tolerate
filler words, false starts, and transcription errors: extract the underlying
factual content, never the verbal stumble.

PRECEDENCE FOR PODCASTS: this material is extracted for precision, not
coverage. The selection rules in this section override every coverage,
per-topic-minimum, re-scan, and "rewrite, do not delete" instruction
elsewhere in this prompt, and they override the core's rescue remedies for
ownerless content: "extract the underlying content instead," "attribute the
person or outlet who cited it," and "extract the asserted finding itself
without inventing a source" do not apply to podcasts; the claim is deleted.
Hedges that state a named study's evidential strength ("a UC Davis study
suggests…") are not forecasts and stay. For podcasts: a passage with no strong fact stays
unrepresented; a topic with fewer than two strong claims is folded into a
neighbour or dropped, never padded; and deleting is the default remedy for a
claim that fails any test below. Few, strong, self-contained claims are the
correct output. A typical one-hour conversation yields 10-30 claims; a
news-digest episode may yield more; an opinion, wellness, or self-help
episode may yield fewer than 10, or none.

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

THE ADMISSION TEST — a claim is extracted only if it passes ALL four:
1. Specific: it names a particular actor, organization, event, work, place,
   quantity, date, or a mechanism attributed to a named theory, law, or
   study. "Negative emotions are stronger than
   positive ones" names nothing; "[Lead author]'s [year] [institution]
   study found participants rated a surgery described as 70% successful
   more favourably than the same surgery described as 30% failing" does.
2. Checkable: a reader could confirm or refute it against a source outside
   this episode (a record, a publication, a filing, a dataset, a dated
   event, an on-record statement by a named public figure).
3. Record-worthy: it would belong in an encyclopedia entry, a news report,
   a biography, or a timeline. Conversational observations, however true,
   do not qualify.
4. Self-contained: it passes the Shuffle Rule with no help from the episode
   title, the topic label, or any other claim.
When a candidate fails any test, delete it. Never rewrite an opinion into a
fact, never add specificity the transcript does not state, and never keep a
claim because a topic would otherwise be empty.

NOT EXTRACTABLE FROM PODCASTS (delete on sight):
- Opinions, evaluations and value judgments: "is a poor strategy," "is
  under-emphasized," "is broken," "is the best," "should," "must," "matters."
- Advice, lessons, principles, and life philosophy: "the quality of your
  relationships determines the quality of your life," "focus on whichever
  production has the biggest problem."
- Generalizations about people, society, markets, or "many": "many people
  struggle to let go of resentment," "most founders underestimate."
- Analogies, metaphors, and framings: "Ethereum's scaling works like a
  franchise model," "a '1 to 100' phase."
- Forecasts and predictions, including hedged ones ("may," "could,"
  "is poised to," "is expected to"). Exception: a specific, dated, on-record
  forecast by a named public figure, company, or institution, where making
  the forecast is itself the noteworthy fact ("In its October 2026 outlook
  the IMF projected…").
- Personal anecdotes and feelings without a verifiable core (a named
  project, a dated event, a measurable outcome).
- Hypotheticals, thought experiments, rhetorical questions, host banter,
  listener questions, and remarks about the episode itself.
- Ownerless frames. These tokens mark a view or report with no named owner
  and make the claim non-extractable as written: "is considered," "is seen
  as," "is described as," "is noted for," "is said to," "is believed,"
  "is thought to," "is expected to," "is viewed as," "reportedly,"
  "according to reports/reporting," "reporting indicates/confirmed," "has
  been accused/criticized/credited" with no accuser named, "some argue,"
  "it is believed/suspected/unclear." Do not rescue such a claim by
  deleting the frame and asserting the content as fact, and do not rescue
  it by writing "the host" or "the guest." Keep the content only when the
  transcript names the public figure, outlet, court, or institution that
  holds the view or made the report, and then attribute it by that name
  ("The New York Times reported on October 9, 2026 that…"). Otherwise
  delete the claim.
- A guest's or host's views about their own field, company, or craft are
  still opinions. What a founder, scientist, or author says about their own
  work is extractable only as a specific, checkable fact: a figure, a date,
  a decision, a result, a named product or publication.

WHAT TO EXTRACT (when it passes the admission test):
- Empirical data and statistics with their source and period.
- Dated events with named actors (what happened, where, when, to whom).
- Biographical facts: roles, dates, affiliations, works, accomplishments.
- Named studies, papers, books, laws, rulings, products, and deals, with
  their essential result or content.
- Technical or scientific definitions and mechanisms, only when tied to a
  named theory, law, researcher, or study, and phrased as that source's
  content ("[Named researcher]'s [year] review found that negative
  events outweigh positive ones of equal size"). An unattributed generalization
  about how people, minds, bodies, markets, or societies work ("negative
  emotions are socially contagious," "the brain overgeneralizes from past
  threats") is not extractable in any wording, even when it sounds
  scientific.
- On-record statements by named public figures or institutions where the
  statement itself is the fact (announced, committed, ruled, testified).

FIX THE CANONICAL NAMES BEFORE WRITING
Podcast speech names its subject once and then uses shorthand for an hour:
"the attack," "the plane," "the co-pilot," "the study," "the company,"
"he." Claims are read one at a time, shuffled, with no title and no
neighbouring claims, so every claim carries the full identity itself. Before
writing, decide the canonical name of each central subject and reuse it
verbatim in every claim:
- The event: WHICH event, by actor plus name, place, or date — "the FlyDubai
  Dubai-Tel Aviv hijacking attempt," not "the attack" or "the incident."
- The people: full names when the transcript gives them. An unnamed person
  is identified by role plus the named event or organization — "the FlyDubai
  co-pilot," "Oman Air's chief executive" — never a bare role ("the
  co-pilot," "the CEO," "the judge," "the prosecutor," "the suspect"). If
  neither a name nor a named event or organization is available, the claim
  is not extractable.
- Organizations, products, studies, laws, places: full proper names.
- Things with no proper name (a lawsuit, a paper, a method, a protocol, a
  dataset): a descriptive identifier built from who/what/when, reused word
  for word in every claim about it. A claim may never rely on an
  introduction made in another claim.
- Anonymized stories: when the source withholds names, identity comes from
  institution, place, and date — "the district attorney handling the October
  2024 Cornell fraternity assault case." A bare "the incident," "the case,"
  "the party," "that night" is prohibited unless the same claim has already
  named it.
- Legal, policy, and news matters: every claim names the matter itself —
  the parties or court and subject of a case, the sponsor and subject of a
  bill, the agency and target of an investigation, the buyer and seller of
  a deal. "The lawsuit," "the litigation," "the ruling," "the
  investigation," "the ads," "the proclamation," "the ceasefire," "the
  defendants" are prohibited unless the same claim has already named the
  matter — including in the second and third claim about it.
- Fronted clauses obey the same rule: a possessive or pronoun may not
  precede its referent even in an opening phrase. Write "After Chris
  Camillo's seven-figure loss, Chris Camillo…," never "Prior to his
  loss, Chris Camillo…."
The topic label and episode title are NOT visible to the reader: carry their
identifying noun phrase into each claim.

TIME IN PODCASTS
- Distinguish interview time from event time: a fact discussed in the episode
  did not necessarily happen when the episode was recorded, and the
  recording date must not be stamped onto facts.
- Resolve relative time from the document's Published date when the
  resolution is unambiguous within about a week, as the core rule allows:
  "on Tuesday," "last week," "yesterday" in a news digest aired October 13,
  2026 become the absolute date; "last year" becomes the named year. A
  month without a year takes the year only when the transcript or the
  Published date makes the year of THAT event unambiguous.
- What cannot be resolved that way is prohibited and the claim is deleted
  or stated without the time word: "currently," "now," "recently," "so
  far," "at the time of recording," present-anchored spans ("over the last
  eight years," "the past decade," "the last 80 years") and relative
  weekdays with no resolvable anchor ("the following Tuesday"). Use the air
  date for "now"/"currently" only when the speaker is plainly describing the
  present, written as "as of October 2026".
- When no absolute anchor exists, state a durable fact in plain present tense
  with no time word; if the figure is meaningless without its date (a price,
  a poll lead, a market share, a ranking), delete the claim.

PODCAST STRENGTH AUDIT (mandatory, before output, claim by claim)
Read each claim alone. Delete it if it:
(a) contains a bare definite reference ("the attack," "the study," "the
    company," "the bill," "the lawsuit," "the ads," "the investigation") or
    a bare role ("the co-pilot," "the CEO") whose identity is not stated in
    that same claim — scan every "the <noun>" in the claim and ask whether a
    stranger could tell which one;
(b) uses a pronoun or possessive before its referent is named in the claim;
(c) contains "the speaker," "the host," "the guest," a reporting frame
    ("X argued that"), or any ownerless-frame token listed above ("is
    considered," "is described as," "reportedly," "has been accused"),
    unless the statement itself is the record-worthy fact and the person,
    outlet, or institution is named;
(d) contains a relative time word, a present-anchored span ("the last 80
    years"), a relative weekday, or a month without a year that the
    transcript or the Published date cannot resolve;
(e) names nothing specific, cannot be checked outside the episode, or is an
    opinion, forecast, analogy, lesson, or generalization in any wording.
Rewrite instead of deleting only for (a)-(d) and only when the transcript
or the document's Published date supplies the missing name or date. Expect to delete a large share of
your first draft; that is the intended outcome. If a summary is requested
and mentions a fact that fails the admission test, remove it from the
summary; never promote it to a claim."""


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
