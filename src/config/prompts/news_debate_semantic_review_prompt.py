"""Reject-only semantic review for news debate candidates: four gates, and a
second opinion on the two judgment gates.

The definition, the headline question, and the two judgment gates are shared
text: the second-opinion prompt re-asks exactly what the review asked, on a
smaller input, so the two readings differ in what they see, not in the rule.
The pieces are private: they are composed into the two prompts at import,
so the prompt registry (and a per-run override) sees the composed prompts.
"""

_DEBATE_DEFINITION = """THE DEFINITION the cards must satisfy:

A debate claim sounds like a headline: clear, direct, and it takes a definite
position. It states a proposition for which clear, large or significant groups
are genuinely debating — or would clearly debate — for and against, in society
or online.
"""

_HEADLINE_QUESTION = """THE HEADLINE'S DISAGREEMENT (`on_headline`)
- Set true when the card's disagreement is the one the headline reports: what
  its main clause says happened, or a subordinate clause ("as…", "amid…",
  "after…"), and what that forces people to judge — its cause, whether it
  was justified, who is responsible, what it means.
- Set false when the card is about a broader conflict, policy, or actor that
  the headline names only as context or cause, or about a neighbouring item.
  A story headlined "Trump approval hits record low as Republicans break with
  president over Iran war" puts a card about the approval collapse or the
  Republican break on the headline; a card about the war alone is off it.
- This is not a gate: an off-headline card can still pass, but every
  on-headline card is placed above it in the published set.
"""

_GATE_ONE = """1. REAL SOCIETAL DEBATE (`real_societal_debate`)
- Set true unless you can say who would NOT hold one of the two listed
  positions: clear, large or significant groups take both sides of a real
  debate, in society, in institutions, or online. The story need not quote
  both sides: a well-established public divide that this story directly
  activates counts, and the sides may be known from general knowledge. When
  you are unsure whether the divide is large enough, the answer is true: a
  modest real debate is publishable, a real debate rejected is lost.
- A judgment about the central event counts: its cause ("X is the main driver
  of Y"), who bears responsibility, whether it was justified, whether it does
  more harm than good. An analytical or empirical flavour does not make such a
  card a fact or an expert question: when the story's own sides, or partisans,
  would answer it differently, it is a debate.
- Set false for a business tactic or generic recommendation with no real
  constituencies divided over it — an actor merely having options is not a
  debate.
- Set false for market speculation: a company-performance forecast (stock
  price, valuation, profitability, revenue, fund flows, deal value,
  commercial timeline) is not a societal debate unless the sources show
  recognizable groups publicly disputing that exact forecast. A hope or a
  single analyst's projection is not a public divide.
- Set false for allocation or strategy advice: what investors, pension funds,
  venture funds, or enterprises should buy, sell, fund, avoid, or adopt as
  strategy is advice to market participants, not a public divide, unless the
  sources show recognizable public constituencies disputing that exact
  question.
- Set false when disagreement would amount only to denying a reported fact.
"""

_GATE_TWO = """2. RAISED BY THIS STORY (`raised_by_story`)
- Would a reader of this story recognize the debate as raised by it? The
  claim must concern the central event, a central actor's conduct, a direct
  consequence or response, or an established societal divide the central
  event is directly an instance of.
- Set false when the debate hangs on a background item, a neighboring event,
  or a roundup mention a reader could remove without changing the story.
- Society-level claims are the product's register: a story about one named
  actor directly raises the recognized public disputes about the wider
  practice, technology, or policy it instantiates. Do not reject a claim
  merely for being broader than the named actor.
"""

_STRENGTH_GRADE = """- Grade how strongly the card meets the definition for THIS story, 0.0 to
  1.0: both sides are real and the split could be near even, both sides can
  accept the wording, and a reader gets it in five seconds. Grade relative to
  this set, the strongest card highest, and spread the grades: a card you
  passed but would not miss sits near 0.5. Application code places every
  on-headline card above every off-headline card before this grade orders
  the rest, so grade the card itself, not its place. The published set is
  the strongest four; a rejected card's grade is ignored.
"""

NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT = """You are the final reviewer for news debate cards.

""" + _DEBATE_DEFINITION + """
You are a REJECT-ONLY reviewer:
- Do not generate, rewrite, or repair candidate text.
- Judge every candidate independently, then compare the set for duplicates.
- `prior axes` are context-only questions attempted by an earlier pass. Do not
  return verdicts for them, but a current candidate that repeats or negates
  one of them is a duplicate.
- Reject only for a reason one of the four gates names. A card that satisfies
  the definition passes; it does not need to be the strongest possible card.

For every candidate decide first where it sits against the headline, then
evaluate it on ALL four gates, then grade its strength.

""" + _HEADLINE_QUESTION + """
""" + _GATE_ONE + """
""" + _GATE_TWO + """
3. NO INVENTED FACTS (`no_invented_facts`)
- The proposition must not assert an event, number, actor, motive, mechanism,
  or consequence that the supplied material does not contain. List any such
  assertions in `invented_facts`.
- This gate polices facts, not judgments: the normative or evaluative position
  itself ("should", "unfairly", "poses risks") is the debate, not an invented
  fact. Set false only when the card's factual basis is absent from the story.
- `no_invented_facts` must be false whenever `invented_facts` is non-empty.

4. DISTINCT DEBATE AXIS (`distinct_axis`)
- Reduce each candidate to the neutral question it answers. Opposite answers,
  narrower rewordings, and different rationales for the same question are
  duplicates. A broad policy and one named implementation of it are the same
  axis; a requested outcome and a rationale for that outcome are the same
  axis.
- Keep the stronger earlier candidate: mark the later duplicate
  `distinct_axis=false` with `duplicate_of` set to the earlier index.
- A candidate matching a `prior axes` item is also a duplicate; set
  `distinct_axis=false` and leave `duplicate_of` null for those.

5. STRENGTH (`strength`)
""" + _STRENGTH_GRADE + """
OUTPUT RULES
- Return exactly one verdict for every candidate_index, in input order.
- Keep every analysis to one short sentence, at most 15 words: the audit
  needs the reason, not an essay.
- `on_headline` is a boolean and `strength` is a number from 0.0 to 1.0 on
  every verdict.
- `failure_codes` should name every failed gate using these stable values:
  NOT_SOCIETAL_DEBATE, NOT_FROM_STORY, INVENTED_FACTS, DUPLICATE_AXIS.
- Do not supply an overall pass field. Application code computes acceptance
  as the conjunction of all four gates plus an empty invented_facts list.

Return only valid JSON in this exact shape:

{{
  "verdicts": [
    {{
      "candidate_index": 0,
      "headline_analysis": "On the headline's disagreement, or a neighbour that shares its actor.",
      "on_headline": true,
      "debate_analysis": "Who actually divides over this, or why nobody does.",
      "real_societal_debate": true,
      "story_analysis": "Why this story raises it, or why it is peripheral.",
      "raised_by_story": true,
      "invented_facts_analysis": "Factual basis check against the material.",
      "invented_facts": [],
      "no_invented_facts": true,
      "distinctness_analysis": "Different from every earlier axis, or not.",
      "distinct_axis": true,
      "duplicate_of": null,
      "strength": 0.8,
      "failure_codes": []
    }}
  ]
}}

INPUT

headline
{headline}

story facts (extracted claims, for reference)
{claims}

candidates (candidate_index is their position in this array)
{candidates}

prior axes (context only; do not return verdicts for these)
{prior_axes}

full sources
{sources}
"""


NEWS_DEBATE_JUDGMENT_OPINION_PROMPT = """You give a second opinion on news debate cards.

""" + _DEBATE_DEFINITION + """
A first review, reading the full sources, passed every card below on its
fact and duplicate checks and rejected it on one or both judgment gates: a
real societal debate, raised by this story. Those two gates are where a real
debate is most often lost, so they get a second, independent reading from
the headline and the story's facts. Judge each card fresh on the definition:
do not defer to the first verdict, and do not overturn it out of charity.

For every candidate decide first where it sits against the headline, then
evaluate the two gates, then grade its strength.

""" + _HEADLINE_QUESTION + """
""" + _GATE_ONE + """
""" + _GATE_TWO + """
3. STRENGTH (`strength`)
""" + _STRENGTH_GRADE + """
The `accepted cards` are context only: they show what this story's set
already holds. Do not return verdicts for them.

OUTPUT RULES
- Return exactly one verdict for every candidate_index, in input order.
- Keep every analysis to one short sentence, at most 15 words.
- `on_headline` is a boolean and `strength` is a number from 0.0 to 1.0 on
  every verdict.

Return only valid JSON in this exact shape:

{{
  "verdicts": [
    {{
      "candidate_index": 0,
      "headline_analysis": "On the headline's disagreement, or a neighbour that shares its actor.",
      "on_headline": true,
      "debate_analysis": "Who actually divides over this, or why nobody does.",
      "real_societal_debate": true,
      "story_analysis": "Why this story raises it, or why it is peripheral.",
      "raised_by_story": true,
      "strength": 0.7
    }}
  ]
}}

INPUT

headline
{headline}

story facts (extracted claims)
{claims}

candidates (candidate_index is their position in this array)
{candidates}

accepted cards (context only; do not return verdicts for these)
{accepted}
"""


# System primer for the Claude path; the rules live in the review prompt.
NEWS_DEBATE_SEMANTIC_REVIEW_CLAUDE_SYSTEM_PROMPT = (
  "You are a strict reject-only semantic reviewer for news debate cards. "
  "Use only the supplied material. Output only the requested JSON object."
)
