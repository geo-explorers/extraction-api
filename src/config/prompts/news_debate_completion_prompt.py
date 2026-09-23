"""Targeted recall pass used when semantic review leaves a single debate."""

NEWS_DEBATE_UNDERFILLED_RESCUE_PROMPT = """You complete an underfilled news Debate collection.

An earlier pass plus independent review left {survivor_count} publishable
debate claims. A useful collection holds 2-5 independent claims. Find at least
{minimum_needed} and at most {maximum_new} NEW claims when the story genuinely
raises them. Returning fewer is correct when it does not; never pad, mirror,
or weaken a claim to reach a count.

THE DEFINITION every new claim must satisfy:

A debate claim sounds like a headline: clear, direct, and it takes a definite
position. It states a proposition for which clear, large or significant groups
are genuinely debating — or would clearly debate — for and against, in society
or online.

Search every unused central fact, actor, consequence, and lens: policy
response, legitimacy/rights/ethics, accountability, cause/consequence, real
tradeoffs, and the societal instance — the established public divide this
story is directly an instance of. Corporate and market stories usually carry
their real debate in that societal lens (regulation of the category, public
risks, market structure, openness, labor, privacy, safety) rather than in the
company's own tactics.

Every new claim must belong to this story's central disagreement: finish
"this belongs under this story because the disagreement is about ______"
with the same answer the accepted claims give. A strong debate about a
neighbouring controversy that merely shares the story's person, institution,
country, or event does not qualify.

The story is the factual basis: a new claim must not assert events, numbers,
motives, or consequences it does not contain. General knowledge is for
recognizing real public divides and naming their sides.

EXCLUSIONS
- Do not repeat, negate, narrow, broaden, or re-rationalize any previously
  attempted neutral question. A proposition and its counterclaim are one axis.
- Do not build on a background item, neighboring event, or roundup mention.
- Do not propose a business tactic or generic recommendation with no real
  constituencies divided over it.
- Do not propose a market-performance forecast (stock price, valuation,
  profitability, revenue, fund flows, deal value, commercial timeline) unless
  the sources show recognizable groups publicly disputing that exact forecast.
- Do not propose allocation or strategy advice: what investors, funds, or
  enterprises should buy, sell, fund, avoid, or adopt as strategy is advice
  to market participants, not a societal debate.
- Do not propose a motive the sources do not state, a pure prediction whose
  only disagreement is guessing what will happen, or a card almost everyone
  would accept.
- Do not smuggle a verdict into the wording ("reckless", "corrupt",
  "betrayal", "severely damaged"): both sides must be able to accept the
  words and disagree only about whether the proposition is true or
  justified.

CARD STYLE
- Direct assertive proposition, never a question; no hedging, no "whether",
  no bare "it", "the administration", or "the president". Aim for 6-10
  words; 20 words is the hard maximum. Drop trailing justification clauses
  and defensive qualifiers unless they are the contested issue.
- Anchor to this story: when a story-specific claim naming the actor and a
  category-general claim are equally strong, prefer the specific one; never
  generalize a claim the story states specifically.
- At most one claim in the finished collection may prescribe ("should",
  "must", "needs to", "ought to" and their equivalents): when an accepted
  claim already prescribes, every new claim states a judgment instead (a
  cause, a justification, a tradeoff, effectiveness, proportionality,
  responsibility, a threshold, a consequence). Never convert a claim into
  another shape to satisfy the rule; find the story's other judgment.
- Draft each new axis in both directions and keep one. When the accepted
  claims all lean the same way (for or against a government, a party,
  enforcement, regulation, one side of a conflict), prefer the opposite
  direction for a new axis when it is equally strong. Never add the
  counterclaim of an accepted claim.
- Provide the neutral question, two real opposing positions naming the
  constituency or worldview behind each side, and the indices of the sources
  the debate arises from.

List the strongest independent candidates first. Return an empty array rather
than filler. Return only valid JSON in this exact shape:

{{
  "debate_claims": [
    {{
      "neutral_question": "A new neutral question not previously attempted.",
      "opposing_positions": ["A recognizable constituency: one position", "An opposing constituency: the other position"],
      "source_indices": [0],
      "text": "A concise independent debate proposition."
    }}
  ]
}}

INPUT

headline
{headline}

story facts (extracted claims, for reference)
{central_claims}

already accepted claims
{surviving_candidates}

previously attempted axes and review outcomes
{attempted_axes}

full sources
{sources}
"""
