"""News debate-card generation, built directly on the product definition."""

NEWS_DEBATE_CLAIM_PROMPT = """You write debate cards for a news product.

THE DEFINITION — every card must satisfy it:

A debate claim sounds like a headline: clear, direct, and it takes a definite
position. It states a proposition for which clear, large or significant groups
are genuinely debating — or would clearly debate — for and against, in society
or online. A reader immediately knows what agreeing and disagreeing mean.

Read the headline, the extracted story facts, and the sources. Then compose
3-5 debate claims that this story raises. Return up to 7 candidates, strongest
first — an independent reviewer picks the published set — and return fewer,
or none, when the story genuinely raises fewer real debates. Never pad,
mirror, or weaken a claim to reach a count.

WHERE TO LOOK

Test every central event, actor, response, consequence, and unresolved tension
through these lenses:
- Policy or response: what should a named government, institution, company, or
  community do about the central situation?
- Legitimacy, rights, or ethics: is the central action justified, fair, safe,
  proportionate, or acceptable?
- Accountability: who bears responsibility, and under what standard?
- Cause or forecast: what disputed interpretation or outcome do informed
  groups genuinely disagree about?
- Tradeoff: which value or approach should take priority, when the story
  presents a real choice?
- Societal instance: what established public divide is this story directly an
  instance of? Corporate and market stories usually carry their real debate
  here — regulation of the category, public risks, market structure, openness,
  labor, privacy, safety — rather than in the company's own tactics.

A story's real debates rarely all share one grammatical form. When candidates
are equally strong on different forms, prefer form variety across the set —
an all-"should" list often means an evaluative, causal, predictive, or
disputed-factual dispute went unfound. Never convert a claim into a different
form to manufacture variety.

The story is the factual basis: claims are composed from what it reports, and
must not assert events, numbers, motives, or consequences it does not contain.
General knowledge is for recognizing the real public divides those facts
activate, and for naming the sides.

WHAT DOES NOT QUALIFY

- A business tactic or generic recommendation ("the company should expand X")
  with no real constituencies divided over it. A decision with two imaginable
  options is not automatically a public debate.
- A market-performance forecast — stock price, valuation, profitability,
  revenue, fund flows, deal value, commercial timeline. A company's hope or an
  analyst's projection is investor speculation, not a societal debate, unless
  the sources show recognizable groups publicly disputing that exact forecast.
- Allocation or strategy advice: what investors, pension funds, venture funds,
  or enterprises should buy, sell, fund, avoid, or adopt as strategy is advice
  to market participants, not a societal debate, unless the sources show
  recognizable public constituencies disputing that exact question.
- A straightforward reported fact someone could merely deny.
- A dispute about a background item, neighboring event, or roundup mention
  rather than this story.

ONE CLAIM PER QUESTION

Every returned claim becomes its OWN debate. Reduce each candidate to the
neutral question it answers and return at most one claim per question:
- A proposition and its counterclaim are one debate, never two cards.
- A broad policy and one of its named implementations are the same axis.
- A requested outcome and a rationale for that same outcome are the same axis.

BAD pair (one mirrored question):
- "Riverton's congestion charge will reduce downtown traffic."
- "Riverton's congestion charge will not reduce downtown traffic."

GOOD pair (two independent questions raised by the story):
- "Riverton's congestion charge will reduce downtown traffic."
- "Riverton's congestion charge unfairly burdens shift workers."

CARD STYLE

- A direct assertive proposition, never a question. No hedging ("may",
  "could", "some argue"), no "whether", no bare "it" or "the policy".
- Aim for 6-10 words; use 11-14 when a named actor or essential distinction
  requires it; 20 words is the hard maximum.
- Name the actor or subject plus the contested action, judgment, or forecast.
  Drop dates, amounts, and trailing justification clauses unless they are the
  contested issue.
- Never use evidentiary-summary framing such as "signals", "proves", or
  "demonstrates" — state the disputed position itself.
- Style-only examples of the shape (never copy their subjects):
  "Orion should disclose its automated hiring criteria." ·
  "Aster's battery design poses unacceptable safety risks." ·
  "Riverton's housing shortage is driven by zoning restrictions." ·
  "Mosaic fusion power will be commercially viable by 2040." ·
  "Vale's recount excluded legally eligible ballots."

For every candidate provide the neutral question it answers, the two real
opposing positions with the constituency, institution, or worldview behind
each side ("support it" / "oppose it" is not enough), and the indices of the
sources this debate arises from.

Return only valid JSON in this exact shape:

{{
  "debate_claims": [
    {{
      "neutral_question": "The neutral question this proposition answers.",
      "opposing_positions": ["A recognizable constituency: one position", "An opposing constituency: the other position"],
      "source_indices": [0],
      "text": "The concise headline-style proposition."
    }}
  ]
}}

Use an empty debate_claims array when nothing satisfies the definition. Do not
include markdown, explanations, or fields outside this JSON object.

INPUT

headline
{headline}

story facts (extracted claims, for reference)
{central_claims}

sources (use each object's index for source_indices)
{sources}
"""
