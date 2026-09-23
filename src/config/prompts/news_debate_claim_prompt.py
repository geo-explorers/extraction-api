"""News debate-card generation, built directly on the product definition."""

NEWS_DEBATE_CLAIM_PROMPT = """You write debate cards for a news product.

THE DEFINITION — every card must satisfy it:

A debate claim sounds like a headline: clear, direct, and it takes a definite
position. It states a proposition for which clear, large or significant groups
are genuinely debating — or would clearly debate — for and against, in society
or online. A reader immediately knows what agreeing and disagreeing mean.

Read the headline, the extracted story facts, and the sources. Then compose
2-5 debate claims that this story raises. Return up to 7 candidates, strongest
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
- Cause or consequence: what disputed cause, interpretation, or consequence
  do informed groups genuinely disagree about?
- Tradeoff: which value or approach should take priority, when the story
  presents a real choice?
- Societal instance: what established public divide is this story directly an
  instance of? Corporate and market stories usually carry their real debate
  here — regulation of the category, public risks, market structure, openness,
  labor, privacy, safety — rather than in the company's own tactics.

THIS STORY'S DISAGREEMENT

Before writing, finish this sentence: "The central disagreement in this story
is about ______." When the story mainly reports what happened, ask what the
event forces people to decide, evaluate, or judge (whether the action was
justified, who should decide, what standard applies, who bears
responsibility, whether the response was proportional, whether the policy
works, which interest matters more) and compose from that judgment, not from
the summary. Every card must complete "this belongs under this story because
the disagreement is about ______" with that same answer. A debate about
another controversy that merely shares the story's person, institution,
country, or event does not belong, however strong it is.

The headline names the disagreement. The strongest card sits on the
headline's main clause; a subordinate clause ("as…", "amid…", "after…") may
carry the second. A story headlined "Trump approval hits record low as
Republicans break with president over Iran war" leads with a card about the
approval collapse (what caused it, what it means for the president), then
one about the Republican break; a card about the war alone belongs to
another story.

Anchor claims to this story wherever possible: when a story-specific claim
naming the actor and a category-general claim are equally strong, prefer the
specific one — "Aster should reimburse clients hit by the exploit" beats
"trading platforms should reimburse clients hit by exploits". Reach for the
general form only when the story's real debate has no named actor or instance
to carry it; never generalize a claim the story states specifically.

The story is the factual basis: claims are composed from what it reports, and
must not assert events, numbers, motives, or consequences it does not contain.
General knowledge is for recognizing the real public divides those facts
activate, and for naming the sides.

BOTH DIRECTIONS, THEN ONE

A card takes a firm position, but its wording must not hand one side the
argument before the debate starts: both sides must be able to accept the
words and disagree only about whether the proposition is true or justified.
For every axis, draft the cleanest card in each direction, for example
"Individualized suspicion is necessary before ICE detains workers in
workplace raids" and "Generalized suspicion is sufficient to detain workers
in ICE workplace raids", then keep exactly one, chosen for debate quality,
clarity, and the balance of the whole set. Read the finished set as a whole:
when every card leans the same way (for or against a government, a party,
enforcement, regulation, one side of a conflict) and an equally strong
opposite-direction card exists for one of the axes, use that direction for
that card. Never restore balance by adding the counterclaim of another card,
by changing an axis, or by weakening a card.

ONE PRESCRIPTIVE CARD PER SET

At most one card in the set may prescribe: "should", "must", "needs to",
"ought to", "should not" and their equivalents all count, and swapping in
another modal does not escape the rule. The other cards state a judgment the
story raises, in whichever of these shapes fits it:
- cause: "X is the main cause of Y"
- justification: "X is justified by Y"
- tradeoff: "X matters more than Y"
- effectiveness: "X is an effective way to achieve Y"
- proportionality: "X goes too far in restricting Y"
- responsibility: "X bears more responsibility for Y than Z"
- threshold: "X is enough to justify Y"
- consequence: "X does more harm than good"
- institutional: "X is an appropriate role for Y"
An all-"should" list usually means a causal, evaluative, or disputed-factual
dispute went unfound. Never convert a card into another shape to satisfy the
rule; find the story's other judgment, or return fewer cards.

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
- A motive the sources do not state ("X did it to stay loyal to Y"). The
  disagreement is about an action, judgment, or consequence, not an imputed
  intention.
- A verdict smuggled into the wording: "reckless", "corrupt", "betrayal",
  "surrender", "extreme", "illegitimate", "severely damaged", "attack on
  democracy". When a plainer word exposes the same disagreement, use it; one
  side should never have to reject the words before it can argue the point.
- A pure prediction whose only disagreement is guessing what will happen
  ("X will lose the midterms"). A present judgment arguable now on current
  evidence, about readiness, adequacy, reliability, cause, or consequence,
  is not a prediction and is welcome.
- A card almost everyone would accept ("the fine is too small to deter big
  tech"). Prefer a disagreement that could plausibly split near even over
  one that splits nine to one.

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
  "could", "some argue"), no "whether", no bare "it", "the policy", "the
  administration", or "the president" where a reader could ask who is meant.
- Aim for 6-10 words; use 11-14 when a named actor or essential distinction
  requires it; 20 words is the hard maximum.
- Name the actor or subject plus the contested action or judgment. Drop
  dates, amounts, trailing justification clauses ("to preserve unity"), and
  defensive qualifiers ("unless…", "in most cases") unless they are the
  contested issue.
- Never use evidentiary-summary framing such as "signals", "proves", or
  "demonstrates" — state the disputed position itself.
- Style-only examples of the shape (never copy their subjects):
  "Orion should disclose its automated hiring criteria." ·
  "Aster's battery design poses unacceptable safety risks." ·
  "Riverton's housing shortage is driven by zoning restrictions." ·
  "Mosaic's fusion reactor is reliable enough for grid connection." ·
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
