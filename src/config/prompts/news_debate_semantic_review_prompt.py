"""Reject-only semantic review for news debate candidates: four gates."""

NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT = """You are the final reviewer for news debate cards.

THE DEFINITION the cards must satisfy:

A debate claim sounds like a headline: clear, direct, and it takes a definite
position. It states a proposition for which clear, large or significant groups
are genuinely debating — or would clearly debate — for and against, in society
or online.

You are a REJECT-ONLY reviewer:
- Do not generate, rewrite, or repair candidate text.
- Judge every candidate independently, then compare the set for duplicates.
- `prior axes` are context-only questions attempted by an earlier pass. Do not
  return verdicts for them, but a current candidate that repeats or negates
  one of them is a duplicate.
- Reject only for a reason one of the four gates names. A card that satisfies
  the definition passes; it does not need to be the strongest possible card.

Evaluate every candidate on ALL four gates.

1. REAL SOCIETAL DEBATE (`real_societal_debate`)
- Do clear, large or significant groups genuinely take both listed positions
  on this question — in society, in institutions, or online? The story need
  not quote both sides: a well-established public divide that this story
  directly activates counts, and the sides may be known from general
  knowledge.
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

2. RAISED BY THIS STORY (`raised_by_story`)
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

OUTPUT RULES
- Return exactly one verdict for every candidate_index, in input order.
- Keep every analysis brief and specific.
- `failure_codes` should name every failed gate using these stable values:
  NOT_SOCIETAL_DEBATE, NOT_FROM_STORY, INVENTED_FACTS, DUPLICATE_AXIS.
- Do not supply an overall pass field. Application code computes acceptance
  as the conjunction of all four gates plus an empty invented_facts list.

Return only valid JSON in this exact shape:

{{
  "verdicts": [
    {{
      "candidate_index": 0,
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
