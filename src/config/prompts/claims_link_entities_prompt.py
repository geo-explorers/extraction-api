"""Prompt for claims.link_entities — facet-parameterized claim annotation.

One user message renders the numbered claims and each facet's numbered
vocabulary. The model selects indices only; names never round-trip, so
misspellings and near-duplicates cannot enter the output. Facet keys and
vocabularies come entirely from the caller — nothing here assumes a domain.

Prompting strategy: definitions first (facet is internal jargon — the model
gets a plain-language definition), then a stepwise procedure, a contrastive
selection standard, and one small WORKED EXAMPLE in a synthetic domain with
deliberately arbitrary facet keys (teaches the mechanics — including that
empty selections are normal and keys are opaque — without biasing toward any
caller's domain). Output-format rules are restated last for recency.
"""

CLAIMS_LINK_ENTITIES_PROMPT = """You are an entity-linking annotator. You will link CLAIMS to entries from fixed candidate lists, by index.

DEFINITIONS
- A CLAIM is a short standalone statement extracted from some source material. Claims may come from any domain; judge each one only by its own text.
- A FACET is a named list of candidate entities (a vocabulary). Facet names are opaque labels chosen by the caller — do not infer meaning from a facet's name beyond its stated criterion. Each facet states how many entries you may select per claim, and may state its own selection criterion.

PROCEDURE
For each claim, independently:
1. Read the claim and identify what its assertion is actually about.
2. For each facet, scan that facet's list and test every plausible candidate against the selection standard below (and the facet's own criterion, if given).
3. Select the qualifying entries' indices, most relevant first, up to that facet's maximum. Selecting nothing from a facet is a normal, frequent, correct outcome.

SELECTION STANDARD — "substantively involved"
Select a candidate only if it is substantively involved in the claim: it is the subject of the assertion, an actor in it, directly affected by it, or the attributed source/speaker of the statement.
- DO select: "Acme Corp laid off 500 workers" → Acme Corp (subject of the assertion).
- DO select: "According to Jane Doe, prices rose 3%" → Jane Doe (attributed source — the statement is hers).
- DO NOT select: a candidate that is merely in the same broad subject area as the claim (thematic adjacency is not involvement).
- A facet's own criterion overrides this default standard for that facet.
- When in doubt, leave it out. Precision beats recall here.

WORKED EXAMPLE (synthetic — illustrates mechanics only)
Claims:
[0] The harbor authority fined two shipping firms for exceeding emission limits.
[1] Analysts expect container traffic to grow 4% next year.
FACET "categories" (select at most 2 per claim):
[0] Maritime regulation
[1] Port infrastructure
[2] Labor disputes
FACET "organizations" (select at most 3 per claim; criterion: organizations that take or receive the claim's action):
[0] Harbor Authority
[1] Dockworkers Union
Correct output:
{{"links": [{{"claim_index": 0, "selections": {{"categories": [0], "organizations": [0]}}}}, {{"claim_index": 1, "selections": {{"categories": [], "organizations": []}}}}]}}
(Claim 0 selects "Maritime regulation" = categories index 0 and "Harbor Authority" = organizations index 0, which takes the claim's action. Claim 1 selects nothing: no listed organization acts in it, and a growth forecast is not regulation, infrastructure, or a labor dispute.)

YOUR TASK
{context_block}CLAIMS:
{claims_block}

{facets_block}

OUTPUT REQUIREMENTS
- Return ONLY a single valid JSON object — no prose, no markdown fences.
- Exactly one entry per claim, in the same order, using each claim's given index.
- Every entry must contain a selection list for EVERY facet key, even when empty.
- Use only indices that appear in the corresponding facet's list above.
Format: {{"links": [{{"claim_index": <int>, "selections": {{{selections_example}}}}}, ...]}}"""


def _facet_block(facet) -> str:
    header = f'FACET "{facet.key}" (select at most {facet.max_per_claim} per claim'
    header += f"; criterion: {facet.criteria})" if facet.criteria else ")"
    if not facet.vocabulary:
        return f"{header}:\n(none)"
    items = "\n".join(f"[{v.index}] {v.name}" for v in facet.vocabulary)
    return f"{header}:\n{items}"


def build_user_prompt(claims: list, facets: list, context: str) -> str:
    """Render the prompt from schema objects (LinkClaim / Facet lists)."""
    claims_block = "\n".join(f"[{c.index}] {c.text}" for c in claims)
    facets_block = "\n\n".join(_facet_block(f) for f in facets)
    selections_example = ", ".join(f'"{f.key}": [<int>, ...]' for f in facets)
    context_block = f"CONTEXT: {context}\n\n" if context else ""
    return CLAIMS_LINK_ENTITIES_PROMPT.format(
        context_block=context_block,
        claims_block=claims_block,
        facets_block=facets_block,
        selections_example=selections_example,
    )
