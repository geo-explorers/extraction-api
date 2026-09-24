"""Prompts for news.review_collections (group, rescue, check, order, review,
source check). Moved here verbatim from news_collection_review_service.py so
the prompt registry can catalog them; the service renders them through
`prompts.get(...)` so a run can override any of them.

All six are str.format templates; ORDER_RULE is plain text spliced into
ORDER_PROMPT's {order_rule} slot.
"""

GROUP_PROMPT = """You group the claims of one news story into blocks. Each claim is published ALONE on a knowledge graph, and claims are shown to readers in blocks under a heading. Your only job is to decide which claims go together and what each block is called. Do not rewrite, drop, or add any claim.

HEADLINE: {headline}

CLAIMS (index. text):
{claims}
{feedback}
A block is a set of claims a reader would expect under one heading because they are about the same specific subject — the same event, decision, actor's conduct, dispute, mechanism or consequence — not merely the same story.

Procedure:
1. For each claim, note (silently) the one specific subject it is about.
2. Group the claims that share a subject — membership FIRST. Two claims belong together only when a reader who has read the block's heading would expect BOTH of them there. Being about the same story, or sharing one keyword with the group, is not belonging. Do not file a claim under the nearest group.
3. Only then name each block from its members: a plain, specific heading of 2-6 words in sentence case (capitalise only the first word and proper nouns) that is true of every claim in it, as an editor would head that section. "These all mention X" is not a heading, and a name alone ("Thomas Kelly") is a label, not a heading. If it is true of only some members, the block is two blocks, or the odd claim is out.
4. Every block holds at least {min_block} claims. A claim that shares its subject with no other claim goes into "lone" — never file it under a heading that does not describe it.
5. A block holds {min_block} to {max_block} claims. A group of more must be split into specific blocks when its claims cleanly separate by subject; only if no clean split exists may it stay larger. Never split a group just to reach a count.

Return JSON only:
{{"blocks": [{{"name": "<heading>", "claims": [<indices>]}}], "lone": [<indices>]}}"""

RESCUE_PROMPT = """You find company for a lone claim from the story's sources. Claims were extracted from the sources of one news story and grouped into blocks shown to readers under a heading. The claims under LONE CLAIMS share their subject with no other claim, so they have no block, and every published claim must sit in a named block of at least two.

For EACH lone claim: search the SOURCES for ONE more substantive fact about the SAME specific subject that is not already among the claims — a different event, figure, named actor's action, decision, or consequence — so that the lone claim and the new fact form a block a reader would expect under one heading. Then name that block.

HEADLINE: {headline}

EXISTING CLAIMS (all of them, so you never restate one):
{claims}

LONE CLAIMS (index. text):
{lone}

SOURCES:
{sources}

CALENDAR (read weekdays and "yesterday" off this; never compute them):
{calendar}

Rules for the new claim:
- Every name, number, date and fact must be traceable to a specific sentence in the SOURCES. Nothing from prior knowledge. If in doubt, leave it out. A weekday or "yesterday" in a source becomes the CALENDAR's date for that source; a year is written only when a source states it or the calendar gives it — code refuses the claim otherwise.
- It must state a DIFFERENT fact from the lone claim and from every existing claim. A restatement, a gloss, or a piece cut from the lone claim's own fact is forbidden — an empty result is better.
- NOT a fact for this purpose, even when the sources state it: what an organisation, product, person or term IS (a definition or profile); how something works in general; a commentator's characterisation of the lone claim's event; a detail of the lone claim's own event (its location, its wallet address, its exact time); how or where the outlet obtained a statement (an interview, a broadcast appearance, a phone call, "spoke exclusively with…"). The reader must learn a second thing that HAPPENED, was DECIDED, or was MEASURED about the same subject.
- At most ONE new claim per lone claim. If the sources carry nothing that qualifies, return no claim for it.
- Self-contained: full proper names, no pronoun before its referent, no "the company"/"the deal"; name the event inside the claim; absolute dates; {min_words}-35 words.
- source_indices: the indices of the sources that state the fact. confidence 0.9+ when explicit. importance for this story (0.3-1.0).
- The heading: 2-6 plain words in sentence case (capitalise only the first word and proper nouns), true of the lone claim and the new claim, as an editor would head that section. Never a name alone ("Thomas Kelly"), never "these all mention X".

Return JSON only:
{{"rescues": [{{"lone": <index>, "name": "<heading>", "claim": {{"text": "...", "source_indices": [0], "confidence": 0.9, "importance": 0.6}}}}]}}
Include every lone index; one whose sources carry nothing gets "claim": null."""

CHECK_PROMPT = """You are the final check on how a news story's claims were grouped into blocks. Each block is shown to readers under its heading. Judge with a HIGH bar and reject anything a careful editor would not publish as is.

HEADLINE: {headline}

BLOCKS:
{blocks}
{lone}
For every block:
- "misfits": the indices of claims that do NOT belong under the heading — a reader who opened that heading would not expect exactly that claim there.
- "purpose": true when the block has one clear shared subject every member contributes to; false for a catch-all or "these all mention X".
- "heading_ok": true when the heading is specific and true of every member; false if it is vague, a catch-all ("Other developments", "Context and reactions"), a bare name with nothing said about it ("Thomas Kelly"), or promises something the claims do not deliver.
- "duplicates": pairs of claim indices inside the block that state the same fact.
- "reason": under 20 words, only when something is wrong.
For every lone claim: "home" = the index of the block whose heading is true of it as written, or -1 if none is.

Return JSON only:
{{"blocks": [{{"index": 0, "misfits": [], "purpose": true, "heading_ok": true, "duplicates": [], "reason": ""}}], "lone": [{{"index": 0, "home": -1}}]}}"""

# The reading order. Armando's example: a story headlined "Trump approval hits
# record low as Republicans break with president over Iran war" opens with the
# approval figures, keeps the cost of living figures from the same poll next
# to them, then runs the Republican break as one thread, and ends with the
# war's background.
ORDER_RULE = """Reading order, for a reader who arrives from the headline:
1. First, the block that states what the headline's MAIN clause states.
2. Then every other block about that same subject — its details, figures, who said it, reactions to it. A block reporting the same poll, vote or measure as the opening block stays next to it.
3. Then each side thread as ONE contiguous run, the thread the headline's subordinate clause ("as…", "amid…", "after…") names first. Inside a thread, cause before consequence.
4. Background and context last."""

ORDER_PROMPT = """You order the blocks of one news story for its readers. Each block is a heading with its claims. Do not rename, drop or merge anything — only order.

HEADLINE: {headline}

BLOCKS:
{blocks}

{order_rule}

Return JSON only: {{"order": [<every block index exactly once, in reading order>]}}"""

REVIEW_PROMPT = """You review the claims of one news story after extraction. Each claim is published ALONE on a knowledge graph, and claims are grouped into named collections shown as blocks. A reader who sees two claims saying the same thing in one block concludes the second was written to fill it. Your only job is to remove that.

HEADLINE: {headline}

CLAIMS (index. text):
{claims}

COLLECTIONS (name: claim indices):
{collections}

Inside each collection, compare every pair of claims and ask: does the reader learn a NEW fact from the second after reading the first?
- Same fact in other words, or with one detail moved → keep the more specific claim; the other is merged into it (no new text needed).
- One fact cut into pieces that share a frame (clauses of one rule, figures of one poll, one measurement at two time points) → write ONE claim carrying every piece. It must be at most {max_words} words and at most {max_growth} words longer than the longest piece. If not everything fits, merge only the two most alike; if even two cannot fit, they are different facts — leave them.
- A merged claim keeps every name, number and date of the claims it replaces and adds nothing.
- Claims that report DIFFERENT facts are never merged, however similar their wording.
- Never drop a claim, never move a claim to another collection, never rename or add a collection. If a merge leaves a collection with one claim, leave it — that is handled after you.

Return JSON only:
{{"merges": [{{"keep": <index>, "drop": [<indices>], "text": "<merged text, or null to keep the kept claim verbatim>"}}]}}"""

SOURCE_CHECK_PROMPT = """Check each sentence below against the story's source articles. Each sentence was written by the system rather than extracted verbatim: a merge of two or three extracted claims into one, or a fact added from the sources to accompany another claim.

SENTENCES:
{sentences}

SOURCES:
{sources}

For each sentence:
- grade: "supported" (every fact, name, number and date is stated in the sources; paraphrase is fine), "partly" (the core is there but a detail is not, or is stated more strongly), "unsupported" (not stated or contradicted).
- false_link: true if the sentence joins facts in a way that implies a relation (cause, sequence, same event, same speaker) that the sources do not state.

Return JSON only: {{"checks": [{{"index": <sentence number>, "grade": "supported"|"partly"|"unsupported", "false_link": true|false}}]}}"""
