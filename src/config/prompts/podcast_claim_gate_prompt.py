"""Podcast claim gate prompts (podcast.extract_claims, gate step).

Added 2026-10-13. The claims prompt alone leaves ~11% of published podcast
claims context-dependent ("the deal", "currently", "is considered"), measured
with a per-claim judge on 893 production claims after the P5 prompt. A model
that reads a claim WITH the transcript in view does not notice what a stranger
misses, so the gate has two calls with different views:

1. DETECT (gemini-2.5-flash, thinking on): claims alone, in batches of ten,
   no transcript. Flags ~20-25% of claims.
2. REPAIR (gemini-2.5-pro): one call per episode with the transcript and the
   episode metadata, flagged claims only. Rewrites minimally, keeps, or
   withholds. Code then re-checks every rewrite (no lost names or numbers,
   word budget, every added identifier present in the transcript).

Measured on a representative 6-episode subset (333 claims): 11.1% -> 3.7%
context-dependent, 97% of claims retained, about $0.14 per episode. The
heavier variant with a separate arbiter and grounding call reached 2.2% at
$0.51 per episode; disabling thinking on the detector collapsed recall (7.4%).
"""

GATE_DETECT_PROMPT = """You are a strict editorial verifier. Each claim below was extracted from a podcast and will be published ALONE on a knowledge graph: no episode title, no date, no speaker names, no neighbouring claims. Judge EACH claim independently, from its wording only (the other claims in this list are NOT context for it). A claim FAILS if:
1. context_dependent: a pronoun or definite reference that nothing inside the claim resolves ("the deal", "the investigation", "the champion", "President Trump's deal", "the search of her home", "the initial event"). Resolved means a proper noun or unique identifier is inside the sentence. Generic or mass nouns and indefinites are fine.
2. relative_time: a present condition, running count or time-relative measurement whose truth depends on when it was said ("currently", "so far", "has been at war for 11 years", "a decade ago", "the previous year") with no absolute anchor. Completed past events sequenced by narrative words ("after his NFL career", "previously") are fine. Plain present tense without a time word is fine.
3. impersonal_frame: carried by an unnamed source or passive frame ("is considered", "is described as", "reportedly", "reporting indicates", "was found to", "the host said").
4. attribution_frame: "X says/notes/argues that Y" where X is a host, pundit or guest and Y is a checkable statement about the world. Not when Y is X's own opinion, prediction, recollection or experience, nor when the utterance itself is the fact of record.
5. conversational_order: "later in the episode", "as mentioned", "in response".
Return JSON only: {{"verdicts":[{{"index":<n>,"fires":true|false,"issue":"<tag or none>","rationale":"<under 20 words>"}}]}} with one entry per claim; "index" is the number printed before the claim (the first claim is 0).

CLAIMS
{claims}"""

GATE_REPAIR_PROMPT = """You are the final editorial gate for claims extracted from a podcast episode. Each claim will be published ALONE on a knowledge graph: no episode title, no date, no speaker names, no neighbouring claims, shuffled among thousands of claims from other sources.

An editorial verifier (which sees each claim alone) flagged the CLAIMS below, each with its reason. For EACH flagged claim decide one of:
- keep: the verifier was wrong; a reader who sees only this sentence can tell exactly who and what it refers to, and when, where that matters.
- rewrite: it fails one of the tags below but can be fixed with a MINIMAL edit grounded in the transcript or metadata.
- withhold: it fails and cannot be fixed without inventing or changing facts, or the content is the speaker's own opinion/prediction that only makes sense as attributed speech and the attribution is missing.

Tags (judge the claim's wording as a stranger would, NOT with your knowledge of the transcript):
1. context_dependent: a pronoun or definite reference that nothing inside the claim resolves ("the deal", "the investigation", "the champion", "President Trump's deal", "the search of her home", "the initial event"). Resolved means a proper noun or unique identifier is inside the sentence. Generic or mass nouns and indefinites are fine.
2. relative_time: a present condition, running count or time-relative measurement whose truth depends on when it was said ("currently", "so far", "has been at war for 11 years", "a decade ago", "the previous year") with no absolute anchor. Completed past events sequenced by narrative words ("after his NFL career", "previously") are fine. Plain present tense without a time word is fine.
3. impersonal_frame: carried by an unnamed source or passive frame ("is considered", "is described as", "reportedly", "reporting indicates", "was found to", "the host said").
4. attribution_frame: "X says/notes/argues that Y" where X is a host, pundit or guest and Y is a checkable statement about the world; strip the frame and state Y. Do NOT strip when Y is X's own opinion, prediction, recollection or experience, or when the utterance itself is the fact of record (officials, witnesses, court filings, on-record responses).
5. conversational_order: "later in the episode", "as mentioned", "in response".

Rewrite rules: change only the failing phrase; keep every other word; never drop a name, date or number the claim carries; use ONLY identifiers that appear in the METADATA or TRANSCRIPT; for relative time use the stated date, or "as of <Month Year>" from the AIR DATE only for present conditions; add at most {max_added_words} words, at most {max_claim_words} words in total; never invent.

Return JSON only: {{"decisions":[{{"index":<n>,"action":"keep"|"rewrite"|"withhold","issue":"context_dependent"|"relative_time"|"impersonal_frame"|"attribution_frame"|"conversational_order"|"none","text":"<rewritten claim, or empty>","reason":"<under 15 words>"}}]}} with one entry per claim, in order; "index" is the number printed before the claim.

METADATA
{metadata}

CLAIMS
{claims}

TRANSCRIPT
{transcript}"""
