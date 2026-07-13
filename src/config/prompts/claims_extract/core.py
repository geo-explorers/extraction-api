"""Shared claim-quality contract for the generalized claims.extract pipeline.

Distilled from the two battle-tested production prompts — the news claim
prompt (Step 4 quality criteria) and the podcast claim prompt (Step 3
criteria). Media-specific behavior lives in media_layers.py; this block is
the media-neutral core every extraction runs under.

No literal JSON braces here: output shape is enforced by response_schema, so
sections stay plain prose and the builder can compose with f-strings.
"""

CORE_CLAIM_RULES = """─────────────────────────────────────────────
CLAIM QUALITY CRITERIA
─────────────────────────────────────────────

Every extracted claim must meet ALL of the following:

Self-Contained (the Shuffle Rule)
Write every claim as if it will be read in isolation, shuffled into a random
order, without the document titles or topic names visible.
- Replace all pronouns (he, she, it, they) with explicit named entities.
- Never use shorthand like "the company," "the speaker," "the theory,"
  "the study," "the report." Write the full proper name every time.
- Include the specific names, dates, locations, and quantities needed for a
  reader to understand the claim entirely on its own.

Atomic — the Split Test
Each claim should express one coherent fact with its essential identifiers
(who, where, when, how many).
Apply this test: "If I delete half this claim, does the remaining half still
make sense as a standalone fact?" If YES, split it into two claims. If NO,
the parts belong together.
- Do not over-fragment: facets of one event (an event and its casualty
  figures, a measurement and its unit and date) stay in one claim.
- Protect high-value standalone facts: named statistics, named actors'
  official responses, specific laws or dates, and concrete consequences each
  deserve their own claim when independently informative.

Attribution Stripping
- Remove reporting verbs: do NOT preface claims with "X said that,"
  "according to X," "X stated," "X noted."
- Extract the fact itself, not the fact-of-statement.
- Exception 1: keep attribution when the act of stating IS the significant
  fact (e.g., an official testifying, a party committing to a position).
- Exception 2: keep the person as subject when the claim is genuinely about
  their biography or actions (e.g., "Jane Smith founded Acme Corp in 2019").

Temporally Grounded
- Use absolute dates ("May 12, 2026") instead of relative references
  ("Monday," "yesterday," "last week").
- Resolve relative dates using the document's date ONLY when the resolution
  is unambiguous within about a week. Phrases that would require guessing the
  year or month are NOT resolvable — preserve the exact original phrasing or
  omit the date. Never invent a year, month, or day the source does not state.
- Distinguish when a fact happened from when it was merely discussed: do not
  stamp a fact with the recording or publication date of the material itself.

Evidence-Appropriate Language
- When a claim rests on a single study, preliminary result, or one
  unconfirmed source, use conditional language: "A study suggests...,"
  "may," "could."
- Reserve declarative language for facts supported by multiple independent
  sources, official statements, or established consensus.
- This matters most for health, science, and medical claims where
  overstatement carries real-world risk.

Concise
- Target 10-30 words per claim. Hard maximum 35 words.
- If a claim exceeds 30 words, re-examine it with the split test.

Verifiable & Source-Grounded (zero hallucination tolerance)
- Every fact, date, name, and statistic must be traceable to a specific
  passage in the provided documents. Do not supplement from prior knowledge,
  even if you believe it to be accurate. When in doubt, omit rather than infer.
- Do not extract opinions, speculation, personal feelings, or hypotheticals
  without factual grounding.
- Preserve the documents' spellings of proper names exactly, even when a
  spelling looks wrong to you — "correcting" a name counts as supplementing
  from prior knowledge and is forbidden.
- If text ends mid-sentence (trailing "…", "[…]", "[&#8230;]", "—", or an
  obvious cut-off), treat everything past the truncation as missing. Never
  complete a name or sentence; use only what verifiably appears before the
  truncation, or skip the fact entirely."""
