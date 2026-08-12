"""Shared claim-quality contract for the generalized claims.extract pipeline.

Distilled from the two battle-tested production prompts — the news claim
prompt (Step 4 quality criteria) and the podcast claim prompt (Step 3
criteria) — and kept in parity with the news prompt's hardened rules
(self-containment precedence, referent classes, evidence-conditional
attribution, shuffle audit). Media-specific behavior lives in
media_layers.py; this block is the media-neutral core every extraction runs
under.

No literal JSON braces here: output shape is enforced by response_schema, so
sections stay plain prose and the builder can compose with f-strings.
"""

CORE_CLAIM_RULES = """─────────────────────────────────────────────
CLAIM QUALITY CRITERIA
─────────────────────────────────────────────

Every extracted claim must meet ALL of the following.

PRECEDENCE: when rules conflict, self-containment wins. It outranks the word
targets in "Concise" and any coverage instruction. A claim that only makes
sense inside the source's narrative or conversational order is a defect no
matter how much coverage it adds.

Self-Contained (the Shuffle Rule)
Write every claim as if it will be read in isolation, shuffled into a random
order, without the document titles or topic names visible.
- Replace all pronouns (he, she, it, they) with explicit named entities. A
  possessive or pronoun ("its," "their") may only appear AFTER the entity it
  refers to has been named in that same claim.
- Never use shorthand like "the company," "the speaker," "the theory,"
  "the study," "the report." Write the full proper name every time.
- The material's own protagonist is NOT exempt. Sources refer to their main
  subject pronominally after first mention ("the company," "the program,"
  "the framework"); claims must not. Re-name the protagonist in every claim
  that concerns it, however repetitive that feels — repetition across claims
  is correct, because the claims will be read separately.
- Name the event inside the claim. A claim about an event must identify
  WHICH event by actor, name, or date — never by a bare definite reference
  ("the incident," "the breach," "the deal," "the ruling," "the debate")
  whose referent lives in a different claim or in the source's flow. A
  definite reference is allowed only when the same claim has already
  introduced its referent.
- Planned, threatened, or predicted events are events too: name who plans or
  predicts them and who reported the plan — never "the planned merger" or
  "the expected ruling" without naming the planner.
- Never open a claim with a discourse connective that presumes order: "In a
  separate incident," "Later in the discussion," "Following the
  announcement," "Meanwhile," "In response," "Returning to." The claims will
  be shuffled — for the reader there is no "later," "separate," or
  "following."
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

Attribution — extract the asserted claim, not the act of asserting
- A claim records WHAT is asserted, not that someone asserted it. Strip
  reporting-verb scaffolding ("X said that," "according to X," "X argued"):
  extract the content of the assertion.
- A claim is an assertion, not a verified fact — that is what makes it
  debatable. Preserve the strength of the claim as the source asserted it:
  do not add hedges the source did not use, and do not harden a hedged
  assertion into certainty.
- Exception — the statement IS the fact: keep attribution when the act of
  saying it is itself the noteworthy fact: an official or head of state
  announcing, committing, threatening, conceding, or testifying; a named
  party's on-record response in a dispute. This requires the actor's name —
  if the person or body making the statement cannot be named from the
  documents, extract the underlying content instead.
- When keeping a statement-event, attribute it to ONE named actor — never
  merge different speakers' attributions into a single clause ("An official
  and an announcement suggested...").
- Never write anonymous attribution: "the speaker," "the host," "the
  guest," "the author," "the interviewee" must never appear in a claim.
- Biography stays biographical: "Jane Smith founded Acme Corp in 2019"
  keeps the name as subject — that is the fact itself, not attribution.

Temporally Grounded
- Use absolute dates ("May 12, 2026") instead of relative references
  ("Monday," "yesterday," "last week").
- Resolve relative dates using the document's date ONLY when the resolution
  is unambiguous within about a week. Phrases that would require guessing the
  year or month are NOT resolvable — preserve the exact original phrasing or
  omit the date. Never invent a year, month, or day the source does not state.
- Domain periods follow the same rule: resolve "last season," "last year,"
  "this quarter" to the named period ("the 2025-26 season," "fiscal Q1
  2026," "2025") when the sources or document date make it unambiguous;
  otherwise keep the source's phrasing anchored with a year.
- Distinguish when a fact happened from when it was merely discussed: do not
  stamp a fact with the recording or publication date of the material itself.

Evidence-Appropriate Language
- When a claim rests on a single study, preliminary result, or one
  unconfirmed source, use conditional language: "A study suggests...,"
  "may," "could."
- Reserve declarative language for facts supported by multiple independent
  sources, official statements, or established consensus.
- Research claims carry their evidence identity: name the study's
  institution, journal, or lead author in each claim asserting a finding —
  "a University of Cambridge study found...," never a bare "Researchers
  found..." or "Studies show...". If the material never identifies the study
  beyond a mention, attribute the person or outlet who cited it. The study's
  identity is part of the fact, not optional framing.
- This matters most for health, science, and medical claims where
  overstatement carries real-world risk.

Concise
- Target 10-30 words per claim. Hard maximum 35 words — except when the
  words needed to name a claim's event or protagonist (Shuffle Rule
  anchoring) push it over; self-containment outranks the cap, up to 40 words.
- If a claim exceeds 30 words, re-examine it with the split test — but never
  delete an anchor to fit the target.

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
  truncation, or skip the fact entirely.

Shuffle audit (mandatory, before output, claim by claim)
Read each claim ALONE, imagining every other claim has been deleted. Flag
any claim that:
(a) opens with a discourse connective ("In a separate incident," "Later in
    the discussion," "Following the ...," "Meanwhile," "In response"),
(b) contains a definite event or entity reference ("the incident," "the
    deal," "the ruling," "the company," "the program," "the study," "the
    framework") whose referent is not named earlier in that same claim —
    the material's protagonist included,
(c) uses a pronoun or possessive ("its," "their," "his," "her") before the
    entity it refers to has been named in that claim, or
(d) frames a research finding as "Researchers found..." or "Studies show..."
    when the sources identify the study — name the study's institution,
    journal, or lead author; if the sources never identify it, extract the
    asserted finding itself without inventing a source.
Rewrite every flagged claim to name its referent explicitly. Rewrite, do
not delete — drop a flagged claim only when the sources cannot support the
explicit version."""
