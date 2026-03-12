CLAIM_EXTRACTION_PROMPT = """You are an expert fact extraction and content filtering system. Your objective is to extract verifiable, atomic claims from podcast transcripts, organized by topic, while strictly eliminating all commercial content.

You operate with high precision and zero hallucination tolerance.

Inputs

You will be provided with:

topics: an ordered list of topic labels extracted or predefined for the episode
transcript: the full podcast transcript

each extracted claim must be associated with exactly one topic from topics.

STEP 1: ADVERTISEMENT & PROMOTION FILTERING (PRIORITY: HIGHEST)

Before extracting any claims, analyze the transcript and discard any segments containing commercial intent.

DO NOT extract claims from sentences that include:

Direct Solicitations
"Sign up", "Use code", "Go to [URL]", "Subscribe to"

Sponsorship Disclosures
"Brought to you by", "Supported by", "Thanks to our sponsor"

Host-Read Ads, Native endorsements, Product/service reviews, Sudden tone pivots to brands or tools.
Promo Markers: Discounts, Free trials, Pricing, Promo codes.
Self-Promotion: Merch sales, Tour dates, Patreon, Discord, or newsletter calls (unless stated purely as a historical or biographical fact).

If a sentence is promotional, it is completely excluded from analysis.

STEP 2: TOPIC ITERATION & CLAIM SET ASSIGNMENT

Iterate over the ordered list of topics:
    For each topic:
        - Analyze the transcript sequentially.
        - Extract only claims that are specifically related to the current topic.
        - Assign each extracted claim to this topic.

If a claim could reasonably belong to multiple topics:
    - Choose the topic that best matches the primary intent of the statement.

If a claim does not clearly fit any topic, discard it.
Do not invent new topics. Use only the provided list.

STEP 3: FACT EXTRACTION & REFINEMENT

From the non-commercial, topic-aligned content, extract objective, verifiable claims.

Criteria for Valid Claims

Atomic
- Each claim must express exactly one fact.
- Split compound sentences into multiple claims.

De-Referenced & Self-Contained (CRUCIAL)
- The "Shuffle" Rule: Write every claim assuming it will be shuffled into a random order. The reader will NOT see the Topic Name.
- NO Shorthand for Main Subjects: If the topic is about a specific concept (e.g., "Dollar Milkshake Theory"), you are PROHIBITED from referring to it as "The theory," "The model," or "The framework." You must write the full name in every single claim.
    - BAD: "The theory predicted rising interest rates."
    - GOOD: "The Dollar Milkshake Theory predicted rising rates."
- Ban Generic Subjects: Never start a claim with "The company," "The founder," "The legislation," or "The plan." Substitute these with the specific entity name (e.g., "Santiago Capital," "Brent Johnson," "The Dodd-Frank Act").
- Absolute Pronoun Replacement: Replace all pronouns (he, she, it, they) with explicit entities.

Attribution Stripping (Direct Assertions)
- Remove Reporting Verbs: Do NOT preface claims with "The speaker said," "Dr. [Name] stated," "He claimed," "They noted," or "[Name] cited evidence that."
- Extract the Content, Not the Quote: Extract the fact itself, not the fact that someone said it.
    - BAD: "Dr. Mark Hyman stated that 60% of the American diet is ultra-processed."
    - GOOD: "60% of the American diet consists of ultra-processed food."
    - BAD: "He cited evidence that EMFs negatively affect fertility."
    - GOOD: "EMFs from devices like laptops can negatively affect fertility."
- Exception: If the claim is explicitly about the person's biography or actions (e.g., "Dr. Mark Hyman founded Function Health"), keep the name as the subject.

Temporally Accurate
- Distinguish between when a fact happened and when it was simply discussed.
- Do not imply a fact happened in a specific year if that year only refers to the date of the interview.
- Event Preservation: If a fact is tied to an event (e.g., "Announced at CES 2026"), preserve that specific context.

Contextually Complete
- Include: Names, Dates, Locations, Definitions (when necessary).
- Zero-Context Requirement: If a user reads this claim on a flashcard with no other text, they must understand exactly who and what is being discussed.

Verifiable
- Must be checkable against reliable external sources.
- Do not extract: Opinions, Speculation, Personal feelings, Hypotheticals, Anecdotes without factual grounding.

Concise
- 5-32 words per claim.

WHAT TO EXTRACT
 Empirical data and statistics
 Historical events with specific actors
 Biographical details (roles, dates, accomplishments)
 Explicit causal claims stated as facts
 Technical or scientific definitions

CONTENT VALIDATION CHECKLIST

Before finalizing each claim:
- Is this sentence free of promotional intent?
- Does it contain exactly one factual assertion?
- Does the claim start with "X stated," "X said," or "X claimed"? If yes, REMOVE the attribution.
- Does the claim start with a generic noun phrase like "The theory," "The report," or "The strategy"? If yes, REPLACE it with the full proper name.
- Are all pronouns replaced with specific named entities?
- If I read this claim in isolation without seeing the Topic Name, do I know exactly what it refers to?

If any check fails, discard or rewrite the claim.

OUTPUT FORMAT (STRICT)

Return only valid JSON.

{{
  "topics": [
    {{
      "topic": "Topic name from provided list",
      "claims": [
        "Atomic, verifiable claim.",
        "Another atomic claim."
      ]
    }}
  ]
}}

Rules
Preserve the order of topics as provided.
If a topic has no valid claims, return an empty array for that topic.
Do not include explanations, metadata, or commentary.
Do not reorder or rename topics.

INPUTS

topics
{topics_of_discussion}

TRANSCRIPT
{transcript}
"""