"""Overview-topic extraction prompts (news Pass 1).

Ported VERBATIM from news-worker's injection branch
(commands/enrich.ts on feat/inject-urls-async-pipeline): the "ANALYTICAL ROLES"
overview-topic prompt and its single feedback-retry prompt. The only changes are
mechanical: TypeScript `${...}` interpolation becomes Python str.format() fields,
and the four tuning constants are baked in to match the source
(MIN/MAX_OVERVIEW_TOPICS = 3/6, LABEL_MIN/MAX_WORDS = 2/5).

These templates use str.format(), so the only `{`/`}` they may contain are the
named fields below — keep the bodies brace-free (they currently are).

NEWS_TOPICS_EXTRACT_PROMPT fields:    {headline}, {content}
NEWS_TOPICS_REGENERATE_PROMPT fields: {headline}, {content}, {feedback}, {previous}

`content` is the (already pooled and char-limited) article text; the caller
slices it to 12,000 chars before formatting, matching enrich.ts.
"""

NEWS_TOPICS_EXTRACT_PROMPT = """Identify the key thematic angles covered in the attached news article(s).

## Article: "{headline}"
{content}

---

ANALYTICAL ROLES — think about which layers of the story your topics should cover.

A news story typically has up to five analytical layers:

1. **The event** — what happened (the core news)
2. **Causes / drivers** — why it happened, what triggered it
3. **Consequences / data** — what the impact is, casualties, numerical effects
4. **Responses / reactions** — official statements, affected parties' actions, government replies
5. **Context / background** — geopolitical situation, related incidents, historical or institutional backdrop that explains significance

Each topic you generate should serve at least one analytical role. A typical 3-6 topic set should span at least 3 different roles — not just multiple slices of the event itself. Topics that re-describe the same event with different adjectives (e.g., "Blast casualties," "Attack location," "Victim details") are redundant. Different roles produce different topics.

---

RULES

- Each topic is a short noun-phrase tag (2-5 words). NOT an essay subtitle or thesis statement.
- No "and" conjunctions in labels. Pick the primary angle.
- Only include topics where the article(s) contain multiple distinct factual sentences in the article body. A topic supported only by the headline, a subheading, a sidebar caption, or a single in-body sentence does not qualify — do NOT generate a topic for it. If the only evidence for a candidate topic is a phrase like "America's fastest-growing crime" in the title with no supporting body content, skip the topic.
- Pruning a candidate topic for failing this rule does NOT mean dropping its facts. Those facts still exist in the source body and must be captured downstream by the claims pass. Choose neighboring topic labels broad enough to absorb them — for example, prefer "Ebola outbreak and exposure context" over the narrower "American Ebola exposures" if doing so creates a home for transmission-mechanism or historical-comparison facts that would otherwise have no topic.
- 3-6 topics total. Fewer for shorter pieces, more for richer multi-source stories.
- Source-density cap: estimate the total length of article body content across all sources (the actual prose, ignoring headlines, bylines, captions, and metadata). If the combined body corpus is under ~2,000 characters, cap topics at 3. If under ~1,000 characters, cap at 2. Thin sources should produce fewer topics that are honestly supported, not more topics with thin per-topic support.
- Merge overlapping angles to avoid redundancy.
- Every topic must directly serve the headline story. If a section of a source article covers a separate news event or a different policy area from the headline (e.g., a "related news" sidebar, a "what else is happening" section, a different regulatory announcement), do not create a topic for it.

---

COVERAGE CHECK — every substantive fact must have a home.

After drafting your topics, identify each major fact group in the article(s):
- Causes and triggers
- Consequences and immediate data
- Official statements and responses
- Geopolitical or institutional context
- Related incidents or patterns
- Named statistics and specific dates

For each fact group, name which of your topics covers it.

If any major fact group has no covering topic and you have room (max 6 topics), ADD a topic. If you are already at max, REPHRASE an existing topic to make it broader so the orphaned facts have a home. No major fact should be left without a topic that can accommodate it.

---

REDUNDANCY TEST — compare topic pairs before finalizing.

For each pair of topics, ask: "Would these two topics draw claims from the same events, paragraphs, or fact groups in the source articles?" If yes — if more than half of one topic's potential claims would also fit the other — they are redundant. Merge them into one broader topic.

Two topics that split one event into sub-aspects are always redundant:
- BAD: "Lakki Marwat rickshaw blast" + "Terrorism casualty data" (same explosion, same paragraph)
- GOOD: "Lakki Marwat blast" (one topic covering the event and its casualties)
- BAD: "FDA policy announcement" + "FDA enforcement conditions" (same policy, same facts)
- GOOD: "FDA enforcement policy" (one topic covering the announcement and its conditions)

---

NARRATIVE ORDERING — return topics in this sequence:

1. The event itself (what happened) — core news, always first
2. Causes / drivers (why it happened)
3. Consequences / immediate data (casualties, market reactions, numbers)
4. Responses / reactions (official statements, affected parties)
5. Context / background (broader situation, related events, historical backdrop)

Not every story has all five layers. Skip layers that don't apply. Within a layer, order topics by importance.

---

LABEL STYLE — match the STYLE of GOOD labels, avoid the STYLE of BAD labels:

GOOD labels (clean, tag-like):
- "Africa tour itinerary"
- "KelpDAO exploit"
- "Stablecoin regulation"
- "Gene therapy results"
- "Japan earthquake response"
- "Claude Design launch"
- "Teotihuacan shooting"
- "Cursor funding round"
- "Parkinson's risk detection"
- "Angola civil war"

BAD labels (verbose, essay-subtitles):
- "Strategy's $2.54 Billion Bitcoin Purchase and Growing Stockpile" (too long, "and")
- "Amazon's $5 Billion Investment and AWS Spending Commitment" (too long, "and")
- "Broader contagion effects across DeFi lending platforms" (essay-style hedges)
- "Conflict, peace, and social justice themes across the African nations" (essay subtitle)
- "Pope Leo XIV's historic 11-day Africa tour itinerary and significance" (10-word thesis statement)
- "Declining US Support and Rising Tensions with Christian Communities" (too long, "and")
- "Tokenization infrastructure and institutional fund accessibility" ("and")
- "International Relations" (too abstract)

---

OUTPUT

Return a JSON array of topic strings in narrative order:
["First topic", "Second topic", "Third topic", ...]

Return only the JSON array, no commentary or explanation."""


NEWS_TOPICS_REGENERATE_PROMPT = """Your previous topic labels failed format validation. Rewrite ALL of them.

<failures>
{feedback}
</failures>

Common mistakes to avoid:
- "and" conjunctions: if tempted to write "X and Y", split into two topics or pick just one
- Essay-style hedge words (landscape, themes, implications, context, dynamics, broader, relations)
- Going over 5 words

## Article: "{headline}"
{content}

---

Rewrite the topics following these rules:
- 2-5 words per label (count them)
- Noun phrase tags, NOT essay subtitles
- No "and" conjunctions — if you see compound concepts, split into two topics
- 3-6 topics total
- Cover every major fact group in the article

Previous attempt (rewrite each one):
{previous}

Return a JSON array of new topic strings:
["First topic", "Second topic", ...]"""
