"""Story-level topic + entity extraction prompt (news "Pass 5").

Ported VERBATIM from news-worker's injection branch
(commands/enrich.ts `extractTopicsAndEntities`, plus ENTITY_TYPE_NAMES from
lib/entity-types.ts). This extracts the whole-story curated/free topics and
related entities — NOT the per-claim grouping topics (that is the separate
news.extract_topics_and_claims task).

The user prompt embeds a literal JSON example with braces, so it is assembled
from three chunks: a header template (headline, summary), the verbatim JSON
example (never formatted, so its braces are safe), and a rules template
(topic rule, entity-type list). Every chunk is fetched through the prompt
registry at build time so a run can override it.

Curated topics are NOT sourced here — extraction-api is stateless. The caller
passes `curated_topic_names` (sourced/cached on its side); the curated-list
system block is marked for prompt caching since it repeats across a batch.
"""

from src.config.overrides import prompts

# Entity type vocabulary offered to the LLM — verbatim from lib/entity-types.ts
# ENTITY_TYPE_NAMES (order preserved). "Company"/"Organization"/"Public figure"/
# "Topic" are intentionally absent (companies -> Project, individuals -> Person,
# broad subjects -> the topics field).
ENTITY_TYPE_NAMES = [
    "Person",
    "Nonprofit", "Foundation", "DAO", "Institution", "Exchange",
    "Project", "Protocol", "DeFi Protocol", "Network", "Token",
    "City", "Country", "Region", "Place",
    "Academic field",
    "Event", "Law", "Policy",
    "Demographic",
]
ENTITY_TYPE_PROMPT = "Available types: " + ", ".join(ENTITY_TYPE_NAMES)

# Base system primer (always present).
SYSTEM_BASE = "You are a knowledge graph analyst. Return only valid JSON."

# Curated-topic system block prefix (the curated names are appended after a
# blank line). Verbatim from enrich.ts:957.
CURATED_SYSTEM_PREFIX = (
    "## Curated Topics\n"
    "The following is the full list of curated topics available for tagging. "
    "You may use EXACT names (case-sensitive) from this list when a topic "
    "directly matches the story's subject.\n"
    "\n"
    "IMPORTANT: Read topic names literally. A topic like \"AI liability\" is "
    "ONLY for stories about liability caused by AI systems — NOT for any lawsuit "
    "involving a company that also works on AI. Similarly, \"AI regulation\" is "
    "only for stories about regulating AI, not tech regulation in general. Only "
    "select a curated topic if the story's core subject matches what the topic "
    "name literally describes. When in doubt, use a free-form topic instead."
)

# Topic rule, two variants (enrich.ts:962-970).
TOPIC_RULE_CURATED = (
    "- topics: 3-10 topic labels relevant to the story.\n"
    "  You may use EXACT names from the Curated Topics list (in system instructions), or use free-form labels.\n"
    "  CURATED TOPIC VALIDATION — before selecting any curated topic, apply this test:\n"
    "    \"Is this article SPECIFICALLY about [the subject in the topic name]?\"\n"
    "    If the answer is \"no, but it's related\" or \"the company involved also does [subject]\", do NOT select it.\n"
    "    Each word in the topic name matters: \"AI liability\" requires the article to be about liability caused by AI, \"AI regulation\" requires the article to be about regulating AI, \"blockchain gaming\" requires the article to be about games on blockchain. A lawsuit or regulation involving a tech company is NOT about AI unless AI itself is the subject.\n"
    "  When no curated topic accurately describes the story's actual subject, use a free-form label. Prefer accuracy over curated coverage."
)
TOPIC_RULE_FREE = "- topics: 3-10 topic labels relevant to the story."

# User prompt, chunk 1 — str.format slots: {headline}, {summary}.
USER_HEADER = (
    "Identify the key topics and entities for this news story.\n"
    "\n"
    "## Story: \"{headline}\"\n"
    "{summary}\n"
    "\n"
)

# User prompt, chunk 2 — literal JSON example; NEVER passed through str.format.
USER_JSON_EXAMPLE = (
        "---\n"
        "\n"
        "Return JSON:\n"
        "```json\n"
        "{\n"
        "  \"topics\": [\n"
        "    {\"name\": \"Regulation\", \"relevance\": 0.95},\n"
        "    {\"name\": \"DeFi\", \"relevance\": 0.7},\n"
        "    {\"name\": \"Stablecoin Depegging\", \"relevance\": 0.8}\n"
        "  ],\n"
        "  \"entities\": [\n"
        "    {\"name\": \"Ripple\", \"type\": \"Project\", \"role\": \"Subject — launched the buyback\"},\n"
        "    {\"name\": \"Brad Garlinghouse\", \"type\": \"Person\", \"role\": \"CEO, announced the initiative\"},\n"
        "    {\"name\": \"San Francisco\", \"type\": \"City\", \"role\": \"Location of company headquarters\"}\n"
        "  ]\n"
        "}\n"
        "```\n"
        "\n"
)

# User prompt, chunk 3 — str.format slots: {topic_rule}, {entity_types}.
USER_RULES = (
        "Rules:\n"
        "{topic_rule}\n"
        "- Entities: People, companies, projects, organizations, cities, countries mentioned\n"
        "  - {entity_types}\n"
        "  - Choose the most specific type that fits (e.g., \"City\" not \"Place\").\n"
        "  - Any company, business, startup, exchange, or for-profit organization → use \"Project\" (never \"Company\" or \"Organization\").\n"
        "  - Any individual person — including public figures, politicians, celebrities, founders, executives — → use \"Person\" (never \"Public figure\").\n"
        "  - role: brief description of their involvement (one phrase)\n"
        "- Use official/full names for entities\n"
        "- Relevance: 0.0-1.0 how central this topic/entity is to the story"
)


def build_curated_system_block(curated_topic_names: list[str]) -> str:
    """The cached system block: the literal prefix, a blank line, then the
    curated names joined by newlines (enrich.ts:957)."""
    prefix = prompts.get("news_topics_entities.curated_prefix")
    return prefix + "\n\n" + "\n".join(curated_topic_names)


def build_user_prompt(headline: str, summary: str, has_curated: bool) -> str:
    """Assemble the user prompt exactly as enrich.ts does (lines 972-1005):
    header + JSON example + rules. The JSON example is concatenated verbatim
    so its literal braces survive."""
    topic_rule = prompts.get(
        "news_topics_entities.topic_rule_curated"
        if has_curated
        else "news_topics_entities.topic_rule_free"
    )
    header = prompts.get("news_topics_entities.user_header").format(
        headline=headline, summary=summary
    )
    json_block = prompts.get("news_topics_entities.user_json_example")
    rules = prompts.get("news_topics_entities.user_rules").format(
        topic_rule=topic_rule, entity_types=ENTITY_TYPE_PROMPT
    )
    return header + json_block + rules
