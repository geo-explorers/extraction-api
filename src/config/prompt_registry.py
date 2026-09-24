"""Catalog of every LLM prompt text in the service, keyed for lookup and override.

Prompts stay where they are written (src/config/prompts/*). This module maps a
stable dotted key to each text so the rest of the code fetches it by name at
render time — through `src.config.overrides.prompts.get(key)`, which consults
the active per-run override map first — and so a caller can list every prompt
(GET /prompts) and override any of them for one run without a code change.

`formatted` marks texts rendered through str.format (or a langchain f-string
template): braces in them are slots, and an override may not introduce a slot
the renderer does not fill. Plain texts are concatenated verbatim and may
contain any braces.

tests/test_overrides.py asserts that every public prompt constant under
src/config/prompts is registered here, so a new prompt cannot be left out.
"""

from dataclasses import dataclass
from string import Formatter
from types import ModuleType
from typing import Dict, List

from src.config.prompts import claim_extraction_prompt as _podcast_claims
from src.config.prompts import claim_keyword_extraction_prompt as _claim_keywords
from src.config.prompts import claims_judge_equivalence_prompt as _judge
from src.config.prompts import claims_link_entities_prompt as _link
from src.config.prompts import guest_extraction_prompt as _guest
from src.config.prompts import host_extraction_prompt as _host
from src.config.prompts import key_takeaways_prompt as _podcast_takeaways
from src.config.prompts import keyword_extraction_prompt as _keywords
from src.config.prompts import media_keyword_extraction_prompt as _media_keywords
from src.config.prompts import news_claim_extract_prompt as _news_claims
from src.config.prompts import news_collection_review_prompt as _review
from src.config.prompts import news_debate_claim_prompt as _debate
from src.config.prompts import news_debate_completion_prompt as _debate_completion
from src.config.prompts import news_debate_semantic_review_prompt as _debate_review
from src.config.prompts import news_topics_entities_prompt as _topics_entities
from src.config.prompts import news_topics_extract_prompt as _topics_extract
from src.config.prompts import space_assignment_prompt as _spaces
from src.config.prompts import topics_of_discussion_extraction_prompt as _podcast_topics
from src.config.prompts.claims_extract import core as _ce_core
from src.config.prompts.claims_extract import media_layers as _ce_media
from src.config.prompts.claims_extract import sections as _ce_sections
from src.config.prompts.claims_extract import takeaways as _ce_takeaways
from src.config.prompts.claims_extract import topics as _ce_topics


def template_slots(text: str) -> List[str]:
    """Named slots of a str.format template, in order of first appearance.

    Raises ValueError on unbalanced braces (the same error str.format would
    raise) and on any slot that is not a plain `{name}`: positional (`{}`,
    `{0}`), attribute or index access (`{a.b}`, `{a[0]}`), conversions and
    format specs. Every renderer here formats by keyword with plain names, and
    an override must not be able to traverse into the format arguments.
    """
    seen: List[str] = []
    for _, field_name, format_spec, conversion in Formatter().parse(text):
        if field_name is None:
            continue
        if not field_name.isidentifier() or format_spec or conversion:
            shown = field_name + ("!" + conversion if conversion else "") + (":" + format_spec if format_spec else "")
            raise ValueError(f"slot {{{shown}}} is not a plain {{name}} slot")
        if field_name not in seen:
            seen.append(field_name)
    return seen


@dataclass(frozen=True)
class PromptEntry:
    key: str
    text: str
    formatted: bool  # rendered via str.format / langchain template: braces are slots
    module: str  # dotted module that defines the default text

    @property
    def slots(self) -> List[str]:
        return template_slots(self.text) if self.formatted else []


PROMPTS: Dict[str, PromptEntry] = {}


def _register(key: str, text: str, *, formatted: bool, module: ModuleType) -> None:
    if key in PROMPTS:
        raise RuntimeError(f"duplicate prompt key {key!r}")
    if not isinstance(text, str) or not text:
        raise RuntimeError(f"prompt {key!r} is not a non-empty string")
    if formatted:
        template_slots(text)  # a default that does not parse is a bug here, not at runtime
    PROMPTS[key] = PromptEntry(key=key, text=text, formatted=formatted, module=module.__name__)


# ── Podcast pipeline (premium extraction + keyword/guest/host endpoints) ──────
_register("podcast_extract.topics", _podcast_topics.TOPICS_OF_DISCUSSION_PROMPT, formatted=True, module=_podcast_topics)
_register("podcast_extract.claims", _podcast_claims.CLAIM_EXTRACTION_PROMPT, formatted=True, module=_podcast_claims)
_register("podcast_extract.takeaways", _podcast_takeaways.KEY_TAKEAWAYS_PROMPT, formatted=True, module=_podcast_takeaways)
_register("keyword_extraction", _keywords.KEYWORD_EXTRACTION_PROMPT, formatted=True, module=_keywords)
_register("media_keyword_extraction", _media_keywords.MEDIA_KEYWORD_EXTRACTION_PROMPT, formatted=True, module=_media_keywords)
_register("claim_keyword_extraction", _claim_keywords.CLAIM_KEYWORD_EXTRACTION_PROMPT, formatted=True, module=_claim_keywords)
_register("guest_extraction", _guest.GUEST_EXTRACTION_PROMPT, formatted=True, module=_guest)
_register("host_extraction", _host.HOST_EXTRACTION_PROMPT, formatted=True, module=_host)

# ── News pipeline ─────────────────────────────────────────────────────────────
_register("news_claim_extract", _news_claims.NEWS_CLAIM_EXTRACT_PROMPT, formatted=True, module=_news_claims)
_register("news_claim_extract.claude_system", _news_claims.NEWS_CLAIM_EXTRACT_CLAUDE_SYSTEM_PROMPT, formatted=False, module=_news_claims)
_register("news_debate_claim", _debate.NEWS_DEBATE_CLAIM_PROMPT, formatted=True, module=_debate)
_register("news_debate_claim.claude_system", _debate.NEWS_DEBATE_CLAIM_CLAUDE_SYSTEM_PROMPT, formatted=False, module=_debate)
_register("news_debate_completion", _debate_completion.NEWS_DEBATE_UNDERFILLED_RESCUE_PROMPT, formatted=True, module=_debate_completion)
_register("news_debate_semantic_review", _debate_review.NEWS_DEBATE_SEMANTIC_REVIEW_PROMPT, formatted=True, module=_debate_review)
_register("news_debate_semantic_review.claude_system", _debate_review.NEWS_DEBATE_SEMANTIC_REVIEW_CLAUDE_SYSTEM_PROMPT, formatted=False, module=_debate_review)
_register("news_topics_extract", _topics_extract.NEWS_TOPICS_EXTRACT_PROMPT, formatted=True, module=_topics_extract)
_register("news_topics_extract.regenerate", _topics_extract.NEWS_TOPICS_REGENERATE_PROMPT, formatted=True, module=_topics_extract)
_register("news_topics_extract.system", _topics_extract.NEWS_TOPICS_SYSTEM_PROMPT, formatted=False, module=_topics_extract)
_register("news_topics_entities.system", _topics_entities.SYSTEM_BASE, formatted=False, module=_topics_entities)
_register("news_topics_entities.curated_prefix", _topics_entities.CURATED_SYSTEM_PREFIX, formatted=False, module=_topics_entities)
_register("news_topics_entities.topic_rule_curated", _topics_entities.TOPIC_RULE_CURATED, formatted=False, module=_topics_entities)
_register("news_topics_entities.topic_rule_free", _topics_entities.TOPIC_RULE_FREE, formatted=False, module=_topics_entities)
_register("news_topics_entities.user_header", _topics_entities.USER_HEADER, formatted=True, module=_topics_entities)
_register("news_topics_entities.user_json_example", _topics_entities.USER_JSON_EXAMPLE, formatted=False, module=_topics_entities)
_register("news_topics_entities.user_rules", _topics_entities.USER_RULES, formatted=True, module=_topics_entities)
_register("news_collection_review.group", _review.GROUP_PROMPT, formatted=True, module=_review)
_register("news_collection_review.rescue", _review.RESCUE_PROMPT, formatted=True, module=_review)
_register("news_collection_review.check", _review.CHECK_PROMPT, formatted=True, module=_review)
_register("news_collection_review.order", _review.ORDER_PROMPT, formatted=True, module=_review)
_register("news_collection_review.order_rule", _review.ORDER_RULE, formatted=False, module=_review)
_register("news_collection_review.review", _review.REVIEW_PROMPT, formatted=True, module=_review)
_register("news_collection_review.source_check", _review.SOURCE_CHECK_PROMPT, formatted=True, module=_review)

# ── Claims tasks (generalized extraction, linking, equivalence) ───────────────
_register("claims_link_entities", _link.CLAIMS_LINK_ENTITIES_PROMPT, formatted=True, module=_link)
_register("claims_judge_equivalence.rubric", _judge.CLAIMS_JUDGE_EQUIVALENCE_RUBRIC, formatted=False, module=_judge)

_register("claims_extract.role", _ce_sections.ROLE_SECTION, formatted=True, module=_ce_sections)
_register("claims_extract.core", _ce_core.CORE_CLAIM_RULES, formatted=False, module=_ce_core)
for _media_type, _layer in _ce_media.MEDIA_LAYERS.items():
    _register(f"claims_extract.media.{_media_type}", _layer, formatted=False, module=_ce_media)
_register("claims_extract.topics", _ce_topics.GENERIC_TOPICS_PROMPT, formatted=True, module=_ce_topics)
_register("claims_extract.takeaways", _ce_takeaways.GENERIC_TAKEAWAYS_PROMPT, formatted=True, module=_ce_takeaways)
_register("claims_extract.grouping", _ce_sections.GROUPING_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.flat", _ce_sections.FLAT_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.quotes", _ce_sections.QUOTES_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.summary", _ce_sections.SUMMARY_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.factuality", _ce_sections.FACTUALITY_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.contestability", _ce_sections.CONTESTABILITY_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.topic_vocabulary", _ce_sections.TOPIC_VOCABULARY_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.consolidation", _ce_sections.CONSOLIDATION_SECTION, formatted=False, module=_ce_sections)
_register("claims_extract.focus_topics", _ce_sections.FOCUS_TOPICS_SECTION, formatted=True, module=_ce_sections)
_register("claims_extract.language", _ce_sections.LANGUAGE_SECTION, formatted=True, module=_ce_sections)
_register("claims_extract.max_claims", _ce_sections.MAX_CLAIMS_SECTION, formatted=True, module=_ce_sections)
_register("claims_extract.custom_instructions", _ce_sections.CUSTOM_INSTRUCTIONS_SECTION, formatted=True, module=_ce_sections)
_register("claims_extract.output_contract", _ce_sections.OUTPUT_CONTRACT_HEADER, formatted=False, module=_ce_sections)
_register("claims_extract.keep_groups_empty", _ce_sections.KEEP_GROUPS_EMPTY, formatted=False, module=_ce_sections)
_register("claims_extract.keep_quotes_empty", _ce_sections.KEEP_QUOTES_EMPTY, formatted=False, module=_ce_sections)
_register("claims_extract.keep_summary_empty", _ce_sections.KEEP_SUMMARY_EMPTY, formatted=False, module=_ce_sections)
_register("claims_extract.keep_factuality_null", _ce_sections.KEEP_FACTUALITY_NULL, formatted=False, module=_ce_sections)
_register("claims_extract.keep_contestability_null", _ce_sections.KEEP_CONTESTABILITY_NULL, formatted=False, module=_ce_sections)
_register("claims_extract.keep_assigned_topics_empty", _ce_sections.KEEP_ASSIGNED_TOPICS_EMPTY, formatted=False, module=_ce_sections)

# ── Geo ───────────────────────────────────────────────────────────────────────
_register("space_assignment.system", _spaces.SPACE_ASSIGNMENT_SYSTEM_PROMPT, formatted=False, module=_spaces)
_register("space_assignment.output", _spaces.SPACE_ASSIGNMENT_OUTPUT_PROMPT, formatted=False, module=_spaces)


def keys() -> List[str]:
    return list(PROMPTS.keys())
