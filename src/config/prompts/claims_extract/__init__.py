"""Composable prompt sections for the generalized claims.extract pipeline.

Assembled by src/extraction/claims_prompt_builder.py:
core rules + media layer + mode section + knob sections + output contract.
"""

from src.config.prompts.claims_extract.core import CORE_CLAIM_RULES
from src.config.prompts.claims_extract.media_layers import MEDIA_LAYERS, MEDIA_NOUNS
from src.config.prompts.claims_extract.topics import GENERIC_TOPICS_PROMPT
from src.config.prompts.claims_extract.takeaways import GENERIC_TAKEAWAYS_PROMPT
from src.config.prompts.claims_extract.sections import (
    GROUPING_SECTION,
    FLAT_SECTION,
    QUOTES_SECTION,
    SUMMARY_SECTION,
    FACTUALITY_SECTION,
    CONSOLIDATION_SECTION,
    FOCUS_TOPICS_SECTION,
    LANGUAGE_SECTION,
    MAX_CLAIMS_SECTION,
    CUSTOM_INSTRUCTIONS_SECTION,
    OUTPUT_CONTRACT_HEADER,
    KEEP_GROUPS_EMPTY,
    KEEP_QUOTES_EMPTY,
    KEEP_SUMMARY_EMPTY,
    KEEP_FACTUALITY_NULL,
)

__all__ = [
    "CORE_CLAIM_RULES",
    "MEDIA_LAYERS",
    "MEDIA_NOUNS",
    "GENERIC_TOPICS_PROMPT",
    "GENERIC_TAKEAWAYS_PROMPT",
    "GROUPING_SECTION",
    "FLAT_SECTION",
    "QUOTES_SECTION",
    "SUMMARY_SECTION",
    "FACTUALITY_SECTION",
    "CONSOLIDATION_SECTION",
    "FOCUS_TOPICS_SECTION",
    "LANGUAGE_SECTION",
    "MAX_CLAIMS_SECTION",
    "CUSTOM_INSTRUCTIONS_SECTION",
    "OUTPUT_CONTRACT_HEADER",
    "KEEP_GROUPS_EMPTY",
    "KEEP_QUOTES_EMPTY",
    "KEEP_SUMMARY_EMPTY",
    "KEEP_FACTUALITY_NULL",
]
