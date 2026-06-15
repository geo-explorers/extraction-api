"""DB-free premium extraction core.

Pure transform: episode metadata + transcript text -> topics, claims, key
takeaways. This module deliberately has NO database, embedding, or persistence
dependencies so it can run inside a stateless worker (Hatchet) as well as behind
the current pipeline. Episode loading and result persistence are the caller's
responsibility.

The chain mirrors the original PremiumExtractionPipeline steps 1-4:
parse -> extract topics -> extract claims per topic -> filter sparse topics ->
assign claim_order -> extract key takeaways.
"""

from dataclasses import dataclass
from typing import List, Optional
import time

from src.config.settings import settings
from src.preprocessing.transcript_parser import TranscriptParser
from src.extraction.premium_claim_extractor import PremiumClaimExtractor
from src.extraction.models import ClaimWithTopic
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# Topics supported by fewer than this many claims are dropped as too thin.
MIN_CLAIMS_PER_TOPIC = 3


@dataclass
class PremiumExtractionCoreResult:
    """Structured output of the DB-free extraction chain.

    Carries everything the caller needs to persist results, minus any
    database-assigned identifiers.
    """
    claims: List[ClaimWithTopic]
    claims_extracted: int
    topic_of_discussion: List[str]
    claim_with_topic: dict[str, list[str]]
    key_takeaways: List[str]
    model_used: str


@dataclass
class TakeawayLink:
    """A key takeaway resolved to the claim it restates (by exact text match)."""
    text: str
    claim_order: Optional[int]


def build_claim_topics(
    topics: List[str],
    claims_with_topics: dict[str, list[str]],
    episode_id: int,
) -> tuple[List[str], dict[str, list[str]], List[ClaimWithTopic], int]:
    """Filter sparse topics and assign sequential claim_order.

    Pure, deterministic — the single source of truth shared by the in-process
    pipeline and the DAG worker. Returns (ordered_topics, filtered_map,
    claim_topics, claims_extracted).
    """
    filtered = {
        topic: claims
        for topic, claims in claims_with_topics.items()
        if len(claims) >= MIN_CLAIMS_PER_TOPIC
    }
    ordered_topics = [t for t in topics if t in filtered]

    claim_topics: List[ClaimWithTopic] = []
    claim_order = 1
    for topic in ordered_topics:
        for claim in filtered.get(topic, []):
            claim_topics.append(
                ClaimWithTopic(
                    claim_text=claim,
                    topic=topic,
                    episode_id=episode_id,
                    claim_order=claim_order,
                )
            )
            claim_order += 1

    claims_extracted = sum(len(c) for c in filtered.values())
    return ordered_topics, filtered, claim_topics, claims_extracted


def format_topics_with_claims(claims_with_topics: dict[str, list[str]]) -> str:
    """Render the topic->claims map into the prompt string for key takeaways."""
    sections = []
    for topic, claims in claims_with_topics.items():
        section = f"Topic: {topic}\n"
        for claim in claims:
            section += f"- {claim}\n"
        sections.append(section)
    return "\n".join(sections)


def link_takeaways_to_claims(
    key_takeaways: List[str],
    claim_topics: List[ClaimWithTopic],
) -> List[TakeawayLink]:
    """Resolve each takeaway to its claim's claim_order by exact text match.

    Done in-process (where both lists are in memory) so downstream consumers
    never have to re-run the fragile string match. Unmatched -> claim_order=None.
    """
    order_by_text = {c.claim_text: c.claim_order for c in claim_topics}
    return [TakeawayLink(text=t, claim_order=order_by_text.get(t)) for t in key_takeaways]


async def run_premium_extraction(
    *,
    episode_id: int,
    title: str,
    description: str,
    transcript: str,
    transcript_format: str,
    parser: Optional[TranscriptParser] = None,
    extractor: Optional[PremiumClaimExtractor] = None,
) -> PremiumExtractionCoreResult:
    """Run the premium extraction chain without any DB access.

    Args:
        episode_id: Episode id, stamped onto each ClaimWithTopic for the caller.
        title: Episode title (prompt input).
        description: Episode description (prompt input).
        transcript: Raw transcript text.
        transcript_format: One of "podscribe" | "bankless" | "assembly".
        parser: Optional shared TranscriptParser (constructed if omitted).
        extractor: Optional shared PremiumClaimExtractor (constructed if omitted).

    Returns:
        PremiumExtractionCoreResult. On no topics or no surviving claims, returns
        a result with an empty `claims` list (claims_extracted=0).
    """
    parser = parser or TranscriptParser()
    extractor = extractor or PremiumClaimExtractor()
    model_used = settings.gemini_premium_model

    # Step 1: Parse transcript
    logger.info("Step 1/5: Parsing transcript...")
    parsed_transcript = parser.parse(transcript, format=transcript_format)
    logger.info(f"  ✓ Parsed {len(parsed_transcript.segments)} segments")

    # Step 2: Extract topics of discussion
    logger.info("Step 2/5: Extracting topics of discussion...")
    stage_start = time.time()
    topics = await extractor.extract_topics_of_discussion_from_episode(
        title=title,
        description=description,
        full_transcript=parsed_transcript.full_text,
    )
    logger.info(
        f"  ✓ Extracted {len(topics)} topics in {time.time() - stage_start:.1f}s"
    )
    if not topics:
        logger.warning("No topics extracted, ending extraction")
        return PremiumExtractionCoreResult(
            claims=[],
            claims_extracted=0,
            topic_of_discussion=[],
            claim_with_topic={},
            key_takeaways=[],
            model_used=model_used,
        )

    # Step 3: Extract claims for each topic
    logger.info("Step 3/5: Extracting claims with topics...")
    stage_start = time.time()
    claims_with_topics = await extractor.extract_claims_with_topics_from_transcript(
        full_transcript=parsed_transcript.full_text,
        topics_of_discussion=topics,
    )
    claims_extracted = sum(len(claim_list) for claim_list in claims_with_topics.values())
    logger.info(
        f"  ✓ Extracted {claims_extracted} claims in {time.time() - stage_start:.1f}s"
    )

    # Filter sparse topics + assign claim_order (shared with the DAG worker).
    topics, claims_with_topics, claim_topics, claims_extracted = build_claim_topics(
        topics, claims_with_topics, episode_id
    )

    if not claim_topics:
        logger.warning("No claims extracted, ending extraction")
        return PremiumExtractionCoreResult(
            claims=[],
            claims_extracted=0,
            topic_of_discussion=topics,
            claim_with_topic={},
            key_takeaways=[],
            model_used=model_used,
        )

    # Step 4: Extract key takeaways from the claim set
    logger.info("Step 4/5 Extract key takeaways...")
    stage_start = time.time()
    key_takeaways = await extractor.extract_key_takeaways_from_claims(
        topics_with_claims=format_topics_with_claims(claims_with_topics)
    )
    logger.info(
        f"  ✓ Extracted {len(key_takeaways)} key takeaways in "
        f"{time.time() - stage_start:.1f}s"
    )

    return PremiumExtractionCoreResult(
        claims=claim_topics,
        claims_extracted=claims_extracted,
        topic_of_discussion=topics,
        claim_with_topic=claims_with_topics,
        key_takeaways=key_takeaways,
        model_used=model_used,
    )
