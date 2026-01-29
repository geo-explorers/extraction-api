"""
Premium extraction pipeline using Gemini 3 Pro for full-transcript processing.

Differences from standard pipeline:
1. NO DSPy (direct Gemini API calls)
2. NO chunking (full transcript processing with 1M context)
3. NO deduplication (simplified)
4. NO ad filtering (simplified)
5. NO quote processing (focus on speed)
6. Single Gemini call instead of parallel chunk processing
7. Much faster (~30-60s vs 5-6 minutes)

Usage:
    from src.pipeline.premium_extraction_pipeline import PremiumExtractionPipeline

    pipeline = PremiumExtractionPipeline()
    result = await pipeline.process_episode(episode_id=123)

    print(f"Extracted {len(result.claims)} claims in {result.processing_time_seconds:.1f}s")
"""

from dataclasses import dataclass
from typing import List, cast, Optional
import time

from src.config.settings import settings
from src.database.claim_episode_repository import ClaimEpisodeRepository
from src.database.tag_map_repository import TagMapRepository
from src.database.connection import get_db_session
from src.database.models import PodcastEpisode
from src.database.claim_repository import ClaimRepository
from src.database.tag_repository import TagRepository
from src.preprocessing.transcript_parser import TranscriptParser
from src.extraction.premium_claim_extractor import PremiumClaimExtractor
from src.extraction.quote_finder import ClaimWithTopic
from src.infrastructure.embedding_service import EmbeddingService
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PremiumPipelineResult:
    """Result from premium pipeline execution."""
    episode_id: int
    claims: list[ClaimWithTopic]
    processing_time_seconds: float
    claims_extracted: int
    model_used: str  # "gemini-3-pro-preview"
    topic_of_discussion: list[str]
    claim_with_topic: dict[str, list[str]]
    key_takeaways: list[str]  # Key takeaways are now stored as tags via TagMap



class PremiumExtractionPipeline:
    """Premium pipeline using Gemini 3 Pro for full-transcript extraction."""

    def __init__(self):
        """Initialize premium pipeline components."""
        logger.info("Initializing PremiumExtractionPipeline")

        self.parser = TranscriptParser()
        self.premium_extractor = PremiumClaimExtractor()

        # Only initialize embedder if embeddings are enabled
        if settings.enable_embeddings:
            self.embedder = EmbeddingService()
            logger.info("Premium pipeline ready (no chunking, no dedup, no quotes)")
        else:
            self.embedder = None
            logger.info("Premium pipeline ready (no chunking, no dedup, no quotes, no embeddings)")

    async def process_episode(
        self,
        episode_id: int,
        save_to_db: bool = True
    ) -> PremiumPipelineResult:
        """
        Process episode through premium pipeline.

        Key differences from standard pipeline:
        - No chunking (processes full transcript)
        - Single Gemini API call
        - Faster processing (~30-60 seconds vs 5-6 minutes)
        - No quote processing
        - No deduplication

        Args:
            episode_id: Episode ID to process
            save_to_db: Whether to save results to database (default True)

        Returns:
            PremiumPipelineResult with claims and stats

        Raises:
            ValueError: If episode not found or has no transcript
            Exception: If pipeline fails
        """
        start_time = time.time()

        logger.info(f"Starting PREMIUM pipeline for episode {episode_id}")

        # Step 1: Load episode
        episode = self._load_episode(episode_id)
        transcript, transcript_format = self._select_transcript(episode)
        transcript_length = len(transcript)

        logger.info(
            f"Loaded episode {episode_id}: '{episode.name}' "
            f"({transcript_length} chars, {transcript_format} format)"
        )

        # Step 2: Parse transcript
        logger.info("Step 1/5: Parsing transcript...")
        parsed_transcript = self.parser.parse(transcript, format=transcript_format)
        logger.info(f"  ✓ Parsed {len(parsed_transcript.segments)} segments")

        logger.info("Step 2/5: Extracting topics of discussion...")
        stage_start = time.time()
        topics = await self.premium_extractor.extract_topics_of_discussion_from_episode(
            title=episode.name,
            description=episode.description,
            full_transcript=parsed_transcript.full_text
        )
        extraction_time = time.time() - stage_start
        topics_extracted = len(topics)
        logger.info(f"  ✓ Extracted {topics_extracted} topics in {extraction_time:.1f}s")
        if not topics:
            logger.warning("No topics extracted, ending pipeline")
            processing_time = time.time() - start_time
            return PremiumPipelineResult(
                episode_id=episode_id,
                claims=[],
                processing_time_seconds=processing_time,
                claims_extracted=0,
                model_used=settings.gemini_premium_model,
                topic_of_discussion=[],
                claim_with_topic={},
                key_takeaways=[]
            )


        logger.info("Step 3/5: Extracting claims with topics...")
        stage_start = time.time()
        claims_with_topics = await self.premium_extractor.extract_claims_with_topics_from_transcript(
            full_transcript=parsed_transcript.full_text,
            topics_of_discussion=topics
        )
        extraction_time = time.time() - stage_start

        claims_extracted = 0
        for _, claimList in claims_with_topics.items():
            claims_extracted += len(claimList)

        logger.info(f"  ✓ Extracted {claims_extracted} claims in {extraction_time:.1f}s")

        # Post-processing: Filter out topics with fewer than 3 claims
        min_claims_per_topic = 3
        filtered_claims_with_topics = {
            topic: claims for topic, claims in claims_with_topics.items()
            if len(claims) >= min_claims_per_topic
        }

        filtered_out_topics = len(claims_with_topics) - len(filtered_claims_with_topics)
        filtered_out_claims = claims_extracted - sum(len(c) for c in filtered_claims_with_topics.values())

        if filtered_out_topics > 0:
            logger.info(
                f"  Filtered out {filtered_out_topics} topics with <{min_claims_per_topic} claims "
                f"({filtered_out_claims} claims removed)"
            )

        # Update variables to use filtered data
        claims_with_topics = filtered_claims_with_topics
        topics = [t for t in topics if t in claims_with_topics]
        claims_extracted = sum(len(c) for c in claims_with_topics.values())

        # Build claim_topics in LLM extraction order:
        # - Topics are ordered as returned by the LLM (preserved in `topics` list)
        # - Claims within each topic are in LLM extraction order
        # - claim_order is assigned sequentially to preserve this order for frontend display
        claim_topics: List[ClaimWithTopic] = []
        claim_order = 1
        for topic in topics:
            claims = claims_with_topics.get(topic, [])
            for claim in claims:
                claim_topics.append(
                    ClaimWithTopic(
                        claim_text=claim,
                        topic=topic,
                        episode_id=episode_id,
                        claim_order=claim_order
                    )
                )
                claim_order += 1

        if not claim_topics:
            logger.warning("No claims extracted, ending pipeline")
            processing_time = time.time() - start_time
            return PremiumPipelineResult(
                episode_id=episode_id,
                claims=[],
                processing_time_seconds=processing_time,
                claims_extracted=0,
                model_used=settings.gemini_premium_model,
                topic_of_discussion=topics,
                claim_with_topic={},
                key_takeaways=[]
            )

        logger.info("Step 4/5 Extract key takeaways...")
        stage_start = time.time()

        # Format claims_with_topics into a structured string for the LLM
        # Each topic with its claims listed below it
        formatted_topics_with_claims = []
        for topic, claims in claims_with_topics.items():
            topic_section = f"Topic: {topic}\n"
            for claim in claims:
                topic_section += f"- {claim}\n"
            formatted_topics_with_claims.append(topic_section)
        topics_with_claims_str = "\n".join(formatted_topics_with_claims)

        key_takeaways_raw = await self.premium_extractor.extract_key_takeaways_from_claims(
            topics_with_claims=topics_with_claims_str
        )
        extraction_time = time.time() - stage_start
        logger.info(f"  ✓ Extracted {len(key_takeaways_raw)} key takeaways in {extraction_time:.1f}s")

        # Key takeaways are now stored as tags (not separate table)
        key_takeaways: list[str] = key_takeaways_raw

        if not key_takeaways:
            logger.warning("No key takeaways extracted, continuing without them")

        if save_to_db:
            logger.info("Step 5/5: Saving results to database...")
            db_session = get_db_session()

            try:
                # Generate embeddings for all claims (if enabled)
                if settings.enable_embeddings:
                    logger.info("  Generating embeddings for claims...")
                    for claim in claim_topics:
                        embedding = await self.embedder.embed_text(claim.claim_text)
                        claim.metadata["embedding"] = embedding
                else:
                    logger.info("  Skipping embedding generation (ENABLE_EMBEDDINGS=false)")

                # Save claims to database
                logger.info("  Saving claims to database...")
                claim_repo = ClaimRepository(db_session)
                saved_claim_topics_with_claim_ids = await claim_repo.save_claims(claim_topics, episode_id)

                # Update embeddings (if enabled)
                if settings.enable_embeddings:
                    saved_claim_ids = [claim.claim_id for claim in saved_claim_topics_with_claim_ids]
                    embeddings_dict = {
                        claim_id: claim.metadata["embedding"]
                        for claim_id, claim in zip(saved_claim_ids, saved_claim_topics_with_claim_ids)
                    }
                    await claim_repo.update_claim_embeddings(embeddings_dict)

                logger.info(f"  ✓ Saved {len(saved_claim_topics_with_claim_ids)} claims to database")

                logger.info("  Saving claim-episode links to database...")

                claim_episode_repo = ClaimEpisodeRepository(db_session)
                saved_claim_topics_with_claim_episode_id = await claim_episode_repo.save_claim_episodes(
                    claim_topics=saved_claim_topics_with_claim_ids,
                    episode_id=episode_id
                )

                logger.info(f"  ✓ Saved {len(saved_claim_topics_with_claim_episode_id)} claim-episode links")


                logger.info("  Saving topic entries to database...")
                tag_repo = TagRepository(db_session)
                saved_claim_topics_with_tag_id = await tag_repo.save_tags(
                    claim_topics=saved_claim_topics_with_claim_episode_id
                )
                logger.info(f"  ✓ Saved {len(saved_claim_topics_with_tag_id)} topic")

                logger.info("  Saving tag map entries to database...")
                tag_map_repo = TagMapRepository(db_session)
                saved_tag_map_topics = await tag_map_repo.save_tag_maps(
                    claim_topics=saved_claim_topics_with_tag_id
                )
                logger.info(f"  ✓ Saved {len(saved_tag_map_topics)} tag map entries")

                # Save key takeaways as tags linked to their corresponding claims
                if key_takeaways:
                    logger.info("  Saving key takeaways as tags...")
                    # Get or create the "Key takeaways" tag once
                    key_takeaway_tag_id = await tag_repo.get_or_create_tag(
                        tag_name="Key takeaways",
                        tag_category="KeyTakeaway"
                    )
                    key_takeaway_tag_count = 0
                    for key_takeaway_text in key_takeaways:
                        # Find the claim that matches this key takeaway
                        for saved_claim_topic in saved_claim_topics_with_tag_id:
                            if key_takeaway_text == saved_claim_topic.claim_text:
                                # Link the claim to the "Key Takeaways" tag via TagMap
                                await tag_map_repo.create_tag_map(
                                    tag_id=key_takeaway_tag_id,
                                    from_claim_episode_id=saved_claim_topic.claim_episode_id
                                )
                                key_takeaway_tag_count += 1
                                break
                    logger.info(f"  ✓ Saved {key_takeaway_tag_count} key takeaway tags")

                # Commit transaction
                db_session.commit()
                logger.info("  ✓ Transaction committed successfully")

            except Exception as e:
                logger.error(f"Error saving to database: {e}", exc_info=True)
                db_session.rollback()
                logger.warning("Transaction rolled back")
                raise
            finally:
                db_session.close()
        else:
            logger.info("Step 5/5: Skipping database save (API mode)")

        processing_time = time.time() - start_time

        logger.info(
            f"✅ PREMIUM pipeline complete for episode {episode_id} "
            f"({processing_time:.1f}s, {len(claim_topics)} claims)"
        )

        return PremiumPipelineResult(
            episode_id=episode_id,
            claims=claim_topics,
            processing_time_seconds=processing_time,
            claims_extracted=claims_extracted,
            model_used=settings.gemini_premium_model,
            topic_of_discussion=topics,
            claim_with_topic=claims_with_topics,
            key_takeaways=key_takeaways
        )

    def _load_episode(self, episode_id: int) -> PodcastEpisode:
        """
        Load episode from database.

        Args:
            episode_id: Episode ID

        Returns:
            PodcastEpisode instance

        Raises:
            ValueError: If episode not found
        """
        session = get_db_session()
        try:
            episode = session.query(PodcastEpisode).filter(
                PodcastEpisode.id == episode_id
            ).first()

            if not episode:
                raise ValueError(f"Episode {episode_id} not found in database")

            return episode

        finally:
            session.close()

    def _select_transcript(self, episode: PodcastEpisode) -> tuple[str, str]:
        """
        Select transcript from episode with priority-based format detection.

        Priority order:
        1. Podscribe transcript (if available)
        2. Bankless transcript (if available)
        3. Assembly transcript (if available)
        4. Raise error if none available

        Args:
            episode: Episode to get transcript from

        Returns:
            Tuple of (transcript_text, format_name)
            format_name is "podscribe", "bankless", or "assembly"

        Raises:
            ValueError: If episode has no transcript in any format
        """
        # Priority 1: Podscribe
        podscribe_transcript = cast(Optional[str], episode.podscribe_transcript)
        if podscribe_transcript:
            return (podscribe_transcript, "podscribe")

        # Priority 2: Bankless
        bankless_transcript = cast(Optional[str], episode.bankless_transcript)
        if bankless_transcript:
            return (bankless_transcript, "bankless")

        # Priority 3: Assembly
        assembly_transcript = cast(Optional[str], episode.assembly_transcript)
        if assembly_transcript:
            return (assembly_transcript, "assembly")

        # No transcript available
        raise ValueError(
            f"Episode {episode.id} has no transcript "
            f"(checked podscribe_transcript, bankless_transcript, and assembly_transcript)"
        )
