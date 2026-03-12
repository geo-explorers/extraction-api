"""
ClaimRepository for database persistence operations.

Handles CRUD operations for claims, quotes, and claim-quote relationships.

Usage:
    from src.database.claim_repository import ClaimRepository

    repo = ClaimRepository(db_session)
    claim_ids = await repo.save_claims(claims_with_quotes, episode_id)
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.database.models import Claim, Quote, ClaimQuote, ClaimEpisode
from src.extraction.quote_finder import ClaimWithTopic, Quote as ExtractedQuote
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimRepository:
    """
    Repository for database persistence of claims and quotes.

    Features:
    - Save claims with embeddings
    - Save unique quotes (reuses existing by position)
    - Create claim-quote junction records
    - Transaction management with rollback
    - Merge quotes to existing claims (for cross-episode dedup)

    Example:
        ```python
        repo = ClaimRepository(db_session)

        # Save new claims
        claim_ids = await repo.save_claims(claims_with_quotes, episode_id)

        # Merge quotes to existing claim (cross-episode dedup)
        await repo.merge_quotes_to_existing_claim(
            existing_claim_id=123,
            new_quotes=quote_list,
            episode_id=456
        )

        # Commit transaction
        db_session.commit()
        ```
    """

    def __init__(self, db_session: Session):
        """
        Initialize the repository.

        Args:
            db_session: SQLAlchemy database session

        Example:
            ```python
            from src.database.connection import get_db_session

            session = get_db_session()
            repo = ClaimRepository(session)
            ```
        """
        self.session = db_session

    async def save_claims(
        self,
        claim_with_topics: List[ClaimWithTopic],
        episode_id: int
    ) -> List[ClaimWithTopic]:
        """
        Save claims with their quotes to database.

        Creates:
        - Claim records with embeddings
        - Quote records (reusing existing when possible)
        - ClaimQuote junction records with relevance scores

        Args:
            claims_with_quotes: List of claims with supporting quotes
            episode_id: Episode ID these claims belong to

        Returns:
            List of claim IDs that were saved

        Example:
            ```python
            repo = ClaimRepository(session)
            claim_ids = await repo.save_claims(claims_with_quotes, episode_id=123)
            print(f"Saved {len(claim_ids)} claims")
            ```
        """
        if not claim_with_topics:
            logger.warning("No claims to save")
            return []

        logger.info(f"Saving {len(claim_with_topics)} claims for episode {episode_id}")

        saved_claim_with_topics: List[ClaimWithTopic] = []

        try:
            for i, claim_with_topic in enumerate(claim_with_topics):
                # Save claim
                claim = Claim(
                    claim_text=claim_with_topic.claim_text,
                    confidence=0.8,
                    source_chunk_id=None,
                    merged_from_chunk_ids=None,
                    embedding=None,  # Will be set after we get the ID
                    confidence_components=None
                )

                self.session.add(claim)
                self.session.flush()  # Get ID without committing
                claim_with_topic.claim_id = claim.id


                saved_claim_with_topics.append(
                    claim_with_topic
                )

                logger.debug(
                    f"Saved claim {i}/{len(claim_with_topics)}: ID={claim.id}, "
                    f"text='{claim.claim_text[:60]}...'"
                )

                # Save quotes and create links
                # if claim_with_topic.quotes:
                #     await self._save_quotes_and_links(
                #         claim.id,
                #         claim_with_topic.quotes,
                #         episode_id
                #     )

            logger.info(f"✅ Saved {len(saved_claim_with_topics)} claims")
            return saved_claim_with_topics

        except Exception as e:
            logger.error(f"Error saving claims: {e}", exc_info=True)
            self.session.rollback()
            raise

    async def _save_quotes_and_links(
        self,
        claim_id: int,
        quotes: List[ExtractedQuote],
        episode_id: int
    ) -> None:
        """
        Save quotes and create claim-quote links.

        Reuses existing quotes when possible (by position).

        Args:
            claim_id: Claim ID to link quotes to
            quotes: List of extracted quotes
            episode_id: Episode ID

        Internal method called by save_claims.
        """
        for quote_obj in quotes:
            # Check if quote already exists (by position)
            existing_quote = (
                self.session.query(Quote)
                .filter(
                    Quote.episode_id == episode_id,
                    Quote.start_position == quote_obj.start_position,
                    Quote.end_position == quote_obj.end_position
                )
                .first()
            )

            if existing_quote:
                quote_id = existing_quote.id
                logger.debug(
                    f"Reusing existing quote ID={quote_id} "
                    f"(pos: {quote_obj.start_position}-{quote_obj.end_position})"
                )
            else:
                # Create new quote
                quote = Quote(
                    episode_id=episode_id,
                    quote_text=quote_obj.quote_text,
                    start_position=quote_obj.start_position,
                    end_position=quote_obj.end_position,
                    speaker=quote_obj.speaker,
                    timestamp_seconds=quote_obj.timestamp_seconds
                )

                self.session.add(quote)
                self.session.flush()

                quote_id = quote.id
                logger.debug(f"Created new quote ID={quote_id}")

            # Create claim-quote link
            claim_quote = ClaimQuote(
                claim_id=claim_id,
                quote_id=quote_id,
                relevance_score=quote_obj.relevance_score,
                match_confidence=quote_obj.relevance_score,  # Same for now
                match_type="reranked",
                entailment_score=quote_obj.entailment_score,
                entailment_relationship=quote_obj.entailment_relationship
            )

            self.session.add(claim_quote)

    async def merge_quotes_to_existing_claim(
        self,
        existing_claim_id: int,
        new_quotes: List[ExtractedQuote],
        episode_id: int
    ) -> int:
        """
        Add new quotes to an existing claim (cross-episode deduplication).

        Used when we find a duplicate claim in another episode and want to
        add the new episode's quotes to the existing claim.

        Args:
            existing_claim_id: ID of existing claim to add quotes to
            new_quotes: List of new quotes to add
            episode_id: Episode ID where new quotes come from

        Returns:
            Number of new quote links created

        Example:
            ```python
            # Found duplicate claim 123 from episode 456
            num_added = await repo.merge_quotes_to_existing_claim(
                existing_claim_id=123,
                new_quotes=new_episode_quotes,
                episode_id=456
            )
            print(f"Added {num_added} quotes to existing claim")
            ```
        """
        logger.info(
            f"Merging {len(new_quotes)} quotes to existing claim {existing_claim_id}"
        )

        added_count = 0

        try:
            for quote_obj in new_quotes:
                # Check if quote already exists
                existing_quote = (
                    self.session.query(Quote)
                    .filter(
                        Quote.episode_id == episode_id,
                        Quote.start_position == quote_obj.start_position,
                        Quote.end_position == quote_obj.end_position
                    )
                    .first()
                )

                if existing_quote:
                    quote_id = existing_quote.id
                else:
                    # Create new quote
                    quote = Quote(
                        episode_id=episode_id,
                        quote_text=quote_obj.quote_text,
                        start_position=quote_obj.start_position,
                        end_position=quote_obj.end_position,
                        speaker=quote_obj.speaker,
                        timestamp_seconds=quote_obj.timestamp_seconds
                    )

                    self.session.add(quote)
                    self.session.flush()
                    quote_id = quote.id

                # Check if link already exists
                existing_link = (
                    self.session.query(ClaimQuote)
                    .filter(
                        ClaimQuote.claim_id == existing_claim_id,
                        ClaimQuote.quote_id == quote_id
                    )
                    .first()
                )

                if existing_link:
                    logger.debug(
                        f"Link already exists for claim {existing_claim_id} "
                        f"and quote {quote_id}"
                    )
                    continue

                # Create new link
                claim_quote = ClaimQuote(
                    claim_id=existing_claim_id,
                    quote_id=quote_id,
                    relevance_score=quote_obj.relevance_score,
                    match_confidence=quote_obj.relevance_score,
                    match_type="cross_ep_rerank",
                    entailment_score=quote_obj.entailment_score,
                    entailment_relationship=quote_obj.entailment_relationship
                )

                self.session.add(claim_quote)
                added_count += 1

            self.session.flush()

            logger.info(
                f"✅ Added {added_count} new quote links to claim {existing_claim_id}"
            )

            return added_count

        except Exception as e:
            logger.error(f"Error merging quotes: {e}", exc_info=True)
            self.session.rollback()
            raise

    async def update_claim_embeddings(
        self,
        claim_ids_and_embeddings: Dict[int, List[float]]
    ) -> None:
        """
        Update embeddings for claims after initial insert.

        Embeddings are updated separately because they're generated after
        claim IDs are assigned.

        Args:
            claim_ids_and_embeddings: Dict mapping claim_id to embedding vector

        Example:
            ```python
            embeddings = {
                123: [0.1, 0.2, ...],  # 768 dims
                124: [0.3, 0.4, ...],
            }
            await repo.update_claim_embeddings(embeddings)
            ```
        """
        logger.info(f"Updating embeddings for {len(claim_ids_and_embeddings)} claims")

        try:
            for claim_id, embedding in claim_ids_and_embeddings.items():
                claim = self.session.query(Claim).get(claim_id)
                if claim:
                    claim.embedding = embedding
                else:
                    logger.warning(f"Claim {claim_id} not found for embedding update")

            self.session.flush()
            logger.info("✅ Embeddings updated")

        except Exception as e:
            logger.error(f"Error updating embeddings: {e}", exc_info=True)
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_claims(claims, episode_id)
                session.commit()
            except Exception as e:
                repo.rollback()
                raise
            ```
        """
        logger.warning("Rolling back transaction")
        self.session.rollback()

    def commit(self) -> None:
        """
        Commit current transaction.

        Example:
            ```python
            await repo.save_claims(claims, episode_id)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()

    def get_claims_by_episodes(
        self,
        episode_ids: List[int],
        include_flagged: bool = False,
        include_verified: bool = False
    ) -> Dict[int, List[Claim]]:
        """
        Get all claims for specific episodes, grouped by episode.

        Args:
            episode_ids: List of episode IDs
            include_flagged: If True, include already flagged claims
            include_verified: If True, include already verified claims

        Returns:
            Dict mapping episode_id to list of claims

        Example:
            ```python
            repo = ClaimRepository(session)

            # Get only unverified, unflagged claims for validation
            claims_by_episode = repo.get_claims_by_episodes([123, 456])

            # Get all claims including verified ones
            all_claims = repo.get_claims_by_episodes([123, 456], include_verified=True)
            ```
        """
        logger.info(
            f"Fetching claims for {len(episode_ids)} episodes "
            f"(include_flagged={include_flagged}, include_verified={include_verified})"
        )

        # Query claims through ClaimEpisode junction table
        query = (
            self.session.query(Claim, ClaimEpisode.episode_id)
            .join(ClaimEpisode, Claim.id == ClaimEpisode.claim_id)
            .filter(ClaimEpisode.episode_id.in_(episode_ids))
        )

        if not include_flagged:
            query = query.filter(Claim.is_flagged == False)

        if not include_verified:
            query = query.filter(Claim.is_verified == False)

        results = query.all()

        # Group by episode
        claims_by_episode: Dict[int, List[Claim]] = {}
        for claim, episode_id in results:
            if episode_id not in claims_by_episode:
                claims_by_episode[episode_id] = []
            claims_by_episode[episode_id].append(claim)

        logger.info(
            f"Found {len(results)} claims across {len(claims_by_episode)} episodes"
        )

        return claims_by_episode

    def get_episode_claim_counts(
        self,
        episode_ids: List[int],
        only_unverified: bool = True
    ) -> Dict[int, int]:
        """
        Get claim counts for specific episodes.

        Args:
            episode_ids: List of episode IDs
            only_unverified: If True, only count unverified claims (default: True)

        Returns:
            Dict mapping episode_id to claim count (unflagged, optionally unverified)

        Example:
            ```python
            repo = ClaimRepository(session)

            # Get count of unverified claims (for validation)
            counts = repo.get_episode_claim_counts([123, 456])
            print(f"Episode 123 has {counts[123]} unverified claims")

            # Get total claim counts
            total_counts = repo.get_episode_claim_counts([123, 456], only_unverified=False)
            ```
        """
        from sqlalchemy import func

        logger.info(
            f"Fetching claim counts for {len(episode_ids)} episodes "
            f"(only_unverified={only_unverified})"
        )

        # Query through ClaimEpisode junction table
        query = (
            self.session.query(
                ClaimEpisode.episode_id,
                func.count(Claim.id).label("count")
            )
            .join(ClaimEpisode, Claim.id == ClaimEpisode.claim_id)
            .filter(
                ClaimEpisode.episode_id.in_(episode_ids),
                Claim.is_flagged == False
            )
        )

        if only_unverified:
            query = query.filter(Claim.is_verified == False)

        results = query.group_by(ClaimEpisode.episode_id).all()

        counts = {episode_id: count for episode_id, count in results}

        # Ensure all requested episodes have an entry (even if 0)
        for episode_id in episode_ids:
            if episode_id not in counts:
                counts[episode_id] = 0

        logger.info(f"Retrieved claim counts for {len(counts)} episodes")

        return counts

    def flag_claims(
        self,
        claim_ids: List[int],
        dry_run: bool = False
    ) -> int:
        """
        Flag claims as bad using batch update.

        Args:
            claim_ids: List of claim IDs to flag
            dry_run: If True, don't actually update the database

        Returns:
            Number of claims flagged

        Example:
            ```python
            repo = ClaimRepository(session)

            # Flag specific claims
            count = repo.flag_claims([123, 456, 789])
            print(f"Flagged {count} claims")

            session.commit()
            ```
        """
        if not claim_ids:
            logger.warning("No claims to flag")
            return 0

        logger.info(f"Flagging {len(claim_ids)} claims (dry_run={dry_run})")

        if dry_run:
            logger.info("[DRY RUN] Would flag claims but not updating database")
            return len(claim_ids)

        try:
            # Batch update using SQLAlchemy
            flagged_count = (
                self.session.query(Claim)
                .filter(Claim.id.in_(claim_ids))
                .update(
                    {"is_flagged": True},
                    synchronize_session=False
                )
            )

            self.session.flush()

            logger.info(f"✅ Flagged {flagged_count} claims")
            return flagged_count

        except Exception as e:
            logger.error(f"Error flagging claims: {e}", exc_info=True)
            self.session.rollback()
            raise

    def mark_claims_verified(
        self,
        claim_ids: List[int],
        dry_run: bool = False
    ) -> int:
        """
        Mark claims as verified (validation completed) using batch update.

        Sets is_verified = TRUE for all provided claim IDs, indicating they
        have been validated by Gemini. This prevents re-validation when
        increasing target counts.

        Args:
            claim_ids: List of claim IDs to mark as verified
            dry_run: If True, don't actually update the database

        Returns:
            Number of claims marked as verified

        Example:
            ```python
            repo = ClaimRepository(session)

            # Mark claims as verified after validation
            claim_ids = [123, 456, 789]
            count = repo.mark_claims_verified(claim_ids)
            print(f"Marked {count} claims as verified")

            session.commit()
            ```
        """
        if not claim_ids:
            logger.warning("No claims to mark as verified")
            return 0

        logger.info(
            f"Marking {len(claim_ids)} claims as verified (dry_run={dry_run})"
        )

        if dry_run:
            logger.info("[DRY RUN] Would mark claims as verified but not updating database")
            return len(claim_ids)

        try:
            # Batch update using SQLAlchemy
            verified_count = (
                self.session.query(Claim)
                .filter(Claim.id.in_(claim_ids))
                .update(
                    {"is_verified": True},
                    synchronize_session=False
                )
            )

            self.session.flush()

            logger.info(f"✅ Marked {verified_count} claims as verified")
            return verified_count

        except Exception as e:
            logger.error(f"Error marking claims as verified: {e}", exc_info=True)
            self.session.rollback()
            raise

    def get_claims_for_episode(
        self,
        episode_id: int,
        include_flagged: bool = False,
        include_verified: bool = True
    ) -> List[Claim]:
        """
        Get all claims for a single episode.

        Args:
            episode_id: Episode ID
            include_flagged: If True, include already flagged claims
            include_verified: If True, include already verified claims (default: True)

        Returns:
            List of claims for the episode

        Example:
            ```python
            repo = ClaimRepository(session)

            # Get all claims for validation (including verified for re-validation)
            claims = repo.get_claims_for_episode(123, include_verified=True)
            ```
        """
        logger.info(
            f"Fetching claims for episode {episode_id} "
            f"(include_flagged={include_flagged}, include_verified={include_verified})"
        )

        # Query through ClaimEpisode junction table (claims no longer have episode_id)
        query = (
            self.session.query(Claim)
            .join(ClaimEpisode, Claim.id == ClaimEpisode.claim_id)
            .filter(ClaimEpisode.episode_id == episode_id)
        )

        if not include_flagged:
            query = query.filter(Claim.is_flagged == False)

        if not include_verified:
            query = query.filter(Claim.is_verified == False)

        claims = query.all()

        logger.info(f"Found {len(claims)} claims for episode {episode_id}")

        return claims
