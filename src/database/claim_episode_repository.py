"""
ClaimEpisodeRepository for database persistence of claim-episode links.

Handles CRUD operations for claim-episode relationships.

Usage:
    from src.database.claim_episode_repository import ClaimEpisodeRepository

    repo = ClaimEpisodeRepository(db_session)
    updated_claims = await repo.save_claim_episodes(claim_topics, episode_id=123)
"""

from typing import List, Dict
from sqlalchemy.orm import Session

from src.database.models import ClaimEpisode
from src.extraction.models import ClaimWithTopic
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimEpisodeRepository:
    """
    Repository for database persistence of claim-episode links.

    Features:
    - Save claim-episode links in batch
    - Query claim IDs by episode
    - Query episode IDs by claim
    - Transaction management with rollback

    Example:
        ```python
        repo = ClaimEpisodeRepository(db_session)

        # Save claim-episode links
        updated_claims = await repo.save_claim_episodes(claim_topics, episode_id=555)

        # Query claim IDs for an episode
        claim_ids = repo.get_claim_ids_by_episode(episode_id=555)

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
            repo = ClaimEpisodeRepository(session)
            ```
        """
        self.session = db_session

    async def save_claim_episodes(
        self,
        claim_topics: List[ClaimWithTopic],
        episode_id: int
    ) -> List[ClaimWithTopic]:
        """
        Save claim-episode links to database.

        Creates ClaimEpisode records for each claim ID, skipping any links
        that already exist for the given episode. Updates claim_episode_id
        on each ClaimWithTopic and persists claim_order values.

        Args:
            claim_topics: ClaimWithTopic items to link to the episode
            episode_id: Episode ID the claims belong to

        Returns:
            List of ClaimWithTopic with claim_episode_id set

        Example:
            ```python
            repo = ClaimEpisodeRepository(session)
            updated_claims = await repo.save_claim_episodes(claim_topics, episode_id=123)
            print(f"Linked {len(updated_claims)} claims to episode")
            ```
        """
        if not claim_topics:
            logger.warning("No claim topics provided to link")
            return []

        logger.info(
            f"Linking {len(claim_topics)} claims to episode {episode_id}"
        )

        try:
            claim_id_to_topics: Dict[int, List[ClaimWithTopic]] = {}
            claim_id_to_order: Dict[int, int] = {}
            ordered_claim_ids: List[int] = []

            for claim_topic in claim_topics:
                claim_id = claim_topic.claim_id
                if claim_id is None:
                    logger.warning(
                        "Claim topic missing claim_id; skipping link creation"
                    )
                    continue

                if claim_id not in claim_id_to_topics:
                    claim_id_to_topics[claim_id] = []
                    ordered_claim_ids.append(claim_id)

                claim_id_to_topics[claim_id].append(claim_topic)

            for claim_id, claim_topic_list in claim_id_to_topics.items():
                claim_order = next(
                    (
                        claim_topic.claim_order
                        for claim_topic in claim_topic_list
                        if claim_topic.claim_order is not None
                    ),
                    None,
                )
                claim_id_to_order[claim_id] = claim_order

            if not ordered_claim_ids:
                logger.warning("No claim IDs provided to link")
                return []

            existing_links = (
                self.session.query(ClaimEpisode)
                .filter(
                    ClaimEpisode.episode_id == episode_id,
                    ClaimEpisode.claim_id.in_(ordered_claim_ids),
                )
                .all()
            )
            existing_by_claim = {
                link.claim_id: link for link in existing_links
            }

            new_claim_ids = [
                claim_id
                for claim_id in ordered_claim_ids
                if claim_id not in existing_by_claim
            ]

            links = []
            if new_claim_ids:
                for claim_id in new_claim_ids:
                    claim_order = claim_id_to_order.get(claim_id)
                    links.append(
                        ClaimEpisode(
                            claim_id=claim_id,
                            episode_id=episode_id,
                            claim_order=claim_order,
                        )
                    )

                self.session.add_all(links)
                self.session.flush()

                for link in links:
                    existing_by_claim[link.claim_id] = link

            for claim_id, claim_order in claim_id_to_order.items():
                link = existing_by_claim.get(claim_id)
                if link is None:
                    continue
                if claim_order is not None:
                    link.claim_order = claim_order

            updated_claim_topics: List[ClaimWithTopic] = []
            for claim_id, claim_topic_list in claim_id_to_topics.items():
                link = existing_by_claim.get(claim_id)
                if link is None:
                    continue
                claim_episode_id = link.id
                for claim_topic in claim_topic_list:
                    claim_topic.claim_episode_id = claim_episode_id
                    updated_claim_topics.append(claim_topic)

            logger.info(
                f"Linked {len(updated_claim_topics)} claims to episode {episode_id}"
            )

            return updated_claim_topics

        except Exception as e:
            logger.error(
                f"Error saving claim-episode links: {e}",
                exc_info=True
            )
            self.session.rollback()
            raise

    def get_claim_ids_by_episode(self, episode_id: int) -> List[int]:
        """
        Get all claim IDs linked to a specific episode.

        Args:
            episode_id: Episode ID to fetch claims for

        Returns:
            List of claim IDs

        Example:
            ```python
            repo = ClaimEpisodeRepository(session)
            claim_ids = repo.get_claim_ids_by_episode(episode_id=123)
            ```
        """
        logger.info(f"Fetching claim IDs for episode {episode_id}")

        claim_ids = [
            row[0]
            for row in (
                self.session.query(ClaimEpisode.claim_id)
                .filter(ClaimEpisode.episode_id == episode_id)
                .all()
            )
        ]

        logger.info(
            f"Found {len(claim_ids)} claim IDs for episode {episode_id}"
        )

        return claim_ids

    def get_episode_ids_by_claims(
        self,
        claim_ids: List[int]
    ) -> Dict[int, List[int]]:
        """
        Get episode IDs for a list of claim IDs.

        Args:
            claim_ids: List of claim IDs to fetch episodes for

        Returns:
            Dict mapping claim_id to list of episode IDs

        Example:
            ```python
            repo = ClaimEpisodeRepository(session)
            episodes_by_claim = repo.get_episode_ids_by_claims([1, 2, 3])
            ```
        """
        if not claim_ids:
            logger.warning("No claim IDs provided to fetch episodes")
            return {}

        logger.info(f"Fetching episode IDs for {len(claim_ids)} claims")

        rows = (
            self.session.query(
                ClaimEpisode.claim_id,
                ClaimEpisode.episode_id
            )
            .filter(ClaimEpisode.claim_id.in_(claim_ids))
            .all()
        )

        episodes_by_claim = {claim_id: [] for claim_id in claim_ids}
        for claim_id, episode_id in rows:
            episodes_by_claim.setdefault(claim_id, []).append(episode_id)

        logger.info(
            f"Found episode links for {len(episodes_by_claim)} claims"
        )

        return episodes_by_claim

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_claim_episodes(claim_topics, episode_id)
                session.commit()
            except Exception:
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
            await repo.save_claim_episodes(claim_topics, episode_id)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
