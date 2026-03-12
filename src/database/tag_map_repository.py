"""
TagMapRepository for database persistence of tag map records.

Handles CRUD operations for tag map relationships.

Usage:
    from src.database.tag_map_repository import TagMapRepository

    repo = TagMapRepository(db_session)
    updated_claims = await repo.save_tag_maps(claim_topics)
"""

from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session

from src.database.models import TagMap
from src.extraction.quote_finder import ClaimWithTopic
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class TagMapRepository:
    """
    Repository for database persistence of tag map records.

    Features:
    - Save tag map entries in batch
    - Query tag categories by claim-episode
    - Query claim-episode IDs by tag categories
    - Transaction management with rollback
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
            repo = TagMapRepository(session)
            ```
        """
        self.session = db_session

    async def save_tag_maps(
        self,
        claim_topics: List[ClaimWithTopic]
    ) -> List[ClaimWithTopic]:
        """
        Save tag map entries to database.

        Creates TagMap records for each claim-episode + tag pair, skipping
        any entries that already exist. Uses a constant tag_category of
        "Topic" and the tag_id from each item.

        Args:
            claim_topics: ClaimWithTopic items with claim_episode_id and tag_id

        Returns:
            List of items with tag map entries saved
        """
        if not claim_topics:
            logger.warning("No items provided to tag map")
            return []

        logger.info(f"Saving tag maps for {len(claim_topics)} items")

        try:
            tag_category = "Topic"
            pair_to_claim_topics: Dict[
                Tuple[int, int],
                List[ClaimWithTopic]
            ] = {}
            ordered_pairs: List[Tuple[int, int]] = []

            for claim_topic in claim_topics:
                claim_episode_id = claim_topic.claim_episode_id
                if claim_episode_id is None:
                    logger.warning(
                        "Item missing claim_episode_id; skipping tag map creation"
                    )
                    continue

                tag_id = claim_topic.tag_id
                if tag_id is None:
                    logger.warning(
                        "Item missing tag_id; skipping tag map creation"
                    )
                    continue

                key = (claim_episode_id, tag_id)
                if key not in pair_to_claim_topics:
                    pair_to_claim_topics[key] = []
                    ordered_pairs.append(key)

                pair_to_claim_topics[key].append(claim_topic)

            if not ordered_pairs:
                logger.warning(
                    "No claim episode IDs or tag IDs provided to tag map"
                )
                return []

            claim_episode_ids = [
                claim_episode_id for claim_episode_id, _ in ordered_pairs
            ]
            tag_ids = [tag_id for _, tag_id in ordered_pairs]
            existing_rows = (
                self.session.query(
                    TagMap.from_claim_episode_id,
                    TagMap.to_tag_id
                )
                .filter(
                    TagMap.from_claim_episode_id.in_(claim_episode_ids),
                    TagMap.to_tag_id.in_(tag_ids),
                    TagMap.tag_category == tag_category,
                )
                .all()
            )
            existing_pairs = {(row[0], row[1]) for row in existing_rows}

            new_pairs = [
                pair
                for pair in ordered_pairs
                if pair not in existing_pairs
            ]

            if new_pairs:
                tag_maps = [
                    TagMap(
                        from_claim_episode_id=claim_episode_id,
                        tag_category=tag_category,
                        to_tag_id=tag_id,
                    )
                    for claim_episode_id, tag_id in new_pairs
                ]
                self.session.add_all(tag_maps)
                self.session.flush()

            saved_claim_topics: List[ClaimWithTopic] = []
            for key, claim_topic_list in pair_to_claim_topics.items():
                if key in existing_pairs or key in new_pairs:
                    saved_claim_topics.extend(claim_topic_list)

            logger.info(
                "Saved %s new tag map entries (out of %s unique)",
                len(new_pairs),
                len(ordered_pairs)
            )

            return saved_claim_topics

        except Exception as e:
            logger.error(f"Error saving tag map entries: {e}", exc_info=True)
            self.session.rollback()
            raise

    def get_tag_categories_by_claim_episode(
        self,
        claim_episode_id: int
    ) -> List[str]:
        """
        Get all tag categories linked to a specific claim-episode record.

        Args:
            claim_episode_id: ClaimEpisode ID to fetch tags for

        Returns:
            List of tag category strings
        """
        logger.info(
            f"Fetching tag categories for claim_episode_id {claim_episode_id}"
        )

        tag_categories = [
            row[0]
            for row in (
                self.session.query(TagMap.tag_category)
                .filter(TagMap.from_claim_episode_id == claim_episode_id)
                .all()
            )
        ]

        logger.info(
            f"Found {len(tag_categories)} tag categories for claim_episode_id {claim_episode_id}"
        )

        return tag_categories

    def get_claim_episode_ids_by_tags(
        self,
        tag_categories: List[str]
    ) -> Dict[str, List[int]]:
        """
        Get claim-episode IDs for a list of tag categories.

        Args:
            tag_categories: List of tag categories to fetch claim-episode IDs for

        Returns:
            Dict mapping tag_category to list of claim-episode IDs
        """
        if not tag_categories:
            logger.warning("No tag categories provided to fetch claim episodes")
            return {}

        logger.info(
            f"Fetching claim-episode IDs for {len(tag_categories)} tag categories"
        )

        rows = (
            self.session.query(
                TagMap.tag_category,
                TagMap.from_claim_episode_id
            )
            .filter(TagMap.tag_category.in_(tag_categories))
            .all()
        )

        claim_episodes_by_tag = {tag_category: [] for tag_category in tag_categories}
        for tag_category, claim_episode_id in rows:
            claim_episodes_by_tag.setdefault(tag_category, []).append(
                claim_episode_id
            )

        logger.info(
            f"Found claim-episode links for {len(claim_episodes_by_tag)} tag categories"
        )

        return claim_episodes_by_tag

    async def create_tag_map(
        self,
        tag_id: int,
        from_claim_episode_id: int,
        tag_category: str = "KeyTakeaway"
    ) -> None:
        """
        Create a single tag map entry.

        Args:
            tag_id: The tag ID to link
            from_claim_episode_id: The claim-episode ID to link from
            tag_category: Category for the tag map (default: "KeyTakeaway")

        Example:
            ```python
            repo = TagMapRepository(session)
            await repo.create_tag_map(
                tag_id=123,
                from_claim_episode_id=456,
                tag_category="KeyTakeaway"
            )
            ```
        """
        try:
            # Check if entry already exists
            existing = (
                self.session.query(TagMap)
                .filter(
                    TagMap.from_claim_episode_id == from_claim_episode_id,
                    TagMap.to_tag_id == tag_id,
                    TagMap.tag_category == tag_category,
                )
                .first()
            )

            if existing:
                logger.debug(
                    f"Tag map entry already exists: claim_episode={from_claim_episode_id}, tag={tag_id}"
                )
                return

            tag_map = TagMap(
                from_claim_episode_id=from_claim_episode_id,
                tag_category=tag_category,
                to_tag_id=tag_id,
            )
            self.session.add(tag_map)
            self.session.flush()

            logger.debug(
                f"Created tag map: claim_episode={from_claim_episode_id}, tag={tag_id}, category={tag_category}"
            )

        except Exception as e:
            logger.error(f"Error creating tag map entry: {e}", exc_info=True)
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_tag_maps(claim_topics)
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
            await repo.save_tag_maps(claim_topics)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
