"""
TagRepository for database persistence of tags.

Handles CRUD operations for tags.

Usage:
    from src.database.tag_repository import TagRepository

    repo = TagRepository(db_session)
    updated_claims = await repo.save_tags(claim_topics)
"""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session

from src.database.models import Tag
from src.extraction.models import ClaimWithTopic
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class TagRepository:
    """
    Repository for database persistence of tags.

    Features:
    - Save unique tags in batch
    - Update tag_id on ClaimWithTopic records
    - Get or create tags by name and category
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
            repo = TagRepository(session)
            ```
        """
        self.session = db_session

    async def save_tags(
        self,
        claim_topics: List[ClaimWithTopic]
    ) -> List[ClaimWithTopic]:
        """
        Save tags to database.

        Creates Tag records for each unique tag text, skipping any tags that
        already exist. Updates tag_id on each item.

        Args:
            claim_topics: ClaimWithTopic items

        Returns:
            List of items with tag_id set
        """
        if not claim_topics:
            logger.warning("No items provided to tag")
            return []

        logger.info(f"Saving tags for {len(claim_topics)} items")

        try:
            ordered_tag_names: List[str] = []
            seen_names = set()

            for claim_topic in claim_topics:
                tag_name = claim_topic.topic
                if not tag_name:
                    logger.warning(
                        "Item missing tag text; skipping tag creation"
                    )
                    continue

                if tag_name not in seen_names:
                    seen_names.add(tag_name)
                    ordered_tag_names.append(tag_name)

            if not ordered_tag_names:
                logger.warning("No tag names provided to save")
                return []

            existing_rows = (
                self.session.query(Tag.name, Tag.id)
                .filter(Tag.name.in_(ordered_tag_names))
                .all()
            )
            existing_by_name: Dict[str, int] = {
                name: tag_id for name, tag_id in existing_rows
            }

            new_tag_names = [
                name for name in ordered_tag_names if name not in existing_by_name
            ]

            if new_tag_names:
                tags = [Tag(name=name) for name in new_tag_names]
                self.session.add_all(tags)
                self.session.flush()

                for tag in tags:
                    existing_by_name[tag.name] = tag.id

            updated_claim_topics: List[ClaimWithTopic] = []
            for claim_topic in claim_topics:
                tag_name = claim_topic.topic
                if not tag_name:
                    continue
                tag_id = existing_by_name.get(tag_name)
                if tag_id is None:
                    continue
                claim_topic.tag_id = tag_id
                updated_claim_topics.append(claim_topic)

            logger.info(
                "Saved %s new tags (out of %s unique)",
                len(new_tag_names),
                len(ordered_tag_names)
            )

            return updated_claim_topics

        except Exception as e:
            logger.error(f"Error saving tags: {e}", exc_info=True)
            self.session.rollback()
            raise

    async def get_or_create_tag(
        self,
        tag_name: str,
        tag_category: Optional[str] = None
    ) -> int:
        """
        Get an existing tag by name or create a new one.

        Args:
            tag_name: The tag name to find or create
            tag_category: Optional category for the tag (stored in metadata)

        Returns:
            The tag ID

        Example:
            ```python
            repo = TagRepository(session)
            tag_id = await repo.get_or_create_tag("KeyTakeaway", tag_category="KeyTakeaway")
            ```
        """
        try:
            existing_tag = (
                self.session.query(Tag)
                .filter(Tag.name == tag_name)
                .first()
            )

            if existing_tag:
                return existing_tag.id

            new_tag = Tag(name=tag_name)
            self.session.add(new_tag)
            self.session.flush()

            logger.info(f"Created new tag: {tag_name} (category: {tag_category})")
            return new_tag.id

        except Exception as e:
            logger.error(f"Error getting/creating tag: {e}", exc_info=True)
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_tags(claim_topics)
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
            await repo.save_tags(claim_topics)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
