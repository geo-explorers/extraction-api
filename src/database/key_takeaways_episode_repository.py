"""
KeyTakeAwaysEpisodeRepository for database persistence of key takeaway links.

Handles CRUD operations for key takeaway-episode relationships.

Usage:
    from src.database.key_takeaways_episode_repository import (
        KeyTakeAwaysEpisodeRepository
    )

    repo = KeyTakeAwaysEpisodeRepository(db_session)
    updated_key_takeaways = await repo.save_key_takeaways(key_takeaways)
"""

from typing import Dict, List, Tuple
from sqlalchemy.orm import Session

from src.database.models import KeyTakeAwaysEpisode
from src.extraction.quote_finder import KeyTakeAwayWithClaim
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class KeyTakeAwaysEpisodeRepository:
    """
    Repository for database persistence of key takeaway-episode links.

    Features:
    - Save key takeaway links in batch
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
            repo = KeyTakeAwaysEpisodeRepository(session)
            ```
        """
        self.session = db_session

    async def save_key_takeaways(
        self,
        key_takeaways: List[KeyTakeAwayWithClaim],
    ) -> List[KeyTakeAwayWithClaim]:
        """
        Save key takeaway-episode links to database.

        Creates KeyTakeAwaysEpisode records for each claim + episode pair,
        skipping any links that already exist. Updates key_takeaway_episode_id
        on each KeyTakeAwayWithClaim and persists claim_order values.

        Args:
            key_takeaways: KeyTakeAwayWithClaim items to link

        Returns:
            List of KeyTakeAwayWithClaim with key_takeaway_episode_id set
        """
        if not key_takeaways:
            logger.warning("No key takeaways provided to link")
            return []

        logger.info(f"Linking {len(key_takeaways)} key takeaways")

        try:
            pair_to_takeaways: Dict[
                Tuple[int, int],
                List[KeyTakeAwayWithClaim]
            ] = {}
            pair_to_order: Dict[Tuple[int, int], int] = {}
            ordered_pairs: List[Tuple[int, int]] = []

            for key_takeaway in key_takeaways:
                claim_id = key_takeaway.claim_id
                episode_id = key_takeaway.episode_id
                if claim_id is None or episode_id is None:
                    logger.warning(
                        "Key takeaway missing claim_id or episode_id; "
                        "skipping link creation"
                    )
                    continue

                key = (claim_id, episode_id)
                if key not in pair_to_takeaways:
                    pair_to_takeaways[key] = []
                    ordered_pairs.append(key)

                pair_to_takeaways[key].append(key_takeaway)

            for key, key_takeaway_list in pair_to_takeaways.items():
                claim_order = next(
                    (
                        take.claim_order
                        for take in key_takeaway_list
                        if take.claim_order is not None
                    ),
                    None,
                )
                pair_to_order[key] = claim_order

            if not ordered_pairs:
                logger.warning("No claim/episode pairs provided to link")
                return []

            claim_ids = list({claim_id for claim_id, _ in ordered_pairs})
            episode_ids = list({episode_id for _, episode_id in ordered_pairs})

            existing_links = (
                self.session.query(KeyTakeAwaysEpisode)
                .filter(
                    KeyTakeAwaysEpisode.claim_id.in_(claim_ids),
                    KeyTakeAwaysEpisode.episode_id.in_(episode_ids),
                )
                .all()
            )
            existing_by_pair = {
                (link.claim_id, link.episode_id): link
                for link in existing_links
            }

            new_pairs = [
                pair
                for pair in ordered_pairs
                if pair not in existing_by_pair
            ]

            links = []
            if new_pairs:
                for claim_id, episode_id in new_pairs:
                    links.append(
                        KeyTakeAwaysEpisode(
                            claim_id=claim_id,
                            episode_id=episode_id,
                            claim_order=pair_to_order.get(
                                (claim_id, episode_id)
                            ),
                        )
                    )

                self.session.add_all(links)
                self.session.flush()

                for link in links:
                    existing_by_pair[(link.claim_id, link.episode_id)] = link

            for key, claim_order in pair_to_order.items():
                if claim_order is None:
                    continue
                link = existing_by_pair.get(key)
                if link is None:
                    continue
                link.claim_order = claim_order

            updated_key_takeaways: List[KeyTakeAwayWithClaim] = []
            for key, key_takeaway_list in pair_to_takeaways.items():
                link = existing_by_pair.get(key)
                if link is None:
                    continue
                key_takeaway_episode_id = link.id
                for key_takeaway in key_takeaway_list:
                    key_takeaway.key_takeaway_episode_id = (
                        key_takeaway_episode_id
                    )
                    updated_key_takeaways.append(key_takeaway)

            logger.info(
                "Linked %s key takeaways (out of %s unique)",
                len(updated_key_takeaways),
                len(ordered_pairs),
            )

            return updated_key_takeaways

        except Exception as e:
            logger.error(
                f"Error saving key takeaway-episode links: {e}",
                exc_info=True,
            )
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_key_takeaways(key_takeaways)
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
            await repo.save_key_takeaways(key_takeaways)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
