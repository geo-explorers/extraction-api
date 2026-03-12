"""
Episode query service for CLI processing.

Handles database queries to find episodes to process, with filtering
for already-processed episodes and podcast selection.

Usage:
    from src.cli.episode_query import EpisodeQueryService

    service = EpisodeQueryService()
    episodes = service.get_episodes_to_process(
        podcast_id=123,
        limit=10,
        force=False
    )
"""

from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from src.database.models import PodcastEpisode, Claim, ClaimEpisode
from src.database.connection import get_db_session
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class EpisodeQueryService:
    """
    Service for querying episodes to process.

    Features:
    - Filter by podcast_id (or all podcasts)
    - Skip already-processed episodes (unless force=True)
    - Order by newest first (air_date DESC)
    - Limit number of episodes
    - Check processing status

    Example:
        ```python
        service = EpisodeQueryService()

        # Get 10 newest unprocessed episodes from podcast 123
        episodes = service.get_episodes_to_process(
            podcast_id=123,
            limit=10,
            force=False
        )

        # Check if specific episode is processed
        is_processed = service.is_episode_processed(episode_id=456)
        ```
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the query service.

        Args:
            db_session: Optional database session. If not provided, creates new session.
        """
        self.session = db_session or get_db_session()
        self._owns_session = db_session is None

    def __del__(self):
        """Close session if we own it."""
        if self._owns_session and hasattr(self, "session"):
            self.session.close()

    def get_episode_by_id(self, episode_id: int) -> Optional[PodcastEpisode]:
        """
        Get a specific episode by ID.

        Args:
            episode_id: Episode ID to fetch

        Returns:
            PodcastEpisode object or None if not found

        Raises:
            ValueError: If episode doesn't have a transcript

        Example:
            ```python
            episode = service.get_episode_by_id(123)
            if episode:
                print(f"Found: {episode.name}")
            ```
        """
        logger.info(f"Fetching episode {episode_id}")

        episode = self.session.query(PodcastEpisode).filter(
            PodcastEpisode.id == episode_id
        ).first()

        if episode is None:
            logger.error(f"Episode {episode_id} not found")
            return None

        if (episode.podscribe_transcript is None and
            episode.bankless_transcript is None and
            episode.assembly_transcript is None):
            logger.error(f"Episode {episode_id} has no transcript")
            raise ValueError(f"Episode {episode_id} has no transcript")

        logger.info(f"Found episode {episode_id}: {episode.name}")
        return episode

    def _count_episodes_with_claims(self, podcast_id: int) -> int:
        """
        Count episodes with claims for a podcast.

        Only counts episodes that have transcripts.

        Args:
            podcast_id: Podcast ID to count for

        Returns:
            Number of episodes with at least one claim
        """
        count = (
            self.session.query(func.count(func.distinct(ClaimEpisode.episode_id)))
            .join(PodcastEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
            .filter(
                PodcastEpisode.podcast_id == podcast_id,
                or_(
                    PodcastEpisode.podscribe_transcript.isnot(None),
                    PodcastEpisode.bankless_transcript.isnot(None),
                    PodcastEpisode.assembly_transcript.isnot(None)
                )
            )
            .scalar()
        )

        return count or 0

    def get_episodes_to_process(
        self, podcast_ids: Optional[List[int]] = None, target: int = 0, force: bool = False
    ) -> List[PodcastEpisode]:
        """
        Get episodes to process with target-based selection.

        When target > 0 and podcast_ids are provided, maintains a rolling window
        of the latest `target` episodes with claims for each podcast.

        Query logic:
        1. For each podcast (when podcast_ids + target specified):
           a. Get the latest `target` episodes with transcripts (by air_date DESC)
           b. Among those, find which don't have claims yet
           c. Return those for processing
        2. When target=0 or no podcast_ids: get all unprocessed episodes with transcripts
        3. All queries filter for transcripts at SQL level
        4. Skip already-processed episodes unless force=True

        Args:
            podcast_ids: Optional list of podcast IDs to filter by (None = all podcasts)
            target: Maintain claims for the latest N episodes per podcast (0 = no limit).
                   When specified with podcast_ids, ensures the latest N episodes all
                   have claims. New episodes will be processed even if target is already met.
                   Requires podcast_ids to be specified.
            force: If True, include already-processed episodes (not yet implemented for target mode)

        Returns:
            List of PodcastEpisode objects to process

        Raises:
            ValueError: If target > 0 but podcast_ids is None

        Example:
            ```python
            # Maintain claims for the latest 100 episodes of podcasts 1,2,3
            # If podcast 1's latest 100 includes 7 without claims, will process those 7
            # If podcast 2's latest 100 all have claims, will skip it
            # If 5 new episodes arrive tomorrow, will process those 5
            episodes = service.get_episodes_to_process(
                podcast_ids=[1,2,3],
                target=100
            )

            # Get all unprocessed episodes with transcripts
            episodes = service.get_episodes_to_process()

            # Get all episodes from specific podcasts (no target)
            episodes = service.get_episodes_to_process(podcast_ids=[1,2,3])
            ```
        """
        logger.info(
            f"Querying episodes: podcast_ids={podcast_ids}, "
            f"target={target}, force={force}"
        )

        # If we have podcast_ids AND a target, maintain rolling window of latest N episodes
        if podcast_ids is not None and target > 0:
            logger.debug(f"Using rolling window selection: maintaining latest {target} episodes with claims for {len(podcast_ids)} podcast(s)")
            all_episodes = []

            for podcast_id in podcast_ids:
                # Step 1: Get IDs of the latest `target` episodes with transcripts
                latest_episode_ids_query = (
                    self.session.query(PodcastEpisode.id)
                    .filter(
                        PodcastEpisode.podcast_id == podcast_id,
                        or_(
                            PodcastEpisode.podscribe_transcript.isnot(None),
                            PodcastEpisode.bankless_transcript.isnot(None),
                            PodcastEpisode.assembly_transcript.isnot(None)
                        )
                    )
                    .order_by(PodcastEpisode.air_date.desc().nulls_last())
                    .limit(target)
                )
                latest_episode_ids = [row[0] for row in latest_episode_ids_query.all()]

                if not latest_episode_ids:
                    logger.info(f"Podcast {podcast_id}: No episodes with transcripts found, skipping")
                    continue

                logger.debug(f"Podcast {podcast_id}: Found {len(latest_episode_ids)} episodes in latest {target} window")

                # Step 2: Among those latest episodes, find which don't have claims
                unprocessed_episodes_query = (
                    self.session.query(PodcastEpisode)
                    .filter(PodcastEpisode.id.in_(latest_episode_ids))
                    .outerjoin(ClaimEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
                    .filter(ClaimEpisode.id.is_(None))  # No claims = unprocessed
                    .order_by(PodcastEpisode.air_date.desc().nulls_last())
                )

                podcast_episodes = unprocessed_episodes_query.all()

                if podcast_episodes:
                    logger.info(f"Podcast {podcast_id}: Processing {len(podcast_episodes)} unprocessed episode(s) from latest {target} window")
                else:
                    logger.info(f"Podcast {podcast_id}: All episodes in latest {target} window have claims, skipping")

                all_episodes.extend(podcast_episodes)

            logger.info(f"Found {len(all_episodes)} total episode(s) to process across {len(podcast_ids)} podcast(s)")
            return all_episodes
        else:
            # Single query for all podcasts or when no target specified

            # Error if target is specified without podcast_ids
            if target > 0 and podcast_ids is None:
                raise ValueError(
                    "target parameter requires podcast_ids to be specified. "
                    "Target-based selection works per podcast, so you must specify which podcasts."
                )

            # Base query - only episodes with transcripts
            query = self.session.query(PodcastEpisode)
            query = query.filter(
                or_(
                    PodcastEpisode.podscribe_transcript.isnot(None),
                    PodcastEpisode.bankless_transcript.isnot(None),
                    PodcastEpisode.assembly_transcript.isnot(None)
                )
            )

            # Filter by podcast_ids if provided
            if podcast_ids is not None:
                query = query.filter(PodcastEpisode.podcast_id.in_(podcast_ids))
                logger.debug(f"Filtering by podcast_ids={podcast_ids}")

            # Skip already-processed episodes unless force=True
            if not force:
                # LEFT JOIN to find episodes without claims
                query = query.outerjoin(
                    ClaimEpisode, ClaimEpisode.episode_id == PodcastEpisode.id
                ).filter(
                    ClaimEpisode.id.is_(None)
                )  # No claims = not processed
                logger.debug("Filtering to unprocessed episodes only (force=False)")

            # Order by newest first (air_date DESC), NULL dates last
            query = query.order_by(PodcastEpisode.air_date.desc().nulls_last())

            # Note: target is not applied in this path (target=0 means no limit)
            # This path is used for: get all unprocessed episodes

            # Execute query
            episodes = query.all()

            logger.info(f"Found {len(episodes)} episodes to process")

            return episodes

    def is_episode_processed(self, episode_id: int) -> bool:
        """
        Check if an episode has already been processed.

        An episode is considered processed if it has any claims in the database.

        Args:
            episode_id: Episode ID to check

        Returns:
            True if episode has claims, False otherwise

        Example:
            ```python
            if service.is_episode_processed(123):
                print("Episode 123 already processed")
            else:
                print("Episode 123 needs processing")
            ```
        """
        count = (
            self.session.query(func.count(ClaimEpisode.id))
            .filter(ClaimEpisode.episode_id == episode_id)
            .scalar()
        )

        is_processed = count > 0

        logger.debug(
            f"Episode {episode_id}: "
            f"{'processed' if is_processed else 'unprocessed'} "
            f"({count} claims)"
        )

        return is_processed

    def get_processing_stats(self, podcast_id: Optional[int] = None) -> Dict[str, int]:
        """
        Get summary statistics for episodes.

        Args:
            podcast_id: Optional podcast ID to filter by

        Returns:
            Dict with statistics:
            - total_episodes: Total episodes with transcripts
            - processed: Episodes with claims
            - unprocessed: Episodes without claims
            - total_claims: Total claims across all episodes

        Example:
            ```python
            stats = service.get_processing_stats(podcast_id=123)
            print(f"Podcast 123: {stats['unprocessed']} episodes to process")
            ```
        """
        # Base filter
        episode_filter = or_(
            PodcastEpisode.podscribe_transcript.isnot(None),
            PodcastEpisode.bankless_transcript.isnot(None),
            PodcastEpisode.assembly_transcript.isnot(None)
        )
        if podcast_id is not None:
            episode_filter = and_(
                episode_filter, PodcastEpisode.podcast_id == podcast_id
            )

        # Total episodes with transcripts
        total_episodes = (
            self.session.query(func.count(PodcastEpisode.id))
            .filter(episode_filter)
            .scalar()
        )

        # Processed episodes (have claims)
        processed = (
            self.session.query(func.count(func.distinct(ClaimEpisode.episode_id)))
            .join(PodcastEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
            .filter(episode_filter)
            .scalar()
        )

        # Total claims
        total_claims = (
            self.session.query(func.count(Claim.id))
            .join(ClaimEpisode, Claim.id == ClaimEpisode.claim_id)
            .join(PodcastEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
            .filter(episode_filter)
            .scalar()
        )

        unprocessed = total_episodes - processed

        stats = {
            "total_episodes": total_episodes,
            "processed": processed,
            "unprocessed": unprocessed,
            "total_claims": total_claims,
        }

        logger.info(
            f"Processing stats (podcast_id={podcast_id}): "
            f"{unprocessed}/{total_episodes} episodes unprocessed, "
            f"{total_claims} total claims"
        )

        return stats

    def get_episodes_to_validate(
        self, podcast_ids: Optional[List[int]] = None, target: int = 0
    ) -> List[PodcastEpisode]:
        """
        Get episodes to validate with target-based selection.

        Mirrors get_episodes_to_process() logic but selects episodes WITH claims
        that have unverified claims (is_verified = FALSE).

        When target > 0 and podcast_ids are provided, maintains a rolling window
        of the latest `target` episodes with claims for each podcast.

        Query logic:
        1. For each podcast (when podcast_ids + target specified):
           a. Get the latest `target` episodes with transcripts (by air_date DESC)
           b. Among those, find which have unverified claims
           c. Return those for validation
        2. When target=0 or no podcast_ids: get all episodes with unverified claims
        3. All queries filter for transcripts at SQL level

        Args:
            podcast_ids: Optional list of podcast IDs to filter by (None = all podcasts)
            target: Maintain validation for the latest N episodes per podcast (0 = no limit).
                   When specified with podcast_ids, ensures the latest N episodes all
                   have validated claims. Requires podcast_ids to be specified.

        Returns:
            List of PodcastEpisode objects to validate

        Raises:
            ValueError: If target > 0 but podcast_ids is None

        Example:
            ```python
            # Validate claims from the latest 20 episodes of podcasts 1,2,3
            episodes = service.get_episodes_to_validate(
                podcast_ids=[1,2,3],
                target=20
            )

            # Get all episodes with unverified claims
            episodes = service.get_episodes_to_validate()
            ```
        """
        logger.info(
            f"Querying episodes for validation: podcast_ids={podcast_ids}, "
            f"target={target}"
        )

        # If we have podcast_ids AND a target, maintain rolling window of latest N episodes
        if podcast_ids is not None and target > 0:
            logger.debug(
                f"Using rolling window selection: validating claims from latest "
                f"{target} episodes for {len(podcast_ids)} podcast(s)"
            )
            all_episodes = []

            for podcast_id in podcast_ids:
                # Step 1: Get IDs of the latest `target` episodes with transcripts
                latest_episode_ids_query = (
                    self.session.query(PodcastEpisode.id)
                    .filter(
                        PodcastEpisode.podcast_id == podcast_id,
                        or_(
                            PodcastEpisode.podscribe_transcript.isnot(None),
                            PodcastEpisode.bankless_transcript.isnot(None),
                            PodcastEpisode.assembly_transcript.isnot(None)
                        )
                    )
                    .order_by(PodcastEpisode.air_date.desc().nulls_last())
                    .limit(target)
                )
                latest_episode_ids = [row[0] for row in latest_episode_ids_query.all()]

                if not latest_episode_ids:
                    logger.info(f"Podcast {podcast_id}: No episodes with transcripts found, skipping")
                    continue

                logger.debug(
                    f"Podcast {podcast_id}: Found {len(latest_episode_ids)} episodes "
                    f"in latest {target} window"
                )

                # Step 2: Among those latest episodes, find which have unverified claims
                episodes_with_unverified_query = (
                    self.session.query(PodcastEpisode)
                    .filter(PodcastEpisode.id.in_(latest_episode_ids))
                    .join(ClaimEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
                    .join(Claim, Claim.id == ClaimEpisode.claim_id)
                    .filter(Claim.is_verified == False)  # Has unverified claims
                    .distinct()
                    .order_by(PodcastEpisode.air_date.desc().nulls_last())
                )

                podcast_episodes = episodes_with_unverified_query.all()

                if podcast_episodes:
                    logger.info(
                        f"Podcast {podcast_id}: Validating {len(podcast_episodes)} "
                        f"episode(s) with unverified claims from latest {target} window"
                    )
                else:
                    logger.info(
                        f"Podcast {podcast_id}: All claims in latest {target} window "
                        f"are verified, skipping"
                    )

                all_episodes.extend(podcast_episodes)

            logger.info(
                f"Found {len(all_episodes)} total episode(s) to validate across "
                f"{len(podcast_ids)} podcast(s)"
            )
            return all_episodes
        else:
            # Single query for all podcasts or when no target specified

            # Error if target is specified without podcast_ids
            if target > 0 and podcast_ids is None:
                raise ValueError(
                    "target parameter requires podcast_ids to be specified. "
                    "Target-based selection works per podcast, so you must specify which podcasts."
                )

            # Base query - only episodes with transcripts AND unverified claims
            query = (
                self.session.query(PodcastEpisode)
                .filter(
                    or_(
                        PodcastEpisode.podscribe_transcript.isnot(None),
                        PodcastEpisode.bankless_transcript.isnot(None),
                        PodcastEpisode.assembly_transcript.isnot(None)
                    )
                )
                .join(ClaimEpisode, ClaimEpisode.episode_id == PodcastEpisode.id)
                .join(Claim, Claim.id == ClaimEpisode.claim_id)
                .filter(Claim.is_verified == False)  # Has unverified claims
                .distinct()
            )

            # Filter by podcast_ids if provided
            if podcast_ids is not None:
                query = query.filter(PodcastEpisode.podcast_id.in_(podcast_ids))
                logger.debug(f"Filtering by podcast_ids={podcast_ids}")

            # Order by newest first (air_date DESC), NULL dates last
            query = query.order_by(PodcastEpisode.air_date.desc().nulls_last())

            # Execute query
            episodes = query.all()

            logger.info(f"Found {len(episodes)} episodes to validate")

            return episodes

    def get_podcast_name(self, podcast_id: int) -> Optional[str]:
        """
        Get podcast name from first episode.

        Note: Podcast model doesn't exist, so we infer name from episodes.

        Args:
            podcast_id: Podcast ID

        Returns:
            Podcast name from first episode, or None if not found
        """
        episode = (
            self.session.query(PodcastEpisode)
            .filter(PodcastEpisode.podcast_id == podcast_id)
            .first()
        )

        if episode:
            # We can extract podcast name from episode metadata
            # For now, just return podcast_id as string
            return f"Podcast {podcast_id}"

        return None
