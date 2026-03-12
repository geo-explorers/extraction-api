"""
Claim deduplicator using embedding similarity and reranker verification.

Deduplicates claims within single episode (batch level) using multi-method
approach: embedding similarity for filtering + reranker for verification.

Usage:
    from src.deduplication.claim_deduplicator import ClaimDeduplicator

    deduplicator = ClaimDeduplicator(embedder, reranker)
    deduplicated = await deduplicator.deduplicate_batch(claims_with_quotes)

    print(f"Deduplicated: {len(claims_with_quotes)} → {len(deduplicated)}")
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session

from src.extraction.quote_finder import ClaimWithQuotes
from src.infrastructure.embedding_service import EmbeddingService
from src.infrastructure.reranker_service import RerankerService
from src.deduplication.quote_deduplicator import QuoteDeduplicator
from src.database.models import Claim, ClaimEpisode
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseDeduplicationResult:
    """
    Result from database-level deduplication check.

    Attributes:
        is_duplicate: Whether claim is duplicate of existing claim
        existing_claim_id: ID of existing claim if duplicate
        existing_claim_text: Text of existing claim if duplicate
        reranker_score: Reranker verification score
        should_merge_quotes: Whether to add new quotes to existing claim
    """
    is_duplicate: bool
    existing_claim_id: Optional[int] = None
    existing_claim_text: Optional[str] = None
    reranker_score: Optional[float] = None
    should_merge_quotes: bool = False


class ClaimDeduplicator:
    """
    Deduplicate claims within episode using embedding similarity and reranker.

    Features:
    - Group similar claims using embedding similarity (>0.85 threshold)
    - Verify duplicates using reranker (>0.9 score = duplicate)
    - Merge duplicate groups intelligently
    - Deduplicate quotes within merged claims
    - Track merge metadata

    Example:
        ```python
        deduplicator = ClaimDeduplicator(embedder, reranker)

        claims = [
            ClaimWithQuotes("Bitcoin reached $69,000", ...),
            ClaimWithQuotes("BTC hit $69k in November 2021", ...),
            ClaimWithQuotes("Bitcoin's price peaked at sixty-nine thousand", ...),
        ]

        deduplicated = await deduplicator.deduplicate_batch(claims)
        # Result: 1 merged claim with combined quotes
        ```
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        reranker: RerankerService,
        embedding_threshold: float = None,
        reranker_threshold: float = None
    ):
        """
        Initialize the claim deduplicator.

        Args:
            embedder: Embedding service for similarity calculation
            reranker: Reranker service for verification
            embedding_threshold: Cosine similarity threshold (default from settings)
            reranker_threshold: Reranker score threshold (default from settings)
        """
        self.embedder = embedder
        self.reranker = reranker
        self.quote_deduplicator = QuoteDeduplicator()

        self.embedding_threshold = embedding_threshold or settings.embedding_similarity_threshold
        self.reranker_threshold = reranker_threshold or settings.reranker_verification_threshold

        logger.info(
            f"Initialized ClaimDeduplicator: "
            f"embedding_threshold={self.embedding_threshold}, "
            f"reranker_threshold={self.reranker_threshold}"
        )

    async def deduplicate_batch(
        self,
        claims_with_quotes: List[ClaimWithQuotes]
    ) -> List[ClaimWithQuotes]:
        """
        Deduplicate claims within episode.

        Algorithm:
        1. Generate embeddings for all claims
        2. Find candidate pairs (cosine similarity > 0.85)
        3. Verify with reranker (batch scoring, threshold > 0.9)
        4. Build groups from verified pairs (union-find)
        5. Merge each group

        Args:
            claims_with_quotes: List of claims with quotes

        Returns:
            List of deduplicated claims

        Example:
            ```python
            deduplicator = ClaimDeduplicator(embedder, reranker)

            claims = [...]  # 15 claims
            deduplicated = await deduplicator.deduplicate_batch(claims)  # 12 claims

            merged_claims = [c for c in deduplicated if c.metadata.get('merged_from_claims')]
            print(f"Merged {len(merged_claims)} claim groups")
            ```
        """
        if len(claims_with_quotes) <= 1:
            logger.info("Only 1 claim, skipping deduplication")
            return claims_with_quotes

        logger.info(f"Deduplicating {len(claims_with_quotes)} claims...")

        claim_texts = [c.claim_text for c in claims_with_quotes]
        embeddings = await self.embedder.embed_texts(claim_texts)

        candidate_pairs = self._find_similar_pairs(embeddings)

        logger.debug(f"Found {len(candidate_pairs)} candidate pairs")

        if not candidate_pairs:
            logger.info("No candidate pairs found, no deduplication needed")
            return claims_with_quotes

        verified_pairs = await self._verify_with_reranker(candidate_pairs, claim_texts)

        logger.debug(f"Verified {len(verified_pairs)} duplicate pairs with reranker")

        if not verified_pairs:
            logger.info("No duplicates verified by reranker")
            return claims_with_quotes

        groups = self._build_duplicate_groups(verified_pairs, len(claims_with_quotes))

        deduplicated = []
        for group in groups:
            if len(group) == 1:
                deduplicated.append(claims_with_quotes[group[0]])
            else:
                group_claims = [claims_with_quotes[i] for i in group]
                merged = self._merge_claim_group(group_claims)
                deduplicated.append(merged)

        logger.info(
            f"Deduplicated {len(claims_with_quotes)} → {len(deduplicated)} claims "
            f"({len(claims_with_quotes) - len(deduplicated)} duplicates removed, "
            f"{100 * (len(claims_with_quotes) - len(deduplicated)) / len(claims_with_quotes):.1f}% reduction)"
        )

        return deduplicated

    def _find_similar_pairs(
        self,
        embeddings: List[List[float]]
    ) -> List[Tuple[int, int]]:
        """
        Find pairs with cosine similarity above threshold.

        Args:
            embeddings: List of claim embeddings

        Returns:
            List of (index1, index2) pairs
        """
        pairs = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self.embedder.cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= self.embedding_threshold:
                    pairs.append((i, j))
                    logger.debug(
                        f"Candidate pair: claims {i} and {j} "
                        f"(similarity: {similarity:.3f})"
                    )

        return pairs

    async def _verify_with_reranker(
        self,
        candidate_pairs: List[Tuple[int, int]],
        claim_texts: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Verify candidate pairs using reranker.

        Args:
            candidate_pairs: List of (index1, index2) candidate pairs
            claim_texts: List of claim texts

        Returns:
            List of verified duplicate pairs
        """
        verified = []

        for i, j in candidate_pairs:
            results = await self.reranker.rerank_quotes(
                claim_texts[i],
                [claim_texts[j]],
                top_k=1
            )

            if results and results[0]["score"] >= self.reranker_threshold:
                verified.append((i, j))
                logger.debug(
                    f"Verified duplicate: claims {i} and {j} "
                    f"(reranker score: {results[0]['score']:.3f})"
                )
            else:
                score = results[0]["score"] if results else 0.0
                logger.debug(
                    f"Not duplicate: claims {i} and {j} "
                    f"(reranker score: {score:.3f} < {self.reranker_threshold})"
                )

        return verified

    def _build_duplicate_groups(
        self,
        verified_pairs: List[Tuple[int, int]],
        num_claims: int
    ) -> List[List[int]]:
        """
        Build groups from pairwise duplicate relationships using union-find.

        Args:
            verified_pairs: List of verified duplicate pairs
            num_claims: Total number of claims

        Returns:
            List of groups (each group is list of claim indices)

        Example:
            ```python
            pairs = [(0, 1), (1, 2), (3, 4)]
            groups = self._build_duplicate_groups(pairs, 5)
            # Result: [[0, 1, 2], [3, 4]]
            # Claims 0, 1, 2 form one group, claims 3, 4 form another
            ```
        """
        parent = list(range(num_claims))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i, j in verified_pairs:
            union(i, j)

        groups_dict = {}
        for i in range(num_claims):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(i)

        groups = list(groups_dict.values())

        logger.debug(
            f"Built {len(groups)} groups from {len(verified_pairs)} pairs: "
            f"{[len(g) for g in groups]}"
        )

        return groups

    def _merge_claim_group(
        self,
        group: List[ClaimWithQuotes]
    ) -> ClaimWithQuotes:
        """
        Merge duplicate claims.

        Merge strategy:
        - Keep claim text with highest confidence (and longest if tied)
        - Combine all quotes from all claims
        - Deduplicate combined quotes
        - Track merge metadata

        Args:
            group: List of duplicate claims to merge

        Returns:
            Merged claim with combined quotes

        Example:
            ```python
            group = [
                ClaimWithQuotes("Bitcoin reached $69,000", confidence=0.85, quotes=[...]),
                ClaimWithQuotes("BTC hit $69k", confidence=0.78, quotes=[...]),
            ]

            merged = self._merge_claim_group(group)
            # Result: claim text from first (higher confidence),
            #         combined and deduplicated quotes from both
            ```
        """
        # Use confidence as primary key, claim length as tiebreaker (longer = more specific)
        best_claim = max(group, key=lambda c: (c.confidence, len(c.claim_text)))

        all_quotes = []
        for claim in group:
            all_quotes.extend(claim.quotes)

        # Collect all source chunk IDs from merged claims
        merged_chunk_ids = []
        for claim in group:
            # Add this claim's source chunk
            if claim.source_chunk_id:
                merged_chunk_ids.append(claim.source_chunk_id)
            # Add any previously merged chunk IDs
            if claim.merged_from_chunk_ids:
                merged_chunk_ids.extend(claim.merged_from_chunk_ids)

        # Deduplicate chunk IDs
        merged_chunk_ids = list(set(merged_chunk_ids))

        logger.debug(f"Merging {len(group)} claims with {len(all_quotes)} total quotes...")

        unique_quotes = self.quote_deduplicator.deduplicate(all_quotes)

        logger.debug(f"Deduplicated quotes: {len(all_quotes)} → {len(unique_quotes)}")

        merged = ClaimWithQuotes(
            claim_text=best_claim.claim_text,
            source_chunk_id=best_claim.source_chunk_id,
            quotes=unique_quotes,
            confidence=best_claim.confidence,
            merged_from_chunk_ids=merged_chunk_ids,  # Track all source chunks
            metadata={
                "merged_from_claims": len(group),
                "original_quote_count": len(all_quotes),
                "deduplicated_quote_count": len(unique_quotes),
                "merged_from_chunk_ids_count": len(merged_chunk_ids)
            }
        )

        logger.debug(
            f"Merged {len(group)} claims: "
            f"'{best_claim.claim_text[:60]}...' "
            f"({len(all_quotes)} quotes → {len(unique_quotes)} unique)"
        )

        return merged

    async def deduplicate_against_database(
        self,
        claim_text: str,
        claim_embedding: List[float],
        episode_id: int,
        db_session: Session
    ) -> DatabaseDeduplicationResult:
        """
        Check if claim is duplicate of existing claims in database.

        Uses pgvector similarity search + reranker verification.

        Algorithm:
        1. pgvector search: Find top 10 similar claims (L2 distance < 0.15)
        2. Exclude claims from current episode (already handled by batch dedup)
        3. Reranker verify: Check each candidate (score > 0.9 = duplicate)
        4. If duplicate found: return info for quote merging
        5. If unique: return not duplicate

        Args:
            claim_text: The claim text to check
            claim_embedding: The claim's embedding vector (768 dims)
            episode_id: Current episode ID (to exclude from search)
            db_session: Database session for querying

        Returns:
            DatabaseDeduplicationResult with duplicate info

        Example:
            ```python
            deduplicator = ClaimDeduplicator(embedder, reranker)

            # Check against database
            result = await deduplicator.deduplicate_against_database(
                claim_text="Bitcoin reached $69,000",
                claim_embedding=embedding_vector,
                episode_id=123,
                db_session=session
            )

            if result.is_duplicate:
                print(f"Duplicate of claim {result.existing_claim_id}")
                print(f"Reranker score: {result.reranker_score:.3f}")
                print(f"Should merge quotes: {result.should_merge_quotes}")
            else:
                print("Unique claim, will insert as new")
            ```
        """
        logger.info(f"Checking database for duplicates of: {claim_text[:60]}...")

        # 1. pgvector similarity search
        # L2 distance < 0.15 ≈ cosine similarity > 0.85
        try:
            # Get claim IDs linked to current episode (to exclude them)
            current_episode_claim_ids_result = (
                db_session.query(ClaimEpisode.claim_id)
                .filter(ClaimEpisode.episode_id == episode_id)
                .all()
            )
            current_episode_claim_ids = [row[0] for row in current_episode_claim_ids_result]

            # Build query to find similar claims
            query = db_session.query(Claim).filter(
                Claim.embedding.l2_distance(claim_embedding) < settings.vector_distance_threshold
            )

            # Exclude current episode's claims if there are any
            if current_episode_claim_ids:
                query = query.filter(~Claim.id.in_(current_episode_claim_ids))

            similar_claims = (
                query
                .order_by(Claim.embedding.l2_distance(claim_embedding))
                .limit(10)
                .all()
            )

            if not similar_claims:
                logger.info("No similar claims found in database (unique claim)")
                return DatabaseDeduplicationResult(is_duplicate=False)

            logger.info(
                f"Found {len(similar_claims)} similar claims in database, "
                f"verifying with reranker..."
            )

            # 2. Reranker verification
            for i, existing_claim in enumerate(similar_claims, 1):
                # Verify with reranker
                reranked = await self.reranker.rerank_quotes(
                    claim_text,
                    [existing_claim.claim_text],
                    top_k=1
                )

                if not reranked:
                    logger.warning(
                        f"Reranker returned no results for claim {existing_claim.id}"
                    )
                    continue

                reranker_score = reranked[0]["score"]

                logger.debug(
                    f"Candidate {i}/{len(similar_claims)}: "
                    f"claim_id={existing_claim.id}, "
                    f"reranker_score={reranker_score:.3f}"
                )

                # Check if duplicate
                if reranker_score >= self.reranker_threshold:
                    logger.info(
                        f"✅ Duplicate found! Existing claim ID: {existing_claim.id}, "
                        f"reranker score: {reranker_score:.3f}"
                    )

                    return DatabaseDeduplicationResult(
                        is_duplicate=True,
                        existing_claim_id=existing_claim.id,
                        existing_claim_text=existing_claim.claim_text,
                        reranker_score=reranker_score,
                        should_merge_quotes=True  # Always merge quotes for duplicates
                    )
                else:
                    logger.debug(
                        f"Not duplicate: reranker score {reranker_score:.3f} "
                        f"< threshold {self.reranker_threshold}"
                    )

            # No duplicates found
            logger.info("No duplicates verified by reranker (unique claim)")
            return DatabaseDeduplicationResult(is_duplicate=False)

        except Exception as e:
            logger.error(
                f"Error in database deduplication: {e}",
                exc_info=True
            )
            # On error, assume unique to avoid blocking processing
            return DatabaseDeduplicationResult(is_duplicate=False)
