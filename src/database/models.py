"""
SQLAlchemy models for the claim-quote extraction pipeline.

Models match the PostgreSQL schema with pgvector support for embeddings.
All tables use the 'crypto' schema.

Models:
    - PodcastEpisode: Podcast episode with transcript
    - TranscriptChunk: Transcript chunk with position tracking
    - Claim: Extracted factual claim with embedding
    - Quote: Text excerpt from transcript
    - ClaimQuote: Many-to-many relationship between claims and quotes
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class PodcastEpisode(Base):
    """
    Podcast episode with transcript.

    Table: crypto.podcast_episodes
    """

    __tablename__ = "podcast_episodes"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    podcast_id = Column(BigInteger, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    episode_number = Column(Integer)
    duration = Column(Integer)
    air_date = Column(Date)
    avatar = Column(Text)
    season = Column(Text)
    episode_type = Column(Text)
    audio_url = Column(Text)
    podscribe_transcript = Column(Text)
    bankless_transcript = Column(Text)
    assembly_transcript = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    transcript_chunks = relationship(
        "TranscriptChunk", back_populates="episode", cascade="all, delete-orphan"
    )
    claim_episodes = relationship(
        "ClaimEpisode", back_populates="episode", cascade="all, delete-orphan"
    )
    quotes = relationship(
        "Quote", back_populates="episode", cascade="all, delete-orphan"
    )


    def __repr__(self) -> str:
        return f"<PodcastEpisode(id={self.id}, name='{self.name[:50]}...')>"


class TranscriptChunk(Base):
    """
    Transcript chunk with position tracking.

    Stores chunks of transcript text that were used for claim extraction.
    This allows tracking which chunk each claim came from for training data
    generation and provenance tracking.

    Table: crypto.transcript_chunks
    """

    __tablename__ = "transcript_chunks"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(
        BigInteger,
        ForeignKey("crypto.podcast_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    start_position = Column(Integer, nullable=False)
    end_position = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    episode = relationship("PodcastEpisode", back_populates="transcript_chunks")
    claims = relationship(
        "Claim", back_populates="source_chunk", foreign_keys="Claim.source_chunk_id"
    )

    def __repr__(self) -> str:
        return f"<TranscriptChunk(id={self.id}, episode_id={self.episode_id}, chunk_index={self.chunk_index}, pos={self.start_position}-{self.end_position})>"


class Claim(Base):
    """
    Extracted factual claim with embedding and confidence score.

    Table: crypto.claims
    """

    __tablename__ = "claims"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_chunk_id = Column(
        BigInteger,
        ForeignKey("crypto.transcript_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    merged_from_chunk_ids = Column(ARRAY(BigInteger), nullable=True)
    claim_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    embedding = Column(Vector(768))  # pgvector for similarity search
    confidence_components = Column(JSONB)
    reranker_scores = Column(JSONB)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_flagged = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_chunk = relationship(
        "TranscriptChunk", back_populates="claims", foreign_keys=[source_chunk_id]
    )
    claim_quotes = relationship(
        "ClaimQuote", back_populates="claim", cascade="all, delete-orphan"
    )
    claim_episodes = relationship(
        "ClaimEpisode", back_populates="claim", cascade="all, delete-orphan"
    )


    def __repr__(self) -> str:
        return f"<Claim(id={self.id}, text='{self.claim_text[:50]}...', confidence={self.confidence:.2f})>"

class ClaimEpisode(Base):
    """
    Many-to-many relationship between claims and episodes.

    Table: crypto.claim_episodes
    """

    __tablename__ = "claim_episodes"
    __table_args__ = (
        UniqueConstraint("claim_id", "episode_id", name="uq_claim_episodes_claim_episode"),
        {"schema": "crypto"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    claim_id = Column(
        BigInteger,
        ForeignKey("crypto.claims.id", ondelete="CASCADE"),
        nullable=False,
    )
    episode_id = Column(
        BigInteger,
        ForeignKey("crypto.podcast_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    claim = relationship("Claim", back_populates="claim_episodes")
    episode = relationship("PodcastEpisode", back_populates="claim_episodes")
    tag_maps = relationship(
        "TagMap", back_populates="claim_episode", cascade="all, delete-orphan"
    )

class Tag(Base):
    """
    Table: crypto.tags
    """

    __tablename__ = "tags"
    __table_args__ = {"schema": "crypto"}
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TagMap(Base):
    """
    Table: crypto.tag_map
    """
    __tablename__ = "tag_map"
    __table_args__ = (
        UniqueConstraint(
            "from_claim_episode_id",
            "tag_category",
            name="uq_tag_map_claim_episode_category",
        ),
        {"schema": "crypto"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tag_category = Column(Text, nullable=False)
    from_claim_episode_id = Column(
        BigInteger,
        ForeignKey("crypto.claim_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    to_tag_id = Column(
        BigInteger,
        ForeignKey("crypto.tags.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Relationships
    claim_episode = relationship("ClaimEpisode", back_populates="tag_maps")
    tag = relationship("Tag")


class Quote(Base):
    """
    Text excerpt from transcript with position tracking.

    Table: crypto.quotes
    """

    __tablename__ = "quotes"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(
        BigInteger,
        ForeignKey("crypto.podcast_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    quote_text = Column(Text, nullable=False)
    start_position = Column(Integer)
    end_position = Column(Integer)
    speaker = Column(String(255))
    timestamp_seconds = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    episode = relationship("PodcastEpisode", back_populates="quotes")
    claim_quotes = relationship(
        "ClaimQuote", back_populates="quote", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Quote(id={self.id}, text='{self.quote_text[:50]}...', speaker='{self.speaker}')>"


class ClaimQuote(Base):
    """
    Many-to-many relationship between claims and quotes.

    Stores relevance scores and entailment validation results.

    Table: crypto.claim_quotes
    """

    __tablename__ = "claim_quotes"
    __table_args__ = {"schema": "crypto"}

    claim_id = Column(
        BigInteger,
        ForeignKey("crypto.claims.id", ondelete="CASCADE"),
        primary_key=True,
    )
    quote_id = Column(
        BigInteger,
        ForeignKey("crypto.quotes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    relevance_score = Column(Float, nullable=False)
    match_confidence = Column(Float, nullable=False)
    match_type = Column(String(20))
    entailment_score = Column(Float)
    entailment_relationship = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    claim = relationship("Claim", back_populates="claim_quotes")
    quote = relationship("Quote", back_populates="claim_quotes")

    def __repr__(self) -> str:
        return (
            f"<ClaimQuote(claim_id={self.claim_id}, quote_id={self.quote_id}, "
            f"relevance={self.relevance_score:.2f}, entailment='{self.entailment_relationship}')>"
        )
