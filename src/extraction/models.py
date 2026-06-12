"""Shared extraction data models used by pipelines and database repositories."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Quote:
    """
    A quote supporting a claim.

    Attributes:
        quote_text: The quote text
        relevance_score: How relevant this quote is to the claim (0.0-1.0)
        start_position: Character position in transcript
        end_position: Character position in transcript
        speaker: Speaker identifier
        timestamp_seconds: Timestamp in seconds
        entailment_score: Entailment confidence score (0.0-1.0), if validated
        entailment_relationship: SUPPORTS/RELATED/NEUTRAL/CONTRADICTS, if validated
    """

    quote_text: str
    relevance_score: float
    start_position: int
    end_position: int
    speaker: str
    timestamp_seconds: int
    entailment_score: Optional[float] = None
    entailment_relationship: Optional[str] = None


@dataclass
class ClaimWithTopic:
    claim_text: str
    topic: str
    episode_id: int
    claim_id: int = None
    claim_episode_id: int = None
    tag_id: int = None
    claim_order: int = None
    metadata: dict = field(default_factory=dict)
