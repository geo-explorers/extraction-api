"""Schemas for claims.link_entities — facet-parameterized claim annotation.

The caller supplies claims plus one or more FACETS: named, numbered
vocabularies of candidate entities (e.g. a "topics" facet holding curated
topic names, a "people" facet holding known person names — but the task
attaches no meaning to keys). The task returns, per claim and per facet, the
vocabulary INDICES the claim is substantively involved with (subject,
actor, affected party, or attributed source — see the prompt's standard).

Design properties:
- Index-selection makes restrictions structural: the model cannot introduce an
  off-vocabulary entity, so policies like "curated only" / "existing only" are
  enforced by what the caller puts in each facet's vocabulary.
- Facets are DATA: a new vocabulary-style annotation field costs the caller a
  new facet config and costs this service nothing. Annotation families with a
  different logic shape (claim<->claim stance, span alignment, classification)
  belong in sibling tasks, not new facets here.
- Stateless and domain-agnostic: no geo ids in or out, no assumptions about
  what the claims are about or which app produced them.
"""

import re
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from src.config.settings import settings

MAX_CLAIMS = 200
MAX_FACETS = 8
# Contract cap per facet vocabulary — env-configurable
# (CLAIMS_LINK_MAX_VOCABULARY), read once at import when the class is built.
MAX_VOCABULARY_ITEMS = settings.claims_link_max_vocabulary

_FACET_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class LinkClaim(BaseModel):
    """One claim to annotate. `index` is the caller's stable handle."""

    index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=2000)


class VocabularyItem(BaseModel):
    """One candidate entity name. `index` is the caller's stable handle."""

    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=300)


class Facet(BaseModel):
    """One named vocabulary to annotate claims against."""

    # Caller-chosen identifier, echoed back in the response (slug: a-z0-9_).
    key: str = Field(min_length=1, max_length=64)
    vocabulary: List[VocabularyItem] = Field(
        default_factory=list, max_length=MAX_VOCABULARY_ITEMS
    )
    max_per_claim: int = Field(default=3, ge=1, le=20)
    # Optional one-line selection criterion for this facet, rendered verbatim
    # into the prompt (e.g. "people who are the subject of or actor in the
    # claim"). Empty uses the generic "substantively involved" bar; a facet
    # criterion OVERRIDES that default for its facet.
    criteria: str = Field(default="", max_length=300)

    @field_validator("key")
    @classmethod
    def _slug_key(cls, v: str) -> str:
        if not _FACET_KEY_RE.match(v):
            raise ValueError("facet key must be a slug: [a-z][a-z0-9_]*")
        return v


class ClaimsLinkEntitiesRequest(BaseModel):
    claims: List[LinkClaim] = Field(min_length=1, max_length=MAX_CLAIMS)
    facets: List[Facet] = Field(min_length=1, max_length=MAX_FACETS)
    # Optional framing for the model (a title or one-line description of the
    # source material) — grounding only, never a source of extra vocabulary.
    context: str = Field(default="", max_length=2000)

    @field_validator("facets")
    @classmethod
    def _unique_keys(cls, v: List[Facet]) -> List[Facet]:
        keys = [f.key for f in v]
        if len(keys) != len(set(keys)):
            raise ValueError("facet keys must be unique")
        return v


class ClaimLink(BaseModel):
    """Annotation for one claim: per facet key, indices into that facet's vocabulary."""

    claim_index: int = Field(ge=0)
    selections: Dict[str, List[int]] = Field(default_factory=dict)


class ClaimsLinkEntitiesResponse(BaseModel):
    # Exactly one entry per request claim, in request order; every entry carries
    # every requested facet key (validator-enforced; empty lists — a claim about
    # none of a facet's vocabulary — are the common case).
    links: List[ClaimLink] = Field(default_factory=list)
    model_used: str = ""
