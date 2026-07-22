"""Schemas for geo.resolve_entities — policy-driven name→entity resolution.

The caller supplies named items, each referencing a POLICY by key. A policy
carries the Geo type id the match must have, an optional tag-entity id the
match must be tagged with (e.g. a "curated" marker), and an optional match
threshold override. Policy keys are opaque caller-chosen labels (like facet
keys in claims.link_entities) — the service attaches no meaning to them.

Read-only and deliberately conservative: the service returns `ambiguous` with
candidates instead of guessing, and never creates anything. Domain-agnostic:
which types, which tags, and what the items mean is entirely the caller's.
"""

import re
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_ITEMS = 200

_POLICY_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_UUID32_RE = re.compile(r"^[0-9a-f]{32}$")


class ResolveItem(BaseModel):
    """One name to resolve. `index` is the caller's stable handle."""

    index: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=300)
    policy: str = Field(min_length=1, max_length=64)


class ResolvePolicy(BaseModel):
    """Constraints a Geo entity must satisfy to match items under this policy."""

    # Geo type id the entity must carry (checked client-side against the
    # entity's types — server-side type filters are unreliable, verified 2026-07-22).
    type_id: str
    # Entity id of a required tag (matched via a Tags relation, e.g. a curated
    # marker). None = no tag restriction.
    restrict_to_tag: Optional[str] = None
    # Per-policy override of the global match threshold (0..1).
    match_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Duplicate-selection strategy when several candidates qualify:
    # team_priority (default) applies the team-wide canonical-selection cascade;
    # strict never chooses and returns `ambiguous`.
    selection: Literal["team_priority", "strict"] = "team_priority"

    @field_validator("type_id", "restrict_to_tag")
    @classmethod
    def _uuid32(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _UUID32_RE.match(v):
            raise ValueError("must be a 32-char lowercase hex Geo id")
        return v


class GeoResolveRequest(BaseModel):
    items: List[ResolveItem] = Field(min_length=1, max_length=MAX_ITEMS)
    policies: Dict[str, ResolvePolicy] = Field(min_length=1)

    @field_validator("policies")
    @classmethod
    def _slug_keys(cls, v: Dict[str, ResolvePolicy]) -> Dict[str, ResolvePolicy]:
        for k in v:
            if not _POLICY_KEY_RE.match(k):
                raise ValueError(f"policy key '{k}' must be a slug: [a-z][a-z0-9_]*")
        return v

    @model_validator(mode="after")
    def _items_reference_known_policies(self) -> "GeoResolveRequest":
        unknown = {i.policy for i in self.items} - set(self.policies)
        if unknown:
            raise ValueError(f"items reference unknown policies: {sorted(unknown)}")
        return self


class ResolveCandidate(BaseModel):
    geo_id: str
    name: str
    score: float
    space_ids: List[str] = Field(default_factory=list)


class Resolution(BaseModel):
    index: int
    status: Literal["resolved", "not_found", "ambiguous", "error"]
    geo_id: Optional[str] = None
    matched_name: Optional[str] = None
    score: Optional[float] = None
    # Populated for `ambiguous` (top candidates, best first) so the caller can
    # decide; empty otherwise.
    candidates: List[ResolveCandidate] = Field(default_factory=list)
    error: Optional[str] = None


class GeoResolveResponse(BaseModel):
    # Exactly one entry per request item, in request order.
    resolutions: List[Resolution] = Field(default_factory=list)
