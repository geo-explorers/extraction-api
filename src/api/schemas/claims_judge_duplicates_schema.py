"""Public contract for claims.judge_duplicates.

Input: claims to check. For each, geo-lens supplies the nearest already-published claims
(vector + text strategies over its claims cache) and one Gemini call judges whether any of
them means EXACTLY the same thing. Output: per claim, the judged candidates with verdicts.

Persistence is the caller's concern (this service stores nothing).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["same", "different", "unsure"]


class ClaimIn(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Geo entity id of the claim when it already exists (excluded from its own candidates).",
    )
    text: str = Field(min_length=1, max_length=2000, description="The claim sentence.")


class ClaimsJudgeDuplicatesInput(BaseModel):
    claims: List[ClaimIn] = Field(min_length=1, max_length=50)
    k: int = Field(default=10, ge=1, le=50, description="Candidates to gather per strategy.")
    min_score: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Vector cosine floor for a candidate to be considered."
    )
    strategies: List[Literal["vector", "text"]] = Field(
        default_factory=lambda: ["vector", "text"],
        description="geo-lens strategies used to gather candidates; results are merged by id.",
    )
    cache: Optional[str] = Field(
        default=None, description="geo-lens cache handle; defaults to GEO_LENS_CLAIMS_CACHE."
    )
    space_ids: List[str] = Field(
        default_factory=list, description="Restrict candidates to claims in these Geo spaces."
    )


class Candidate(BaseModel):
    id: str
    text: str
    score: float = Field(description="Best score across the strategies that returned it (cosine for vector).")
    sources: List[str] = Field(description="Strategies that returned this candidate.")


class Match(BaseModel):
    id: str
    text: str
    score: float
    sources: List[str]
    verdict: Verdict
    rationale: str = ""


class ClaimResult(BaseModel):
    claim: ClaimIn
    candidates_considered: int
    matches: List[Match] = Field(description="Every judged candidate, best score first.")
    same: List[str] = Field(description="Ids judged to mean exactly the same thing.")
    unsure: List[str] = Field(default_factory=list)


class ClaimsJudgeDuplicatesResult(BaseModel):
    results: List[ClaimResult]
    cache: str
    model_used: str
    llm_calls: int
