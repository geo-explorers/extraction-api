"""Public contract for claims.judge_equivalence.

A pure judge: ONE claim and a group of CANDIDATE claims in; the candidates that are logically
equivalent to the claim out. "Equivalent" means the two sentences have the same truth
conditions — not similar, not very close, not one entailing the other.

Retrieval is deliberately NOT part of this task. Whoever needs "the published claims that mean
exactly this" first asks geo-lens (``POST /caches/{handle}/query``, vector and/or text strategy)
for candidates, then sends claim + candidates here. That keeps geo-lens and extraction-api
loosely coupled: either can change without the other knowing.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["equivalent", "not_equivalent", "unsure"]


class ClaimText(BaseModel):
    id: Optional[str] = Field(default=None, description="Caller-opaque id (e.g. the Geo entity id); echoed back.")
    text: str = Field(min_length=1, max_length=2000)


class ClaimsJudgeEquivalenceInput(BaseModel):
    claim: ClaimText
    candidates: List[ClaimText] = Field(min_length=1, max_length=50)


class JudgedCandidate(BaseModel):
    index: int = Field(description="Position in the input `candidates` list.")
    id: Optional[str]
    text: str
    verdict: Verdict
    rationale: str = Field(description="One sentence: the decisive difference, or why they are equivalent.")


class ClaimsJudgeEquivalenceResult(BaseModel):
    claim: ClaimText
    equivalent: List[JudgedCandidate] = Field(description="Candidates judged logically equivalent to the claim.")
    unsure: List[JudgedCandidate] = Field(default_factory=list)
    judged: List[JudgedCandidate] = Field(description="Every candidate with its verdict, in input order.")
    model_used: str
