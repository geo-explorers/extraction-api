from typing import List, Dict
from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    id: str
    text: str


class StandaloneClaimKeywordsRequest(BaseModel):
    claims: List[ClaimInput] = Field(..., min_length=1)
    min_keywords: int = Field(default=1, ge=1, le=10)
    max_keywords: int = Field(default=5, ge=1, le=10)


class StandaloneClaimKeywordsResponse(BaseModel):
    claim_keywords: Dict[str, List[str]] | None
    error: str | None = None
