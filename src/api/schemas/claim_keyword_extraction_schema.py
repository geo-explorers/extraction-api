from pydantic import BaseModel
from typing import List


class ClaimInput(BaseModel):
    id: str
    text: str


class ClaimKeywordExtractionRequest(BaseModel):
    claims: List[ClaimInput]
    title: str
    description: str
    topics_list: List[str]
    min_keywords: int = 0
    max_keywords: int = 10
    min_topics: int = 0
    max_topics: int = 10
