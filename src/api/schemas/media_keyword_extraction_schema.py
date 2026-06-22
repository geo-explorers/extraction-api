from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class MediaKeywordExtractionRequest(BaseModel):
  media: Dict[str, Any]
  media_type: Optional[str] = None
  topics_list: List[str]
  min_keywords: int = 5
  max_keywords: int = 15
  min_topics: int = 0
  max_topics: int = 10
