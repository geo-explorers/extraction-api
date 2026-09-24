from pydantic import BaseModel
from src.api.schemas.overrides_schema import OverridesMixin
from typing import Dict, Any, List

class KeywordExtractionRequest(OverridesMixin):
  episode: Dict[str, Any]
  topics_list: List[str]
  min_keywords: int = 5
  max_keywords: int = 15
  min_topics: int = 0
  max_topics: int = 10
