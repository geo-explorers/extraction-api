from typing import List, Optional
from pydantic import BaseModel
from src.api.schemas.overrides_schema import OverridesMixin

class HostExtractionRequest(OverridesMixin):
  title: str
  description: str
  truncated_transcript: str
  possible_hosts: Optional[List[str]] = None
