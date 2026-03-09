from typing import List, Optional
from pydantic import BaseModel

class HostExtractionRequest(BaseModel):
  title: str
  description: str
  truncated_transcript: str
  possible_hosts: Optional[List[str]] = None
