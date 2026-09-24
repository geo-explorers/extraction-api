from pydantic import BaseModel
from src.api.schemas.overrides_schema import OverridesMixin

class GuestExtractionRequest(OverridesMixin):
  title: str
  description: str
  truncated_transcript: str = ""
