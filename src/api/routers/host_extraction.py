from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.services.host_extraction_service import extract_podcast_hosts
from src.api.schemas.host_extraction_schema import (
  HostExtractionRequest,
)

router = APIRouter()

@router.post("/hosts")
def host_extraction(request: HostExtractionRequest) -> JSONResponse:
  try:
    hosts = extract_podcast_hosts(
      title=request.title,
      description=request.description,
      truncated_transcript=request.truncated_transcript,
      possible_hosts=request.possible_hosts,
    )
    return JSONResponse(
      content={
        "error":None,
        "hosts":hosts
      },
      status_code=status.HTTP_200_OK
    )
  except Exception as e:
    return JSONResponse(
      content={
        "error":"An internal error occurred. Please try again later.",
        "hosts":None
      },
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
