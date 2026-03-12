from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.services.guest_extraction_service import extract_podcast_guests
from src.api.schemas.guest_extraction_schema import (
  GuestExtractionRequest,
)

router = APIRouter()

@router.post("/guests")
def guest_extraction(request: GuestExtractionRequest) -> JSONResponse:
  try:
    guests = extract_podcast_guests(
      title=request.title,
      description=request.description,
      truncated_transcript=request.truncated_transcript,
    )
    return JSONResponse(
      content={
        "error":None,
        "guests":guests
      }, 
      status_code=status.HTTP_200_OK
    )
  except Exception as e:
    return JSONResponse(
      content={
        "error":"An internal error occurred. Please try again later.",
        "guests":None
      }, 
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )
  

  