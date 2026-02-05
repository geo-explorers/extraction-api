from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.api.schemas.standalone_claim_keywords_schema import (
    StandaloneClaimKeywordsRequest,
)
from src.api.services.standalone_claim_keywords_service import (
    extract_standalone_claim_keywords,
    TooManyClaimsError,
)

router = APIRouter()


@router.post("/standalone-claim-keywords")
def standalone_claim_keywords_extraction(
    request: StandaloneClaimKeywordsRequest,
) -> JSONResponse:
    claims_data = [{"id": c.id, "text": c.text} for c in request.claims]

    try:
        claim_keywords = extract_standalone_claim_keywords(
            claims=claims_data,
            min_keywords=request.min_keywords,
            max_keywords=request.max_keywords,
        )
        return JSONResponse(
            content={
                "claim_keywords": claim_keywords,
                "error": None,
            },
            status_code=status.HTTP_200_OK,
        )
    except TooManyClaimsError as e:
        return JSONResponse(
            content={
                "claim_keywords": None,
                "error": str(e),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except Exception:
        return JSONResponse(
            content={
                "claim_keywords": None,
                "error": "An internal error occurred. Please try again later.",
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
