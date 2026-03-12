"""Custom exceptions and error handlers for the API."""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import requests


class EpisodeNotFoundError(HTTPException):
    """Raised when an episode is not found in the database."""

    def __init__(self, episode_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Episode {episode_id} not found in database"
        )


class PodcastNotFoundError(HTTPException):
    """Raised when a podcast has no episodes."""

    def __init__(self, podcast_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No episodes found for podcast {podcast_id}"
        )


class ProcessingTimeoutError(HTTPException):
    """Raised when processing exceeds the timeout."""

    def __init__(self, timeout_seconds: int):
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Processing exceeded timeout of {timeout_seconds} seconds"
        )


class ProcessingError(HTTPException):
    """Raised when extraction pipeline fails."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {detail}"
        )


async def database_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle database errors."""
    from src.infrastructure.logger import get_logger
    logger = get_logger(__name__)
    logger.error(f"Database error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Database error occurred",
            "error": str(exc)
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc),
            "type": type(exc).__name__
        }
    )


async def requests_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle HTTP request errors to external services."""
    from src.infrastructure.logger import get_logger
    logger = get_logger(__name__)
    logger.error(f"External service error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "External service error",
            "error": str(exc)
        }
    )
