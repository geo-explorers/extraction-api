"""Main FastAPI application."""

import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import requests

from src.api.routers import extraction, guest_extraction, host_extraction, keyword_extraction, media_keyword_extraction, validation, claim_keyword_extraction
from src.api.exceptions import (
    database_exception_handler,
    generic_exception_handler,
    requests_exception_handler,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


# Signal handlers for graceful shutdown on Windows
def handle_shutdown_signal(signum, frame):
    """Handle shutdown signals (Ctrl+C) gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    # Close DSPy connections immediately
    try:
        from src.config.dspy_config import shutdown_dspy_configuration
        shutdown_dspy_configuration()
    except Exception as e:
        logger.error(f"Error closing DSPy connections: {e}")
    sys.exit(0)


# Register signal handlers for both SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, handle_shutdown_signal)
signal.signal(signal.SIGTERM, handle_shutdown_signal)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("=" * 80)
    logger.info("Podcast Extraction API Starting")
    logger.info("=" * 80)
    logger.info(f"Host: {settings.api_host}:{settings.port}")
    logger.info(f"Timeout: {settings.api_timeout}s (0 = no timeout)")
    logger.info(f"Database: {settings.database_url}")
    logger.info(f"Ollama: {settings.ollama_url}")
    logger.info(f"Ollama Embedding: {settings.ollama_embedding_url}")
    logger.info(
        f"Reranker: {settings.reranker_url} (enabled={settings.enable_reranker})"
    )
    logger.info(f"Quote Processing: {settings.enable_quote_processing}")
    logger.info("=" * 80)

    # Validate DSPy configuration (fail-fast if Ollama unreachable)
    logger.info("Validating DSPy configuration...")
    try:
        from src.config.dspy_startup import validate_dspy_configuration
        validate_dspy_configuration()
        logger.info("DSPy validation complete - models will initialize lazily on first request")
    except Exception as e:
        logger.error(f"DSPy validation failed: {e}", exc_info=True)
        logger.critical("API cannot start without valid DSPy configuration")
        raise

    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Podcast Extraction API shutting down")

    # Close DSPy connections to prevent hanging on shutdown
    try:
        from src.config.dspy_config import shutdown_dspy_configuration
        shutdown_dspy_configuration()
    except Exception as e:
        logger.error(f"Error during DSPy shutdown: {e}", exc_info=True)

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Podcast Extraction API",
    description="""
    API for extracting claims, guests, and keywords from podcast transcripts.

    ## Features
    - **Claim Extraction**: Extract claims and supporting quotes from podcast transcripts using DSPy
    - **Guest Extraction**: Extract podcast guest names and URLs using Gemini
    - **Keyword Extraction**: Extract keywords and topics from episode data using Gemini
    - Batch processing across multiple podcasts
    - Automatic quote finding and validation
    - Confidence scoring and filtering

    ## Processing
    - Claims are extracted using DSPy LLM models (Ollama)
    - Guests and keywords are extracted using Google Gemini

    ## Authentication
    All endpoints require API key authentication via X-API-Key header.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS (needed for Swagger UI and external API access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production with API key auth)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Key Authentication Middleware
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify API key for all endpoints except docs."""
    # Allow access to documentation without authentication
    public_paths = ["/", "/docs", "/redoc", "/openapi.json"]
    if request.url.path in public_paths:
        return await call_next(request)

    # Check for API key in header
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Missing API key. Include 'X-API-Key' header in your request."
            }
        )

    # Validate API key
    if api_key != settings.api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Invalid API key"
            }
        )

    # API key is valid, proceed with request
    return await call_next(request)


# Register routers
app.include_router(extraction.router, tags=["claims"])
app.include_router(guest_extraction.router, prefix="/extract", tags=["guests"])
app.include_router(host_extraction.router, prefix="/extract", tags=["hosts"])
app.include_router(keyword_extraction.router, prefix="/extract", tags=["keywords"])
app.include_router(media_keyword_extraction.router, prefix="/extract", tags=["keywords"])
app.include_router(claim_keyword_extraction.router, prefix="/extract", tags=["claim-keywords"])
app.include_router(validation.router, tags=["validation"])

# Register exception handlers
app.add_exception_handler(SQLAlchemyError, database_exception_handler)
app.add_exception_handler(requests.RequestException, requests_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")
