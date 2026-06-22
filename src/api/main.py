"""Main FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import requests

from src.api.routers import extraction, guest_extraction, host_extraction, keyword_extraction, media_keyword_extraction, claim_keyword_extraction, news_claim_extract, tasks
from src.api.exceptions import (
    database_exception_handler,
    generic_exception_handler,
    requests_exception_handler,
)
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


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
    logger.info(f"Embeddings: {settings.enable_embeddings}")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Podcast Extraction API shutting down")
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Podcast Extraction API",
    description="""
    API for extracting claims, guests, and keywords from podcast transcripts.

    ## Features
    - **Claim Extraction**: Extract claims from podcast transcripts using Gemini (full-context)
    - **News Claim Extraction**: Extract claims, quotes, and collections from news articles
    - **Guest/Host Extraction**: Extract podcast guest and host names using Gemini
    - **Keyword Extraction**: Extract keywords and topics from episode or media data using Gemini
    - Batch processing across multiple podcasts

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
app.include_router(news_claim_extract.router, prefix="/extract", tags=["news-claim-extract"])
app.include_router(tasks.router, tags=["tasks"])

# Register exception handlers
app.add_exception_handler(SQLAlchemyError, database_exception_handler)
app.add_exception_handler(requests.RequestException, requests_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")
