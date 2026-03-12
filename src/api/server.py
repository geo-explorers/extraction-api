"""Server entry point for running the FastAPI application."""

import uvicorn
from src.config.settings import settings


def main():
    """Run the FastAPI server with uvicorn."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=True,  # Set to True for development
        log_level="info",
        timeout_keep_alive=settings.api_timeout,
    )


if __name__ == "__main__":
    main()
