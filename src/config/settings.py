"""
Configuration settings for the claim-quote extraction pipeline.

Loads configuration from environment variables using Pydantic Settings.
All settings can be overridden via .env file or environment variables.

Usage:
    from src.config.settings import settings

    print(settings.database_url)
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/podcast_db",
        description="PostgreSQL connection string",
    )

    # Embeddings (optional; used by the premium pipeline when enabled)
    enable_embeddings: bool = Field(
        default=True,
        description="Enable embedding generation and storage (disable for deployment without embedding service)"
    )
    ollama_embedding_url: str = Field(
        default="http://localhost:11435",
        description="Ollama API endpoint for embedding operations",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Embedding model (768 dimensions)"
    )

    # Caching (embedding service LRU cache)
    cache_max_size: int = Field(
        default=10000, description="Maximum number of entries in LRU caches"
    )
    cache_ttl_hours: int = Field(
        default=1, description="Cache entry time-to-live in hours"
    )

    # Logging
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str = Field(default="logs/extraction.log", description="Log file path")

    # API keys for LLM providers
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key"
    )

    # Gemini Guest/Keyword Extraction
    gemini_extraction_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for guest/keyword extraction"
    )
    gemini_extraction_temperature: float = Field(
        default=0.0,
        description="Temperature for extraction tasks (0 = deterministic)"
    )

    # Premium Claim Extraction (Gemini 3)
    gemini_premium_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini 2 model for premium claim extraction"
    )
    gemini_premium_temperature: float = Field(
        default=0.2,
        description="Temperature for premium extraction (0 = deterministic)"
    )
    premium_extraction_max_parallel_episodes: int = Field(
        default=20,
        description="Maximum number of episodes processed in parallel for premium extraction"
    )
    premium_extraction_gemini_calls_per_episode: int = Field(
        default=3,
        description="Estimated Gemini calls per episode for premium extraction rate limiting"
    )
    premium_extraction_rate_limit_max_tokens: int = Field(
        default=100,
        description="Maximum Gemini calls allowed per rate limit window for premium extraction"
    )
    premium_extraction_rate_limit_window_seconds: float = Field(
        default=60.0,
        description="Rate limit window in seconds for premium extraction calls"
    )

    # News Claim Extraction (POST /extract/news/claims) — scoped to the news
    # endpoint only; deliberately separate from gemini_premium_model so the
    # podcast premium pipeline is unaffected. gemini-3.5-flash with a low
    # thinking level gives ~2-5x faster extraction at equal/better claim
    # quality vs gemini-2.5-pro (benchmarked 2026-05-27). Called directly via
    # the google-genai SDK (not langchain build_chain) because thinking_level
    # requires the consolidated SDK that langchain 3.x does not expose.
    gemini_news_claim_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini model for news claim extraction (/extract/news/claims)"
    )
    gemini_news_claim_temperature: float = Field(
        default=0.2,
        description="Temperature for news claim extraction"
    )
    gemini_news_claim_thinking_level: str = Field(
        default="low",
        description="Gemini 3+ thinking level for news claim extraction: minimal|low|medium|high. 'low' is adaptive (fast on light stories, thinks on dense ones) and holds claim coverage; 'minimal' degrades dense multi-source stories."
    )

    # News Claim Extraction — Claude fallback (POST /extract/news/claims/claude)
    # Runs the EXACT SAME NEWS_CLAIM_EXTRACT_PROMPT as the Gemini endpoint, just
    # on Anthropic Claude. news-worker calls this only when the Gemini path
    # errors out, so a Gemini outage no longer drops the pipeline onto a weaker
    # locally-defined prompt (root cause of the 2026-06-10 inverted-claim
    # incident). Requires anthropic_api_key (ANTHROPIC_API_KEY env).
    news_claim_claude_model: str = Field(
        default="claude-sonnet-4-5",
        description="Anthropic Claude model for the news-claim Claude fallback endpoint. Same strong prompt as the Gemini path."
    )
    news_claim_claude_max_tokens: int = Field(
        default=32000,
        description="Max output tokens for the Claude news-claim fallback (dense multi-source stories can produce many claims)."
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8000,
        description="API server port (Railway sets this via PORT env var)"
    )
    api_timeout: int = Field(
        default=0,
        description="Maximum request timeout in seconds (0 = no timeout)"
    )
    api_key: str = Field(
        default="change-me-in-production",
        description="API key for authentication (X-API-Key header)"
    )


# Global settings instance
settings = Settings()
