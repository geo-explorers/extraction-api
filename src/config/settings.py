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

    # Hatchet worker.
    # The Hatchet client reads HATCHET_CLIENT_TOKEN / HATCHET_CLIENT_HOST_PORT /
    # HATCHET_CLIENT_TLS_STRATEGY directly from the environment (see .env.example).
    hatchet_worker_slots: int = Field(
        default=10,
        description="Max concurrent task runs per worker (the provider rate limit is the real throttle)"
    )
    # Global provider rate limits (calls/min), enforced by Hatchet across ALL
    # workers via static keys. Set below the true provider quota — Hatchet uses
    # fixed windows, so leave headroom vs Gemini's sliding-window quota.
    gemini_global_rate_per_min: int = Field(
        default=100,
        description="Global Gemini calls/min across all workers (Hatchet static key 'gemini_global')"
    )
    claude_global_rate_per_min: int = Field(
        default=100,
        description="Global Claude calls/min across all workers (Hatchet static key 'claude_global')"
    )
    # Spend circuit breaker: hard hourly ceiling on LLM calls per provider.
    # 0 disables it. Distinct from the rate limiter — this caps total volume/$.
    llm_max_calls_per_hour: int = Field(
        default=0,
        description="Per-provider hourly LLM call ceiling (0 = disabled). Backstop against runaway loops."
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

    # Podcast publishing (postgres_to_geo "Export API").
    # The podcast.export task forwards to this service's POST /api/export. In
    # prod this MUST be the *.railway.internal address: a synchronous publish
    # runs ~25 min, and only Railway's private network (no L7 edge proxy) lets a
    # connection idle that long without a 502. The public *.up.railway.app URL
    # would still time out.
    postgrestogeo_url: str = Field(
        default="http://localhost:3000",
        description="Base URL of the postgres_to_geo export service (prod: http://postgrestogeo.railway.internal:3000)"
    )
    postgrestogeo_api_key: str | None = Field(
        default=None,
        description="X-API-Key for the postgres_to_geo /api/export endpoint"
    )

    # Geo / Hypergraph knowledge graph (read side).
    # geo.fetch_entities and the geo.assign_spaces_to_sheet DAG read canonical
    # spaces and typed entities from the Geo GraphQL API. A headless worker cannot
    # use the HyperGraph MCP, so it queries this HTTP endpoint directly. Reads are
    # UNAUTHENTICATED (matching the geo-explorers reference clients).
    hypergraph_graphql_url: str = Field(
        default="https://testnet-api.geobrowser.io/graphql",
        description="Geo GraphQL endpoint for reading spaces/entities (mainnet: https://api.geobrowser.io/graphql)"
    )
    hypergraph_api_key: str | None = Field(
        default=None,
        description="Optional bearer key for the Geo GraphQL endpoint. Reads are normally unauthenticated; leave unset."
    )
    geo_root_space_id: str = Field(
        default="a19c345ab9866679b001d7d2138d88a1",
        description="Geo root space id (the 'Geo' space). Canonical spaces are its subspaces, fetched dynamically."
    )
    geo_canonical_space_ids: str | None = Field(
        default=None,
        description="Optional override: comma-separated space ids to assign against. Unset = subspaces of geo_root_space_id (dynamic)."
    )

    # Google Sheets export (service-account auth).
    # sheets.export_table and geo.assign_spaces_to_sheet create a NEW spreadsheet
    # via gspread. A headless worker authenticates with a service-account key:
    # provide it inline as JSON (preferred for Railway env vars) or as a file path.
    # The created sheet lives in the service account's own Drive, so it is shared
    # with google_sheets_share_email (and/or created inside google_drive_folder_id)
    # to be visible to a human.
    google_service_account_json: str | None = Field(
        default=None,
        description="Service-account key as inline JSON (preferred). Takes precedence over the file path."
    )
    google_service_account_file: str | None = Field(
        default=None,
        description="Path to a service-account JSON key file (fallback when google_service_account_json is unset)."
    )
    google_sheets_share_email: str | None = Field(
        default=None,
        description="Email the created spreadsheet is shared with (writer role), so it is visible outside the service account."
    )
    google_drive_folder_id: str | None = Field(
        default=None,
        description="Optional Drive/Shared-Drive folder ID to create the spreadsheet in."
    )

    # Geo entity->space assignment (Gemini, schema-enforced JSON).
    # The assign_spaces step classifies each fetched entity against the fetched
    # canonical spaces. Kept separate from the other Gemini model settings so this
    # pipeline can be tuned independently.
    gemini_space_assignment_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for assigning canonical spaces to entities"
    )
    gemini_space_assignment_temperature: float = Field(
        default=0.0,
        description="Temperature for space assignment (0 = deterministic)"
    )
    space_assignment_batch_size: int = Field(
        default=50,
        description="Entities per Gemini call in the space-assignment step. Kept modest because each item now emits a reasoning field (larger output → truncation risk at big batches)."
    )
    space_assignment_concurrency: int = Field(
        default=8,
        ge=1,
        description="Max concurrent Gemini calls in the space-assignment step (batches run in a bounded thread pool)."
    )


# Global settings instance
settings = Settings()
