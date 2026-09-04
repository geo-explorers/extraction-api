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

    # claims.judge_equivalence: one Gemini call judges which candidates are logically
    # equivalent to a claim. Candidates come from the caller (typically geo-lens).
    claims_equivalence_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for claims.judge_equivalence (one call per claim, all candidates)",
    )
    claims_equivalence_temperature: float = Field(default=0.0)

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

    # News Debate Extraction — a separate definition-led pass. Keeping these
    # controls independent lets us spend more reasoning on the comparatively
    # small compose-and-review decision without slowing the large factual
    # extraction prompt. It defaults to the same proven Flash model.
    gemini_news_debate_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini model for source-grounded news debate generation"
    )
    gemini_news_debate_temperature: float = Field(
        default=0.0,
        description="Temperature for news debate generation (0 = stable candidate gating)"
    )
    gemini_news_debate_thinking_level: str = Field(
        default="medium",
        description="Gemini 3+ thinking level for source-grounded news debates: minimal|low|medium|high. Empty disables."
    )
    gemini_news_debate_review_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini model for reject-only semantic review of news debates"
    )
    gemini_news_debate_review_temperature: float = Field(
        default=0.0,
        description="Temperature for news debate semantic review"
    )
    gemini_news_debate_review_thinking_level: str = Field(
        default="high",
        description="Gemini 3+ thinking level for news debate semantic review. Empty disables."
    )
    news_debate_semantic_review_enforced: bool = Field(
        default=True,
        description="Reject candidates that fail semantic review. False runs the reviewer in shadow mode."
    )
    news_debate_zero_retry_enabled: bool = Field(
        default=True,
        description="When first-pass review accepts zero debate candidates but generation produced some, take one fresh generation + review draw. Thin-supply stories sit at the 3-floor with no margin, so a single review flip otherwise zeroes the collection."
    )
    news_debate_underfilled_rescue_enabled: bool = Field(
        default=True,
        description="Run one additional grounded candidate + semantic-review pass when one or two debates survive."
    )

    # News Claim Extraction — Claude fallback (POST /extract/news/claims/claude)
    # Runs the same factual + grounded-debate contracts as Gemini, just on
    # Anthropic Claude. news-worker calls this only when the Gemini path
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

    # Generalized claim extraction (claims.extract DAG) — one pipeline for any
    # text media: debate transcripts, research papers, news articles, podcasts,
    # talks, generic documents. Deliberately separate from gemini_premium_model
    # and gemini_news_claim_model so each pipeline stays independently tunable.
    # Defaults to the pro tier rather than flash: this endpoint serves
    # unbenchmarked media (multi-hour debates, dense papers), so quality
    # headroom beats the flash speedup. gemini-2.5-pro is the latest STABLE
    # pro on the API as of 2026-07-13 (3.x pro exists only as -preview;
    # gemini-3.5-pro not yet GA — swap it in via CLAIMS_EXTRACT_MODEL once
    # released, and set CLAIMS_EXTRACT_THINKING_LEVEL=low with it).
    # Cost/speed downshift: CLAIMS_EXTRACT_MODEL=gemini-3.5-flash
    # (news-benchmark-proven, 2026-05-27).
    claims_extract_model: str = Field(
        default="gemini-2.5-pro",
        description="Gemini model for the generalized claims.extract DAG"
    )
    claims_extract_temperature: float = Field(
        default=0.2,
        description="Temperature for generalized claim extraction"
    )
    # Empty by default because the default model is 2.5-era: thinking_level is
    # a Gemini 3+ control and attaching it to a 2.5 model errors. Set to
    # minimal|low|medium|high only alongside a Gemini 3+ CLAIMS_EXTRACT_MODEL.
    claims_extract_thinking_level: str = Field(
        default="",
        description="Gemini 3+ thinking level for claims.extract: minimal|low|medium|high. Empty disables (required with 2.5-era models)."
    )

    # Claim entity linking (claims.link_entities) — index-selection over a
    # caller-supplied vocabulary. A light classification task, so it defaults
    # to the flash tier with adaptive low thinking (same pairing proven for
    # news claim extraction, 2026-05-27); independently tunable per the
    # one-knob-per-pipeline convention.
    claims_link_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini model for claims.link_entities annotation"
    )
    claims_link_temperature: float = Field(
        default=0.1,
        description="Temperature for claim entity linking (index selection wants determinism)"
    )
    claims_link_thinking_level: str = Field(
        default="low",
        description="Gemini 3+ thinking level for claims.link_entities: minimal|low|medium|high. Empty disables (required if overriding to a 2.5-era model)."
    )
    claims_link_max_vocabulary: int = Field(
        default=2000,
        description="Max vocabulary items per facet in claims.link_entities (contract cap, applied at import)"
    )

    # Geo entity resolution (geo.resolve_entities) — read-only matching knobs.
    geo_resolve_match_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for geo.resolve_entities name matching (news-worker parity)"
    )
    geo_resolve_max_candidates: int = Field(
        default=30,
        description="Candidates fetched per name query in geo.resolve_entities"
    )
    # Team-wide graph constants for the team_priority selection cascade —
    # properties of the shared Geo graph (agreed dedup rules), not caller
    # domain concepts. Overridable for other graph deployments.
    geo_featured_tag_id: str = Field(
        default="b69b8b1659df4e6d99d79956a30e8932",
        description="Featured-marker tag entity id (cascade tier)"
    )
    geo_curated_tag_id: str = Field(
        default="7f796eb5bfc5449c98649bf7d996a2ca",
        description="Curated-marker tag entity id (cascade tier)"
    )
    geo_score_property_id: str = Field(
        default="85a4668a42fa4f488969c0a9de0c294b",
        description="Score property id; >=2 scored candidates escalate to ambiguous"
    )
    geo_catchall_space_id: str = Field(
        default="b5a31f8182b042437ede0f84ee02f104",
        description="Catch-all space demoted by the properly-placed cascade tier"
    )
    geo_dataset_space_ids: str = Field(
        default="5908c73ad336472ccbd983491d2d17e4,941964642f4d3e70ef48f54a3915277d,44eb138f564fbed6ed9ce543de1b849c,da96a4c26e718bfa6c27c3b1f3c316cd,1b3d2963d14de99d4e440000125edb65",
        description="Comma-separated dataset space ids (excluded from canonical selection)"
    )

    @property
    def geo_dataset_space_ids_list(self) -> list[str]:
        return [s.strip() for s in self.geo_dataset_space_ids.split(",") if s.strip()]

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
        default="https://api-testnet.geobrowser.io/graphql",
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
