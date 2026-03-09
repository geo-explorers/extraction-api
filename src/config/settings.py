"""
Configuration settings for the claim-quote extraction pipeline.

Loads configuration from environment variables using Pydantic Settings.
All settings can be overridden via .env file or environment variables.

Usage:
    from src.config.settings import settings

    print(settings.database_url)
    print(settings.ollama_url)
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

    # Ollama
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint for LLM operations (qwen)",
    )
    ollama_embedding_url: str = Field(
        default="http://localhost:11435",
        description="Ollama API endpoint for embedding operations",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct-q4_0",
        description="LLM model for claim extraction and entailment",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Embedding model (768 dimensions)"
    )

    # Reranker
    enable_reranker: bool = Field(
        default=True, description="Enable reranker service for high-precision scoring"
    )
    reranker_url: str = Field(
        default="http://localhost:8080", description="Reranker service endpoint"
    )
    reranker_timeout: int = Field(
        default=5000, description="Reranker request timeout in milliseconds"
    )

    # Quote Processing
    enable_quote_processing: bool = Field(
        default=True,
        description="Enable quote finding and entailment validation (disable for faster extraction without quotes)"
    )

    # Embeddings
    enable_embeddings: bool = Field(
        default=True,
        description="Enable embedding generation and storage (disable for deployment without embedding service)"
    )

    # Chunking
    chunk_size: int = Field(
        default=16000, description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=1000, description="Overlap between chunks in characters"
    )

    # Processing (DSPy asyncify concurrency limits)
    max_claim_extraction_concurrency: int = Field(
        default=3,
        description="Maximum concurrent claim extraction calls (DSPy asyncify)",
    )
    max_entailment_concurrency: int = Field(
        default=10,
        description="Maximum concurrent entailment validation calls (DSPy asyncify)",
    )
    max_ad_classification_concurrency: int = Field(
        default=10,
        description="Maximum concurrent ad classification calls (DSPy asyncify)",
    )
    ad_classification_batch_size: int = Field(
        default=10,
        description="Number of claims to classify per LLM call (true batching)",
    )

    # Claim Quality Filtering
    enable_claim_specificity_filter: bool = Field(
        default=True,
        description="Enable specificity filtering (requires numbers, proper nouns, or dates)",
    )
    filter_lowercase_claims: bool = Field(
        default=True,
        description="Filter out claims that start with a lowercase letter",
    )
    max_claims_per_chunk: int = Field(
        default=10,
        description="Maximum claims to extract per chunk (limits dense content, keeps longest claims)",
    )

    # Ad Classification
    filter_advertisement_claims: bool = Field(
        default=True,
        description="Enable advertisement claim filtering in extraction pipeline",
    )
    ad_classifier_model_path: str = Field(
        default="models/ad_classifier_v1.json",
        description="Path to trained ad classification model",
    )
    ad_classification_threshold: float = Field(
        default=0.7, description="Minimum confidence to classify claim as advertisement"
    )

    # DSPy Model Paths
    claim_extractor_model_path: str = Field(
        default="models/claim_extractor_llm_judge_v1.json",
        description="Path to trained claim extraction model",
    )
    entailment_validator_model_path: str = Field(
        default="models/entailment_validator_v1.json",
        description="Path to trained entailment validation model",
    )

    # Deduplication Thresholds
    enable_cross_episode_deduplication: bool = Field(
        default=True,
        description="Enable cross-episode claim deduplication via database similarity search"
    )
    embedding_similarity_threshold: float = Field(
        default=0.85, description="Cosine similarity threshold for claim deduplication"
    )
    reranker_verification_threshold: float = Field(
        default=0.9, description="Reranker score threshold for duplicate verification"
    )
    string_similarity_threshold: float = Field(
        default=0.95, description="Jaccard similarity threshold for quote deduplication"
    )
    vector_distance_threshold: float = Field(
        default=0.15, description="pgvector L2 distance threshold for database search"
    )

    # Scoring
    min_confidence: float = Field(
        default=0.3, description="Minimum confidence score to save claims"
    )
    max_quotes_per_claim: int = Field(
        default=10, description="Maximum number of quotes to link per claim"
    )
    min_quote_relevance: float = Field(
        default=0.85, description="Minimum relevance score to link quote to claim"
    )

    # Caching
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

    # MIPROv2 Optimization (optional cloud model for instruction proposal)
    mipro_prompt_model: str | None = Field(
        default=None,
        description="Anthropic model for MIPROv2 instruction proposal (e.g., 'anthropic/claude-3-5-sonnet-20241022')"
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude models"
    )

    # Gemini Claim Validation
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key for claim validation"
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model for claim validation (gemini-2.0-flash-exp recommended)"
    )
    gemini_validation_batch_size: int = Field(
        default=50,
        description="Number of claims to validate per Gemini API call"
    )
    gemini_timeout: int = Field(
        default=30,
        description="Gemini API request timeout in seconds"
    )
    min_claims_for_validation: int = Field(
        default=5,
        description="Minimum number of unverified claims an episode must have to be eligible for validation"
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
