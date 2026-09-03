# Extraction API

FastAPI service that extracts claims, keywords, topics, guests, and hosts from podcast and news content using Google Gemini (with an Anthropic Claude fallback for news claims). Results are written to a shared PostgreSQL database (podcast pipeline) or returned to the caller (news pipeline).

Deployed on Railway (project "Geo daily", service "Extraction API") from `master`.

## Endpoints

All endpoints require an `X-API-Key` header. Interactive docs at `/docs`.

| Endpoint | Purpose | Caller |
|----------|---------|--------|
| `POST /extract/claims/premium` | Full-context claim extraction from podcast episodes (topics → claims → key takeaways), saved to Postgres | pg-migrations (daily ETL) |
| `POST /extract/keywords` | Keyword + topic extraction for podcast episodes | pg-migrations |
| `POST /extract/guests` | Guest name/URL extraction | pg-migrations |
| `POST /extract/hosts` | Host extraction | pg-migrations |
| `POST /extract/news/claims` | Single-pass news claim extraction: claims, quotes, topic/perspective collections, summary (Gemini) | news-worker (`CLAIM_FRESH_EXTRACT_ENABLED`) |
| `POST /extract/news/claims/claude` | Same prompt on Claude — fallback when the Gemini path fails | news-worker (main) |
| `POST /extract/media/keywords` | Media-type-agnostic keyword/topic extraction (articles, papers, …) | news-worker (in progress) |
| `POST /extract/claim-keywords` | Per-claim keyword/topic extraction | (reserved) |

## Setup

```bash
# Install uv, then:
uv sync

# Configure environment
cp .env.example .env   # fill in DATABASE_URL, GEMINI_API_KEY, ANTHROPIC_API_KEY, API_KEY

# Run the server
uv run python -m src.api.server
```

## Project Structure

```
src/
├── api/
│   ├── main.py                # FastAPI app, auth middleware, router registration
│   ├── server.py              # uvicorn entrypoint (Railway start command)
│   ├── routers/               # One router per endpoint group
│   ├── services/              # Endpoint orchestration
│   └── schemas/               # Request/response models
├── cli/
│   └── episode_query.py       # Episode selection queries (used by premium pipeline)
├── config/
│   ├── settings.py            # Pydantic settings (env-driven)
│   └── prompts/               # All LLM prompts
├── database/                  # SQLAlchemy models + repositories (shared crypto schema)
├── extraction/
│   ├── premium_claim_extractor.py  # Gemini structured-output calls
│   └── models.py              # Shared dataclasses (ClaimWithTopic, Quote)
├── infrastructure/            # Logger, embedding service
├── pipeline/
│   └── premium_extraction_pipeline.py  # Topics → claims → takeaways → DB save
└── preprocessing/
    └── transcript_parser.py   # Transcript segment/speaker parsing
```

## Configuration

See `.env.example` for all settings. Key flags:

- `GEMINI_PREMIUM_MODEL` — model for podcast claim extraction
- `GEMINI_NEWS_CLAIM_MODEL` / `GEMINI_NEWS_CLAIM_THINKING_LEVEL` — news claim extraction (scoped separately from the podcast pipeline)
- `NEWS_CLAIM_CLAUDE_MODEL` — Claude fallback for news claims
- `ENABLE_EMBEDDINGS` — optional claim embeddings via an Ollama embedding service (off in production)

## Tests

```bash
uv run pytest tests/
```
