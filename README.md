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

## Prompt and LLM overrides (hand-testing without a deploy)

Every prompt is cataloged by key in `src/config/prompt_registry.py`, and every task input
and sync request body accepts two optional maps:

| Field | Value | Effect |
|-------|-------|--------|
| `prompt_overrides` | `{ "<prompt key>": "<replacement text>" }` | That prompt text, for this run only |
| `llm_overrides` | `{ "<setting name>": <value> }` | That model / temperature / thinking level / max tokens, for this run only |

`GET /prompts` lists every key with its format slots, plus every overridable LLM setting with
its current value. `GET /prompts/{key}?format=text` returns one prompt's default text for
copy-paste. The loop is: fetch, edit, paste into the Hatchet dashboard's trigger payload
(or `POST /tasks`, or a sync endpoint body):

```json
{
  "media_type": "debate",
  "documents": [{"content": "..."}],
  "prompt_overrides": {"claims_extract.factuality": "...edited section..."},
  "llm_overrides": {"claims_extract_model": "gemini-3.5-pro", "claims_extract_thinking_level": "low"}
}
```

Rules, enforced at enqueue (422 at the facade; input-validation failure on the worker for a
dashboard trigger): the key must exist; a formatted prompt's override may use only the slots
its default uses (`{headline}` etc., literal braces as `{{ }}`); an LLM setting name must be
one `GET /prompts` lists, with a value of the right type and range. `claims.extract` is
overridden per section (`claims_extract.core`, `claims_extract.media.debate`,
`claims_extract.factuality`, ...) because the builder renders documents and vocabulary into
the prompt itself.

Each run logs one line per task (or per DAG step) naming the overrides that applied, any that
went unused, and every prompt key and setting the step read; the worker forwards its logging
into the run's log, so the line is visible in the Hatchet dashboard. A model override is also
reported in `model_used` where the result carries it.


Under the hood: `src/config/overrides.py` holds a contextvars scope activated by the task
runner (`src/tasks/base.py`, `with_overrides`) and by the HTTP handlers; prompt consumers call
`prompts.get(key)` and LLM call sites call `llm.get(setting)` at render time instead of
importing constants. The scope follows `asyncio.to_thread`; work handed to a bare
`ThreadPoolExecutor` goes through `bind_context()`.

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
│   ├── prompt_registry.py     # Key -> prompt text catalog (GET /prompts, prompt_overrides)
│   ├── overrides.py           # Per-run prompt/LLM override scope (prompts.get / llm.get)
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
- `GEMINI_NEWS_DEBATE_MODEL` / `GEMINI_NEWS_DEBATE_THINKING_LEVEL` — dedicated news debate-generation pass
- `GEMINI_NEWS_DEBATE_REVIEW_MODEL` / `NEWS_DEBATE_SEMANTIC_REVIEW_ENFORCED` — reject-only semantic review (set enforcement false for shadow mode)
- `NEWS_DEBATE_UNDERFILLED_RESCUE_ENABLED` — one best-effort extra candidate/review pass when one or two debates survive
- `NEWS_DEBATE_ZERO_RETRY_ENABLED` — one fresh generation+review draw when review approves zero candidates despite generation producing some
- `NEWS_CLAIM_CLAUDE_MODEL` — Claude fallback for news claims
- `ENABLE_EMBEDDINGS` — optional claim embeddings via an Ollama embedding service (off in production)

## Tests

```bash
uv run pytest tests/
```
