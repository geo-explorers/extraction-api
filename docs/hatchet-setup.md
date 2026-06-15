# Hatchet self-hosted setup (Railway)

The extraction worker runs long LLM tasks on a self-hosted Hatchet plane. This
runbook covers the one-time Railway provisioning the code can't do for you. After
this, enqueuing the `ping` task and seeing it run proves the plane end-to-end.

> Phase 1 of the Hatchet task-layer refactor. The application code (client,
> worker, ping task) lives in `src/hatchet_client.py`, `src/worker.py`,
> `src/tasks/`. This document is the infra half.

## Services to create (project: "Geo daily")

### 1. `hatchet-postgres` — dedicated database
A Postgres instance used **only** by Hatchet (queue, runs, step checkpoints).
Keep it isolated from the application `crypto` database — it is the blast-radius
boundary for any Hatchet-side DB issue. Note its private connection string.

### 2. `hatchet-lite` — engine + dashboard
Single container: engine + REST + gRPC + dashboard.
- **Image:** `ghcr.io/hatchet-dev/hatchet/hatchet-lite:<PINNED_TAG>` — pin an
  explicit version tag, never `:latest`.
- **Required env:**
  - `DATABASE_URL` → the `hatchet-postgres` private URL
  - `SERVER_AUTH_BASIC_AUTH_ENABLED=true`
  - `SERVER_ALLOW_SIGNUP=false` — **must set**; the default allows anyone reaching
    the dashboard to create a tenant
  - `SERVER_ENCRYPTION_MASTER_KEYSET` + JWT keysets — generate once, store as
    Railway secrets (keeps the container stateless and restart-safe)
  - `SERVER_GRPC_BROADCAST_ADDRESS` / `SERVER_URL` → the private host
    (e.g. `hatchet-lite.railway.internal:7077`)
- **Networking:** expose gRPC (`:7077`) and REST on the **private** network only.
  The dashboard should be reached via a Railway-authenticated/private domain — do
  not expose it publicly.

After it boots, log into the dashboard, create the tenant, and generate a
**tenant API token** (a JWT starting `ey...`). You'll need it below.

### 3. `extraction-worker` — the Python worker
Same Docker image as `extraction-api` (one image, two start commands).
- **Start command:** `uv run python -m src.worker`
- **Env:**
  - `HATCHET_CLIENT_TOKEN` = the tenant token from step 2
  - `HATCHET_CLIENT_HOST_PORT` = `hatchet-lite.railway.internal:7077`
  - `HATCHET_CLIENT_TLS_STRATEGY` = `none` (private networking, no TLS)
  - `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, model settings (as the API service)
  - `HATCHET_WORKER_SLOTS` = `10` (optional; the provider rate limit is the real throttle)
  - `RAILWAY_DEPLOYMENT_DRAINING_SECONDS` = `600` — on redeploy the worker stops
    pulling new runs and finishes in-flight ones before SIGKILL
  - **No `DATABASE_URL`** — the worker is stateless and never touches the app DB
- **No public domain** — it only makes outbound gRPC to `hatchet-lite`.

### `extraction-api` (existing service)
From Phase 2 it hosts the enqueue facade (`POST /tasks`, `GET /tasks/{id}`), so it
also needs `HATCHET_CLIENT_TOKEN` / `HATCHET_CLIENT_HOST_PORT` /
`HATCHET_CLIENT_TLS_STRATEGY`. The facade constructs the client lazily, so a
missing token degrades only `/tasks`, not the whole API.

## Verify the plane (Phase 1 acceptance)

1. Deploy `hatchet-lite`, `hatchet-postgres`, and `extraction-worker`.
2. In the dashboard, confirm `extraction-worker` appears under Workers with the
   `ping` task registered.
3. Trigger a run of `ping` from the dashboard with input `{"message": "hello"}`;
   confirm it succeeds with `{"reply": "pong: hello"}`.
4. Redeploy `extraction-worker` while a run is in flight; confirm the in-flight
   run drains and completes rather than erroring (validates the draining setting).

## Notes
- The SDK reads `HATCHET_CLIENT_*` from the environment; it decodes the token as a
  JWT at construction to derive endpoints, so a real token is required wherever the
  client is constructed (worker always; API only when the facade is used).
- Local dev: run `hatchet-lite` via its docker-compose for a real token, or skip
  the worker and exercise extraction through the existing sync HTTP endpoints.
