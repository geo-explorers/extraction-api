# Hatchet (full self-hosted stack) + extraction worker — Docker Compose IaC

The full, MIT-licensed self-hosted Hatchet control plane + the extraction worker,
as portable Docker Compose. The same `docker-compose.prod.yml` runs on a **VM**,
on **Railway** (Docker-Compose support), or any Docker host — no vendor-specific
config. Adapted from Hatchet's official self-host compose, hardened for prod and
simplified to a **Postgres message broker** (no RabbitMQ). Replaces the old prose
runbook: the know-how is in code + one script; only secret *values* live outside git.

## Services
- `postgres` — Hatchet state + the message queue (broker = Postgres)
- `migration` — one-shot DB schema setup (`hatchet-migrate`)
- `setup-config` — one-shot config + keyset generation into the `hatchet_config` volume (`hatchet-admin quickstart`)
- `hatchet-engine` — the orchestration engine (gRPC :7077→7070)
- `hatchet-dashboard` — web UI (:8080)
- `extraction-worker` — our worker (same image as extraction-api, `src.worker`)

## Files
- `docker-compose.prod.yml` — the stack above
- `.env.example` — every variable, documented (copy → `.env`, gitignored)
- `gen-token.sh` — one-time tenant-token bootstrap

## Bring-up
```bash
cd infra/hatchet
cp .env.example .env && $EDITOR .env     # fill: PG password, cookie secrets, admin creds, URLs, LLM keys
# postgres → migration → setup-config (generates keysets) → engine + dashboard come up:
docker compose -f docker-compose.prod.yml up -d postgres migration setup-config hatchet-engine hatchet-dashboard
./gen-token.sh                            # prints a tenant token
#   → paste into .env as HATCHET_CLIENT_TOKEN, and set the SAME token on the extraction-api service
docker compose -f docker-compose.prod.yml up -d extraction-worker
```

## Verify
- Dashboard (`HATCHET_SERVER_URL`, default `:8080`) → log in with `HATCHET_ADMIN_*` → Workers shows `extraction-worker` registering the task types.
- Enqueue a `ping` via the facade (`POST /tasks {"type":"ping","payload":{"message":"hi"}}`) → `pong`.

## Operate
- **Persist + back up** `hatchet_postgres_data` (task state + tenant) **and** `hatchet_config` (keysets). On Railway, attach a persistent Volume for each — a restart on ephemeral storage regenerates keysets and locks the worker out. (`hatchet_certs` is empty here — `--skip certs` → insecure gRPC on the private network.)
- **Redeploys drain**: `stop_grace_period: 600s` lets in-flight tasks finish before SIGKILL.
- **Networking**: expose only the dashboard (behind TLS/auth); keep gRPC `:7077` on the private network.
- **Rate limits**: set `*_GLOBAL_RATE_PER_MIN` below the real provider quota.

## Notes
- **Broker = Postgres** (`SERVER_MSGQUEUE_KIND=postgres`); RabbitMQ from the upstream compose is removed. Fine for our volume; re-add RabbitMQ if throughput ever demands it.
- **Image version**: pin all Hatchet components together via `HATCHET_VERSION` (default `v0.89.6`, validated locally).
- **extraction-api facade**: deployed separately (its own service); add `HATCHET_CLIENT_TOKEN` / `HATCHET_CLIENT_HOST_PORT` (= `HATCHET_GRPC_BROADCAST`) / `HATCHET_CLIENT_TLS_STRATEGY=none` so it can enqueue.
