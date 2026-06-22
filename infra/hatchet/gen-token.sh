#!/usr/bin/env bash
# Generate a Hatchet tenant API token for the worker + the extraction-api facade.
#
# Run ONCE, after `up -d` has brought up postgres + migration + setup-config + engine
# (setup-config has written the keysets to the hatchet_config volume). This runs a
# one-off hatchet-admin against that same volume + database to mint the token.
# Paste the printed token into .env as HATCHET_CLIENT_TOKEN, set it on the
# extraction-api service too, then `docker compose up -d extraction-worker`.
#
# Usage:  ./gen-token.sh [tenant-id]
# The default tenant id is the one setup-config seeds; confirm it in the dashboard
# (Settings → Tenant) if you created/use a different tenant.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE=(docker compose -f "$DIR/docker-compose.prod.yml")
TENANT="${1:-707d0855-80ab-4e1f-a156-f1c4546cbf52}"

echo "Generating tenant token (tenant=$TENANT)..." >&2
# `run --no-deps` reuses setup-config's image + the hatchet_config volume + DATABASE_URL,
# without re-triggering migration (postgres must already be up from `up -d`).
"${COMPOSE[@]}" run --rm --no-deps setup-config \
  /hatchet/hatchet-admin token create --config /hatchet/config --tenant-id "$TENANT"
