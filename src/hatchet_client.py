"""Shared Hatchet client singleton.

Both the worker (`src/worker.py`) and the API enqueue facade import the same
client instance. Configuration is read from the environment by the SDK
(`HATCHET_CLIENT_*` vars); see `.env.example` and `docs/hatchet-setup.md`.

Construction does NOT open a gRPC connection — that happens lazily on the first
run/enqueue or when the worker starts — so importing this module is cheap and
safe in the API process.
"""

from hatchet_sdk import Hatchet

# Singleton. The SDK reads HATCHET_CLIENT_TOKEN / HATCHET_CLIENT_HOST_PORT /
# HATCHET_CLIENT_TLS_STRATEGY from the environment.
hatchet = Hatchet()
