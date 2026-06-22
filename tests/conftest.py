"""Test configuration.

The Hatchet client decodes HATCHET_CLIENT_TOKEN as a JWT at construction (to
derive endpoints), so importing the task registry needs a structurally-valid
token even though tests never connect to an engine. Provide an unsigned,
local-only JWT before any task module is imported.
"""

import base64
import json
import os


def _fake_hatchet_token() -> str:
    def b64url(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    header = b64url({"alg": "none", "typ": "JWT"})
    payload = b64url(
        {
            "sub": "test-tenant",
            "aud": "https://localhost",
            "exp": 9999999999,
            "grpc_broadcast_address": "localhost:7077",
            "server_url": "http://localhost:8080",
        }
    )
    return f"{header}.{payload}.sig"


os.environ.setdefault("HATCHET_CLIENT_TOKEN", _fake_hatchet_token())
os.environ.setdefault("HATCHET_CLIENT_HOST_PORT", "localhost:7077")
os.environ.setdefault("HATCHET_CLIENT_TLS_STRATEGY", "none")
