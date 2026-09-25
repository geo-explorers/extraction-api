"""API-key authentication: legacy API_KEY plus the per-caller API_KEYS list."""

import logging

import pytest
from fastapi.testclient import TestClient

from src.api.auth import ApiKey, configured_keys, describe, parse_api_keys, resolve_caller
from src.config.settings import settings


def test_parse_labels_bare_keys_and_blanks():
    keys = parse_api_keys(" news-worker:k1 , k2 ,, agent.ops:k3 ,bad label:k4, k5:")
    assert keys == [
        ApiKey("news-worker", "k1"),
        ApiKey("key-2", "k2"),
        ApiKey("agent.ops", "k3"),
        ApiKey("key-4", "bad label:k4"),  # invalid label -> the whole entry is the key
        ApiKey("key-5", "k5:"),  # empty value after the colon -> the whole entry is the key
    ]
    assert parse_api_keys("") == [] and parse_api_keys(None) == []


def test_configured_keys_puts_legacy_first(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy")
    monkeypatch.setattr(settings, "api_keys", "a:one,b:two")
    keys = configured_keys(settings)
    assert [k.label for k in keys] == ["api_key", "a", "b"]
    monkeypatch.setattr(settings, "api_key", "")
    assert [k.label for k in configured_keys(settings)] == ["a", "b"]


def test_resolve_caller_returns_label_or_none():
    keys = [ApiKey("api_key", "legacy"), ApiKey("a", "one"), ApiKey("b", "one")]
    assert resolve_caller("legacy", keys) == "api_key"
    assert resolve_caller("one", keys) == "a"  # first matching label wins
    assert resolve_caller("nope", keys) is None
    assert resolve_caller("", keys) is None and resolve_caller(None, keys) is None
    assert resolve_caller("legacy", []) is None


def test_describe_never_includes_values():
    text = describe([ApiKey("api_key", "s3cret"), ApiKey("agent", "t0ken")])
    assert text == "2 API key(s) configured: api_key, agent"
    assert "s3cret" not in text and "t0ken" not in text


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "legacy-key")
    monkeypatch.setattr(settings, "api_keys", "news-worker:nw-key,agent-ops:agent-key")
    from src.api.main import app

    return TestClient(app)


def test_middleware_accepts_legacy_and_listed_keys(client):
    assert client.get("/prompts").status_code == 401
    assert client.get("/prompts", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/prompts", headers={"X-API-Key": "legacy-key"}).status_code == 200
    assert client.get("/prompts", headers={"X-API-Key": "nw-key"}).status_code == 200
    assert client.get("/prompts", headers={"X-API-Key": "agent-key"}).status_code == 200
    assert client.get("/docs").status_code == 200  # public path, no key needed


def test_removing_an_entry_revokes_only_that_caller(client, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "news-worker:nw-key")
    assert client.get("/prompts", headers={"X-API-Key": "agent-key"}).status_code == 401
    assert client.get("/prompts", headers={"X-API-Key": "nw-key"}).status_code == 200
    assert client.get("/prompts", headers={"X-API-Key": "legacy-key"}).status_code == 200


def test_unset_api_keys_keeps_the_legacy_key_working(client, monkeypatch):
    # The production shape today: API_KEY set, API_KEYS absent.
    monkeypatch.setattr(settings, "api_keys", "")
    assert client.get("/prompts", headers={"X-API-Key": "legacy-key"}).status_code == 200
    assert client.get("/prompts", headers={"X-API-Key": "agent-key"}).status_code == 401


def test_enqueue_log_names_the_caller(client, caplog, monkeypatch):
    import src.tasks.registry as registry

    class FakeRef:
        workflow_run_id = "run-123"

    entry = registry.get_task("ping")
    monkeypatch.setattr(entry, "runnable", type("R", (), {"run": staticmethod(lambda *a, **k: FakeRef())})())
    with caplog.at_level(logging.INFO, logger="src.api.routers.tasks"):
        resp = client.post("/tasks", headers={"X-API-Key": "agent-key"}, json={"type": "ping", "payload": {"message": "hi"}})
    assert resp.status_code == 201, resp.text
    assert any("Enqueued ping -> run run-123 (caller=agent-ops)" in r.getMessage() for r in caplog.records)
