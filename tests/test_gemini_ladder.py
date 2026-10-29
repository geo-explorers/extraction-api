"""The Gemini ladder without Gemini: which errors end a rung, the rung order, the
skips that keep later calls off a dead rung, their expiry, and the client and
LangChain wrappers that route through it."""

from datetime import datetime, timedelta, timezone

import pytest
from google.api_core import exceptions as api_core
from google.genai import errors, types

from src.infrastructure import gemini_ladder as ladder
from src.infrastructure.gemini_ladder import KEY_OUT, MODEL_OUT, GeminiClient, GeminiUnavailable, classify

PRO, FLASH = "gemini-3.1-pro-preview", "gemini-3.8-flash"


def _google_error(code: int, message: str, quota_id: str = "") -> errors.APIError:
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}] if quota_id else []
    status = {429: "RESOURCE_EXHAUSTED", 403: "PERMISSION_DENIED", 401: "UNAUTHENTICATED", 400: "INVALID_ARGUMENT"}.get(code, "INTERNAL")
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": status, "details": details}})


def daily():
    return _google_error(429, "You exceeded your current quota", "GenerateRequestsPerDayPerProjectPerModel")


def per_minute():
    return _google_error(429, "You exceeded your current quota", "GenerateRequestsPerMinutePerProjectPerModel")


def spending_cap():
    return _google_error(429, "Your project has exceeded its monthly spending cap. Please go to AI Studio")


@pytest.fixture(autouse=True)
def two_keys(monkeypatch):
    monkeypatch.setattr(ladder.settings, "gemini_api_key", "key-a")
    monkeypatch.setattr(ladder.settings, "gemini_fallback_api_key", "key-b")
    monkeypatch.setattr(ladder.settings, "gemini_fallback_model", FLASH)
    ladder._skips.clear()
    yield
    ladder._skips.clear()


class Fake:
    """call(key, model) that raises the scripted error for a rung, else answers."""

    def __init__(self, **fail):
        self.fail = {tuple(k.split("__")): v for k, v in fail.items()}
        self.calls = []

    def __call__(self, key, model):
        self.calls.append((key, model))
        err = self.fail.get((key.replace("-", "_"), "pro" if model == PRO else "flash"))
        if err:
            raise err()
        return f"{model} on {key}"


# ── which errors end a rung ──────────────────────────────────────────────────


def test_classify_reads_the_quota_and_status_of_google_errors():
    assert classify(daily()) == MODEL_OUT
    assert classify(spending_cap()) == KEY_OUT
    assert classify(_google_error(403, "Permission denied: Consumer has been suspended")) == KEY_OUT
    assert classify(_google_error(401, "API keys are not supported by this API")) == KEY_OUT
    assert classify(_google_error(400, "API key not valid. Please pass a valid API key.")) == KEY_OUT
    assert classify(_google_error(402, "Your prepayment credits are depleted")) == KEY_OUT
    # Errors a retry can clear stay with the caller's own retries
    assert classify(per_minute()) is None
    assert classify(_google_error(500, "Internal error")) is None
    assert classify(_google_error(400, "Request contains an invalid argument.")) is None
    assert classify(ValueError("Empty response from Gemini API")) is None


def test_classify_reads_the_langchain_path_errors_too():
    assert classify(api_core.ResourceExhausted("Quota exceeded for quota metric 'generate_content_requests_per_day'")) == MODEL_OUT
    assert classify(api_core.ResourceExhausted("Quota exceeded for requests per minute")) is None
    assert classify(api_core.PermissionDenied("Consumer suspended")) == KEY_OUT


# ── rung order and skips ─────────────────────────────────────────────────────


def test_rungs_go_key_by_key_then_to_the_fallback_model():
    assert [(label, m) for label, _, m in ladder.rungs(PRO)] == [
        ("primary", PRO), ("fallback", PRO), ("primary", FLASH), ("fallback", FLASH),
    ]
    # A model that is already the fallback model has no model rung
    assert [m for _, _, m in ladder.rungs(FLASH)] == [FLASH, FLASH]


def test_unset_fallbacks_leave_one_rung(monkeypatch):
    monkeypatch.setattr(ladder.settings, "gemini_fallback_api_key", None)
    monkeypatch.setattr(ladder.settings, "gemini_fallback_model", "")
    assert ladder.rungs(PRO) == [("primary", "key-a", PRO)]
    with pytest.raises(GeminiUnavailable) as e:
        ladder.run(PRO, Fake(key_a__pro=daily))
    assert isinstance(e.value.__cause__, errors.APIError)


def test_a_used_up_day_moves_to_the_fallback_key_and_stays_there():
    fake = Fake(key_a__pro=daily)
    assert ladder.run(PRO, fake) == f"{PRO} on key-b"
    assert ladder.run(PRO, fake) == f"{PRO} on key-b"
    # The dead rung was asked once; the second call went straight to key B
    assert fake.calls == [("key-a", PRO), ("key-b", PRO), ("key-b", PRO)]


def test_a_daily_limit_is_per_model_so_the_same_key_still_serves_flash():
    fake = Fake(key_a__pro=daily, key_b__pro=daily)
    assert ladder.run(PRO, fake) == f"{FLASH} on key-a"
    assert ladder.run(FLASH, fake) == f"{FLASH} on key-a"


def test_a_dead_key_is_skipped_on_every_model():
    fake = Fake(key_a__pro=spending_cap, key_b__pro=daily)
    assert ladder.run(PRO, fake) == f"{FLASH} on key-b"
    assert fake.calls == [("key-a", PRO), ("key-b", PRO), ("key-b", FLASH)]


def test_an_ordinary_error_is_raised_and_leaves_the_rung_open():
    fake = Fake(key_a__pro=per_minute)
    with pytest.raises(errors.ClientError):
        ladder.run(PRO, fake)
    assert ladder._skips == {}


def test_every_rung_out_raises_and_later_calls_spend_no_request():
    fake = Fake(key_a__pro=daily, key_b__pro=daily, key_a__flash=daily, key_b__flash=daily)
    with pytest.raises(GeminiUnavailable, match="every Gemini rung is out"):
        ladder.run(PRO, fake)
    asked = len(fake.calls)
    with pytest.raises(GeminiUnavailable):
        ladder.run(PRO, fake)
    assert len(fake.calls) == asked == 4


def test_skips_end_at_the_daily_reset_and_after_key_retry(monkeypatch):
    t = datetime(2026, 10, 28, 19, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(ladder, "_now", lambda: t)
    ladder.run(PRO, Fake(key_a__pro=daily))
    ladder.run(FLASH, Fake(key_a__flash=spending_cap))
    assert ladder._skips[(ladder._fingerprint("key-a"), PRO)][0] == datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc)
    assert ladder._skips[(ladder._fingerprint("key-a"), None)][0] == t + ladder.KEY_RETRY

    fake = Fake()
    monkeypatch.setattr(ladder, "_now", lambda: datetime(2026, 10, 29, 8, 1, tzinfo=timezone.utc))
    assert ladder.run(PRO, fake) == f"{PRO} on key-a"


def test_the_daily_reset_is_the_next_0800_utc():
    at = lambda h: datetime(2026, 10, 29, h, 30, tzinfo=timezone.utc)
    assert ladder._next_daily_reset(at(6)) == datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc)
    assert ladder._next_daily_reset(at(9)) == datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)


def test_an_explicit_key_replaces_the_primary():
    assert [k for _, k, _ in ladder.rungs(FLASH, api_key="key-x")] == ["key-x", "key-b"]


# ── the wrappers ─────────────────────────────────────────────────────────────


def test_client_answers_from_the_fallback_model_and_names_it(monkeypatch):
    seen = []

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            self.models = self
            self.key = api_key

        def generate_content(self, *, model, contents, config=None):
            seen.append((self.key, model))
            if model == PRO:
                raise daily()
            return types.GenerateContentResponse(model_version=f"{model}-001")

    monkeypatch.setattr(ladder.genai, "Client", FakeClient)
    response = GeminiClient().models.generate_content(model=PRO, contents="x")
    assert response.model_version == FLASH
    assert seen == [("key-a", PRO), ("key-b", PRO), ("key-a", FLASH)]


def test_langchain_chain_routes_through_the_ladder(monkeypatch):
    from langchain_core.messages import AIMessage

    from src.api.utils import llm_model

    seen = []

    class FakeChat:
        def __init__(self, model, temperature, api_key, max_retries):
            self.model, self.key = model, api_key

        def invoke(self, prompt_value):
            seen.append((self.key, self.model))
            if self.key == "key-a":
                raise api_core.PermissionDenied("Consumer suspended")
            return AIMessage(content=f"hello from {self.model}")

    monkeypatch.setattr(llm_model, "ChatGoogleGenerativeAI", FakeChat)
    chain = llm_model.build_chain("Say hello to {name}", model_name=FLASH, temperature=0.0)
    assert chain.invoke({"name": "Geo"}) == f"hello from {FLASH}"
    assert seen == [("key-a", FLASH), ("key-b", FLASH)]
