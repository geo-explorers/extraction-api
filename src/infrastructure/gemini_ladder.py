"""Gemini calls that step down a ladder instead of failing when a key runs out.

The rungs, in order:

1. the requested model on GEMINI_API_KEY
2. the same model on GEMINI_FALLBACK_API_KEY
3. GEMINI_FALLBACK_MODEL on GEMINI_API_KEY
4. GEMINI_FALLBACK_MODEL on GEMINI_FALLBACK_API_KEY

A rung is skipped only after an error that retrying will not clear:

- **The model's per-day request quota is used up** (a 429 that names a PerDay
  quota). Google resets these at midnight Pacific, so that model is skipped on
  that key until the reset. The quota is per project and per model, which is
  why the same key can still serve the fallback model.
- **The key no longer works at all**: the monthly spending cap, 402 (prepaid
  credit used up), 401/403, or an invalid key. The key is skipped on every
  model and tried again after KEY_RETRY, in case someone fixed it.

Any other error (a per-minute 429, a 5xx, a bad request) is raised unchanged,
so each caller's own retries behave as before. An unset fallback key or model
simply removes its rungs. Skip state is per process, which matches the
worker's and the API's scale today (same caveat as spend_guard).
"""

import hashlib
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from google import genai
from google.genai import types

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# How long a key that refused every call is skipped before it is tried again.
KEY_RETRY = timedelta(hours=1)
# Per-day quotas reset at midnight Pacific: 07:00 UTC in summer, 08:00 UTC in
# winter. 08:00 UTC is past both, at the cost of up to one extra hour on the
# next rung in summer, and needs no time-zone database in the image.
_DAILY_RESET_HOUR_UTC = 8

MODEL_OUT = "model"  # this model on this key, until the daily reset
KEY_OUT = "key"  # every model on this key, for KEY_RETRY

_lock = threading.Lock()
# (key fingerprint, model, or None for the whole key) -> (skip until, reason)
_skips: Dict[Tuple[str, Optional[str]], Tuple[datetime, str]] = {}


class GeminiUnavailable(Exception):
    """Every rung of the ladder is out for this call."""


def classify(err: Exception) -> Optional[str]:
    """MODEL_OUT or KEY_OUT when `err` ends a rung, None when it is an ordinary error.

    Reads the HTTP status from `.code`, which both google-genai errors and the
    google.api_core errors behind the LangChain path carry, and the quota or
    reason from the error text, where both put Google's error details.
    """
    code = getattr(err, "code", None)
    text = str(err).lower()
    if "spending cap" in text or "api key not valid" in text or "api_key_invalid" in text:
        return KEY_OUT
    if code in (401, 402, 403):
        return KEY_OUT
    # GenerateRequestsPerDayPerProjectPerModel, ..._requests_per_day, "per day"
    if code == 429 and "perday" in text.replace("_", "").replace(" ", ""):
        return MODEL_OUT
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def _next_daily_reset(now: datetime) -> datetime:
    reset = now.replace(hour=_DAILY_RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    return reset if reset > now else reset + timedelta(days=1)


def _skip_reason(fp: str, model: str, now: datetime) -> Optional[str]:
    with _lock:
        for scope in ((fp, None), (fp, model)):
            hit = _skips.get(scope)
            if hit and hit[0] > now:
                return hit[1]
    return None


def rungs(model: str, api_key: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """(key label, key, model) per rung, in order. `api_key` replaces GEMINI_API_KEY."""
    keys: List[Tuple[str, str]] = []
    for label, key in (("primary", api_key or settings.gemini_api_key), ("fallback", settings.gemini_fallback_api_key)):
        key = (key or "").strip()
        if key and key not in (k for _, k in keys):
            keys.append((label, key))
    models = [model]
    fallback_model = (settings.gemini_fallback_model or "").strip()
    if fallback_model and fallback_model != model:
        models.append(fallback_model)
    return [(label, key, m) for m in models for label, key in keys]


def run(model: str, call: Callable[[str, str], T], api_key: Optional[str] = None) -> T:
    """Return `call(key, model)` from the first rung that is not out.

    Steps down past a rung on a MODEL_OUT or KEY_OUT error and remembers it, so
    later calls skip that rung without spending a request on it. Raises
    GeminiUnavailable when every rung is out.
    """
    requested = rungs(model, api_key)
    if not requested:
        raise GeminiUnavailable("GEMINI_API_KEY is not configured")
    out: List[str] = []
    last: Optional[Exception] = None
    for i, (label, key, m) in enumerate(requested):
        fp = _fingerprint(key)
        now = _now()
        reason = _skip_reason(fp, m, now)
        if reason is None:
            try:
                result = call(key, m)
            except Exception as err:
                scope = classify(err)
                if scope is None:
                    raise
                last = err
                if scope == MODEL_OUT:
                    until, reason = _next_daily_reset(now), f"per-day quota for {m} used up"
                else:
                    until, reason = now + KEY_RETRY, f"key refused (HTTP {getattr(err, 'code', None) or '?'})"
                with _lock:
                    _skips[(fp, m if scope == MODEL_OUT else None)] = (until, reason)
                logger.warning(
                    f"Gemini ladder: {m} on the {label} key is out ({reason}); "
                    f"skipping it until {until:%Y-%m-%d %H:%M} UTC and stepping down"
                )
            else:
                if i:
                    logger.info(f"Gemini ladder: {model} answered by {m} on the {label} key (rung {i + 1})")
                return result
        out.append(f"{m} on the {label} key: {reason}")
    raise GeminiUnavailable(f"every Gemini rung is out for {model}: " + "; ".join(out)) from last


class _Models:
    def __init__(self, owner: "GeminiClient"):
        self._owner = owner

    def generate_content(self, *, model: str, **kwargs: Any) -> types.GenerateContentResponse:
        return self._owner._generate_content(model, kwargs)


class GeminiClient:
    """Stands in for genai.Client where only `models.generate_content` is used.

    Each call runs down the ladder. The response's `model_version` is set to the
    model that answered, so a caller can report a step down to the fallback model.
    """

    def __init__(self, api_key: Optional[str] = None, http_options: Optional[types.HttpOptions] = None):
        self._api_key = api_key
        self._http_options = http_options
        self._clients: Dict[str, genai.Client] = {}
        self.models = _Models(self)

    def _client(self, key: str) -> genai.Client:
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key, http_options=self._http_options)
        return self._clients[key]

    def _generate_content(self, model: str, kwargs: Dict[str, Any]) -> types.GenerateContentResponse:
        def call(key: str, m: str) -> types.GenerateContentResponse:
            response = self._client(key).models.generate_content(model=m, **kwargs)
            response.model_version = m
            return response

        return run(model, call, self._api_key)
