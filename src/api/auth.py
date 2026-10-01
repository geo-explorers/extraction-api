"""API-key authentication: the legacy single key plus a revocable per-caller list.

    API_KEY   the original single key; still accepted, so existing callers
              (pg-migrations, news-worker, ...) keep working unchanged.
    API_KEYS  comma-separated extra keys, each "label:key", "label:ro:key"
              or a bare key, e.g. "news-worker:k1,agent-ops:ro:k2". Any
              listed key is accepted; deleting an entry revokes that caller
              alone. The optional "ro" scope makes a key read-only: it may
              only make GET requests (inspect prompts, runs, inputs, step
              outputs, logs) and can never enqueue, extract or export.

Comparison is constant-time and every configured key is checked, so response
timing reveals neither the key nor which entry matched. The label identifies
the caller in logs (request.state.api_caller); key values are never logged.
"""

import hmac
import re
from dataclasses import dataclass
from typing import List, Optional

LEGACY_LABEL = "api_key"
_LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")
SCOPE_READ_WRITE = "rw"
SCOPE_READ_ONLY = "ro"
SCOPES = (SCOPE_READ_WRITE, SCOPE_READ_ONLY)
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class ApiKey:
    label: str
    value: str
    scope: str = SCOPE_READ_WRITE

    @property
    def read_only(self) -> bool:
        return self.scope == SCOPE_READ_ONLY


def parse_api_keys(raw: str) -> List[ApiKey]:
    """Parse the API_KEYS setting. An entry is "label:key", "label:ro:key"
    (or "label:rw:key") or a bare key; bare keys get positional labels
    (key-1, key-2, ...) and every key is read-write unless marked "ro".
    Blank entries are ignored."""
    keys: List[ApiKey] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        label, sep, rest = item.partition(":")
        if sep and _LABEL_RE.fullmatch(label.strip()) and rest.strip():
            scope, sep2, value = rest.partition(":")
            if sep2 and scope.strip() in SCOPES and value.strip():
                keys.append(ApiKey(label.strip(), value.strip(), scope.strip()))
            else:
                keys.append(ApiKey(label.strip(), rest.strip()))
        else:
            keys.append(ApiKey(f"key-{len(keys) + 1}", item))
    return keys


def configured_keys(settings) -> List[ApiKey]:
    """Every key the API accepts right now: the legacy API_KEY first, then API_KEYS."""
    keys: List[ApiKey] = []
    if settings.api_key:
        keys.append(ApiKey(LEGACY_LABEL, settings.api_key))
    keys.extend(parse_api_keys(settings.api_keys))
    return keys


def resolve_key(presented: Optional[str], keys: List[ApiKey]) -> Optional[ApiKey]:
    """The configured key `presented` matches, or None. Constant-time per key,
    and every key is compared regardless of an earlier match."""
    if not presented:
        return None
    match: Optional[ApiKey] = None
    presented_bytes = presented.encode()
    for key in keys:
        if hmac.compare_digest(presented_bytes, key.value.encode()) and match is None:
            match = key
    return match


def resolve_caller(presented: Optional[str], keys: List[ApiKey]) -> Optional[str]:
    """The label of the matching key, or None."""
    key = resolve_key(presented, keys)
    return key.label if key else None


def describe(keys: List[ApiKey]) -> str:
    """Startup summary: count, labels and scopes only, never values."""
    names = ", ".join(k.label + (" (ro)" if k.read_only else "") for k in keys) or "none"
    return f"{len(keys)} API key(s) configured: {names}"
