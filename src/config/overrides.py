"""Per-run prompt and LLM-parameter overrides.

A task run — or a request to one of the sync HTTP endpoints — may carry
`prompt_overrides` and `llm_overrides` in its payload (see
src/api/schemas/overrides_schema.py). `activate()` installs them in a
contextvars scope for the duration of the run, and the two accessors every
prompt consumer and LLM call site use read that scope first:

    prompts.get("news_claim_extract")   -> override text, else the registered default
    llm.get("claims_extract_model")     -> override value, else settings.claims_extract_model

The scope follows the run through asyncio tasks and `asyncio.to_thread`,
which copy the context. A bare ThreadPoolExecutor does not, so work
submitted to one goes through `bind_context()`.

The scope also records which prompts and settings were read and which
overrides took effect, so the run's log says exactly what a hand-test
changed — and flags an override no step consumed.

Validation (`validate_prompt_overrides`, `validate_llm_overrides`) runs at
enqueue through OverridesMixin, so a bad override is a 422 at the facade (or
an input-validation failure on the worker for a dashboard trigger), never a
KeyError inside a billed LLM call.
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set

from src.config.settings import Settings, settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

THINKING_LEVELS = ("", "minimal", "low", "medium", "high")
MAX_PROMPT_OVERRIDE_CHARS = 200_000
MAX_TEMPERATURE = 2.0
MAX_TOKENS_CEILING = 1_000_000

# A settings field is an overridable LLM parameter when its name ends in one of
# these and it is not an embedding model or a rate-limit knob.
_LLM_SUFFIXES = ("_model", "_temperature", "_thinking_level", "_max_tokens")
_LLM_EXCLUDED_SUBSTRINGS = ("embedding", "rate_limit")


def llm_setting_names() -> List[str]:
    """Every settings field a run may override, in declaration order."""
    return [
        name
        for name in Settings.model_fields
        if name.endswith(_LLM_SUFFIXES)
        and not any(s in name for s in _LLM_EXCLUDED_SUBSTRINGS)
    ]


@dataclass
class OverrideScope:
    prompt_overrides: Dict[str, str]
    llm_overrides: Dict[str, Any]
    applied: Set[str] = field(default_factory=set)  # override keys that took effect
    prompts_read: List[str] = field(default_factory=list)
    llm_read: List[str] = field(default_factory=list)

    @property
    def given(self) -> Set[str]:
        return set(self.prompt_overrides) | {f"llm:{k}" for k in self.llm_overrides}

    @property
    def unused(self) -> Set[str]:
        return self.given - self.applied

    def summary(self, label: str) -> str:
        """One log line: what this scope changed and what it read. For a DAG
        the line is per step, so an override "unused" in one step may be the
        one another step applies."""
        parts = [f"overrides[{label}]"]
        if self.given:
            parts.append(f"applied={sorted(self.applied)}")
            parts.append(f"unused={sorted(self.unused)}")
        else:
            parts.append("none given")
        parts.append(f"prompts_read={self.prompts_read}")
        parts.append(f"llm_read={self.llm_read}")
        return " ".join(parts)


_scope: contextvars.ContextVar[Optional[OverrideScope]] = contextvars.ContextVar(
    "extraction_overrides", default=None
)


def current_scope() -> Optional[OverrideScope]:
    return _scope.get()


@contextmanager
def activate(
    prompt_overrides: Optional[Mapping[str, str]] = None,
    llm_overrides: Optional[Mapping[str, Any]] = None,
) -> Iterator[OverrideScope]:
    """Install an override scope for the duration of the block."""
    scope = OverrideScope(dict(prompt_overrides or {}), dict(llm_overrides or {}))
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def overrides_of(payload: Any) -> tuple[Dict[str, str], Dict[str, Any]]:
    """The two override maps carried by a task input / request model (empty
    for a model that does not declare them)."""
    return (
        dict(getattr(payload, "prompt_overrides", None) or {}),
        dict(getattr(payload, "llm_overrides", None) or {}),
    )


@contextmanager
def activate_for(payload: Any, label: str) -> Iterator[OverrideScope]:
    """activate() from a payload model, logging the scope summary on exit."""
    prompt_ov, llm_ov = overrides_of(payload)
    with activate(prompt_ov, llm_ov) as scope:
        try:
            yield scope
        finally:
            logger.info(scope.summary(label))


def bind_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap `fn` so every call runs under a copy of the CALLER's context — the
    active override scope included. For ThreadPoolExecutor, whose worker
    threads start with an empty context. Each call gets its own Context
    object, since one Context cannot be entered by two threads at once."""
    snapshot = contextvars.copy_context()

    def run(*args: Any, **kwargs: Any) -> Any:
        def _inner() -> Any:
            for var, value in snapshot.items():
                var.set(value)
            return fn(*args, **kwargs)

        return contextvars.Context().run(_inner)

    return run


class _Prompts:
    def get(self, key: str) -> str:
        """The prompt text for `key`: the active override if one is set, else
        the registered default. Unknown keys raise KeyError (a code bug, since
        consumers pass literals; callers' keys are validated at enqueue)."""
        # Lazy: prompt modules import this module, and the registry imports them.
        from src.config.prompt_registry import PROMPTS

        entry = PROMPTS[key]
        scope = _scope.get()
        if scope is None:
            return entry.text
        if key not in scope.prompts_read:
            scope.prompts_read.append(key)
        text = scope.prompt_overrides.get(key)
        if text is None:
            return entry.text
        scope.applied.add(key)
        return text


class _Llm:
    def get(self, name: str) -> Any:
        """The LLM setting `name` (a settings field): the active override if
        one is set, else the settings value."""
        scope = _scope.get()
        if scope is None:
            return getattr(settings, name)
        if name not in scope.llm_read:
            scope.llm_read.append(name)
        if name in scope.llm_overrides:
            scope.applied.add(f"llm:{name}")
            return scope.llm_overrides[name]
        return getattr(settings, name)


prompts = _Prompts()
llm = _Llm()


# ── Validation (run at enqueue by OverridesMixin) ─────────────────────────────


def validate_prompt_overrides(value: Mapping[str, Any]) -> Dict[str, str]:
    from src.config.prompt_registry import PROMPTS, template_slots

    cleaned: Dict[str, str] = {}
    for key, text in value.items():
        entry = PROMPTS.get(key)
        if entry is None:
            raise ValueError(
                f"unknown prompt key {key!r}; GET /prompts lists the valid keys"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt override {key!r} must be a non-empty string")
        if len(text) > MAX_PROMPT_OVERRIDE_CHARS:
            raise ValueError(
                f"prompt override {key!r} is {len(text)} chars; "
                f"the maximum is {MAX_PROMPT_OVERRIDE_CHARS}"
            )
        if entry.formatted:
            try:
                slots = template_slots(text)
            except ValueError as e:
                raise ValueError(
                    f"prompt override {key!r} has unbalanced braces ({e}); this "
                    "prompt is a format template, so write literal braces as "
                    "{{ and }}"
                ) from e
            unknown = [s for s in slots if s not in entry.slots]
            if unknown:
                raise ValueError(
                    f"prompt override {key!r} uses slots {unknown} that the "
                    f"renderer does not fill; available slots: {entry.slots}"
                )
        cleaned[key] = text
    return cleaned


def validate_llm_overrides(value: Mapping[str, Any]) -> Dict[str, Any]:
    valid = llm_setting_names()
    cleaned: Dict[str, Any] = {}
    for name, v in value.items():
        if name not in valid:
            raise ValueError(
                f"unknown llm setting {name!r}; valid names: {valid}"
            )
        if isinstance(v, bool):
            raise ValueError(f"llm override {name!r} must not be a boolean")
        if name.endswith("_model"):
            if not isinstance(v, str) or not v.strip() or len(v) > 200:
                raise ValueError(f"llm override {name!r} must be a non-empty model name")
            cleaned[name] = v.strip()
        elif name.endswith("_temperature"):
            if not isinstance(v, (int, float)) or not 0 <= v <= MAX_TEMPERATURE:
                raise ValueError(
                    f"llm override {name!r} must be a number in [0, {MAX_TEMPERATURE}]"
                )
            cleaned[name] = float(v)
        elif name.endswith("_thinking_level"):
            if not isinstance(v, str) or v.strip().lower() not in THINKING_LEVELS:
                raise ValueError(
                    f"llm override {name!r} must be one of {list(THINKING_LEVELS)} "
                    "(empty disables thinking config)"
                )
            cleaned[name] = v.strip().lower()
        elif name.endswith("_max_tokens"):
            if not isinstance(v, int) or not 1 <= v <= MAX_TOKENS_CEILING:
                raise ValueError(
                    f"llm override {name!r} must be an integer in [1, {MAX_TOKENS_CEILING}]"
                )
            cleaned[name] = v
        else:  # unreachable: llm_setting_names() only admits the suffixes above
            cleaned[name] = v
    return cleaned
