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
import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Set

from src.config.settings import Settings, settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

THINKING_LEVELS = ("", "minimal", "low", "medium", "high")
MAX_PROMPT_OVERRIDE_CHARS = 200_000
# The Anthropic ceiling. The gemini_news_* temperatures also feed the Claude
# fallback paths, and no pipeline here samples above 1, so one bound serves
# every provider and an in-range override can never fail inside the call.
MAX_TEMPERATURE = 1.0
MAX_TOKENS_CEILING = 1_000_000

# The settings a run may override, explicitly. A new *_model / *_temperature /
# *_thinking_level / *_max_tokens field must be added here or to
# LLM_SETTINGS_NOT_OVERRIDABLE (tests/test_overrides.py enforces the choice),
# so nothing becomes overridable by naming accident.
LLM_SETTINGS: tuple[str, ...] = (
    "gemini_extraction_model",
    "gemini_extraction_temperature",
    "gemini_premium_model",
    "gemini_premium_temperature",
    "gemini_news_claim_model",
    "gemini_news_claim_temperature",
    "gemini_news_claim_thinking_level",
    "gemini_news_debate_model",
    "gemini_news_debate_temperature",
    "gemini_news_debate_thinking_level",
    "gemini_news_debate_review_model",
    "gemini_news_debate_review_temperature",
    "gemini_news_debate_review_thinking_level",
    "news_collection_review_model",
    "news_collection_review_temperature",
    "news_claim_claude_model",
    "news_claim_claude_max_tokens",
    "claims_extract_model",
    "claims_extract_temperature",
    "claims_extract_thinking_level",
    "claims_link_model",
    "claims_link_temperature",
    "claims_link_thinking_level",
    "claims_equivalence_model",
    "claims_equivalence_temperature",
    "claims_equivalence_thinking_level",
    "gemini_space_assignment_model",
    "gemini_space_assignment_temperature",
)
# Look like LLM knobs by name, but are not per-run model parameters.
LLM_SETTINGS_NOT_OVERRIDABLE: tuple[str, ...] = (
    "ollama_embedding_model",  # embedding dimensions are baked into stored vectors
    "premium_extraction_rate_limit_max_tokens",  # a rate-limit budget, not a call parameter
)
LLM_SETTING_SUFFIXES = ("_model", "_temperature", "_thinking_level", "_max_tokens")

_unknown = [n for n in LLM_SETTINGS if n not in Settings.model_fields]
if _unknown:
    raise RuntimeError(f"LLM_SETTINGS names fields Settings does not have: {_unknown}")


def llm_setting_names() -> List[str]:
    """Every settings field a run may override."""
    return list(LLM_SETTINGS)


@dataclass
class OverrideScope:
    prompt_overrides: Dict[str, str]
    llm_overrides: Dict[str, Any]
    # Sets, not lists: bind_context() shares one scope across pool threads and
    # set.add is atomic, so concurrent reads cannot duplicate entries.
    applied: Set[str] = field(default_factory=set)  # override keys that took effect
    prompts_read: Set[str] = field(default_factory=set)
    llm_read: Set[str] = field(default_factory=set)

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
        parts.append(f"prompts_read={sorted(self.prompts_read)}")
        parts.append(f"llm_read={sorted(self.llm_read)}")
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
    threads start with an empty context. Each call gets its own copy, since
    one Context cannot be entered by two threads at once (this is what
    asyncio.to_thread does per call)."""
    snapshot = contextvars.copy_context()

    def run(*args: Any, **kwargs: Any) -> Any:
        return snapshot.copy().run(fn, *args, **kwargs)

    return run


def with_payload_overrides(label: str):
    """Decorator for an HTTP handler (sync or async): activate the override
    scope carried by its request model — the first argument that declares
    `prompt_overrides` — for the duration of the call, and log the summary.
    Put it under the route decorator; functools.wraps keeps the signature
    FastAPI reads."""

    def payload_in(args: tuple, kwargs: dict) -> Any:
        for value in (*args, *kwargs.values()):
            if hasattr(value, "prompt_overrides"):
                return value
        return None

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with activate_for(payload_in(args, kwargs), label):
                    return await fn(*args, **kwargs)

            async_wrapper.overrides_label = label  # type: ignore[attr-defined]
            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with activate_for(payload_in(args, kwargs), label):
                return fn(*args, **kwargs)

        wrapper.overrides_label = label  # type: ignore[attr-defined]
        return wrapper

    return decorate


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
        scope.prompts_read.add(key)
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
        scope.llm_read.add(name)
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
                    f"prompt override {key!r} is not a valid format template "
                    f"({e}); this prompt is rendered with str.format, so use only "
                    f"plain {{name}} slots from {entry.slots} and write literal "
                    "braces as {{ and }}"
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
