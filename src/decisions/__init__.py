"""Decision models: the interface (base.py) and the providers behind it.

`get_decision_model(provider)` is the one place a provider name becomes an
implementation. Adding a provider is one entry in `_PROVIDERS`.
"""

from functools import lru_cache
from typing import Callable, Dict

from src.decisions.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionError,
    DecisionModel,
    Question,
    YesNo,
    YesNoAnswer,
)
from src.decisions.perplexity import PerplexityDecisionModel

_PROVIDERS: Dict[str, Callable[[], DecisionModel]] = {
    "perplexity": PerplexityDecisionModel,
}


def provider_names() -> list[str]:
    return list(_PROVIDERS)


@lru_cache(maxsize=None)
def get_decision_model(provider: str) -> DecisionModel:
    """The decision model for `provider`, built once per process. Construction
    is where a missing API key surfaces, so it happens on first use, not at
    import."""
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(f"unknown decision provider {provider!r}; known providers: {provider_names()}")
    return factory()


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionError",
    "DecisionModel",
    "Question",
    "YesNo",
    "YesNoAnswer",
    "get_decision_model",
    "provider_names",
]
