"""Payload fields that carry per-run prompt and LLM overrides.

Every task input model — and the request models the sync HTTP endpoints share
with the tasks — inherits OverridesMixin, so any run can be triggered from the
Hatchet dashboard, the /tasks facade or curl with a hand-edited prompt or a
different model, and nothing else changes. Both maps default to empty, so
existing callers are unaffected.

Validation happens here, at enqueue: unknown keys, malformed templates and
out-of-range values are rejected before anything is queued. GET /prompts lists
the prompt keys (with their slots and default text) and the LLM setting names.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

# For a handler that forwards its input verbatim to another service: the
# override maps are for THIS service's run only and must not travel on.
OVERRIDE_FIELDS = frozenset({"prompt_overrides", "llm_overrides"})


class OverridesMixin(BaseModel):

    prompt_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Prompt key -> replacement text, for this run only. A formatted "
            "prompt's override may use only the slots its default uses. "
            "GET /prompts lists keys, slots and default texts."
        ),
    )
    llm_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "LLM setting name -> value, for this run only: any *_model, "
            "*_temperature, *_thinking_level or *_max_tokens setting "
            "(e.g. claims_extract_model). GET /prompts lists the names."
        ),
    )

    @field_validator("prompt_overrides")
    @classmethod
    def _validate_prompt_overrides(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            return {}
        from src.config.overrides import validate_prompt_overrides  # lazy: avoids import cycles

        return validate_prompt_overrides(v)

    @field_validator("llm_overrides")
    @classmethod
    def _validate_llm_overrides(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            return {}
        from src.config.overrides import validate_llm_overrides

        return validate_llm_overrides(v)
