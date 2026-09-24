"""Read-only catalog of prompts and LLM settings, for hand-testing overrides.

    GET /prompts               -> every prompt key (module, slots, size) and
                                  every overridable LLM setting with its value
    GET /prompts/{key}         -> one prompt's default text (JSON)
    GET /prompts/{key}?format=text -> the same as text/plain, for copy-paste

The loop is: fetch a prompt here, edit it, and pass it back as
`prompt_overrides[key]` in a task payload (Hatchet dashboard or POST /tasks)
or in a sync endpoint's request body.
"""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import PlainTextResponse

from src.config.overrides import llm_setting_names
from src.config.prompt_registry import PROMPTS
from src.config.settings import settings

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("")
def list_prompts() -> dict:
    return {
        "prompts": [
            {
                "key": e.key,
                "module": e.module,
                "formatted": e.formatted,
                "slots": e.slots,
                "chars": len(e.text),
            }
            for e in PROMPTS.values()
        ],
        "llm_settings": [
            {"name": name, "value": getattr(settings, name)}
            for name in llm_setting_names()
        ],
        "usage": {
            "prompt_overrides": "{<key>: <replacement text>} in a task payload or request body",
            "llm_overrides": "{<setting name>: <value>} in a task payload or request body",
        },
    }


@router.get("/{key}")
def get_prompt(key: str, format: str = Query(default="json", pattern="^(json|text)$")):
    entry = PROMPTS.get(key)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown prompt key: {key}"
        )
    if format == "text":
        return PlainTextResponse(entry.text)
    return {
        "key": entry.key,
        "module": entry.module,
        "formatted": entry.formatted,
        "slots": entry.slots,
        "text": entry.text,
    }
