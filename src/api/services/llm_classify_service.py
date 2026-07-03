"""Generic LLM classification helper: assign each item to 0+ categories from a
supplied vocabulary, with Gemini schema-enforced JSON output.

Engine-agnostic and reusable — no Hatchet import. Space-assignment is one caller;
the same helper works for topics, tags, or any taxonomy. The caller builds a
prompt that lists the categories and items and instructs the model to return, per
item, the ids of the categories that apply; this module constrains the response
to ``ClassificationResult`` so parsing is guaranteed.

The google-genai call is blocking; callers on an event loop offload via
``asyncio.to_thread``.
"""

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

_GEMINI_TIMEOUT_MS = 60 * 3 * 1000


class ItemAssignment(BaseModel):
    """One item's assigned category ids (0 or more)."""

    item_id: str = Field(description="The id of the item being classified")
    category_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the categories assigned to this item (0 or more). "
        "Empty list if none apply. Use only ids from the provided vocabulary.",
    )


class ClassificationResult(BaseModel):
    assignments: list[ItemAssignment] = Field(default_factory=list)


def classify_items(
    *,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> list[ItemAssignment]:
    """Run a Gemini schema-enforced classification and return the per-item
    assignments. ``prompt`` must instruct the model to return, for each item, its
    ``item_id`` and the ``category_ids`` that apply.

    Synchronous (blocking). Offload via ``asyncio.to_thread`` on an event loop.
    """
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY required for LLM classification")

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=_GEMINI_TIMEOUT_MS),
    )
    config = types.GenerateContentConfig(
        temperature=(
            settings.gemini_space_assignment_temperature
            if temperature is None
            else temperature
        ),
        response_mime_type="application/json",
        response_schema=ClassificationResult,
    )
    response = client.models.generate_content(
        model=model or settings.gemini_space_assignment_model,
        contents=prompt,
        config=config,
    )
    if not response or not response.text or not response.text.strip():
        raise ValueError("Empty response from Gemini during classification")

    result = ClassificationResult.model_validate_json(response.text)
    logger.info(f"classify_items: {len(result.assignments)} assignments returned")
    return result.assignments
