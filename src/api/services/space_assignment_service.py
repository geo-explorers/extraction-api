"""Assign canonical spaces to entities (thin wrapper over the generic classifier).

Builds the assignment prompt from the fetched spaces + entities, runs the generic
Gemini classifier, then joins each returned space id back to its ``Space.name`` to
produce table-ready ``AssignedRow`` objects (one per entity, in input order). No
Hatchet import; the blocking Gemini call is offloaded by the caller.
"""

from src.api.schemas.geo_spaces_schema import AssignedRow, Entity, Space
from src.api.services.llm_classify_service import classify_items
from src.config.prompts.space_assignment_prompt import build_space_assignment_prompt
from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def assign_spaces(spaces: list[Space], entities: list[Entity]) -> list[AssignedRow]:
    """Assign 0+ canonical spaces to each entity via Gemini, returning one
    table-ready row per entity (in input order).

    Synchronous (blocking Gemini call); offload via ``asyncio.to_thread`` on an
    event loop.
    """
    # One row per entity, seeded unassigned; dict preserves input order.
    rows_by_entity: dict[str, AssignedRow] = {
        e.id: AssignedRow(entity_id=e.id, entity_name=e.name, entity_type=e.type)
        for e in entities
    }
    if not entities:
        return []
    if not spaces:
        # No vocabulary to assign against — return unassigned rows rather than
        # calling the LLM with an empty space list.
        logger.warning("assign_spaces: no spaces provided; returning unassigned rows")
        return list(rows_by_entity.values())

    valid_space_ids = {s.id for s in spaces}
    name_by_id = {s.id: s.name for s in spaces}

    prompt = build_space_assignment_prompt(spaces, entities)
    assignments = classify_items(
        prompt=prompt,
        model=settings.gemini_space_assignment_model,
        temperature=settings.gemini_space_assignment_temperature,
    )

    for a in assignments:
        row = rows_by_entity.get(a.item_id)
        if row is None:
            continue  # model returned an unknown entity id — ignore
        seen: set[str] = set()
        for sid in a.category_ids:
            # Keep only valid space ids; dedup; preserve model order.
            if sid in valid_space_ids and sid not in seen:
                seen.add(sid)
                row.assigned_space_ids.append(sid)
                row.assigned_space_names.append(name_by_id[sid])

    assigned = sum(1 for r in rows_by_entity.values() if r.assigned_space_ids)
    logger.info(f"assign_spaces: {assigned}/{len(entities)} entities got >=1 space")
    return list(rows_by_entity.values())
