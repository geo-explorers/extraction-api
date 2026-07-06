"""Assign canonical spaces to entities (thin wrapper over the generic classifier).

Builds the assignment prompt from the fetched spaces + entities, runs the generic
Gemini classifier, then joins each returned space id back to its ``Space.name`` to
produce table-ready ``AssignedRow`` objects (one per entity, in input order). No
Hatchet import; the blocking Gemini call is offloaded by the caller.
"""

from concurrent.futures import ThreadPoolExecutor

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

    # Batch entities across Gemini calls so a large entity set never overflows a
    # single prompt. Each batch gets the full spaces vocabulary + its slice of
    # entities.
    batch_size = max(1, settings.space_assignment_batch_size)
    concurrency = max(1, settings.space_assignment_concurrency)
    chunks = [
        entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
    ]

    def _classify(chunk: list[Entity]):
        return classify_items(
            prompt=build_space_assignment_prompt(spaces, chunk),
            model=settings.gemini_space_assignment_model,
            temperature=settings.gemini_space_assignment_temperature,
        )

    # Run the per-batch Gemini calls in a bounded thread pool (each is a blocking
    # SDK request) so large entity sets don't run serially. Results are merged
    # sequentially afterwards, so no lock is needed on rows_by_entity.
    if concurrency == 1 or len(chunks) <= 1:
        results = [_classify(c) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(chunks))) as pool:
            results = list(pool.map(_classify, chunks))

    for assignments in results:
        for a in assignments:
            row = rows_by_entity.get(a.item_id)
            if row is None:
                continue  # model returned an unknown entity id — ignore
            seen = set(row.assigned_space_ids)
            for sid in a.category_ids:
                # Keep only valid space ids; dedup; preserve model order.
                if sid in valid_space_ids and sid not in seen:
                    seen.add(sid)
                    row.assigned_space_ids.append(sid)
                    row.assigned_space_names.append(name_by_id[sid])

    assigned = sum(1 for r in rows_by_entity.values() if r.assigned_space_ids)
    logger.info(
        f"assign_spaces: {assigned}/{len(entities)} entities got >=1 space "
        f"({len(chunks)} batches, batch_size={batch_size}, concurrency={concurrency})"
    )
    return list(rows_by_entity.values())
