"""Prompt builder for entity -> space assignment.

Given the canonical spaces (the vocabulary) and the entities to classify, produce
a prompt instructing the model to assign each entity to 0 or more space ids. Built
by string concatenation (not ``str.format``) so any literal JSON braces in the
embedded entity payload are safe.
"""

import json

from src.api.schemas.geo_spaces_schema import Entity, Space

_SYSTEM = (
    "You are a knowledge-graph curator. Assign each ENTITY to the SPACES it belongs to.\n"
    "\n"
    "A space is a topical collection. Assign a space to an entity only when the "
    "entity's PRIMARY subject matter clearly falls within that space's scope — judge "
    "by the space's DESCRIPTION, not just its name.\n"
    "\n"
    "An entity may belong to zero, one, or several spaces:\n"
    "- Zero is common and correct. If no space is a clear fit, return an empty list. "
    "Do NOT force an assignment.\n"
    "- Assign several only when the entity genuinely spans them (e.g. a project that "
    "is fundamentally about both crypto and AI -> both).\n"
    "\n"
    "Rules:\n"
    "- Base the decision on the entity's name, type, description and properties vs. "
    "each space's DESCRIPTION.\n"
    "- Do NOT assign for tangential or incidental relevance (e.g. a company that "
    "merely *uses* a technology is not in that technology's space unless that is its "
    "core subject).\n"
    "- When uncertain, prefer precision: leave it unassigned rather than assign a "
    "weak match.\n"
    "- Use ONLY the exact space ids listed. Never invent, modify, or guess an id.\n"
    "- Return every entity exactly once, echoing its id verbatim as item_id.\n"
)


def _space_line(s: Space) -> str:
    detail = s.description.strip()
    if s.entity_types:
        types_str = ", ".join(s.entity_types[:20])
        detail = (
            f"{detail} | contains types: {types_str}"
            if detail
            else f"contains types: {types_str}"
        )
    return f'- id="{s.id}" name="{s.name}"' + (f" — {detail}" if detail else "")


def _entity_block(e: Entity) -> dict:
    # Compact, token-frugal view; include description + any non-empty properties.
    block: dict = {"id": e.id, "name": e.name, "type": e.type}
    if e.description:
        block["description"] = e.description
    props = {
        k: v for k, v in (e.properties or {}).items() if v not in (None, "", [], {})
    }
    if props:
        block["properties"] = props
    return block


def build_space_assignment_prompt(spaces: list[Space], entities: list[Entity]) -> str:
    space_lines = "\n".join(_space_line(s) for s in spaces)
    entities_json = json.dumps(
        [_entity_block(e) for e in entities], ensure_ascii=False, indent=2
    )
    return (
        _SYSTEM
        + "\n### SPACES (assign only these exact ids)\n"
        + space_lines
        + "\n\n### ENTITIES (JSON)\n"
        + entities_json
        + "\n\n### OUTPUT\n"
        + "For every entity return an object with `item_id` (the entity id, verbatim), "
        + "`reasoning` (one brief sentence justifying the choice), and `category_ids` "
        + "(the list of space ids it belongs to; empty list if none apply). Use only "
        + "ids from ### SPACES. Include every entity exactly once."
    )
