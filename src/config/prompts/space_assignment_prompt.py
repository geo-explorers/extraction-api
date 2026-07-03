"""Prompt builder for entity -> space assignment.

Given the canonical spaces (the vocabulary) and the entities to classify, produce
a prompt instructing the model to assign each entity to 0 or more space ids. Built
by string concatenation (not ``str.format``) so any literal JSON braces in the
embedded entity payload are safe.
"""

import json

from src.api.schemas.geo_spaces_schema import Entity, Space

_SYSTEM = (
    "You are a knowledge-graph curator. You are given a fixed set of canonical "
    "SPACES (topical knowledge-graph collections) and a list of ENTITIES. For "
    "each entity, decide which spaces it belongs to based on its name, type, and "
    "properties. An entity may belong to zero, one, or several spaces. Only use "
    "space ids from the provided list — never invent a space. Prefer precision: "
    "assign a space only when the entity clearly fits its topic.\n"
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
        + "\nSPACES (assign only these ids):\n"
        + space_lines
        + "\n\nENTITIES to classify (JSON):\n"
        + entities_json
        + "\n\nReturn, for every entity, an object with its `item_id` (the entity "
        + "id) and `category_ids` (the list of space ids it belongs to; empty list "
        + "if none apply). Include every entity exactly once."
    )
