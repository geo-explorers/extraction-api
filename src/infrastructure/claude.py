"""Helpers every Claude path shares: the text of a Messages response, and the JSON in it.

One parser for all of them, so the news services and the moment reel tasks read the same Claude
output the same way.
"""

import json
import re
from typing import Any


def claude_text(message) -> str:
    """Concatenate the text blocks of an Anthropic Messages response."""
    return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def parse_json(raw: str) -> Any:
    """The JSON in `raw`: the whole text, or else its first fenced block."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw or "", re.DOTALL | re.I)
        if not match:
            raise
        return json.loads(match.group(1))


__all__ = ["claude_text", "parse_json"]
