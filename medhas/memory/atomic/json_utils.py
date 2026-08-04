"""Leaf JSON helpers for memory layers.

This module deliberately imports nothing from the ``memory`` package so it can be
imported by sibling modules (insert_fact, search_facts, memory_crud) without
creating a circular import through ``memory.atomic.__init__``.
"""
import json
from typing import Any


def _coerce_json(value: Any) -> Any:
    """Normalize a JSONB column read back from asyncpg into a Python object.

    asyncpg returns jsonb/text columns as Python ``str`` unless a JSON codec is
    registered on the connection. Several call sites used ``dict(row["metadata"])``
    which raises ``ValueError: dictionary update sequence element #0 has length 1``
    when the value is a JSON *string*. We accept str, bytes, or already-decoded
    objects and always return a decoded structure (or None for empty/null).
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        if text.strip() == "":
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
    return value
