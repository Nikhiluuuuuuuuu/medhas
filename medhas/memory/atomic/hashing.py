"""Leaf helper: content hashing for fact dedup (no package imports — avoids cycles).

Mirrors Mem0's md5 content hash (mem0/memory/main.py:991) used for exact-dedup.
Kept in a leaf module so both the atomic fact layer and the async extractor can
share it without creating a circular import.
"""

import hashlib


def content_hash(text: str) -> str:
    """Stable md5 of normalized fact text (Mem0 main.py:991 md5 dedup)."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
