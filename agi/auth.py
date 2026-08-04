"""E22/E23 — Multi-tenant API auth + rate limiting.

E22: HMAC-signed API keys scoped per tenant/user (rows created in api_keys table).
E23: in-memory token-bucket rate limiter (per user/key) guarding the write paths so a
     single noisy client cannot starve the memory store.
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.db import DatabasePool
from utils import log_error, log_atomic

_SECRET = os.getenv("MEDHAS_API_SECRET", "medhas-dev-api-secret").encode()

# E23 default buckets: (capacity, refill_per_sec)
DEFAULT_LIMITS = {"write": (30, 1.0), "read": (120, 4.0), "admin": (20, 0.5)}


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key(user_id: str, tenant_id: str = "default",
                     scopes: Optional[list] = None) -> Tuple[str, str]:
    """Create a new API key. Returns (plain_key, key_hash)."""
    import secrets
    plain = "mk_" + secrets.token_urlsafe(32)
    key_hash = _hash_key(plain)
    import asyncio
    asyncio.run(_persist_key(key_hash, user_id, tenant_id, scopes or ["read", "write"]))
    return plain, key_hash


async def _persist_key(key_hash: str, user_id: str, tenant_id: str, scopes: list) -> None:
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (key_hash, user_id, tenant_id, scopes)
                VALUES ($1,$2,$3,$4) ON CONFLICT (key_hash) DO NOTHING;
                """,
                key_hash, user_id, tenant_id, scopes,
            )
    except Exception as e:
        log_error(f"persist_key failed: {e}")


async def authenticate(raw_key: str) -> Optional[Dict[str, Any]]:
    """Resolve a raw API key to its record. Returns None if invalid/inactive."""
    if not raw_key or not raw_key.startswith("mk_"):
        return None
    key_hash = _hash_key(raw_key)
    try:
        async with DatabasePool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, tenant_id, scopes, is_active
                FROM api_keys WHERE key_hash = $1;
                """,
                key_hash,
            )
        if not row or not row["is_active"]:
            return None
        return {"user_id": row["user_id"], "tenant_id": row["tenant_id"],
                "scopes": list(row["scopes"] or ["read"]), "active": row["is_active"]}
    except Exception as e:
        log_error(f"authenticate failed: {e}")
        return None


def authorize(record: Optional[Dict[str, Any]], action: str) -> bool:
    if not record:
        return False
    if action == "admin":
        return "admin" in record["scopes"]
    return action in record["scopes"] or "write" in record["scopes"] or "read" in record["scopes"]


# ----------------------------------------------------------------- E23

@dataclass
class _Bucket:
    capacity: float
    refill: float
    tokens: float
    last: float


class RateLimiter:
    """Token-bucket per key, per action (E23)."""

    def __init__(self, limits: Optional[Dict[str, Tuple[int, float]]] = None) -> None:
        self.limits = limits or DEFAULT_LIMITS
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}

    def _bucket(self, key: str, action: str) -> _Bucket:
        cap, refill = self.limits.get(action, (20, 1.0))
        b = self._buckets.get((key, action))
        now = time.monotonic()
        if b is None:
            b = _Bucket(cap, refill, float(cap), now)
            self._buckets[(key, action)] = b
        else:
            elapsed = now - b.last
            b.tokens = min(b.capacity, b.tokens + elapsed * b.refill)
            b.last = now
        return b

    def allow(self, key: str, action: str = "write") -> bool:
        b = self._bucket(key, action)
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True
        return False

    def status(self, key: str, action: str = "write") -> Dict[str, float]:
        b = self._bucket(key, action)
        return {"tokens": round(b.tokens, 2), "capacity": b.capacity,
                "refill_per_sec": b.refill}


#: process-wide limiter
rate_limiter = RateLimiter()
