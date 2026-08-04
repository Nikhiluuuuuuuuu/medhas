"""E31 — Sensory / perceptual buffer + E6 multimodal ingestion.

A short-TTL pre-attentive staging area. Raw percepts (captions, OCR text, audio
transcript fragments, tool observations) land here first; only what survives an
attention filter is promoted into episodic/semantic memory. Everything else
expires. This prevents raw perceptual noise from polluting long-term storage.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from infrastructure.db import DatabasePool
from infrastructure.llm import FastEmbeddingProvider
from utils import log_atomic, log_error, measure_latency

embedder = FastEmbeddingProvider()

DEFAULT_TTL_SECONDS = 900          # 15 minutes in the buffer
PROMOTE_MIN_CHARS = 12
SUPPORTED_MODALITIES = ("text", "image", "audio", "video", "document", "tool")


async def buffer_percept(
    user_id: str,
    raw_caption: str,
    *,
    modality: str = "text",
    session_id: Optional[UUID] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    embed: bool = True,
) -> Optional[UUID]:
    """Stage a raw percept in the sensory buffer (E31/E6)."""
    text = (raw_caption or "").strip()
    if not text:
        return None
    if modality not in SUPPORTED_MODALITIES:
        modality = "text"
    expires = datetime.now(timezone.utc) + timedelta(seconds=max(30, ttl_seconds))
    async with measure_latency("agi.sensory.buffer_percept"):
        try:
            vec = None
            if embed:
                try:
                    e = await embedder.embed_text(text)
                    vec = f"[{','.join(str(x) for x in e)}]"
                except Exception:
                    vec = None
            async with DatabasePool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO percept_buffer
                        (user_id, session_id, modality, raw_caption, embedding, expires_at)
                    VALUES ($1,$2,$3,$4,$5::vector,$6)
                    RETURNING id;
                    """,
                    user_id, session_id, modality, text, vec, expires,
                )
            return row["id"] if row else None
        except Exception as e:
            log_error(f"buffer_percept failed: {e}")
            return None


def attention_filter(caption: str, modality: str = "text") -> bool:
    """Decide whether a buffered percept deserves promotion to long-term memory.

    Rejects trivially short, boilerplate or purely decorative perceptual content.
    """
    text = (caption or "").strip()
    if len(text) < PROMOTE_MIN_CHARS:
        return False
    lc = text.lower()
    noise = ("blank image", "no text detected", "screenshot", "untitled",
             "image.png", "loading", "n/a", "none")
    if any(lc == n or lc.startswith(n) for n in noise):
        return False
    # Require some information-bearing content: a verb-ish token or a number/proper noun.
    if len(text.split()) < 3 and not any(c.isdigit() for c in text):
        return False
    return True


async def promote_percepts(
    user_id: str,
    session_id: Optional[UUID] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Run the attention filter over unexpired percepts and promote survivors to facts."""
    from memory.atomic import insert_fact  # local import avoids circular dependency

    promoted: List[str] = []
    dropped = 0
    async with measure_latency("agi.sensory.promote_percepts"):
        try:
            async with DatabasePool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, modality, raw_caption
                    FROM percept_buffer
                    WHERE user_id=$1 AND expires_at > CURRENT_TIMESTAMP
                      AND ($2::uuid IS NULL OR session_id = $2)
                    ORDER BY created_at ASC LIMIT $3;
                    """,
                    user_id, session_id, limit,
                )
            for r in rows:
                if attention_filter(r["raw_caption"], r["modality"]):
                    text = (f"[{r['modality'].capitalize()}] {r['raw_caption']}"
                            if r["modality"] != "text" else r["raw_caption"])
                    try:
                        await insert_fact(
                            user_id, text, session_id=session_id,
                            memory_type="episodic",
                            metadata={"source": "sensory_buffer", "modality": r["modality"]},
                        )
                        promoted.append(text[:80])
                    except Exception as ie:
                        log_error(f"percept promotion insert failed: {ie}")
                else:
                    dropped += 1
                async with DatabasePool.acquire() as conn:
                    await conn.execute("DELETE FROM percept_buffer WHERE id=$1;", r["id"])
            if promoted or dropped:
                log_atomic(f"E31 sensory: promoted={len(promoted)} filtered={dropped}")
            return {"promoted": len(promoted), "filtered": dropped, "items": promoted}
        except Exception as e:
            log_error(f"promote_percepts failed: {e}")
            return {"promoted": 0, "filtered": 0, "error": str(e)}


async def sweep_expired() -> int:
    """Delete percepts past their TTL (buffer hygiene)."""
    try:
        async with DatabasePool.acquire() as conn:
            s = await conn.execute("DELETE FROM percept_buffer WHERE expires_at <= CURRENT_TIMESTAMP;")
            return int(s.split()[-1]) if s else 0
    except Exception as e:
        log_error(f"sweep_expired failed: {e}")
        return 0


async def list_buffer(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        async with DatabasePool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, modality, raw_caption, expires_at, created_at
                FROM percept_buffer
                WHERE user_id=$1 AND expires_at > CURRENT_TIMESTAMP
                ORDER BY created_at DESC LIMIT $2;
                """,
                user_id, limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:
        log_error(f"list_buffer failed: {e}")
        return []
