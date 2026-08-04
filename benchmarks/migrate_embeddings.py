"""F3/F4: Embedding model upgrade migration — bge-small (384) -> bge-base-en-v1.5 (768).

Safe, idempotent, runnable against the live Medhas DB (unified_memory / medhas_test):
  1. ALTER every vector(384) column to vector(768) (IF NOT EXISTS-style via exception guard).
  2. NULL out stale embeddings so no 384-dim vector remains in a 768-dim column.
  3. Re-embed all atomic_facts, graph_nodes, archival_memory, messages from their text.
  4. Report counts.

Run: POSTGRES_DB=medhas_test python -m benchmarks.migrate_embeddings
"""
import asyncio
from typing import List, Tuple

from medhas.storage import DatabasePool, initialize_schema
from medhas.embeddings import FastEmbeddingProvider
from medhas.utils import logger


# (table, text_column, id_column) — all embedding-bearing tables
TABLES: List[Tuple[str, str, str]] = [
    ("atomic_facts", "fact_text", "id"),
    ("graph_nodes", "name", "id"),
    ("archival_memory", "content", "id"),
    ("messages", "content", "id"),
    ("percept_buffer", "raw_caption", "id"),
]


async def _alter_and_null(conn, table: str) -> None:
    # 1) NULL stale vectors FIRST so the ALTER doesn't try to cast old 384-dim vecs to 768
    try:
        await conn.execute(f"UPDATE {table} SET embedding = NULL WHERE embedding IS NOT NULL;")
    except Exception as e:
        logger.warning(f"null {table}.embedding skipped: {e}")
    # 2) widen the column (no rows to cast now -> safe). Already-768 is a no-op.
    try:
        await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector(768);")
    except Exception as e:
        logger.warning(f"alter {table}.embedding skipped: {e}")


async def main() -> None:
    await DatabasePool.initialize()
    await initialize_schema()  # also runs agi_schema (now 768)

    provider = FastEmbeddingProvider()  # lazy-loads bge-base-en-v1.5 (768)
    print(f"Embedding model: {provider.model_name}")

    async with DatabasePool.acquire() as conn:
        for table, text_col, id_col in TABLES:
            # does the table + column exist?
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name=$1 AND column_name='embedding'",
                table,
            )
            if not exists:
                print(f"skip {table} (no embedding column)")
                continue
            try:
                await _alter_and_null(conn, table)
            except Exception as e:
                print(f"alter/null {table} error: {e}")

            rows = await conn.fetch(
                f"SELECT {id_col}, {text_col} FROM {table} WHERE {text_col} IS NOT NULL"
            )
            ids = [r[id_col] for r in rows]
            texts = [r[text_col] for r in rows]
            if not texts:
                print(f"{table}: 0 rows, nothing to embed")
                continue
            # embed in batches
            batch = 256
            done = 0
            for i in range(0, len(texts), batch):
                chunk_ids = ids[i:i + batch]
                chunk_txt = texts[i:i + batch]
                vecs = await provider.embed_batch(chunk_txt)
                for rid, vec in zip(chunk_ids, vecs):
                    vector_str = f"[{','.join(str(x) for x in vec)}]"
                    await conn.execute(
                        f"UPDATE {table} SET embedding = $1::vector WHERE {id_col} = $2",
                        vector_str, rid,
                    )
                done += len(chunk_txt)
            print(f"{table}: re-embedded {done} rows -> vector(768)")

    # sanity: any non-null embedding with wrong dim?
    async with DatabasePool.acquire() as conn:
        bad = 0
        for table, _, id_col in TABLES:
            try:
                n = await conn.fetchval(
                    f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL "
                    f"AND octet_length(embedding::text) > 0 AND vector_dims(embedding) <> 768"
                )
                bad += int(n or 0)
            except Exception:
                pass
        print(f"rows with wrong dim remaining: {bad}")
    await DatabasePool.close()
    print("migration complete")


if __name__ == "__main__":
    asyncio.run(main())
