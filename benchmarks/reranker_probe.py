"""B: measure reranker contribution + diagnose the 'products' recall failure.

Runs the core search_facts path (what engine.recall uses) with and without the
cross-encoder reranker, on a seeded fact set, and reports:
  - top-1/recall@5 with reranker ON vs OFF
  - the similarity + rrf scores so we can see if FACT_SIMILARITY_THRESHOLD (0.70)
    is silently dropping valid hits
Run: POSTGRES_DB=medhas_test python benchmarks/reranker_probe.py
"""
import asyncio
import json
from datetime import datetime, timezone

from medhas.storage import DatabasePool, initialize_schema
from medhas.config import settings
from medhas.memory.atomic import search_facts
from medhas.engine import engine


UID = "rerank_user"
FACTS = [
    "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
    "Kraionyx products are KareOS, KrAiGita, Doclave and Svaani",
    "Nikhil prefers concise, direct answers and native Hermes integrations",
    "The team moved KOS-7 ticket to In Progress on 2026-07-31",
    "Nikhil lives in Hyderabad and uses Tailscale machine nikhil71",
]
QUERIES = {
    "products": "which products does Kraionyx build",
    "cofound": "what company did Nikhil co-found",
    "live": "where does Nikhil live",
}


async def main():
    await DatabasePool.initialize()
    await initialize_schema()
    for f in FACTS:
        await engine.remember(UID, f, source="user")

    for label, q in QUERIES.items():
        # reranker ON (current config)
        on = await search_facts(UID, q, limit=5)
        # reranker OFF (force fusion only)
        settings.FACT_RERANKER_ENABLED = False
        off = await search_facts(UID, q, limit=5)
        settings.FACT_RERANKER_ENABLED = True

        def fmt(rs):
            return [(round(float(r.similarity), 3), round(float(r.rrf_score), 3),
                     (r.fact_text[:40])) for r in rs]

        print(f"\n### {label}: {q}")
        print("  reranker ON :", fmt(on)[:3])
        print("  reranker OFF:", fmt(off)[:3])

    async with DatabasePool.acquire() as c:
        for t in ("atomic_facts", "episodes"):
            await c.execute(f"DELETE FROM {t} WHERE user_id=$1", UID)
    await DatabasePool.close()


if __name__ == "__main__":
    asyncio.run(main())
