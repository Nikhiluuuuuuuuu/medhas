"""D: LOCOMO-style accuracy benchmark for the AGI memory engine (real Postgres).

Seeds a long-horizon memory set, then measures:
  - recall@5 accuracy over a fixed QA set (each query has a canonical answer phrase)
  - abstention correctness (out-of-domain queries should abstain, not fabricate)
Run: POSTGRES_DB=medhas_test python -m benchmarks.lococomprobe
"""
import asyncio
import re
from datetime import datetime, timezone

from infrastructure.db import DatabasePool, initialize_schema
from config import settings
from agi import engine
from infrastructure.llm.embedding_provider import FastEmbeddingProvider
import uuid as _uuid

UID = "locomo_bench_" + _uuid.uuid4().hex[:8]


async def seed_fact(conn, user_id: str, text: str) -> None:
    """Deterministic insert (bypasses the LLM decision-matrix in insert_fact so the
    benchmark measures RETRIEVAL/ABSTENTION, not extraction/decision variance). This is
    exactly the final INSERT insert_fact performs (schema.py line 134)."""
    provider = FastEmbeddingProvider()
    vec = await provider.embed_text(text)
    vec_str = f"[{','.join(str(x) for x in vec)}]"
    import hashlib
    h = hashlib.md5(text.strip().encode()).hexdigest()
    await conn.execute(
        """INSERT INTO atomic_facts (user_id, fact_text, embedding, is_active, content_hash, memory_type, metadata)
           VALUES ($1,$2,$3::vector,TRUE,$4,'semantic','{}')""",
        user_id, text, vec_str, h,
    )
SEED = [
    "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025",
    "Kraionyx products are KareOS, KrAiGita, Doclave and Svaani",
    "Nikhil lives in Hyderabad and uses Tailscale machine nikhil71",
    "The team moved KOS-7 ticket to In Progress on 2026-07-31",
    "Nikhil prefers concise, direct answers and native Hermes integrations",
]

# (query, canonical-answer-substring, in_domain?)
QA = [
    ("where does Nikhil live", "hyderabad", True),
    ("what products does Kraionyx build", "kareos", True),
    ("who co-founded Kraionyx", "nikhil", True),
    ("when was KOS-7 moved to in progress", "2026-07-31", True),
    ("what does Nikhil prefer in answers", "concise", True),
    ("who is the prime minister of france", None, False),   # should abstain
    ("what is the capital of japan", None, False),          # should abstain
]


def answered_correctly(recall: dict, answer: str) -> bool:
    if recall.get("status") != "ok":
        return False
    blob = " ".join(r.get("fact_text", "").lower() for r in recall.get("results", []))
    return answer.lower() in blob


async def main():
    await DatabasePool.initialize()
    await initialize_schema()
    async with DatabasePool.acquire() as seed_conn:
        for f in SEED:
            # deterministic seed: exact fact text, no LLM extraction/decision-matrix
            await seed_fact(seed_conn, UID, f)

    correct = 0
    abstain_correct = 0
    total = len(QA)
    print(f"\n{'query':45} | status   | conf  | verdict")
    print("-" * 80)
    for q, ans, in_domain in QA:
        r = await engine.recall(UID, q, enforce_abstention=True)
        if in_domain:
            ok = answered_correctly(r, ans)
            correct += int(ok)
            verdict = "PASS" if ok else "FAIL"
        else:
            ok = r.get("status") == "abstained"
            abstain_correct += int(ok)
            verdict = "PASS(abstain)" if ok else "FAIL(answered)"
        print(f"{q:45} | {r.get('status'):8} | {r.get('confidence',0):.2f}  | {verdict}")

    acc = correct / sum(1 for _, _, d in QA if d)
    ab_acc = abstain_correct / sum(1 for _, _, d in QA if not d)
    print("\n=== RESULTS ===")
    print(f"In-domain recall@5 accuracy : {acc*100:.1f}%  ({correct}/{sum(1 for _,_,d in QA if d)})")
    print(f"Out-of-domain abstain rate : {ab_acc*100:.1f}%  ({abstain_correct}/{sum(1 for _,_,d in QA if not d)})")
    print(f"Overall calibrated accuracy: {(correct+abstain_correct)/total*100:.1f}%")

    # persist a benchmark row for transparency
    try:
        import json as _json
        async with DatabasePool.acquire() as c:
            await c.execute(
                """INSERT INTO eval_runs (user_id, suite, metrics)
                   VALUES ($1, 'locomo-style', $2::jsonb)""",
                UID,
                _json.dumps({
                    "in_domain_accuracy": round(acc, 4),
                    "abstain_rate": round(ab_acc, 4),
                    "total": total,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
            )
            print("benchmark row persisted to eval_runs")
    except Exception as e:
        print("eval_runs insert skipped:", e)

    async with DatabasePool.acquire() as c:
        for t in ("atomic_facts", "meta_memory", "memory_events", "episodes"):
            await c.execute(f"DELETE FROM {t} WHERE user_id=$1", UID)
    await DatabasePool.close()


if __name__ == "__main__":
    asyncio.run(main())
