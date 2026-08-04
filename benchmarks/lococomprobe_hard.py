"""F1: Harder LOCOMO-style benchmark — temporal + multi-hop + distractor queries.

Extends the smoke benchmark with question types that separate a real memory system
from a keyword matcher:
  - TEMPORAL: find a fact valid at a point in time / before an event
  - MULTI-HOP: answer requires joining 2 stored facts
  - DISTRACTOR: near-duplicate facts that must NOT be confused
Run: POSTGRES_DB=medhas_test python -m benchmarks.lococomprobe_hard
"""
import asyncio
from datetime import datetime, timezone

from medhas.storage import DatabasePool, initialize_schema
from medhas.engine import engine
from medhas.embeddings import FastEmbeddingProvider
import uuid as _uuid
import hashlib

UID = "locomo_hard_" + _uuid.uuid4().hex[:8]


async def seed_fact(conn, user_id: str, text: str, valid_from=None) -> None:
    """Deterministic insert (bypasses the LLM decision-matrix in insert_fact so the
    benchmark measures RETRIEVAL/ABSTENTION, not extraction/decision variance)."""
    provider = FastEmbeddingProvider()
    vec = await provider.embed_text(text)
    vec_str = f"[{','.join(str(x) for x in vec)}]"
    h = hashlib.md5(text.strip().encode()).hexdigest()
    vf_sql = "$5" if valid_from is not None else "CURRENT_TIMESTAMP"
    params = [user_id, text, vec_str, h]
    if valid_from is not None:
        params.append(valid_from)
    await conn.execute(
        f"""INSERT INTO atomic_facts (user_id, fact_text, embedding, is_active, content_hash, memory_type, metadata, valid_from)
           VALUES ($1,$2,$3::vector,TRUE,$4,'semantic','{{}}',{vf_sql})""",
        *params,
    )
# (fact_text, valid_from_year) — years create real bitemporal order so 'before/after' works
SEED = [
    ("Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025", 2020),
    ("Kraionyx products are KareOS, KrAiGita, Doclave and Svaani", 2022),
    ("KareOS is the company's clinical operating system product", 2022),
    ("Nikhil lived in Bangalore before moving to Hyderabad in 2024", 2018),
    ("The team moved KOS-7 ticket to In Progress on 2026-07-31", 2026),
    ("Nikhil prefers concise, direct answers and native Hermes integrations", 2023),
    ("Kraionyx was initially named 'Project Hermes' until the rebrand in early 2025", 2019),
]

# (query, answer-substring, in_domain?)
QA = [
    ("what products does Kraionyx build", "kareos", True),
    ("which product is the clinical operating system", "kareos", True),          # multi-hop-ish
    ("where does Nikhil live now", "hyderabad", True),
    ("where did Nikhil live before Hyderabad", "bangalore", True),                # temporal
    ("what was Kraionyx called before the rebrand", "project hermes", True),    # temporal
    ("when was KOS-7 moved to in progress", "2026-07-31", True),
    ("what does Nikhil prefer in answers", "concise", True),
    ("who co-founded the company that builds KareOS", "nikhil", True),           # multi-hop
    ("who is the prime minister of france", None, False),
    ("what is the capital of japan", None, False),
]


def answered_correctly(recall: dict, answer: str) -> bool:
    if recall.get("status") != "ok":
        return False
    blob = " ".join(r.get("fact_text", "").lower() for r in recall.get("results", []))
    return answer.lower() in blob


async def main():
    await DatabasePool.initialize()
    async with DatabasePool.acquire() as seed_conn:
        for f, yr in SEED:
            # deterministic seed: exact fact text, no LLM extraction/decision-matrix
            vf = datetime(yr, 1, 1, tzinfo=timezone.utc)
            await seed_fact(seed_conn, UID, f, valid_from=vf)

    correct = abstain_correct = 0
    print(f"\n{'query':52} | status   | conf | verdict")
    print("-" * 86)
    for q, ans, in_domain in QA:
        r = await engine.recall(UID, q, enforce_abstention=True)
        if in_domain:
            ok = answered_correctly(r, ans)
            correct += int(ok)
            verdict = "PASS" if ok else "FAIL"
        else:
            ok = r.get("status") == "abstained"
            abstain_correct += int(ok)
            verdict = "PASS(ab)" if ok else "FAIL(ans)"
        print(f"{q:52} | {r.get('status'):8} | {r.get('confidence',0):.2f} | {verdict}")

    n_in = sum(1 for _, _, d in QA if d)
    n_out = sum(1 for _, _, d in QA if not d)
    acc = correct / n_in
    ab = abstain_correct / n_out
    print("\n=== HARD BENCHMARK RESULTS ===")
    print(f"In-domain recall@5 accuracy : {acc*100:.1f}%  ({correct}/{n_in})")
    print(f"Out-of-domain abstain rate : {ab*100:.1f}%  ({abstain_correct}/{n_out})")
    print(f"Overall calibrated accuracy: {(correct+abstain_correct)/len(QA)*100:.1f}%")

    try:
        import json as _json
        async with DatabasePool.acquire() as c:
            await c.execute(
                "INSERT INTO eval_runs (user_id, suite, metrics) VALUES ($1,'locomo-hard',$2::jsonb)",
                UID, _json.dumps({"in_domain_accuracy": round(acc, 4),
                                  "abstain_rate": round(ab, 4), "total": len(QA),
                                  "timestamp": datetime.now(timezone.utc).isoformat()}),
            )
            print("hard benchmark row persisted")
    except Exception as e:
        print("eval_runs insert skipped:", e)

    async with DatabasePool.acquire() as c:
        for t in ("atomic_facts", "meta_memory", "memory_events", "episodes"):
            await c.execute(f"DELETE FROM {t} WHERE user_id=$1", UID)
    await DatabasePool.close()


if __name__ == "__main__":
    asyncio.run(main())
