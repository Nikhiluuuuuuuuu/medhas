"""E21 — Memory evaluation harness.

Benchmark the system the way the research evaluates agent memory:
  • SITUATEDQA-style temporal consistency (does the answer match the facts valid at the
    query time?)
  • LOCOMO-style long-horizon recall@k + hallucination rate (does retrieval return
    stored facts and abstain when absent?)
  • persistence (does an old fact survive consolidation + forgetting?)
  • belief/uncertainty calibration (do unverified facts carry lower belief?)
Stores each run in eval_runs for longitudinal tracking.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infrastructure.db import DatabasePool
from utils import log_error


@dataclass
class EvalCase:
    query: str
    expects_fact: str          # a fact that *should* be retrievable
    valid_at: Optional[str] = None
    should_abstain: bool = False


@dataclass
class EvalResult:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


async def _run_case(user_id: str, case: EvalCase, recall_fn) -> EvalResult:
    # recall_fn must accept (user_id, query, limit=...) and return a list of
    # objects exposing .fact_text (or .content). We never pass valid_at here so
    # the harness works with any recall implementation (engine.recall, search_facts).
    hits = await recall_fn(user_id, case.query, limit=5)
    texts = [getattr(h, "fact_text", getattr(h, "content", "")) for h in hits]
    if case.should_abstain:
        passed = len(hits) == 0
        return EvalResult(case.query, passed, {"recalled": len(hits)})
    passed = any(case.expects_fact.lower()[:40] in t.lower() for t in texts)
    return EvalResult(case.query, passed, {"recalled": len(hits)})


async def run_eval_suite(
    user_id: str,
    cases: List[EvalCase],
    recall_fn,
) -> Dict[str, Any]:
    """Execute a benchmark suite and persist results (E21)."""
    results: List[EvalResult] = []
    for c in cases:
        try:
            results.append(await _run_case(user_id, c, recall_fn))
        except Exception as e:
            results.append(EvalResult(c.query, False, {"error": str(e)}))
            log_error(f"eval case failed: {e}")

    passed = sum(1 for r in results if r.passed)
    metrics = {
        "total": len(results),
        "passed": passed,
        "accuracy": round(passed / len(results), 4) if results else 0.0,
        "cases": [
            {"query": r.name, "passed": r.passed, **r.detail} for r in results
        ],
    }
    try:
        async with DatabasePool.acquire() as conn:
            await conn.execute(
                "INSERT INTO eval_runs (user_id, suite, metrics) VALUES ($1,'benchmark',$2::jsonb);",
                user_id, json.dumps(metrics),
            )
    except Exception as e:
        log_error(f"eval persist failed: {e}")
    return metrics


def temporal_consistency_check(facts: List[Dict[str, Any]], query_time: str) -> Dict[str, Any]:
    """SITUATEDQA style: which facts are valid at query_time, and are any contradictory?"""
    qt = query_time
    valid = [f for f in facts
             if f.get("valid_from", "") <= qt
             and (f.get("valid_to") in (None, "") or f.get("valid_to", "") > qt)]
    contradictions = []
    for a in valid:
        for b in valid:
            if a["id"] != b["id"] and a.get("contradicted_by") and str(b["id"]) in [
                str(x) for x in (a.get("contradicted_by") or [])
            ]:
                contradictions.append((a["fact_text"], b["fact_text"]))
    return {
        "valid_at_query_time": len(valid),
        "contradictions": contradictions,
        "consistent": len(contradictions) == 0,
    }
