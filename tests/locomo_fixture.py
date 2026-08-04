"""LOCOMO-style benchmark dataset for Medhas (zero-LLM, deterministic).

A small synthetic multi-session corpus inspired by the LOCOMO long-term
conversational-memory benchmark. It is seeded directly into atomic_facts via
SQL (no Groq), with facts spread across three "sessions" at different valid_from
times so temporal queries are meaningful.

The eval harness in agi/eval.py runs these cases against search_facts (the
deterministic retrieval layer) — no abstention gate, stable for CI.

Each CASE: (query, expects_substring_lower, kind)
  kind in: single | multihop | temporal | contradiction
"""

# Three sessions, each a few facts. valid_from ISO dates.
SESSIONS = {
    "S1": "2024-01-10",   # early: company founding
    "S2": "2024-06-15",   # mid: product + team
    "S3": "2025-03-20",   # late: pivot / update
}

FACTS = [
    # S1 — founding
    ("S1", "Nikhil co-founded Kraionyx AI with 4 friends in Hyderabad in 2025", 1.0),
    ("S1", "Kraionyx AI was started to build clinical decision-support software", 1.0),
    # S2 — products + team
    ("S2", "Kraionyx builds KareOS as its flagship hospital operating system", 1.0),
    ("S2", "KrAiGita is a Sanskrit learning assistant made by Kraionyx", 1.0),
    ("S2", "Priya leads the KareOS engineering team at Kraionyx", 1.0),
    ("S2", "Rahul handles go-to-market for Kraionyx products", 1.0),
    # S3 — pivot / update (contradiction of an earlier fact over time)
    ("S3", "Kraionyx renamed KrAiGita to Svaani, a broader language tutor", 0.9),
    ("S3", "Kraionyx opened a second office in Bangalore in 2025", 0.9),
    ("S3", "Doclave became Kraionyx's medical transcription product", 0.9),
    # temporal-only fact (precedes S1)
    ("S0", "Nikhil studied computer science at a university in 2019", 1.0),
]

# Eval cases: query -> expected substring (lowercased, first 40 chars matched)
CASES = [
    # single-hop
    ("who co-founded Kraionyx AI", "nikhil co-founded kraionyx", "single"),
    ("what is KareOS", "kareos", "single"),
    ("who leads the KareOS engineering team", "priya", "single"),
    # multi-hop (entity bridging across facts)
    ("what product does the company founded by Nikhil build", "kareos", "multihop"),
    ("who works on go-to-market at the company that makes KrAiGita", "rahul", "multihop"),
    # temporal (pre-S1)
    ("what did Nikhil do before founding Kraionyx", "studied computer science", "temporal"),
    # contradiction / update (latest value should win in retrieval)
    ("what is KrAiGita called now", "svaani", "contradiction"),
    ("where did Kraionyx open a second office", "bangalore", "single"),
    # product line
    ("what does Doclave do", "medical transcription", "single"),
    ("name a Sanskrit learning product from Kraionyx", "svaani", "multihop"),
]
