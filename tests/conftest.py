"""Pytest configuration for Medhas memory-engine regression suite.

Uses a dedicated test database (medhas_test) so the suite never touches dev
data. Connection settings come from config.settings, overridden here to point
at the test DB.

Schema initialization is session-scoped (NOT per-test): every test module used
to declare its own function-scoped autouse fixture that called initialize_schema(),
and under pytest-asyncio the module fixtures ran concurrently against the shared
asyncpg pool, causing `Schema DDL execution failed: cannot perform operation:
another operation is in progress` cascades. Now there is ONE place that runs DDL,
guarded by a global lock, and the pool is opened once per session and closed once.
"""

import os
import sys

import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the suite at the dedicated test database so it never touches dev data.
# The suite now runs ONLINE (no MEDHAS_OFFLINE): every LLM-backed path exercises the
# real Groq provider, matching production. A valid GROQ_API_KEY must be present in the
# environment / .env for the LLM-dependent tests to pass.
os.environ.setdefault("POSTGRES_DB", "medhas_test")
os.environ.setdefault("GROQ_API_KEY", "test_key_placeholder")

from infrastructure.db import DatabasePool, initialize_schema  # noqa: E402

# ---- Single session-scoped schema init --------------------------------
# Previously every test module declared its own function-scoped autouse fixture that
# called initialize_schema(), and under pytest-asyncio those ran concurrently against
# the shared asyncpg pool, causing `cannot perform operation: another operation is in
# progress` cascades. Now DDL runs exactly once, in a session-scoped fixture, before any
# test executes. initialize_schema() itself uses a single acquire, so there is no
# concurrent-DDL risk.


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _session_db():
    await DatabasePool.initialize()
    await initialize_schema()
    yield
    await DatabasePool.close()


@pytest.fixture
def user_id():
    import uuid
    return f"pytest_{uuid.uuid4().hex}"
