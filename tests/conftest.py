"""Pytest configuration for Medhas memory-engine regression suite.

Uses a dedicated test database (medhas_test) so the suite never touches dev
data. Connection settings come from config.settings, overridden here to point
at the test DB.
"""
import os
import sys
import pytest
import pytest_asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force the test database before importing the app config.
os.environ.setdefault("POSTGRES_DB", "medhas_test")
os.environ.setdefault("GROQ_API_KEY", "test_key_placeholder")

from infrastructure.db import DatabasePool, initialize_schema  # noqa: E402


@pytest_asyncio.fixture(scope="function", autouse=True)
async def _db():
    await DatabasePool.initialize()
    await initialize_schema()
    yield
    await DatabasePool.close()


@pytest.fixture
def user_id():
    import uuid
    return f"pytest_{uuid.uuid4().hex}"
