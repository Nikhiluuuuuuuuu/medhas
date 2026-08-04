"""High-performance asyncpg connection pool manager."""

import asyncpg
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from config import settings
from utils import logger, measure_latency
from core.exceptions import DatabaseConnectionError

class DatabasePool:
    """Singleton asyncpg Connection Pool Manager."""
    
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the connection pool with optimized settings.

        Idempotent and loop-safe: if the process previously created a pool bound to a
        CLOSED event loop (e.g. a prior pytest session, or pytest-asyncio re-creating the
        loop), it is discarded and rebuilt on the current loop. This prevents the
        "pool attached to a different loop" / "connection was closed" failures that occur
        when a singleton asyncpg pool outlives the event loop that created it.
        """
        import asyncio
        try:
            cur_loop = asyncio.get_running_loop()
        except RuntimeError:
            cur_loop = None
        if cls._pool is not None:
            # If the existing pool is bound to a dead/foreign loop, drop it so we rebuild.
            bound_loop = getattr(cls._pool, "_loop", None)
            if bound_loop is None or bound_loop.is_closed() or (
                cur_loop is not None and bound_loop is not cur_loop
            ):
                try:
                    await cls._pool.close()
                except Exception:
                    pass
                cls._pool = None
        if cls._pool is None:
            try:
                async with measure_latency("DatabasePool.initialize"):
                    cls._pool = await asyncpg.create_pool(
                        dsn=settings.database_url,
                        min_size=settings.DB_POOL_MIN_SIZE,
                        max_size=settings.DB_POOL_MAX_SIZE,
                        timeout=settings.DB_POOL_TIMEOUT,
                        command_timeout=60.0,
                    )
                logger.info("✅ PostgreSQL connection pool initialized successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize database connection pool: {e}")
                raise DatabaseConnectionError(f"Database pool initialization error: {e}")

    @classmethod
    async def close(cls) -> None:
        """Close all connections in the pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("Closed PostgreSQL connection pool.")

    @classmethod
    @asynccontextmanager
    async def acquire(cls) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool."""
        if cls._pool is None:
            await cls.initialize()
        
        assert cls._pool is not None, "Database pool failed to initialize"
        async with cls._pool.acquire() as connection:
            yield connection


async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Helper generator to acquire DB connection."""
    async with DatabasePool.acquire() as conn:
        yield conn
