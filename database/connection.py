"""
ATLAS Database Connection Pool
================================
Async connection pool using asyncpg for TimescaleDB.
"""

from __future__ import annotations

import asyncpg
from loguru import logger

from config.settings import settings

_pool: asyncpg.Pool | None = None


async def init_pool(min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """Initialize and return the global connection pool."""
    global _pool
    if _pool is None:
        logger.info("Initialising TimescaleDB connection pool …")
        _pool = await asyncpg.create_pool(
            dsn=settings.db_dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
        )
        logger.success("TimescaleDB pool ready (min={}, max={})", min_size, max_size)
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Return the existing pool, initialising it if necessary."""
    if _pool is None:
        return await init_pool()
    return _pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("TimescaleDB connection pool closed.")


async def execute(query: str, *args) -> str:
    """Execute a single query (INSERT / UPDATE / DELETE / DDL)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args) -> list[asyncpg.Record]:
    """Fetch multiple rows."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetchrow(query: str, *args) -> asyncpg.Record | None:
    """Fetch a single row."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetchval(query: str, *args):
    """Fetch a single scalar value."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)
