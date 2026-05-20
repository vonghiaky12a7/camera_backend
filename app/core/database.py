# backend/app/core/database.py
# =============================================================================
# SQLAlchemy async engine and session factory targeting the Odoo PostgreSQL DB.
#
# Design notes:
#   - Uses asyncpg driver (postgresql+asyncpg DSN) for non-blocking I/O inside
#     FastAPI and Celery async tasks.
#   - pgvector's Vector type is registered globally via `pgvector.sqlalchemy`.
#   - A single engine + sessionmaker is created at import time (module singleton).
#     FastAPI lifespan disposes the pool cleanly on shutdown.
#   - get_db() is an async generator suitable for FastAPI Depends().
#   - get_db_session() is a plain async context manager for use in Celery tasks
#     (which cannot use FastAPI dependency injection).
# =============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Declarative base – shared by all SQLAlchemy models in this project
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Project-wide declarative base. Import this in every model file."""

    pass


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def _build_engine() -> AsyncEngine:
    db_url_str = str(settings.odoo_database_url)

    # Nếu đang chạy bằng Celery Worker thì không dùng Pool để tránh cạn kiệt Connection
    if settings.service_role in ("worker", "beat"):
        return create_async_engine(db_url_str, echo=False, poolclass=NullPool)

    # Dành cho FastAPI (API Role)
    return create_async_engine(
        db_url_str,
        echo=False,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )


engine: AsyncEngine = _build_engine()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep loaded attributes accessible after commit
    autoflush=False,  # Explicit flush gives us control over write timing
    autocommit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency  →  use with `Depends(get_db)`
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an AsyncSession for use in FastAPI route handlers.

    Example::

        @router.get("/partners/{partner_id}")
        async def read_partner(
            partner_id: int,
            db: AsyncSession = Depends(get_db),
        ):
            result = await db.get(ResPartner, partner_id)
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Celery / script helper  →  use as `async with get_db_session() as db:`
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside FastAPI (Celery tasks, CLI scripts).

    Example::

        async with get_db_session() as db:
            result = await db.execute(select(ResPartner).limit(5))
            partners = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Lifespan helpers called from app/main.py
# ---------------------------------------------------------------------------
async def connect_db() -> None:
    """Validate the DB connection at application startup."""
    from sqlalchemy import text

    async with engine.connect() as conn:
        row = await conn.execute(text("SELECT version()"))
        version = row.scalar_one()
        logger.info("Odoo DB connection OK: %s", version)

        # Verify pgvector extension is installed
        row = await conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        ext = row.scalar_one_or_none()
        if ext is None:
            raise RuntimeError(
                "pgvector extension is NOT installed in the Odoo database. "
                "Run: CREATE EXTENSION IF NOT EXISTS vector;"
            )
        logger.info("pgvector extension found: v%s", ext)


async def disconnect_db() -> None:
    """Dispose the connection pool on application shutdown."""
    await engine.dispose()
    logger.info("Odoo DB connection pool disposed.")
