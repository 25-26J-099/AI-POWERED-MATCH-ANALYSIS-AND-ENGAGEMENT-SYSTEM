"""Async SQLAlchemy engine and session factory."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
_db_init_lock = asyncio.Lock()
_db_initialized = False


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


async def init_db():
    """Create all tables (dev convenience; use Alembic in production)."""
    global _db_initialized

    # Import models before create_all so SQLAlchemy metadata is populated.
    from app.models import models  # noqa: F401

    async with _db_init_lock:
        if _db_initialized:
            return

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _db_initialized = True


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields an async session."""
    await init_db()

    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
