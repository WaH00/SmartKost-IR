import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Engine async untuk operasi CRUD via FastAPI ---
# pool_pre_ping=True: cek koneksi mati sebelum digunakan (cegah idle timeout error)
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,   # Log SQL query di mode debug (matikan di production)
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# --- Session factory ---
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Kelas dasar untuk seluruh SQLAlchemy ORM model aplikasi."""
    pass


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injector untuk async database session.
    Menjamin session di-commit, di-rollback jika error, dan ditutup
    setelah setiap request HTTP selesai (Resource cleanup otomatis).

    Penggunaan di route handler:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(select(KosListing))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database session error: {exc}")
            raise
        finally:
            await session.close()